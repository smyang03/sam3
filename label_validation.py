#!/usr/bin/env python3
"""
라벨 검증 및 누락 감지 유틸리티

sam3_offline.py에 통합하여 실시간으로 라벨 누락을 감지하고 경고
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


logger = logging.getLogger(__name__)


class FrameLabelValidator:
    """프레임별 라벨 검증 클래스"""

    def __init__(self, expected_classes: List[str], warning_threshold: float = 0.5):
        """
        Args:
            expected_classes: 예상되는 클래스 이름 리스트
            warning_threshold: 객체 수가 이전 프레임 대비 이 비율 이하로 떨어지면 경고
        """
        self.expected_classes = set(expected_classes)
        self.warning_threshold = warning_threshold

        # 통계 수집
        self.total_frames = 0
        self.empty_frames = []
        self.class_missing_stats = defaultdict(int)
        self.class_total_detections = defaultdict(int)
        self.prev_results = None

        # 중간 누락 패턴
        self.middle_missing_frames = []

    def validate_frame(
        self,
        frame_idx: int,
        frame_name: str,
        results_by_prompt: Dict,
        class_mapping: Dict[str, int]
    ) -> Tuple[bool, List[str]]:
        """
        프레임 라벨 검증

        Returns:
            (is_valid, warnings): 유효성 여부와 경고 메시지 리스트
        """
        warnings = []
        self.total_frames += 1

        # 1. 클래스 매핑 검증
        unmapped = self._check_class_mapping(results_by_prompt, class_mapping)
        if unmapped:
            warnings.append(
                f"Frame {frame_idx} ({frame_name}): "
                f"다음 클래스가 class_mapping에 없음: {unmapped}"
            )

        # 2. 전체 객체 수 확인
        total_objects = sum(
            len(result.get('boxes', [])) if result else 0
            for result in results_by_prompt.values()
        )

        if total_objects == 0:
            self.empty_frames.append({
                'index': frame_idx,
                'name': frame_name
            })

            # 이전 프레임에 객체가 있었다면 경고
            if self.prev_results and self._prev_had_objects():
                warnings.append(
                    f"Frame {frame_idx} ({frame_name}): "
                    f"전체 라벨 누락 (이전 프레임에는 객체 존재)"
                )
                self.middle_missing_frames.append({
                    'index': frame_idx,
                    'name': frame_name,
                    'type': 'all_missing'
                })

        # 3. 클래스별 누락 확인
        class_warnings = self._check_class_missing(
            frame_idx, frame_name, results_by_prompt
        )
        warnings.extend(class_warnings)

        # 4. 급격한 객체 수 감소 확인
        if self.prev_results:
            decrease_warnings = self._check_sudden_decrease(
                frame_idx, frame_name, results_by_prompt
            )
            warnings.extend(decrease_warnings)

        # 통계 업데이트
        self._update_statistics(results_by_prompt)

        # 현재 결과 저장
        self.prev_results = results_by_prompt.copy()

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def _check_class_mapping(
        self,
        results_by_prompt: Dict,
        class_mapping: Dict[str, int]
    ) -> List[str]:
        """클래스 매핑 확인"""
        unmapped = []

        for prompt_name in results_by_prompt.keys():
            if prompt_name not in class_mapping:
                unmapped.append(prompt_name)

        return unmapped

    def _prev_had_objects(self) -> bool:
        """이전 프레임에 객체가 있었는지 확인"""
        if not self.prev_results:
            return False

        total = sum(
            len(result.get('boxes', [])) if result else 0
            for result in self.prev_results.values()
        )
        return total > 0

    def _check_class_missing(
        self,
        frame_idx: int,
        frame_name: str,
        results_by_prompt: Dict
    ) -> List[str]:
        """특정 클래스 누락 확인"""
        warnings = []

        if not self.prev_results:
            return warnings

        for class_name in self.expected_classes:
            # 이전 프레임에서 이 클래스가 있었는지
            prev_count = len(
                self.prev_results.get(class_name, {}).get('boxes', [])
            )

            # 현재 프레임에서 이 클래스 수
            curr_count = len(
                results_by_prompt.get(class_name, {}).get('boxes', [])
            )

            # 이전에는 있었는데 지금은 없음 → 누락 의심
            if prev_count > 0 and curr_count == 0:
                warnings.append(
                    f"Frame {frame_idx} ({frame_name}): "
                    f"'{class_name}' 클래스 누락 (이전: {prev_count}개)"
                )
                self.class_missing_stats[class_name] += 1

                self.middle_missing_frames.append({
                    'index': frame_idx,
                    'name': frame_name,
                    'type': 'class_missing',
                    'class': class_name,
                    'prev_count': prev_count
                })

        return warnings

    def _check_sudden_decrease(
        self,
        frame_idx: int,
        frame_name: str,
        results_by_prompt: Dict
    ) -> List[str]:
        """급격한 객체 수 감소 확인"""
        warnings = []

        if not self.prev_results:
            return warnings

        for class_name in self.expected_classes:
            prev_count = len(
                self.prev_results.get(class_name, {}).get('boxes', [])
            )
            curr_count = len(
                results_by_prompt.get(class_name, {}).get('boxes', [])
            )

            if prev_count == 0:
                continue

            # 객체 수가 threshold 이하로 감소
            decrease_ratio = curr_count / prev_count
            if decrease_ratio < self.warning_threshold and curr_count > 0:
                warnings.append(
                    f"Frame {frame_idx} ({frame_name}): "
                    f"'{class_name}' 급격한 감소 "
                    f"({prev_count} → {curr_count}, {decrease_ratio:.1%})"
                )

        return warnings

    def _update_statistics(self, results_by_prompt: Dict):
        """통계 업데이트"""
        for class_name, result in results_by_prompt.items():
            if result and 'boxes' in result:
                count = len(result['boxes'])
                self.class_total_detections[class_name] += count

    def get_summary(self) -> Dict:
        """검증 통계 요약"""
        empty_rate = (
            len(self.empty_frames) / self.total_frames
            if self.total_frames > 0 else 0
        )

        class_missing_rate = {}
        for class_name in self.expected_classes:
            missing = self.class_missing_stats.get(class_name, 0)
            rate = missing / self.total_frames if self.total_frames > 0 else 0
            class_missing_rate[class_name] = {
                'missing_frames': missing,
                'missing_rate': rate,
                'total_detections': self.class_total_detections.get(class_name, 0)
            }

        return {
            'total_frames': self.total_frames,
            'empty_frames': len(self.empty_frames),
            'empty_rate': empty_rate,
            'middle_missing_patterns': len(self.middle_missing_frames),
            'class_statistics': class_missing_rate
        }

    def print_summary(self):
        """요약 출력"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("라벨 검증 요약")
        print("="*70)
        print(f"\n총 처리 프레임: {summary['total_frames']}")
        print(f"빈 프레임: {summary['empty_frames']} ({summary['empty_rate']:.1%})")
        print(f"중간 누락 패턴: {summary['middle_missing_patterns']}")

        print("\n클래스별 통계:")
        for class_name, stats in summary['class_statistics'].items():
            print(f"  {class_name}:")
            print(f"    총 감지: {stats['total_detections']}")
            print(f"    누락 프레임: {stats['missing_frames']} ({stats['missing_rate']:.1%})")

        print("="*70)


def validate_class_mapping_complete(
    prompts: List[str],
    class_mapping: Dict[str, int]
) -> bool:
    """
    클래스 매핑이 완전한지 검증 (처리 시작 전 호출)

    Args:
        prompts: 사용할 프롬프트 리스트
        class_mapping: 클래스 이름 → ID 매핑

    Returns:
        True if valid, raises ValueError otherwise
    """
    unmapped = [p for p in prompts if p not in class_mapping]

    if unmapped:
        raise ValueError(
            f"다음 프롬프트가 class_mapping에 없습니다: {unmapped}\n"
            f"현재 매핑: {class_mapping}\n"
            f"모든 프롬프트: {prompts}"
        )

    # 역매핑도 확인 (중복 ID)
    id_to_class = defaultdict(list)
    for name, idx in class_mapping.items():
        id_to_class[idx].append(name)

    duplicates = {idx: names for idx, names in id_to_class.items() if len(names) > 1}
    if duplicates:
        raise ValueError(
            f"중복된 클래스 ID 발견: {duplicates}"
        )

    return True


def detect_interpolatable_frames(
    all_results: List[Dict],
    iou_threshold: float = 0.3
) -> List[Dict]:
    """
    보간 가능한 누락 프레임 감지

    빈 프레임이 정상 프레임 사이에 끼어있고, 이전/다음 프레임의 객체가
    위치적으로 유사하면 보간 후보로 추천

    Args:
        all_results: 모든 프레임의 처리 결과 리스트
        iou_threshold: 이전/다음 프레임 객체 간 최소 IoU

    Returns:
        보간 후보 프레임 리스트
    """
    interpolatable = []

    for i in range(1, len(all_results) - 1):
        curr = all_results[i]
        prev = all_results[i-1]
        next_frame = all_results[i+1]

        # 현재 프레임이 비어있고, 이전/다음은 정상
        if (curr.get('num_objects', 0) == 0 and
            prev.get('num_objects', 0) > 0 and
            next_frame.get('num_objects', 0) > 0):

            # 클래스별로 이전/다음 객체 비교
            can_interpolate = _check_interpolatable(prev, next_frame, iou_threshold)

            if can_interpolate:
                interpolatable.append({
                    'frame_index': i,
                    'prev_frame': i - 1,
                    'next_frame': i + 1,
                    'confidence': can_interpolate['confidence']
                })

    return interpolatable


def _check_interpolatable(
    prev_result: Dict,
    next_result: Dict,
    iou_threshold: float
) -> Optional[Dict]:
    """
    이전/다음 프레임이 보간 가능한지 확인

    간단한 버전: 클래스별 객체 수가 같으면 보간 가능으로 판정
    """
    # 간단한 휴리스틱: 클래스 구성이 유사하면 보간 가능
    prev_classes = set(prev_result.get('results_by_prompt', {}).keys())
    next_classes = set(next_result.get('results_by_prompt', {}).keys())

    common_classes = prev_classes & next_classes

    if not common_classes:
        return None

    # 공통 클래스가 있으면 보간 가능
    confidence = len(common_classes) / max(len(prev_classes), len(next_classes))

    return {
        'confidence': confidence,
        'common_classes': list(common_classes)
    }


if __name__ == '__main__':
    # 테스트 예제
    validator = FrameLabelValidator(
        expected_classes=['person', 'car', 'bicycle'],
        warning_threshold=0.5
    )

    # 시뮬레이션 데이터
    test_frames = [
        {'person': {'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]]}, 'car': {'boxes': [[500, 500, 600, 600]]}},
        {'person': {'boxes': [[105, 105, 205, 205], [305, 305, 405, 405]]}, 'car': {'boxes': [[505, 505, 605, 605]]}},
        {'person': {'boxes': []}, 'car': {'boxes': [[510, 510, 610, 610]]}},  # person 누락
        {'person': {'boxes': [[110, 110, 210, 210], [310, 310, 410, 410]]}, 'car': {'boxes': [[515, 515, 615, 615]]}},
    ]

    class_mapping = {'person': 0, 'car': 1, 'bicycle': 2}

    for idx, frame_data in enumerate(test_frames):
        is_valid, warnings = validator.validate_frame(
            frame_idx=idx,
            frame_name=f'frame_{idx:04d}',
            results_by_prompt=frame_data,
            class_mapping=class_mapping
        )

        if warnings:
            for w in warnings:
                print(f"⚠️  {w}")

    validator.print_summary()
