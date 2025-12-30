#!/usr/bin/env python3
"""실시간 검증 데모 (numpy 없이)"""

import sys
from pathlib import Path

# label_validation에서 필요한 부분만 간단히 구현
class FrameLabelValidator:
    def __init__(self, expected_classes, warning_threshold=0.5):
        self.expected_classes = set(expected_classes)
        self.warning_threshold = warning_threshold
        self.total_frames = 0
        self.empty_frames = []
        self.class_missing_stats = {}
        self.class_total_detections = {}
        self.prev_results = None
        self.middle_missing_frames = []

    def validate_frame(self, frame_idx, frame_name, results_by_prompt, class_mapping):
        warnings = []
        self.total_frames += 1

        # 1. 전체 객체 수 확인
        total_objects = sum(
            len(result.get('boxes', [])) if result else 0
            for result in results_by_prompt.values()
        )

        if total_objects == 0:
            self.empty_frames.append({'index': frame_idx, 'name': frame_name})
            if self.prev_results and self._prev_had_objects():
                warnings.append(
                    f"Frame {frame_idx} ({frame_name}): "
                    f"전체 라벨 누락 (이전 프레임에는 객체 존재)"
                )

        # 2. 클래스별 누락 확인
        if self.prev_results:
            for class_name in self.expected_classes:
                prev_count = len(self.prev_results.get(class_name, {}).get('boxes', []))
                curr_count = len(results_by_prompt.get(class_name, {}).get('boxes', []))

                if prev_count > 0 and curr_count == 0:
                    warnings.append(
                        f"Frame {frame_idx} ({frame_name}): "
                        f"'{class_name}' 클래스 누락 (이전: {prev_count}개)"
                    )
                    self.class_missing_stats[class_name] = \
                        self.class_missing_stats.get(class_name, 0) + 1

        # 통계 업데이트
        for class_name, result in results_by_prompt.items():
            if result and 'boxes' in result:
                count = len(result['boxes'])
                self.class_total_detections[class_name] = \
                    self.class_total_detections.get(class_name, 0) + count

        self.prev_results = results_by_prompt.copy()
        return len(warnings) == 0, warnings

    def _prev_had_objects(self):
        if not self.prev_results:
            return False
        total = sum(
            len(result.get('boxes', [])) if result else 0
            for result in self.prev_results.values()
        )
        return total > 0

    def print_summary(self):
        print("\n" + "="*70)
        print("라벨 검증 요약")
        print("="*70)
        print(f"\n총 처리 프레임: {self.total_frames}")
        print(f"빈 프레임: {len(self.empty_frames)} "
              f"({len(self.empty_frames)/self.total_frames*100:.1f}%)")

        print("\n클래스별 통계:")
        for class_name in sorted(self.expected_classes):
            total = self.class_total_detections.get(class_name, 0)
            missing = self.class_missing_stats.get(class_name, 0)
            rate = missing / self.total_frames * 100 if self.total_frames > 0 else 0
            print(f"  {class_name}:")
            print(f"    총 감지: {total}개")
            print(f"    누락 프레임: {missing}번 ({rate:.1f}%)")
        print("="*70)


# 테스트 시뮬레이션
def simulate_processing():
    print("="*70)
    print("실시간 검증 데모 (sam3_offline.py 통합 시뮬레이션)")
    print("="*70)

    # 검증기 초기화
    validator = FrameLabelValidator(
        expected_classes=['person', 'car', 'bicycle'],
        warning_threshold=0.5
    )

    class_mapping = {'person': 0, 'car': 1, 'bicycle': 2}

    # 시뮬레이션 데이터 (프레임 처리 결과)
    test_frames = [
        # frame 0: 정상
        {'person': {'boxes': [[100, 100, 200, 200], [300, 300, 400, 400]]},
         'car': {'boxes': [[500, 500, 600, 600]]}},

        # frame 1: 정상
        {'person': {'boxes': [[105, 105, 205, 205], [305, 305, 405, 405]]},
         'car': {'boxes': [[505, 505, 605, 605]]}},

        # frame 2: person 클래스 누락!
        {'person': {'boxes': []},
         'car': {'boxes': [[510, 510, 610, 610]]}},

        # frame 3: 정상
        {'person': {'boxes': [[110, 110, 210, 210], [310, 310, 410, 410]]},
         'car': {'boxes': [[515, 515, 615, 615]]}},

        # frame 4: 전체 누락!
        {'person': {'boxes': []},
         'car': {'boxes': []}},

        # frame 5: 정상 + bicycle 추가
        {'person': {'boxes': [[115, 115, 215, 215]]},
         'car': {'boxes': [[520, 520, 620, 620]]},
         'bicycle': {'boxes': [[700, 700, 800, 800]]}},
    ]

    print("\n처리 시작...\n")

    # 프레임 처리 시뮬레이션
    for idx, frame_data in enumerate(test_frames):
        frame_name = f'frame_{idx:04d}'

        # 프레임 처리 (실제로는 process_single_image_batch 결과)
        total_objs = sum(len(r['boxes']) for r in frame_data.values())
        print(f"[{idx+1}/{len(test_frames)}] {frame_name}.jpg")
        print(f"  처리 완료: {total_objs}개 객체 감지")

        # ← 실시간 검증!
        is_valid, warnings = validator.validate_frame(
            frame_idx=idx,
            frame_name=frame_name,
            results_by_prompt=frame_data,
            class_mapping=class_mapping
        )

        # 경고 출력
        if warnings:
            for w in warnings:
                print(f"  ⚠️  {w}")
        else:
            print(f"  ✓ 검증 통과")
        print()

    # 처리 완료 후 요약
    print("\n처리 완료!\n")
    validator.print_summary()


if __name__ == '__main__':
    simulate_processing()
