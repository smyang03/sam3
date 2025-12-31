#!/usr/bin/env python3
"""
프레임별 라벨 누락 분석 스크립트

YOLO 형식 라벨 파일을 분석하여:
1. 프레임 중간에 전체 라벨이 누락되는 경우 감지
2. 특정 클래스 라벨만 누락되는 경우 감지
3. 누락 패턴 통계 및 시각화
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import json


def parse_yolo_label(label_path):
    """YOLO 라벨 파일 파싱

    Returns:
        list of dict: [{'class_id': int, 'bbox': [x_center, y_center, w, h]}, ...]
    """
    annotations = []

    if not os.path.exists(label_path):
        return None  # 파일 자체가 없음

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]

            annotations.append({
                'class_id': class_id,
                'bbox': bbox
            })

        return annotations

    except Exception as e:
        print(f"오류 - {label_path}: {e}")
        return None


def analyze_label_directory(label_dir, class_names=None):
    """라벨 디렉토리 분석

    Args:
        label_dir: YOLO 라벨 파일들이 있는 디렉토리
        class_names: 클래스 ID to 이름 매핑 dict (optional)

    Returns:
        dict: 분석 결과
    """
    label_dir = Path(label_dir)
    label_files = sorted(label_dir.glob('*.txt'))

    if not label_files:
        return {
            'error': f'라벨 파일을 찾을 수 없습니다: {label_dir}'
        }

    # 프레임별 정보 수집
    frames_info = []
    missing_file_frames = []
    empty_frames = []
    class_counts_per_frame = []
    all_classes_seen = set()

    for idx, label_file in enumerate(label_files):
        frame_name = label_file.stem
        annotations = parse_yolo_label(label_file)

        if annotations is None:
            # 파일이 없거나 읽기 실패
            missing_file_frames.append({
                'index': idx,
                'name': frame_name,
                'path': str(label_file)
            })
            frames_info.append({
                'index': idx,
                'name': frame_name,
                'status': 'missing',
                'num_objects': 0,
                'classes': []
            })
            class_counts_per_frame.append({})
            continue

        if len(annotations) == 0:
            # 빈 라벨 파일 (객체 없음)
            empty_frames.append({
                'index': idx,
                'name': frame_name,
                'path': str(label_file)
            })
            frames_info.append({
                'index': idx,
                'name': frame_name,
                'status': 'empty',
                'num_objects': 0,
                'classes': []
            })
            class_counts_per_frame.append({})
            continue

        # 정상 프레임
        class_ids = [ann['class_id'] for ann in annotations]
        class_count = defaultdict(int)
        for cid in class_ids:
            class_count[cid] += 1
            all_classes_seen.add(cid)

        frames_info.append({
            'index': idx,
            'name': frame_name,
            'status': 'ok',
            'num_objects': len(annotations),
            'classes': list(class_count.keys())
        })
        class_counts_per_frame.append(dict(class_count))

    # 누락 패턴 분석
    total_frames = len(label_files)
    missing_count = len(missing_file_frames)
    empty_count = len(empty_frames)
    ok_count = total_frames - missing_count - empty_count

    # 중간 프레임 누락 감지 (연속된 정상 프레임 사이에 누락/빈 프레임이 있는 경우)
    middle_missing_patterns = []

    for i in range(1, len(frames_info) - 1):
        prev_status = frames_info[i-1]['status']
        curr_status = frames_info[i]['status']
        next_status = frames_info[i+1]['status']

        if curr_status in ['missing', 'empty'] and prev_status == 'ok' and next_status == 'ok':
            middle_missing_patterns.append({
                'index': i,
                'name': frames_info[i]['name'],
                'type': curr_status,
                'prev': frames_info[i-1]['name'],
                'next': frames_info[i+1]['name']
            })

    # 특정 클래스 누락 패턴 분석
    class_missing_patterns = analyze_class_missing_patterns(
        frames_info, class_counts_per_frame, all_classes_seen
    )

    # 결과 정리
    result = {
        'summary': {
            'total_frames': total_frames,
            'ok_frames': ok_count,
            'empty_frames': empty_count,
            'missing_file_frames': missing_count,
            'all_classes_seen': sorted(list(all_classes_seen))
        },
        'missing_files': missing_file_frames,
        'empty_frames': empty_frames,
        'middle_missing_patterns': middle_missing_patterns,
        'class_missing_patterns': class_missing_patterns,
        'frames_info': frames_info
    }

    return result


def analyze_class_missing_patterns(frames_info, class_counts_per_frame, all_classes):
    """특정 클래스가 중간 프레임에서 누락되는 패턴 분석"""

    patterns = []

    for class_id in all_classes:
        # 각 클래스별로 프레임 시퀀스 확인
        class_presence = []  # True: 존재, False: 없음

        for frame_info, class_count in zip(frames_info, class_counts_per_frame):
            if frame_info['status'] != 'ok':
                class_presence.append(None)  # 프레임 자체가 누락/빈 경우
            else:
                class_presence.append(class_id in class_count)

        # 중간 누락 감지: True - False - True 패턴
        missing_ranges = []
        for i in range(1, len(class_presence) - 1):
            if (class_presence[i-1] == True and
                class_presence[i] == False and
                class_presence[i+1] == True):

                missing_ranges.append({
                    'frame_index': i,
                    'frame_name': frames_info[i]['name'],
                    'prev_count': class_counts_per_frame[i-1].get(class_id, 0),
                    'next_count': class_counts_per_frame[i+1].get(class_id, 0)
                })

        if missing_ranges:
            patterns.append({
                'class_id': class_id,
                'missing_occurrences': len(missing_ranges),
                'missing_details': missing_ranges
            })

    return patterns


def print_analysis_report(result, class_names=None):
    """분석 결과 출력"""

    print("\n" + "="*70)
    print("프레임별 라벨 누락 분석 보고서")
    print("="*70)

    if 'error' in result:
        print(f"\n오류: {result['error']}")
        return

    summary = result['summary']

    # 요약 정보
    print(f"\n[요약]")
    print(f"  총 프레임 수: {summary['total_frames']}")
    print(f"  정상 프레임: {summary['ok_frames']} ({summary['ok_frames']/summary['total_frames']*100:.1f}%)")
    print(f"  빈 프레임 (객체 없음): {summary['empty_frames']} ({summary['empty_frames']/summary['total_frames']*100:.1f}%)")
    print(f"  누락 파일: {summary['missing_file_frames']} ({summary['missing_file_frames']/summary['total_frames']*100:.1f}%)")
    print(f"  감지된 클래스: {summary['all_classes_seen']}")

    # 중간 프레임 전체 누락
    middle_missing = result['middle_missing_patterns']
    print(f"\n[중간 프레임 전체 누락]")
    if middle_missing:
        print(f"  감지된 패턴 수: {len(middle_missing)}")
        for pattern in middle_missing[:10]:  # 최대 10개만 표시
            print(f"    - 프레임 {pattern['index']} ({pattern['name']}): {pattern['type']}")
            print(f"      이전: {pattern['prev']}, 다음: {pattern['next']}")
        if len(middle_missing) > 10:
            print(f"    ... 외 {len(middle_missing) - 10}개")
    else:
        print("  감지 안됨 ✓")

    # 특정 클래스 누락
    class_missing = result['class_missing_patterns']
    print(f"\n[특정 클래스 중간 누락]")
    if class_missing:
        for pattern in class_missing:
            class_id = pattern['class_id']
            class_name = class_names.get(class_id, f"class_{class_id}") if class_names else f"class_{class_id}"
            print(f"  클래스 {class_id} ({class_name}): {pattern['missing_occurrences']}번 누락")

            for detail in pattern['missing_details'][:5]:  # 최대 5개만 표시
                print(f"    - 프레임 {detail['frame_index']} ({detail['frame_name']})")
                print(f"      이전 프레임 객체 수: {detail['prev_count']}, 다음 프레임 객체 수: {detail['next_count']}")

            if len(pattern['missing_details']) > 5:
                print(f"    ... 외 {len(pattern['missing_details']) - 5}개")
    else:
        print("  감지 안됨 ✓")

    # 빈 프레임 상세
    if result['empty_frames']:
        print(f"\n[빈 프레임 상세 (최대 10개)]")
        for empty in result['empty_frames'][:10]:
            print(f"  - 프레임 {empty['index']}: {empty['name']}")
        if len(result['empty_frames']) > 10:
            print(f"  ... 외 {len(result['empty_frames']) - 10}개")

    # 누락 파일 상세
    if result['missing_files']:
        print(f"\n[누락 파일 상세]")
        for missing in result['missing_files']:
            print(f"  - 프레임 {missing['index']}: {missing['name']}")

    print("\n" + "="*70)


def save_analysis_json(result, output_path):
    """분석 결과를 JSON으로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n분석 결과 저장됨: {output_path}")


def parse_class_mapping(class_str):
    """클래스 매핑 문자열 파싱

    Format: "name1:0,name2:1,name3:2"
    """
    if not class_str:
        return None

    mapping = {}
    pairs = class_str.split(',')

    for pair in pairs:
        pair = pair.strip()
        if ':' in pair:
            name, idx = pair.split(':')
            mapping[int(idx.strip())] = name.strip()

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description='YOLO 라벨 파일의 프레임별 누락 분석',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 분석
  python check_missing_labels.py --label-dir ./labels

  # 클래스 이름 포함 분석
  python check_missing_labels.py --label-dir ./labels --classes "person:0,car:1,dog:2"

  # JSON 결과 저장
  python check_missing_labels.py --label-dir ./labels --output analysis.json
        """
    )

    parser.add_argument('--label-dir', '-d', required=True,
                        help='YOLO 라벨 파일이 있는 디렉토리')
    parser.add_argument('--classes', '-c', type=str,
                        help='클래스 매핑 (예: "person:0,car:1,dog:2")')
    parser.add_argument('--output', '-o', type=str,
                        help='결과를 저장할 JSON 파일 경로')

    args = parser.parse_args()

    # 클래스 매핑 파싱
    class_names = parse_class_mapping(args.classes)

    # 분석 실행
    print(f"\n라벨 디렉토리 분석 중: {args.label_dir}")
    result = analyze_label_directory(args.label_dir, class_names)

    # 결과 출력
    print_analysis_report(result, class_names)

    # JSON 저장
    if args.output:
        save_analysis_json(result, args.output)

    # 종료 코드
    if 'error' in result:
        return 1

    # 누락이 발견되면 경고 코드 반환
    has_issues = (
        result['middle_missing_patterns'] or
        result['class_missing_patterns'] or
        result['missing_files']
    )

    return 2 if has_issues else 0


if __name__ == '__main__':
    exit(main())
