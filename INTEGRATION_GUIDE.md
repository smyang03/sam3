# 라벨 검증 통합 가이드

## 개요

`label_validation.py`를 `sam3_offline.py`에 통합하여 실시간으로 라벨 누락을 감지하고 경고하는 방법

---

## 1. 빠른 시작

### 1.1 label_validation.py를 sam3_offline.py에 통합

`sam3_offline.py` 상단에 import 추가:

```python
# sam3_offline.py 상단
from label_validation import (
    FrameLabelValidator,
    validate_class_mapping_complete
)
```

### 1.2 main() 함수 수정

```python
def main():
    args = parse_args()

    # ... (기존 코드)

    # 클래스 매핑 검증 (처리 시작 전)
    try:
        validate_class_mapping_complete(prompts, class_mapping)
        print("✓ 클래스 매핑 검증 완료")
    except ValueError as e:
        print(f"❌ 클래스 매핑 오류: {e}")
        return 1

    # 검증기 초기화
    validator = FrameLabelValidator(
        expected_classes=prompts,
        warning_threshold=0.5  # 50% 이하로 감소하면 경고
    )

    # ... (기존 처리 코드)

    # 처리 루프
    for idx, image_path in enumerate(image_files):
        # ... (기존 처리)

        result = process_single_image_batch(...)

        if result['success']:
            # 검증 수행
            is_valid, warnings = validator.validate_frame(
                frame_idx=idx,
                frame_name=Path(image_path).stem,
                results_by_prompt=results_by_prompt,  # process_single_image_batch에서 반환 필요
                class_mapping=class_mapping
            )

            # 경고 출력
            for warning in warnings:
                print(f"⚠️  {warning}")

    # 처리 완료 후 요약
    validator.print_summary()
```

---

## 2. process_single_image_batch() 수정

results_by_prompt를 반환하도록 수정:

```python
def process_single_image_batch(...):
    # ... (기존 코드)

    return {
        'success': True,
        'num_objects': num_objects,
        'results_by_prompt': results_by_prompt,  # ← 추가
        'image_size': (img_width, img_height),
        # ... (나머지)
    }
```

---

## 3. 통합 예제 (완전한 코드)

### 3.1 전체 통합 코드

```python
# sam3_offline.py에 추가할 코드

import sys
from pathlib import Path

# label_validation 임포트
sys.path.insert(0, str(Path(__file__).parent))
from label_validation import (
    FrameLabelValidator,
    validate_class_mapping_complete,
    detect_interpolatable_frames
)


def main():
    args = parse_args()

    # === 기존 초기화 코드 ===
    device = setup_device(args)
    model_cfg = get_model_config(args)
    model = load_model(model_cfg, args.checkpoint, device)
    postprocessor = create_postprocessor(args)

    prompts = args.prompts.split(',')
    class_mapping = parse_class_mapping(args.classes)

    # === 클래스 매핑 검증 (새로 추가) ===
    try:
        validate_class_mapping_complete(prompts, class_mapping)
        print("✓ 클래스 매핑 검증 완료")
        print(f"  프롬프트: {prompts}")
        print(f"  매핑: {class_mapping}")
    except ValueError as e:
        print(f"❌ 클래스 매핑 오류:\n{e}")
        return 1

    # === 검증기 초기화 (새로 추가) ===
    validator = FrameLabelValidator(
        expected_classes=prompts,
        warning_threshold=args.warning_threshold if hasattr(args, 'warning_threshold') else 0.5
    )
    print(f"✓ 라벨 검증기 초기화 (경고 임계값: {validator.warning_threshold})")

    # === 이미지 파일 수집 ===
    image_files = collect_image_files(args)
    print(f"총 {len(image_files)}개 이미지 처리 예정")

    # === 처리 결과 저장 (보간 감지용) ===
    all_results = []

    # === 이미지 처리 루프 ===
    for idx, image_path in enumerate(image_files):
        print(f"\n[{idx+1}/{len(image_files)}] {image_path}")

        # 이미지 처리
        result = process_single_image_batch(
            image_path=image_path,
            prompts=prompts,
            class_mapping=class_mapping,
            model=model,
            postprocessor=postprocessor,
            device=device,
            output_dir=args.output_dir,
            # ... (기타 인자)
        )

        all_results.append(result)

        if not result['success']:
            print(f"  ❌ 처리 실패: {result.get('error', 'Unknown')}")
            continue

        # === 라벨 검증 (새로 추가) ===
        if 'results_by_prompt' in result:
            is_valid, warnings = validator.validate_frame(
                frame_idx=idx,
                frame_name=Path(image_path).stem,
                results_by_prompt=result['results_by_prompt'],
                class_mapping=class_mapping
            )

            # 경고 출력
            if warnings:
                for warning in warnings:
                    print(f"  ⚠️  {warning}")
            else:
                print(f"  ✓ 검증 통과 ({result['num_objects']}개 객체)")

        # 진행 상황
        if (idx + 1) % 10 == 0:
            temp_summary = validator.get_summary()
            print(f"\n--- 중간 요약 ({idx+1}개 처리) ---")
            print(f"  빈 프레임: {temp_summary['empty_frames']} ({temp_summary['empty_rate']:.1%})")

    # === 처리 완료 후 요약 ===
    print("\n" + "="*70)
    print("처리 완료")
    print("="*70)

    validator.print_summary()

    # === 보간 가능 프레임 감지 (선택) ===
    if args.detect_interpolatable:
        interpolatable = detect_interpolatable_frames(all_results)

        if interpolatable:
            print(f"\n보간 가능 프레임: {len(interpolatable)}개")
            for item in interpolatable[:10]:
                print(f"  프레임 {item['frame_index']} "
                      f"(신뢰도: {item['confidence']:.2f})")

            # 보간 후보 저장
            import json
            with open('interpolatable_frames.json', 'w') as f:
                json.dump(interpolatable, f, indent=2)
            print("  → interpolatable_frames.json에 저장됨")

    return 0


if __name__ == '__main__':
    sys.exit(main())
```

---

## 4. 커맨드라인 인자 추가

`parse_args()` 함수에 검증 옵션 추가:

```python
def parse_args():
    parser = argparse.ArgumentParser()

    # ... (기존 인자)

    # 라벨 검증 옵션
    parser.add_argument('--warning-threshold', type=float, default=0.5,
                        help='객체 수 감소 경고 임계값 (기본: 0.5)')
    parser.add_argument('--detect-interpolatable', action='store_true',
                        help='보간 가능 프레임 감지 활성화')
    parser.add_argument('--validation-summary', type=str,
                        help='검증 요약을 저장할 JSON 파일 경로')

    return parser.parse_args()
```

---

## 5. 로깅 설정 (권장)

경고를 로그 파일로도 저장:

```python
import logging

def setup_logging(log_file='label_validation.log'):
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# main()에서 사용
logger = setup_logging()
logger.info("라벨 검증 시작")

# 경고 출력 시
for warning in warnings:
    logger.warning(warning)
```

---

## 6. 사용 예시

### 6.1 기본 사용

```bash
python sam3_offline.py \
  --checkpoint ./weights/sam3.pth \
  --prompts "person,car,bicycle" \
  --classes "person:0,car:1,bicycle:2" \
  --input-dir ./frames \
  --output-dir ./labels
```

**출력**:
```
✓ 클래스 매핑 검증 완료
  프롬프트: ['person', 'car', 'bicycle']
  매핑: {'person': 0, 'car': 1, 'bicycle': 2}
✓ 라벨 검증기 초기화 (경고 임계값: 0.5)

[1/100] ./frames/frame_0000.jpg
  ✓ 검증 통과 (5개 객체)

[2/100] ./frames/frame_0001.jpg
  ✓ 검증 통과 (5개 객체)

[3/100] ./frames/frame_0002.jpg
  ⚠️  Frame 2 (frame_0002): 'person' 클래스 누락 (이전: 2개)
  처리 완료 (3개 객체)

...

=====================================================================
라벨 검증 요약
=====================================================================

총 처리 프레임: 100
빈 프레임: 5 (5.0%)
중간 누락 패턴: 8

클래스별 통계:
  person:
    총 감지: 180
    누락 프레임: 8 (8.0%)
  car:
    총 감지: 95
    누락 프레임: 3 (3.0%)
  bicycle:
    총 감지: 12
    누락 프레임: 15 (15.0%)
=====================================================================
```

### 6.2 보간 감지 활성화

```bash
python sam3_offline.py \
  --checkpoint ./weights/sam3.pth \
  --prompts "person,car" \
  --classes "person:0,car:1" \
  --input-dir ./frames \
  --output-dir ./labels \
  --detect-interpolatable \
  --validation-summary validation_report.json
```

**출력**:
```
보간 가능 프레임: 3개
  프레임 10 (신뢰도: 0.85)
  프레임 25 (신뢰도: 0.92)
  프레임 67 (신뢰도: 0.78)
  → interpolatable_frames.json에 저장됨
```

---

## 7. 검증 요약 JSON 저장

`validation_summary.json` 예시:

```json
{
  "total_frames": 100,
  "empty_frames": 5,
  "empty_rate": 0.05,
  "middle_missing_patterns": 8,
  "class_statistics": {
    "person": {
      "missing_frames": 8,
      "missing_rate": 0.08,
      "total_detections": 180
    },
    "car": {
      "missing_frames": 3,
      "missing_rate": 0.03,
      "total_detections": 95
    }
  }
}
```

---

## 8. 고급 사용

### 8.1 임계값 조정

객체 수가 30% 이하로 감소하면 경고:

```bash
python sam3_offline.py ... --warning-threshold 0.3
```

### 8.2 클래스 매핑 오류 발견 예시

잘못된 매핑:
```bash
python sam3_offline.py \
  --prompts "person,car,dog" \
  --classes "person:0,car:1"  # dog 누락!
```

**출력**:
```
❌ 클래스 매핑 오류:
다음 프롬프트가 class_mapping에 없습니다: ['dog']
현재 매핑: {'person': 0, 'car': 1}
모든 프롬프트: ['person', 'car', 'dog']
```

---

## 9. 후속 작업

### 9.1 누락 프레임 재처리

`check_missing_labels.py`로 누락 프레임 목록 생성 후 재처리:

```bash
# 1. 누락 분석
python check_missing_labels.py \
  --label-dir ./labels \
  --output missing_analysis.json

# 2. 누락 프레임만 추출하여 재처리 (낮은 threshold로)
# (별도 스크립트 작성 필요)
```

### 9.2 통계 시각화

```python
import json
import matplotlib.pyplot as plt

# 검증 요약 로드
with open('validation_report.json') as f:
    summary = json.load(f)

# 클래스별 누락률 시각화
classes = list(summary['class_statistics'].keys())
missing_rates = [
    summary['class_statistics'][c]['missing_rate']
    for c in classes
]

plt.bar(classes, missing_rates)
plt.ylabel('Missing Rate')
plt.title('Class-wise Label Missing Rate')
plt.savefig('missing_rate.png')
```

---

## 10. 문제 해결

### Q1: "module 'label_validation' not found"

**A**: `label_validation.py`가 `sam3_offline.py`와 같은 디렉토리에 있는지 확인

### Q2: 너무 많은 경고가 출력됨

**A**: `--warning-threshold` 값을 낮추거나 (예: 0.3), 로깅 레벨 조정:

```python
logging.getLogger('label_validation').setLevel(logging.ERROR)
```

### Q3: 특정 클래스만 자주 누락됨

**A**: 해당 클래스의 프롬프트나 detection_threshold 조정 필요.
클래스별 threshold를 다르게 설정하는 것도 고려.

---

## 11. 참고 자료

- `label_validation.py` - 검증 유틸리티 구현
- `check_missing_labels.py` - 오프라인 라벨 분석 도구
- `LABEL_MISSING_ANALYSIS.md` - 누락 패턴 분석 보고서
