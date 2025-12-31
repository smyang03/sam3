# 라벨 누락 해결책 구현 매핑

## 제공된 해결책과 실제 구현의 연결

---

## ✅ 1. 실시간 검증: 처리 중 즉시 누락 감지

### 구현 위치: `label_validation.py`

#### 핵심 클래스: `FrameLabelValidator`

```python
# label_validation.py:18-70
class FrameLabelValidator:
    """프레임별 라벨 검증 클래스"""

    def validate_frame(
        self,
        frame_idx: int,
        frame_name: str,
        results_by_prompt: Dict,
        class_mapping: Dict[str, int]
    ) -> Tuple[bool, List[str]]:
        """
        프레임 라벨 검증 - 실시간으로 누락 감지

        Returns:
            (is_valid, warnings): 유효성 여부와 경고 메시지 리스트
        """
```

### 사용 방법: `sam3_offline.py`에 통합

```python
# INTEGRATION_GUIDE.md 참고
# sam3_offline.py의 처리 루프에 추가

# 1. 검증기 초기화 (처리 시작 전 1회)
validator = FrameLabelValidator(
    expected_classes=['person', 'car', 'bicycle'],
    warning_threshold=0.5
)

# 2. 각 프레임 처리 후 즉시 검증
for idx, image_path in enumerate(image_files):
    result = process_single_image_batch(...)  # 기존 처리

    # ← 여기서 실시간 검증!
    is_valid, warnings = validator.validate_frame(
        frame_idx=idx,
        frame_name=Path(image_path).stem,
        results_by_prompt=result['results_by_prompt'],
        class_mapping=class_mapping
    )

    # 즉시 경고 출력
    for warning in warnings:
        print(f"⚠️  {warning}")
        # 또는 로그 파일에 기록
        logger.warning(warning)
```

### 실제 동작 예시

**프레임 처리 중 실시간 출력**:
```
[1/100] ./frames/frame_0000.jpg
  처리 완료: 5개 객체
  ✓ 검증 통과

[2/100] ./frames/frame_0001.jpg
  처리 완료: 5개 객체
  ✓ 검증 통과

[3/100] ./frames/frame_0002.jpg
  처리 완료: 3개 객체
  ⚠️  Frame 2 (frame_0002): 'person' 클래스 누락 (이전: 2개)
  ⚠️  Frame 2 (frame_0002): 'car' 급격한 감소 (2 → 1, 50%)
```

→ **프레임 처리 직후 바로 문제 발견!**

---

## ✅ 2. 사전 검증: 클래스 매핑 오류 미리 차단

### 구현 위치: `label_validation.py:255-288`

#### 핵심 함수: `validate_class_mapping_complete()`

```python
def validate_class_mapping_complete(
    prompts: List[str],
    class_mapping: Dict[str, int]
) -> bool:
    """
    클래스 매핑이 완전한지 검증 (처리 시작 전 호출)

    검증 내용:
    1. 모든 프롬프트가 class_mapping에 있는지
    2. 클래스 ID 중복이 없는지

    Returns:
        True if valid, raises ValueError otherwise
    """
    # 1. 누락된 프롬프트 확인
    unmapped = [p for p in prompts if p not in class_mapping]

    if unmapped:
        raise ValueError(
            f"다음 프롬프트가 class_mapping에 없습니다: {unmapped}\n"
            f"현재 매핑: {class_mapping}\n"
            f"모든 프롬프트: {prompts}"
        )

    # 2. 중복 ID 확인
    id_to_class = defaultdict(list)
    for name, idx in class_mapping.items():
        id_to_class[idx].append(name)

    duplicates = {idx: names for idx, names in id_to_class.items() if len(names) > 1}
    if duplicates:
        raise ValueError(f"중복된 클래스 ID 발견: {duplicates}")

    return True
```

### 사용 방법: 처리 시작 전 호출

```python
# sam3_offline.py의 main() 함수 시작 부분

def main():
    args = parse_args()

    # 프롬프트와 클래스 매핑 파싱
    prompts = args.prompts.split(',')  # ['person', 'car', 'bicycle']
    class_mapping = parse_class_mapping(args.classes)  # {'person': 0, 'car': 1}

    # ← 여기서 사전 검증! (이미지 처리 전)
    try:
        validate_class_mapping_complete(prompts, class_mapping)
        print("✓ 클래스 매핑 검증 완료")
    except ValueError as e:
        print(f"❌ 클래스 매핑 오류:\n{e}")
        return 1  # 프로그램 종료

    # 이제 안전하게 처리 시작
    for image_path in image_files:
        ...
```

### 실제 동작 예시

#### ❌ 잘못된 입력 (프롬프트 누락)
```bash
python sam3_offline.py \
  --prompts "person,car,dog" \
  --classes "person:0,car:1"
```

**즉시 오류 발생 (이미지 처리 전)**:
```
❌ 클래스 매핑 오류:
다음 프롬프트가 class_mapping에 없습니다: ['dog']
현재 매핑: {'person': 0, 'car': 1}
모든 프롬프트: ['person', 'car', 'dog']

→ 프로그램 종료 (시간 낭비 방지!)
```

#### ❌ 잘못된 입력 (ID 중복)
```bash
python sam3_offline.py \
  --prompts "person,car" \
  --classes "person:0,car:0"
```

**즉시 오류 발생**:
```
❌ 클래스 매핑 오류:
중복된 클래스 ID 발견: {0: ['person', 'car']}

→ 프로그램 종료
```

#### ✅ 올바른 입력
```bash
python sam3_offline.py \
  --prompts "person,car,dog" \
  --classes "person:0,car:1,dog:2"
```

**성공**:
```
✓ 클래스 매핑 검증 완료
  프롬프트: ['person', 'car', 'dog']
  매핑: {'person': 0, 'car': 1, 'dog': 2}

→ 처리 시작
```

---

## ✅ 3. 상세 로깅: 프레임/클래스별 누락 기록

### 구현 위치: `label_validation.py`

#### A. 프레임별 누락 기록

```python
# label_validation.py:118-136
def _check_class_missing(self, frame_idx, frame_name, results_by_prompt):
    """특정 클래스 누락 확인"""
    warnings = []

    for class_name in self.expected_classes:
        prev_count = len(self.prev_results.get(class_name, {}).get('boxes', []))
        curr_count = len(results_by_prompt.get(class_name, {}).get('boxes', []))

        # 이전에는 있었는데 지금은 없음 → 누락!
        if prev_count > 0 and curr_count == 0:
            # ← 여기서 상세 기록
            warning_msg = (
                f"Frame {frame_idx} ({frame_name}): "
                f"'{class_name}' 클래스 누락 (이전: {prev_count}개)"
            )
            warnings.append(warning_msg)

            # 통계에 기록
            self.class_missing_stats[class_name] += 1

            # 중간 누락 패턴 저장
            self.middle_missing_frames.append({
                'index': frame_idx,
                'name': frame_name,
                'type': 'class_missing',
                'class': class_name,
                'prev_count': prev_count
            })

    return warnings
```

#### B. 로그 파일 저장

```python
# INTEGRATION_GUIDE.md:193-209 참고
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('label_validation.log', encoding='utf-8'),  # ← 파일 저장
        logging.StreamHandler()  # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

# 경고 발생 시 자동으로 파일에 기록
for warning in warnings:
    logger.warning(warning)
```

### 실제 로그 파일 예시: `label_validation.log`

```
2025-12-30 10:15:23 [INFO] 라벨 검증 시작
2025-12-30 10:15:23 [INFO] 클래스 매핑 검증 완료
2025-12-30 10:15:30 [WARNING] Frame 12 (frame_0012): 'person' 클래스 누락 (이전: 2개)
2025-12-30 10:15:45 [WARNING] Frame 25 (frame_0025): 전체 라벨 누락 (이전 프레임에는 객체 존재)
2025-12-30 10:15:52 [WARNING] Frame 31 (frame_0031): 'bicycle' 클래스 누락 (이전: 1개)
2025-12-30 10:16:10 [WARNING] Frame 48 (frame_0048): 'car' 급격한 감소 (3 → 1, 33%)
2025-12-30 10:20:15 [INFO] 처리 완료: 총 100 프레임
```

→ **언제든지 파일을 열어서 어떤 프레임/클래스에서 문제가 있었는지 확인 가능!**

---

## ✅ 4. 통계 수집: 누락률, 패턴 분석

### 구현 위치: `label_validation.py:186-218`

#### 핵심 메서드: `get_summary()` 및 `print_summary()`

```python
# label_validation.py:186-218
def get_summary(self) -> Dict:
    """검증 통계 요약"""
    empty_rate = (
        len(self.empty_frames) / self.total_frames
        if self.total_frames > 0 else 0
    )

    # 클래스별 누락률 계산
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
```

### 사용 방법: 처리 완료 후 호출

```python
# sam3_offline.py 처리 루프 종료 후

# 모든 프레임 처리 완료
validator.print_summary()  # ← 콘솔에 요약 출력

# 또는 JSON으로 저장
summary = validator.get_summary()
with open('validation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
```

### 실제 출력 예시

#### 콘솔 출력
```
======================================================================
라벨 검증 요약
======================================================================

총 처리 프레임: 100
빈 프레임: 5 (5.0%)
중간 누락 패턴: 8

클래스별 통계:
  person:
    총 감지: 180개
    누락 프레임: 8 (8.0%)
  car:
    총 감지: 95개
    누락 프레임: 3 (3.0%)
  bicycle:
    총 감지: 12개
    누락 프레임: 15 (15.0%)  ← 문제 있음!
======================================================================
```

#### JSON 파일: `validation_summary.json`
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
    },
    "bicycle": {
      "missing_frames": 15,
      "missing_rate": 0.15,
      "total_detections": 12
    }
  }
}
```

→ **누락률이 높은 클래스(bicycle: 15%)를 파악하여 재처리 가능!**

---

## ✅ 5. 오프라인 분석: 기존 데이터 품질 검증

### 구현 위치: `check_missing_labels.py`

#### 독립 실행형 도구

```python
# check_missing_labels.py:41-153
def analyze_label_directory(label_dir, class_names=None):
    """
    라벨 디렉토리 분석

    분석 내용:
    1. 파일이 없는 프레임 (missing_file_frames)
    2. 빈 라벨 파일 (empty_frames)
    3. 중간 프레임 전체 누락 패턴
    4. 특정 클래스만 중간에 누락되는 패턴

    Returns:
        dict: 분석 결과
    """
    label_files = sorted(label_dir.glob('*.txt'))

    # 프레임별 정보 수집
    for idx, label_file in enumerate(label_files):
        annotations = parse_yolo_label(label_file)

        if annotations is None:
            missing_file_frames.append(...)  # 파일 없음
        elif len(annotations) == 0:
            empty_frames.append(...)  # 빈 파일
        else:
            # 정상 프레임 - 클래스 통계 수집
            ...

    # 중간 누락 패턴 감지
    for i in range(1, len(frames_info) - 1):
        if (frames_info[i]['status'] in ['missing', 'empty'] and
            frames_info[i-1]['status'] == 'ok' and
            frames_info[i+1]['status'] == 'ok'):
            middle_missing_patterns.append(...)

    # 클래스별 누락 패턴 분석
    class_missing_patterns = analyze_class_missing_patterns(...)

    return result
```

### 사용 방법: 커맨드라인 실행

```bash
# 이미 생성된 라벨 파일들을 분석
python check_missing_labels.py \
  --label-dir ./output/labels \
  --classes "person:0,car:1,bicycle:2" \
  --output analysis_report.json
```

### 실제 출력 예시

```
======================================================================
프레임별 라벨 누락 분석 보고서
======================================================================

라벨 디렉토리 분석 중: ./output/labels

[요약]
  총 프레임 수: 500
  정상 프레임: 475 (95.0%)
  빈 프레임 (객체 없음): 20 (4.0%)
  누락 파일: 5 (1.0%)
  감지된 클래스: [0, 1, 2]

[중간 프레임 전체 누락]
  감지된 패턴 수: 12
    - 프레임 15 (video1_frame_0015): empty
      이전: video1_frame_0014, 다음: video1_frame_0016
    - 프레임 42 (video1_frame_0042): empty
      이전: video1_frame_0041, 다음: video1_frame_0043
    - 프레임 78 (video1_frame_0078): missing
      이전: video1_frame_0077, 다음: video1_frame_0079
    ... 외 9개

[특정 클래스 중간 누락]
  클래스 0 (person): 15번 누락
    - 프레임 23 (video1_frame_0023)
      이전 프레임 객체 수: 2, 다음 프레임 객체 수: 2
    - 프레임 67 (video1_frame_0067)
      이전 프레임 객체 수: 3, 다음 프레임 객체 수: 3
    ... 외 13개

  클래스 2 (bicycle): 28번 누락
    - 프레임 10 (video1_frame_0010)
      이전 프레임 객체 수: 1, 다음 프레임 객체 수: 1
    ... 외 27개

[빈 프레임 상세 (최대 10개)]
  - 프레임 15: video1_frame_0015
  - 프레임 42: video1_frame_0042
  - 프레임 78: video1_frame_0078
  ... 외 17개

[누락 파일 상세]
  - 프레임 100: video1_frame_0100
  - 프레임 250: video1_frame_0250

======================================================================

분석 결과 저장됨: analysis_report.json
```

### JSON 리포트: `analysis_report.json`

```json
{
  "summary": {
    "total_frames": 500,
    "ok_frames": 475,
    "empty_frames": 20,
    "missing_file_frames": 5,
    "all_classes_seen": [0, 1, 2]
  },
  "middle_missing_patterns": [
    {
      "index": 15,
      "name": "video1_frame_0015",
      "type": "empty",
      "prev": "video1_frame_0014",
      "next": "video1_frame_0016"
    }
  ],
  "class_missing_patterns": [
    {
      "class_id": 0,
      "missing_occurrences": 15,
      "missing_details": [
        {
          "frame_index": 23,
          "frame_name": "video1_frame_0023",
          "prev_count": 2,
          "next_count": 2
        }
      ]
    }
  ]
}
```

→ **이 리포트로 어떤 프레임을 재처리해야 할지 파악!**

---

## 📊 전체 워크플로우

### 시나리오 1: 새로운 비디오 처리 (실시간 검증)

```bash
# 1. sam3_offline.py 실행 (label_validation 통합)
python sam3_offline.py \
  --checkpoint ./weights/sam3.pth \
  --prompts "person,car,bicycle" \
  --classes "person:0,car:1,bicycle:2" \
  --input-dir ./frames \
  --output-dir ./labels \
  --validation-summary validation.json
```

**처리 중 실시간 출력**:
```
✓ 클래스 매핑 검증 완료  ← 해결책 #2: 사전 검증
✓ 라벨 검증기 초기화

[1/100] frame_0000.jpg
  ✓ 검증 통과 (5개 객체)  ← 해결책 #1: 실시간 검증

[10/100] frame_0009.jpg
  ⚠️  'person' 클래스 누락  ← 해결책 #3: 상세 로깅

--- 중간 요약 (50개 처리) ---
  빈 프레임: 2 (4.0%)  ← 해결책 #4: 통계 수집

처리 완료
======================================================================
라벨 검증 요약  ← 해결책 #4: 통계 수집
======================================================================
총 처리 프레임: 100
빈 프레임: 5 (5.0%)
...
```

### 시나리오 2: 기존 라벨 품질 검증 (오프라인 분석)

```bash
# 2. 기존 라벨 검증 (독립 도구)
python check_missing_labels.py \
  --label-dir ./labels \
  --classes "person:0,car:1,bicycle:2" \
  --output quality_report.json
```

**출력**:
```
======================================================================
프레임별 라벨 누락 분석 보고서  ← 해결책 #5: 오프라인 분석
======================================================================

[요약]
  총 프레임 수: 100
  정상 프레임: 95 (95.0%)
  빈 프레임: 5 (5.0%)

[중간 프레임 전체 누락]
  감지된 패턴 수: 3  ← 문제 발견!

[특정 클래스 중간 누락]
  클래스 2 (bicycle): 12번 누락  ← 심각한 문제!
```

### 시나리오 3: 문제 프레임 재처리

```bash
# 3. quality_report.json을 보고 문제 프레임만 추출
# 4. 낮은 threshold로 재처리
python sam3_offline.py \
  --checkpoint ./weights/sam3.pth \
  --prompts "bicycle" \
  --classes "bicycle:2" \
  --input-list missing_frames.txt \  # 누락 프레임 목록
  --output-dir ./labels_fixed \
  --detection-threshold 0.3  # 더 낮은 임계값
```

---

## 💡 요약

| 해결책 | 구현 도구 | 핵심 코드 | 실행 시점 |
|--------|----------|----------|----------|
| **1. 실시간 검증** | `label_validation.py` | `FrameLabelValidator.validate_frame()` | 각 프레임 처리 직후 |
| **2. 사전 검증** | `label_validation.py` | `validate_class_mapping_complete()` | 처리 시작 전 |
| **3. 상세 로깅** | `label_validation.py` | `_check_class_missing()` + logging | 실시간 + 파일 저장 |
| **4. 통계 수집** | `label_validation.py` | `get_summary()`, `print_summary()` | 처리 완료 후 |
| **5. 오프라인 분석** | `check_missing_labels.py` | `analyze_label_directory()` | 독립 실행 |

---

## 🔧 통합 상태

현재 제공된 것:
- ✅ **도구 코드**: `label_validation.py`, `check_missing_labels.py`
- ✅ **통합 가이드**: `INTEGRATION_GUIDE.md`
- ✅ **분석 보고서**: `LABEL_MISSING_ANALYSIS.md`

필요한 작업:
- ⚠️ **`sam3_offline.py` 실제 통합**: `INTEGRATION_GUIDE.md`를 따라 직접 수정 필요

통합 후:
- ✅ 실시간 검증 자동 실행
- ✅ 모든 해결책 활성화
