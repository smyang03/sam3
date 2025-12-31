# 특정 클래스 전체 누락 근본 원인 분석

## 문제 상황

```
프레임 N-1: 사람 ✓, 헬멧 ✓ (3개)
프레임 N:   사람 ✓, 헬멧 ✗ (0개) ← 이상함!
프레임 N+1: 사람 ✓, 헬멧 ✓ (3개)
```

**왜 이상한가?**
- 헬멧 3개 중 2개만 누락 → 이해 가능 (가려짐, 모델 실수)
- 헬멧 3개가 **한 프레임에서 전부 사라짐** → 이상함!

---

## 근본 원인

### 1. Presence Score의 영향 (가장 의심됨)

**코드 위치**: `sam3/eval/postprocessors.py:100-102`

```python
if self.use_presence:
    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    out_probs = out_probs * presence_score  # ← 핵심!
```

**작동 방식**:
- `presence_logit_dec`: 모델이 판단한 "이 프롬프트에 해당하는 객체가 이미지에 존재하는가?"
- sigmoid 후 0~1 사이 값
- **모든 객체의 확률에 곱해짐**

**문제 시나리오**:
```python
# 프레임 N-1
헬멧 프롬프트:
  presence_score = 0.9  # 모델: "헬멧 있음"
  객체 1 score: 0.7 * 0.9 = 0.63
  객체 2 score: 0.6 * 0.9 = 0.54
  객체 3 score: 0.5 * 0.9 = 0.45

# 프레임 N (문제 프레임)
헬멧 프롬프트:
  presence_score = 0.2  # 모델: "헬멧 별로 없음" ← 오판!
  객체 1 score: 0.7 * 0.2 = 0.14  ← threshold(0.3) 이하
  객체 2 score: 0.6 * 0.2 = 0.12  ← threshold 이하
  객체 3 score: 0.5 * 0.2 = 0.10  ← threshold 이하
  → 3개 전부 필터링됨!

# 프레임 N+1
헬멧 프롬프트:
  presence_score = 0.85  # 모델: "헬멧 있음"
  객체 1 score: 0.7 * 0.85 = 0.595
  객체 2 score: 0.6 * 0.85 = 0.51
  객체 3 score: 0.5 * 0.85 = 0.425
```

**결과**:
- 프레임 N에서 presence_score가 낮으면 **모든 헬멧이 동시에** threshold 이하로 떨어짐
- 개별 객체는 잘 감지되었지만, **presence score 때문에 전부 필터링**

---

### 2. Detection Threshold 필터링

**코드 위치**: `sam3/eval/postprocessors.py:242-250`

```python
if self.detection_threshold > 0:
    # Filter out the boxes with scores below the detection threshold
    keep = scores > self.detection_threshold

    boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
    scores = [s[k.to(s.device)] for s, k in zip(scores, keep)]
    labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]
```

**시나리오**:
```python
# detection_threshold = 0.3인 경우

# 프레임 N (presence_score = 0.2)
헬멧 객체들의 최종 점수: [0.14, 0.12, 0.10]
keep = [False, False, False]  # 모두 0.3 이하
boxes = []  # 빈 배열
scores = []
labels = []
```

---

### 3. 전체 처리 흐름

```
1. 모델 추론
   ├─ pred_logits: [헬멧1: 0.7, 헬멧2: 0.6, 헬멧3: 0.5]
   └─ presence_logit_dec: 0.2 (낮음!)

2. Sigmoid 적용
   ├─ out_probs: [0.7, 0.6, 0.5]
   └─ presence_score: 0.2

3. Presence Score 곱셈 (postprocessors.py:102)
   └─ out_probs = [0.7, 0.6, 0.5] * 0.2 = [0.14, 0.12, 0.10]

4. 최대값 선택 (postprocessors.py:225)
   └─ scores = [0.14, 0.12, 0.10]

5. Threshold 필터링 (postprocessors.py:242-250)
   └─ keep = scores > 0.3 = [False, False, False]
   └─ boxes = [] (빈 배열)

6. sam3_offline.py에서 결과 처리 (869-874)
   └─ results_by_prompt['helmet'] = {'boxes': [], 'scores': []}

7. YOLO 라벨 저장 (476-507)
   └─ len(result['boxes']) == 0이므로 continue
   └─ 헬멧 라벨 0개 저장
```

---

## 왜 이런 일이 발생하는가?

### Presence Score가 낮아지는 이유

1. **이미지 품질 문제**
   - 블러, 어두움, 노출 과다
   - 프레임 N만 유독 품질이 나쁨

2. **객체 배치/구도**
   - 헬멧이 프레임 가장자리에 위치
   - 다른 객체에 부분적으로 가려짐
   - 모델이 "전체적으로 헬멧이 적다"고 판단

3. **모델의 일시적 오판**
   - SAM3 모델의 attention mechanism이 특정 프레임에서 헬멧 특징을 놓침
   - 배경이나 다른 객체에 집중

4. **프롬프트 처리 순서**
   - Chunk 단위 처리 (기본 4개씩)
   - GPU 메모리 상태, 배치 처리 순서에 따라 결과 변동 가능

---

## 확인 방법

### 1. Presence Score 로깅 추가

`sam3_offline.py`에 디버깅 코드 추가:

```python
# Line 853 이후
processed_results = postprocessor.process_results(output, batch.find_metadatas)

# 디버깅: presence score 확인
if 'presence_logit_dec' in output:
    presence_scores = output['presence_logit_dec'].sigmoid()
    for idx, (prompt_name, p_score) in enumerate(zip(chunk_prompts, presence_scores)):
        p_score_val = p_score.item() if hasattr(p_score, 'item') else p_score
        if p_score_val < 0.5:  # 낮은 presence score 경고
            print(f"⚠️ 낮은 presence score - {prompt_name}: {p_score_val:.3f}")
```

### 2. 원시 출력 저장

```python
# 모델 출력을 저장하여 나중에 분석
if frame_idx in [N-1, N, N+1]:  # 문제 프레임 주변
    torch.save({
        'pred_logits': output['pred_logits'],
        'presence_logit_dec': output['presence_logit_dec'],
        'pred_boxes': output['pred_boxes'],
        'frame': frame_idx,
        'prompts': chunk_prompts
    }, f'debug_frame_{frame_idx}.pt')
```

### 3. 개별 객체 점수 확인

```python
# postprocessor 이전 점수 확인
out_probs_before = output['pred_logits'].sigmoid()
print(f"Helmet 객체 점수 (presence 곱셈 전): {out_probs_before}")

# presence 곱셈 후
if self.use_presence:
    presence_score = output['presence_logit_dec'].sigmoid()
    out_probs_after = out_probs_before * presence_score
    print(f"Presence score: {presence_score}")
    print(f"Helmet 객체 점수 (곱셈 후): {out_probs_after}")
```

---

## 해결 방법

### ✅ 방법 1: Presence Score 비활성화 (가장 효과적)

**설정 변경**:
```python
# sam3_offline.py에서 postprocessor 생성 시
postprocessor = PostProcessImage(
    max_dets_per_img=100,
    detection_threshold=0.3,
    use_presence=False  # ← False로 변경!
)
```

**장점**:
- Presence score의 영향 제거
- 개별 객체 점수만으로 판단

**단점**:
- False positive 증가 가능 (없는 객체 감지)

---

### ✅ 방법 2: Detection Threshold 낮추기

**설정 변경**:
```python
postprocessor = PostProcessImage(
    detection_threshold=0.1  # 0.3 → 0.1
)
```

**효과**:
- Presence score가 낮아도 살아남을 가능성 증가
- 예: presence 0.2 * 객체 0.7 = 0.14 → threshold 0.1 통과

**단점**:
- False positive 증가
- 낮은 품질 감지 포함

---

### ✅ 방법 3: 클래스별 다른 Threshold 적용

현재는 모든 클래스에 동일한 threshold를 사용하는데, 헬멧처럼 작은 객체는 낮은 threshold를 적용:

**구현 필요** (postprocessor 수정):
```python
class PostProcessImageWithClassThresholds(PostProcessImage):
    def __init__(self, class_thresholds: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.class_thresholds = class_thresholds

    def _process_boxes_and_labels(self, ...):
        # 클래스별로 다른 threshold 적용
        for class_name, threshold in self.class_thresholds.items():
            ...
```

**사용**:
```python
postprocessor = PostProcessImageWithClassThresholds(
    class_thresholds={
        'person': 0.3,
        'helmet': 0.1,  # 헬멧은 낮은 threshold
        'car': 0.3
    }
)
```

---

### ✅ 방법 4: Presence Score에 하한선 설정

Presence score가 너무 낮게 떨어지는 것을 방지:

**postprocessors.py 수정**:
```python
if self.use_presence:
    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    # 하한선 설정: 최소 0.3은 보장
    presence_score = torch.clamp(presence_score, min=0.3)
    out_probs = out_probs * presence_score
```

**효과**:
- Presence score 최소값 보장
- 극단적인 필터링 방지

---

### ✅ 방법 5: 시간적 일관성 보정

이전/다음 프레임 정보를 활용:

```python
def smooth_presence_scores(presence_scores_sequence, window=3):
    """이동 평균으로 presence score 스무딩"""
    smoothed = []
    for i, score in enumerate(presence_scores_sequence):
        window_scores = presence_scores_sequence[max(0, i-window):i+window+1]
        smoothed_score = np.mean(window_scores)
        smoothed.append(smoothed_score)
    return smoothed
```

---

## 즉시 테스트 방법

### 1단계: Presence Score 확인

문제 프레임들의 원시 모델 출력 저장:

```bash
python sam3_offline.py \
  --save-debug-outputs \  # 디버그 출력 저장
  --debug-frames "100,101,102"  # 문제 프레임
```

### 2단계: Presence Score 비활성화 테스트

```bash
python sam3_offline.py \
  --use-presence false \  # presence 비활성화
  --input-dir ./frames \
  --output-dir ./labels_no_presence
```

기존 결과와 비교:
```bash
diff -r ./labels ./labels_no_presence
```

### 3단계: Threshold 조정 테스트

```bash
python sam3_offline.py \
  --detection-threshold 0.1 \  # 낮은 threshold
  --output-dir ./labels_low_threshold
```

---

## 결론

**헬멧 3개가 한 프레임에서 전부 사라지는 이유**:

1. **근본 원인**: Presence Score가 낮게 나옴 (모델 오판)
2. **악화 요인**: Presence score와 객체 점수 곱셈 → 모두 threshold 이하
3. **결과**: 개별 객체는 감지되었지만 후처리에서 **전부 필터링**

**이것은 버그가 아니라 설계된 동작**입니다:
- Presence score는 "이 클래스가 이미지에 있는가?"를 판단
- 낮으면 모든 객체 점수를 낮춤 → 의도된 동작
- 하지만 **오판 시 전체 클래스 누락**이라는 부작용

**권장 해결책**:
1. **즉시**: `use_presence=False` 테스트
2. **단기**: `detection_threshold` 낮춤 (0.1~0.15)
3. **장기**: 클래스별 threshold 또는 시간적 보정 구현
