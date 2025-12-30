# Presence Score 완전 분석

## TL;DR (요약)

**Presence Score**는:
- **이미지 전체에 이 클래스가 존재하는가?**에 대한 점수
- 각 **박스(객체)의 분류 확률에 곱해짐**
- 박스 자체의 존재 확률이 아님!

---

## 정확한 역할

### 1. 모델 출력 구조

```python
outputs = {
    'pred_boxes': [batch, num_queries, 4],      # 박스 좌표
    'pred_logits': [batch, num_queries, num_classes],  # 각 박스의 클래스 점수
    'presence_logit_dec': [batch, num_prompts],  # 이미지에 클래스 존재 여부
    'pred_masks': ...
}
```

**용어 설명**:
- `num_queries`: 모델이 감지한 객체 후보 개수 (예: 100개)
- `num_classes`: 클래스 수 (또는 1)
- `num_prompts`: 질의한 프롬프트 개수 (예: "헬멧", "사람")

---

### 2. 두 가지 점수의 차이

#### A. `pred_logits` (박스별 분류 점수)

**의미**: "이 박스가 헬멧인가?"

```python
pred_logits = [
    [0.7, 0.2, 0.1],  # 박스 1: 헬멧 70%, 사람 20%, 배경 10%
    [0.6, 0.3, 0.1],  # 박스 2: 헬멧 60%, 사람 30%, 배경 10%
    [0.5, 0.4, 0.1],  # 박스 3: 헬멧 50%, 사람 40%, 배경 10%
]
```

→ **개별 박스가 어떤 클래스인지**에 대한 점수

---

#### B. `presence_logit_dec` (이미지별 존재 점수)

**의미**: "이 이미지에 헬멧이 있나?"

```python
presence_logit_dec = [
    0.9,  # 프롬프트1 (헬멧): 이미지에 헬멧 있을 확률 90%
    0.8,  # 프롬프트2 (사람): 이미지에 사람 있을 확률 80%
]
```

→ **이미지 전체에 이 클래스가 존재하는지**에 대한 점수

---

### 3. 실제 처리 과정

```python
# postprocessors.py:96-102

# 1. 박스별 분류 확률
out_logits = outputs["pred_logits"]  # [batch, num_queries, num_classes]
out_probs = out_logits.sigmoid()     # [batch, num_queries, num_classes]

# 2. 이미지별 존재 확률
if self.use_presence:
    presence_score = outputs["presence_logit_dec"].sigmoid()  # [batch, num_prompts]
    presence_score = presence_score.unsqueeze(1)  # [batch, 1, num_prompts]

    # 3. 곱셈!
    out_probs = out_probs * presence_score
```

---

## 구체적인 예시

### 헬멧 감지 시나리오

#### 입력
- 이미지: 헬멧 3개 있는 공사장
- 프롬프트: "helmet"

#### 모델 출력

```python
# 박스별 점수 (각 박스가 헬멧일 확률)
pred_logits = [
    [0.7],  # 박스 1이 헬멧일 확률
    [0.6],  # 박스 2가 헬멧일 확률
    [0.5],  # 박스 3이 헬멧일 확률
    [0.1],  # 박스 4 (false positive)
    ...
]

# 이미지 레벨 존재 점수
presence_logit_dec = [0.9]  # 이미지에 헬멧 있음: 90%
```

#### 처리 과정

**정상 프레임 (presence 높음)**:
```python
presence_score = sigmoid(0.9) = 0.71

# 각 박스 점수 계산
박스1: sigmoid(0.7) × 0.71 = 0.67 × 0.71 = 0.48
박스2: sigmoid(0.6) × 0.71 = 0.64 × 0.71 = 0.45
박스3: sigmoid(0.5) × 0.71 = 0.62 × 0.71 = 0.44
박스4: sigmoid(0.1) × 0.71 = 0.52 × 0.71 = 0.37

# Threshold 0.3으로 필터링
→ 박스1, 2, 3, 4 모두 통과 (4개 감지)
```

**문제 프레임 (presence 낮음)**:
```python
presence_score = sigmoid(-1.4) = 0.20  # ← 낮음!

# 각 박스 점수 계산 (박스는 똑같이 감지됨)
박스1: sigmoid(0.7) × 0.20 = 0.67 × 0.20 = 0.13
박스2: sigmoid(0.6) × 0.20 = 0.64 × 0.20 = 0.13
박스3: sigmoid(0.5) × 0.20 = 0.62 × 0.20 = 0.12
박스4: sigmoid(0.1) × 0.20 = 0.52 × 0.20 = 0.10

# Threshold 0.3으로 필터링
→ 전부 미달! (0개 감지)
```

---

## 핵심 포인트

### ❌ Presence는 박스 존재 확률이 아님

**잘못된 이해**:
```
presence_score = "이 박스가 존재하는가?"
```

**올바른 이해**:
```
presence_score = "이 이미지에 이 클래스가 존재하는가?"
```

---

### ✅ Presence는 클래스 존재 확률

**역할**:
1. 모델이 이미지를 전체적으로 보고 "헬멧이 있는 이미지인가?"를 판단
2. 이 판단을 **모든 박스의 점수에 곱함**
3. 낮으면 → 모든 박스 점수가 낮아짐 → 전부 필터링

**설계 의도**:
- False positive 줄이기
- 예: "고양이" 프롬프트로 강아지 사진 질의
  - 박스들이 어정쩡하게 감지될 수 있음
  - Presence: "이미지에 고양이 없음" (낮음)
  - 모든 박스 점수 낮춤 → 잘못된 감지 제거

**부작용**:
- Presence 오판 시 → 올바른 감지도 전부 제거

---

## 왜 헬멧만 자주 문제가 되나?

### 작은 객체 특성

1. **박스 점수 자체가 낮음**
   ```
   큰 객체 (사람): pred_logits = [0.9, 0.8, 0.85]
   작은 객체 (헬멧): pred_logits = [0.6, 0.5, 0.55]
   ```

2. **Presence도 낮게 나올 가능성 높음**
   - 작아서 눈에 안 띔
   - 배경/다른 객체와 구분 어려움
   - 모델이 "헬멧 이미지"라고 확신하기 어려움

3. **곱셈 효과**
   ```
   사람: 0.8 × 0.7 (presence) = 0.56 → threshold 통과
   헬멧: 0.5 × 0.3 (presence) = 0.15 → threshold 미달
   ```

---

## 코드로 확인

### 어디서 곱해지나?

```python
# sam3/eval/postprocessors.py:100-102

if self.use_presence:
    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
    out_probs = out_probs * presence_score  # ← 여기!
```

### 어디서 필터링되나?

```python
# sam3/eval/postprocessors.py:242-249

if self.detection_threshold > 0:
    keep = scores > self.detection_threshold  # ← 여기서 걸림

    boxes = [b[k] for b, k in zip(boxes, keep)]
    scores = [s[k] for s, k in zip(scores, keep)]
```

---

## 해결 방법 다시 정리

### 방법 1: Presence 비활성화 (권장)

```python
use_presence=False
```

**효과**: 박스별 점수만으로 판단 (곱셈 없음)

**장점**:
- Presence 오판 영향 제거
- 박스가 잘 감지되면 살아남음

**단점**:
- False positive 증가 가능
- 없는 클래스를 감지할 수 있음

---

### 방법 2: Threshold 낮추기

```python
detection_threshold=0.1  # 기존 0.3
```

**효과**: Presence가 낮아도 살아남을 가능성

```python
헬멧: 0.5 × 0.2 = 0.10
→ threshold 0.1: 통과 ✓
→ threshold 0.3: 미달 ✗
```

---

### 방법 3: Presence 하한선

```python
presence_score = torch.clamp(presence_score, min=0.3)
```

**효과**: Presence가 0.3 미만으로 떨어지는 것 방지

```python
# 기존
presence = 0.2
헬멧: 0.5 × 0.2 = 0.10 → 미달

# 하한선 적용
presence = max(0.2, 0.3) = 0.3
헬멧: 0.5 × 0.3 = 0.15 → 여전히 미달

# 하한선 0.5로 올리면
presence = max(0.2, 0.5) = 0.5
헬멧: 0.5 × 0.5 = 0.25 → threshold 0.2면 통과
```

---

## 비유로 이해하기

### 식당 예약 시스템

**박스 점수 (pred_logits)**:
```
"이 사람이 예약자인가?"
- 사람1: 70% 확실
- 사람2: 60% 확실
- 사람3: 50% 확실
```

**Presence 점수**:
```
"이 식당에 예약이 있는가?"
- 높음 (90%): "예약 있는 식당" → 모든 사람 받아들임
- 낮음 (20%): "예약 없는 식당" → 모든 사람 거부
```

**결과**:
```
예약 있는 식당:
  사람1: 70% × 90% = 63% → 입장 ✓
  사람2: 60% × 90% = 54% → 입장 ✓
  사람3: 50% × 90% = 45% → 입장 ✓

예약 없는 식당:
  사람1: 70% × 20% = 14% → 거부 ✗
  사람2: 60% × 20% = 12% → 거부 ✗
  사람3: 50% × 20% = 10% → 거부 ✗
```

---

## 결론

### Presence Score는:

1. **이미지 레벨의 클래스 존재 확률**
2. **모든 박스의 분류 점수에 곱해짐**
3. **낮으면 모든 박스가 필터링됨**

### 헬멧 전체 누락 이유:

1. Presence가 "이 이미지에 헬멧 없음"으로 오판 (0.2)
2. 각 헬멧 박스는 잘 감지됨 (0.7, 0.6, 0.5)
3. 곱셈: [0.14, 0.12, 0.10]
4. Threshold(0.3) 미달 → **전부 제거**

### 해결책:

**즉시**: `use_presence=False` (1줄 수정)
