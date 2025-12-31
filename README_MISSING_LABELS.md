# 라벨 누락 문제 해결 가이드

## 문제 요약

**현상**: 프레임 중간에 특정 클래스(예: 헬멧)의 **모든 객체가 동시에 사라졌다가** 다음 프레임에서 다시 나타남

```
프레임 100: 사람 ✓, 헬멧 ✓ (3개)
프레임 101: 사람 ✓, 헬멧 ✗ (0개) ← 이상함!
프레임 102: 사람 ✓, 헬멧 ✓ (3개)
```

**근본 원인**: SAM3 모델의 **Presence Score**가 낮게 나와서 해당 클래스 전체가 후처리 단계에서 필터링됨

---

## 빠른 해결 방법

### 1️⃣ Presence Score 비활성화 (가장 효과적)

`sam3_offline.py` 또는 `run.py`에서 postprocessor 생성 부분 수정:

```python
# 기존 코드 찾기
postprocessor = PostProcessImage(
    max_dets_per_img=100,
    detection_threshold=0.3,
    use_presence=True  # ← 기존 설정
)

# 수정
postprocessor = PostProcessImage(
    max_dets_per_img=100,
    detection_threshold=0.3,
    use_presence=False  # ← False로 변경!
)
```

**효과**: Presence score의 영향을 제거하여 개별 객체 점수만으로 판단

---

### 2️⃣ Detection Threshold 낮추기

```python
postprocessor = PostProcessImage(
    detection_threshold=0.1  # 0.3 → 0.1로 낮춤
)
```

**효과**: Presence score가 낮아도 더 많은 객체가 살아남음

---

## 상세 분석

### Presence Score란?

- SAM3 모델이 판단하는 "이 프롬프트(클래스)에 해당하는 객체가 이미지에 존재하는가?"
- 0~1 사이 값 (sigmoid 출력)
- **모든 객체의 확률에 곱해짐**

### 왜 문제가 되는가?

```python
# 정상 프레임
헬멧 presence_score = 0.9
객체 1: 0.7 * 0.9 = 0.63 → threshold(0.3) 통과 ✓
객체 2: 0.6 * 0.9 = 0.54 → 통과 ✓
객체 3: 0.5 * 0.9 = 0.45 → 통과 ✓

# 문제 프레임
헬멧 presence_score = 0.2  ← 모델 오판!
객체 1: 0.7 * 0.2 = 0.14 → threshold(0.3) 미달 ✗
객체 2: 0.6 * 0.2 = 0.12 → 미달 ✗
객체 3: 0.5 * 0.2 = 0.10 → 미달 ✗
→ 3개 전부 필터링!
```

---

## 전체 해결 방법

| 방법 | 난이도 | 효과 | 부작용 |
|------|--------|------|--------|
| **1. Presence 비활성화** | ⭐ 쉬움 | ⭐⭐⭐ 높음 | False positive 증가 |
| **2. Threshold 낮추기** | ⭐ 쉬움 | ⭐⭐ 중간 | False positive 증가 |
| **3. Presence 하한선 설정** | ⭐⭐ 보통 | ⭐⭐⭐ 높음 | 코드 수정 필요 |
| **4. 클래스별 Threshold** | ⭐⭐⭐ 어려움 | ⭐⭐⭐ 높음 | 구현 필요 |
| **5. 시간적 보정** | ⭐⭐⭐ 어려움 | ⭐⭐⭐ 높음 | 복잡한 구현 |

---

## 디버깅 방법

### 1. Presence Score 확인

`sam3_offline.py`에 디버깅 코드 추가 (Line 853 이후):

```python
processed_results = postprocessor.process_results(output, batch.find_metadatas)

# 디버깅: presence score 로깅
if 'presence_logit_dec' in output:
    presence_scores = output['presence_logit_dec'].sigmoid()
    for prompt_name, p_score in zip(chunk_prompts, presence_scores):
        p_score_val = p_score.item() if hasattr(p_score, 'item') else p_score
        if p_score_val < 0.5:
            print(f"⚠️ {image_path} - {prompt_name}: presence={p_score_val:.3f}")
```

### 2. 디버그 출력 저장

문제 프레임의 원시 출력 저장:

```python
# Line 845 이후
output = model(batch)

# 문제 프레임 저장
if 'frame_0101' in image_path:  # 문제 프레임
    torch.save({
        'pred_logits': output['pred_logits'].cpu(),
        'presence_logit_dec': output['presence_logit_dec'].cpu(),
        'pred_boxes': output['pred_boxes'].cpu(),
        'prompts': chunk_prompts,
        'image_path': image_path
    }, 'debug_frame_0101.pt')
```

### 3. 디버그 파일 분석

```bash
python fix_presence_score.py --mode debug --input debug_frame_0101.pt
```

**출력 예시**:
```
Presence Scores:
  person        : 0.8542 ✓
  helmet        : 0.1823 ⚠️ 낮음
  car           : 0.7234 ✓

개별 객체 점수:
  helmet        : 3개 객체, 최대=0.7123, 평균=0.5821
    → Presence 곱셈 후: 최대=0.1298, 평균=0.1061
    → Threshold 0.3: 0/3개 통과  ← 문제 확인!
```

---

## 고급 해결 방법

### 방법 3: Presence Score 하한선 설정

`sam3/eval/postprocessors.py` 수정 (Line 100-102):

```python
if self.use_presence:
    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)

    # 패치: 최소값 보장
    MIN_PRESENCE_SCORE = 0.3  # 조정 가능
    presence_score = torch.clamp(presence_score, min=MIN_PRESENCE_SCORE)

    out_probs = out_probs * presence_score
```

**효과**: Presence score가 0.3 미만으로 떨어지는 것 방지

---

### 방법 4: 클래스별 다른 Threshold

작은 객체(헬멧)는 낮은 threshold 적용:

```python
class PostProcessImageWithClassThresholds(PostProcessImage):
    def __init__(self, class_thresholds: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.class_thresholds = class_thresholds
```

**사용**:
```python
postprocessor = PostProcessImageWithClassThresholds(
    class_thresholds={
        'person': 0.3,
        'helmet': 0.1,  # 헬멧만 낮게
        'car': 0.3
    }
)
```

---

## 테스트 및 비교

### Before/After 비교

```bash
# 1. 기존 설정으로 처리
python sam3_offline.py \
  --prompts "person,helmet,car" \
  --classes "person:0,helmet:1,car:2" \
  --input-dir ./frames \
  --output-dir ./labels_original

# 2. Presence 비활성화
python sam3_offline.py \
  --prompts "person,helmet,car" \
  --classes "person:0,helmet:1,car:2" \
  --input-dir ./frames \
  --output-dir ./labels_no_presence \
  --use-presence false  # ← 인자 추가 필요

# 3. 차이 확인
python check_missing_labels.py --label-dir ./labels_original
python check_missing_labels.py --label-dir ./labels_no_presence

# 4. 비교
diff -r ./labels_original ./labels_no_presence
```

---

## 관련 파일

1. **HELMET_MISSING_ROOT_CAUSE.md** - 상세 원인 분석
2. **LABEL_MISSING_ANALYSIS.md** - 전체 누락 패턴 분석
3. **INTEGRATION_GUIDE.md** - 검증 도구 통합 가이드
4. **fix_presence_score.py** - 디버깅 유틸리티
5. **check_missing_labels.py** - 라벨 누락 분석 도구

---

## FAQ

### Q1: Presence score를 비활성화하면 부작용은?

**A**: False positive(없는 객체를 감지)가 증가할 수 있습니다. 하지만 누락보다는 나은 경우가 많습니다. 후처리에서 NMS 등으로 걸러낼 수 있습니다.

### Q2: 왜 모델이 presence score를 잘못 판단하나요?

**A**:
- 이미지 품질 (블러, 어두움)
- 객체 배치 (가장자리, 가려짐)
- 모델의 일시적 attention 실수
- 학습 데이터에 없던 구도/각도

### Q3: 개별 객체는 잘 감지되는데 왜 필터링되나요?

**A**: Presence score가 낮으면 **모든** 객체 점수에 곱해지기 때문입니다. 개별 객체 점수가 0.7이어도 presence 0.2를 곱하면 0.14가 되어 threshold(0.3)를 통과하지 못합니다.

### Q4: 헬멧만 자주 누락되는 이유는?

**A**: 헬멧은 작고, 사람 머리에 붙어있어서:
- 배경/사람과 구분이 어려움
- Presence score가 낮게 나올 확률이 높음
- 개별 객체 점수도 상대적으로 낮음
- Presence 곱셈 후 threshold 미달 가능성 높음

---

## 권장 조치

### 즉시 (5분)
1. ✅ `use_presence=False` 설정
2. ✅ 문제 프레임 재처리
3. ✅ 결과 비교

### 단기 (1시간)
4. 📊 Presence score 로깅 추가
5. 🔍 문제 프레임 디버그 출력 분석
6. ⚙️ Threshold 최적화

### 장기 (1일~)
7. 🛠️ Presence score 하한선 패치 적용
8. 🎯 클래스별 threshold 구현
9. 📹 시간적 보정 메커니즘 추가
