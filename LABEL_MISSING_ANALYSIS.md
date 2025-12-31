# 프레임 라벨 누락 분석 보고서

## 요약

SAM3 기반 YOLO 라벨 생성 시스템의 코드를 분석한 결과, **프레임 중간에 전체 라벨 또는 특정 라벨이 누락될 수 있는 여러 시나리오**를 발견했습니다.

---

## 1. 라벨 누락 시나리오

### 1.1 전체 라벨 누락 (Empty Frame)

**발생 위치**: `sam3_offline.py:476-507` (`save_yolo_annotation` 함수)

#### 원인 1: 모든 클래스에서 감지 실패
```python
# sam3_offline.py:484-486
for prompt_name, result in results_by_prompt.items():
    if result is None or len(result['boxes']) == 0:
        continue  # 이 클래스는 건너뜀
```

**시나리오**:
- 모든 프롬프트(클래스)에서 `boxes`가 비어있으면
- `lines` 리스트가 비어있는 상태로 파일에 저장됨
- **결과**: 빈 `.txt` 파일 생성 (객체 없음)

#### 원인 2: 프롬프트 결과가 딕셔너리에 없음
```python
# sam3_offline.py:869-874
if prompt_id in processed_results:
    # 정상 처리
else:
    if prompt_name not in results_by_prompt:
        results_by_prompt[prompt_name] = {
            'boxes': np.array([]),
            'scores': np.array([])
        }
```

**시나리오**:
- 모델 추론 결과 `processed_results`에 해당 `prompt_id`가 없으면
- 빈 배열로 초기화
- 모든 프롬프트가 이 상태면 → **빈 라벨 파일**

#### 원인 3: 이미지 로드 실패
```python
# sam3_offline.py:942-949 (예외 처리)
except Exception as e:
    return {
        'success': False,
        'error': str(e),
        'num_objects': 0,
    }
```

**시나리오**:
- 이미지 파일이 손상되었거나 읽을 수 없는 경우
- 처리 실패 → 라벨 파일이 생성되지 않거나 비어있음

---

### 1.2 특정 클래스 라벨 누락

**발생 위치**: `sam3_offline.py:488-490`

```python
class_id = class_mapping.get(prompt_name, -1)
if class_id < 0:
    continue  # 매핑에 없는 클래스는 건너뜀
```

#### 원인 1: 클래스 매핑 누락
**시나리오**:
- 프롬프트 이름이 `class_mapping`에 없으면
- 해당 클래스의 감지 결과가 있어도 라벨에 저장 안됨

#### 원인 2: 특정 클래스만 감지 실패
```python
# sam3_offline.py:484-486
for prompt_name, result in results_by_prompt.items():
    if result is None or len(result['boxes']) == 0:
        continue
```

**시나리오**:
- 프레임 N-1: person(2개), car(1개) 감지
- 프레임 N: person(0개), car(1개) 감지 ← **person 클래스 누락**
- 프레임 N+1: person(2개), car(1개) 감지

**이유**:
1. 모델이 해당 프레임에서 객체를 감지하지 못함 (낮은 신뢰도)
2. 후처리 단계에서 필터링됨 (threshold)

---

### 1.3 비디오 트래킹에서의 객체 누락

**발생 위치**: `sam3/model/sam3_tracking_predictor.py:565-583`

```python
# 객체가 temp_output_dict_per_obj에 없으면
if out is None:
    out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
if out is None:
    out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
if out is None:
    # 객체가 이 프레임에서 완전히 누락됨
    # NO_OBJ_SCORE로 채워짐
    consolidated_out["pred_masks"][obj_idx] = NO_OBJ_SCORE
```

#### 시나리오: 비디오 추적 중 객체 소실
- 프레임 10: 객체 추적 중
- 프레임 11: 객체가 일시적으로 가려지거나 화면 밖으로 나가면 → `NO_OBJ_SCORE` 할당
- 프레임 12: 객체 다시 나타남

**처리 방식**:
- `NO_OBJ_SCORE = -1024.0` (placeholder)
- 메모리 인코더에 더미 포인터 사용

---

### 1.4 평가/변환 단계에서의 프레임 스킵

**발생 위치**: `sam3/eval/conversion_util.py:169-171`

```python
# Skip frames with missing objects (None or zero bbox)
if bbox is None or all(x == 0 for x in bbox):
    continue
```

#### 시나리오
- COCO 형식 변환 시 bbox가 None이거나 모두 0이면
- 해당 프레임을 건너뜀 → **평가 데이터셋에서 제외**

---

## 2. 후처리 단계 필터링

**발생 위치**: `sam3/eval/postprocessors.py`

### 2.1 신뢰도 임계값 필터링
```python
# PostProcessImage 클래스
detection_threshold  # 낮은 신뢰도 감지 필터링
max_dets_per_img     # 최대 감지 개수 제한
```

**시나리오**:
- 감지는 되었으나 `detection_threshold`보다 낮은 점수
- 후처리 단계에서 제거됨 → **라벨에 포함 안됨**

---

## 3. 누락 패턴 예시

### 패턴 A: 중간 프레임 전체 누락
```
Frame 100: [person:2, car:1] ✓
Frame 101: []              ← 전체 누락
Frame 102: [person:2, car:1] ✓
```

**가능한 원인**:
- 이미지 품질 저하 (블러, 어두움)
- 모델 추론 실패
- 모든 클래스의 신뢰도가 임계값 미만

### 패턴 B: 특정 클래스만 누락
```
Frame 100: [person:2, car:1] ✓
Frame 101: [car:1]         ← person 클래스만 누락
Frame 102: [person:2, car:1] ✓
```

**가능한 원인**:
- 사람이 일시적으로 가려짐
- 사람 감지 신뢰도만 낮음
- 클래스 특화된 감지 실패

### 패턴 C: 점진적 누락
```
Frame 100: [person:2, car:1] ✓
Frame 101: [person:1, car:1] ← person 1개 누락
Frame 102: [person:0, car:1] ← person 전체 누락
Frame 103: [person:2, car:1] ✓
```

**가능한 원인**:
- 객체가 화면에서 점점 사라짐
- 비디오 트래킹 품질 저하

---

## 4. 근본 원인 분석

### 4.1 모델 측면
1. **SAM3 모델의 텍스트 기반 감지 특성**
   - 프롬프트 품질에 따라 감지 성능 변동
   - 일부 프레임에서 특정 객체를 놓칠 수 있음

2. **신뢰도 임계값**
   - 과도하게 높은 threshold → 많은 감지 누락
   - 너무 낮은 threshold → 잘못된 감지 포함

### 4.2 구현 측면
1. **빈 결과 처리**
   - 현재: 빈 배열로 초기화 후 건너뜀
   - 문제: 명시적 경고나 로깅 부족

2. **클래스 매핑 검증 부족**
   - `class_mapping`에 없는 프롬프트는 조용히 무시됨
   - 오타나 설정 오류 발견 어려움

3. **프레임 간 일관성 검증 부재**
   - 이전/다음 프레임과 비교하는 로직 없음
   - 급격한 객체 수 변화 감지 안됨

---

## 5. 개선 방안

### 5.1 즉시 적용 가능한 개선

#### A. 누락 감지 및 경고 추가
```python
# sam3_offline.py에 추가
def validate_frame_results(frame_idx, results_by_prompt, prev_results=None):
    """프레임 결과 검증 및 경고"""

    # 1. 전체 누락 검사
    total_objects = sum(len(r['boxes']) for r in results_by_prompt.values())
    if total_objects == 0:
        logger.warning(f"Frame {frame_idx}: 모든 객체 누락")

    # 2. 이전 프레임과 비교
    if prev_results:
        for class_name in prev_results.keys():
            prev_count = len(prev_results[class_name]['boxes'])
            curr_count = len(results_by_prompt.get(class_name, {}).get('boxes', []))

            if prev_count > 0 and curr_count == 0:
                logger.warning(
                    f"Frame {frame_idx}: {class_name} 클래스 누락 "
                    f"(이전: {prev_count}개)"
                )

    return total_objects
```

#### B. 클래스 매핑 검증
```python
def validate_class_mapping(prompts, class_mapping):
    """클래스 매핑 사전 검증"""
    unmapped = [p for p in prompts if p not in class_mapping]

    if unmapped:
        raise ValueError(
            f"다음 프롬프트가 class_mapping에 없습니다: {unmapped}\n"
            f"현재 매핑: {class_mapping}"
        )
```

#### C. 빈 프레임 통계 수집
```python
def collect_empty_frame_statistics(all_results):
    """빈 프레임 통계 수집 및 보고"""
    empty_frames = []
    class_missing_stats = defaultdict(int)

    for idx, result in enumerate(all_results):
        if result['num_objects'] == 0:
            empty_frames.append(idx)

        # 클래스별 누락 통계
        for class_name in expected_classes:
            if class_name not in result or len(result[class_name]) == 0:
                class_missing_stats[class_name] += 1

    return {
        'empty_frames': empty_frames,
        'empty_rate': len(empty_frames) / len(all_results),
        'class_missing_stats': dict(class_missing_stats)
    }
```

### 5.2 중기 개선

#### A. 시간적 일관성 보정
- 이전/다음 프레임 정보를 활용한 보간
- 갑작스런 객체 소실/출현 필터링

#### B. 적응형 임계값
- 프레임별 동적 threshold 조정
- 이전 프레임 신뢰도 고려

#### C. 멀티패스 감지
- 첫 pass에서 누락된 프레임 식별
- 두 번째 pass에서 낮은 threshold로 재처리

### 5.3 장기 개선

#### A. 트래킹 기반 보정
- 비디오 시퀀스에서 객체 트래킹
- 트래킹 정보로 누락 프레임 보완

#### B. 앙상블 접근
- 여러 프롬프트/모델 결과 결합
- 보수적으로 누락 최소화

---

## 6. 검증 도구

### 생성된 스크립트: `check_missing_labels.py`

**기능**:
1. 프레임별 라벨 파일 스캔
2. 전체 라벨 누락 감지 (빈 파일, 파일 없음)
3. 특정 클래스 누락 패턴 감지
4. 통계 및 상세 보고서 생성

**사용법**:
```bash
# 기본 분석
python check_missing_labels.py --label-dir ./labels

# 클래스 이름 포함
python check_missing_labels.py \
  --label-dir ./labels \
  --classes "person:0,car:1,bicycle:2"

# JSON 결과 저장
python check_missing_labels.py \
  --label-dir ./labels \
  --output analysis.json
```

**출력 예시**:
```
[요약]
  총 프레임 수: 1000
  정상 프레임: 950 (95.0%)
  빈 프레임: 30 (3.0%)
  누락 파일: 20 (2.0%)

[중간 프레임 전체 누락]
  감지된 패턴 수: 15
    - 프레임 105 (frame_105): empty
      이전: frame_104, 다음: frame_106

[특정 클래스 중간 누락]
  클래스 0 (person): 8번 누락
    - 프레임 120 (frame_120)
      이전 프레임 객체 수: 2, 다음 프레임 객체 수: 2
```

---

## 7. 결론

### 현재 상태
✅ **시스템은 작동하지만, 프레임 라벨 누락 가능성 존재**

누락 발생 가능 시나리오:
1. 모델 감지 실패 (전체 또는 특정 클래스)
2. 신뢰도 임계값 필터링
3. 클래스 매핑 오류
4. 비디오 트래킹 중 객체 소실
5. 이미지 로드/처리 오류

### 권장 조치

#### 즉시 조치
1. ✅ `check_missing_labels.py`로 기존 라벨 데이터 검증
2. 📝 누락 패턴 로깅 추가 (`validate_frame_results`)
3. 🔍 클래스 매핑 사전 검증 추가

#### 단기 조치
4. 📊 처리 중 실시간 통계 수집
5. ⚠️ 누락률이 높은 프레임/클래스 경고
6. 🔄 누락 프레임 재처리 옵션

#### 장기 조치
7. 🎯 시간적 일관성 보정 구현
8. 🤖 적응형 임계값 시스템
9. 📹 트래킹 기반 보완 메커니즘

---

## 8. 참고 파일

- `sam3_offline.py:476-507` - YOLO 라벨 저장 로직
- `sam3_offline.py:869-902` - 빈 결과 처리
- `sam3/model/sam3_tracking_predictor.py:500-629` - 비디오 추적 누락 처리
- `sam3/eval/conversion_util.py:169-171` - 프레임 스킵 로직
- `sam3/eval/postprocessors.py` - 후처리 필터링
- `check_missing_labels.py` - 누락 분석 도구 (신규 생성)
