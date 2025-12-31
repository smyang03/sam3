# Config 파일 사용 가이드

## 개요

Presence score와 클래스별 detection threshold를 config 파일로 관리할 수 있습니다.

---

## 빠른 시작

### 1. Config 파일 생성

`config.json`:
```json
{
  "model_dir": "./models",
  "image_dir": "./data/images",
  "label_dir": "./data/labels",

  "classes": {
    "person": 0,
    "helmet": 1,
    "car": 2
  },

  "detection_config": {
    "use_presence": false,
    "default_threshold": 0.3,
    "class_thresholds": {
      "person": 0.3,
      "helmet": 0.15,
      "car": 0.3
    }
  },

  "inference": {
    "chunk_size": 4,
    "max_dets_per_img": 100
  }
}
```

### 2. 실행

```bash
python sam3_offline.py --config config.json --gpu 0
```

---

## Config 파일 구조

### detection_config (핵심 설정)

| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `use_presence` | boolean | `true` | Presence score 사용 여부 |
| `default_threshold` | float | `0.3` | 기본 detection threshold |
| `class_thresholds` | object | `{}` | 클래스별 threshold |
| `max_dets_per_img` | int | `100` | 이미지당 최대 감지 개수 |

### video_config (동영상 처리 설정)

| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `video_source` | string | `null` | 동영상 파일/폴더 경로 |
| `fps` | int | `1` | 프레임 추출 FPS (0/-1: 전체) |
| `jpeg_dir` | string | `./data/JPEGImages` | 추출된 프레임 저장 경로 |

### inference (추론 설정)

| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `chunk_size` | int | `4` | 프롬프트 청크 크기 |

### output (출력 설정)

| 항목 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `save_viz` | boolean | `false` | 시각화 이미지 저장 여부 |
| `show` | boolean | `false` | 실시간 결과 표시 여부 |

---

## 사용 예시

### 예시 1: Presence Score 비활성화 (권장)

헬멧 전체 누락 문제 해결:

```json
{
  "detection_config": {
    "use_presence": false,
    "default_threshold": 0.3
  }
}
```

**효과**: Presence score가 낮아도 개별 객체 점수만으로 판단

---

### 예시 2: 클래스별 다른 Threshold

작은 객체(헬멧)는 낮은 threshold 적용:

```json
{
  "classes": {
    "person": 0,
    "helmet": 1,
    "vest": 2,
    "car": 3
  },

  "detection_config": {
    "use_presence": false,
    "default_threshold": 0.3,
    "class_thresholds": {
      "person": 0.3,
      "helmet": 0.15,
      "vest": 0.2,
      "car": 0.3
    }
  }
}
```

**효과**:
- 헬멧: threshold 0.15 (낮음) → 더 많이 감지
- 조끼: threshold 0.2 (중간)
- 사람, 차: threshold 0.3 (높음)

---

### 예시 3: Presence + 낮은 Threshold 조합

```json
{
  "detection_config": {
    "use_presence": true,
    "default_threshold": 0.1,
    "class_thresholds": {
      "helmet": 0.05
    }
  }
}
```

**효과**: Presence를 사용하되, threshold를 낮춰서 살아남을 확률 증가

---

## 동작 방식

### 1. use_presence = false

```python
# Presence score 곱셈 없음
박스 점수: [0.7, 0.6, 0.5]
threshold: 0.3
→ 결과: [0.7, 0.6, 0.5] → 3개 모두 통과
```

### 2. use_presence = true

```python
# Presence score 곱셈 있음
박스 점수: [0.7, 0.6, 0.5]
presence: 0.2
최종 점수: [0.14, 0.12, 0.10]
threshold: 0.3
→ 결과: [] → 모두 미달
```

### 3. 클래스별 Threshold

```python
# 헬멧 클래스
박스 점수: [0.5, 0.4]
threshold (helmet): 0.15
→ 결과: [0.5, 0.4] → 2개 통과

# 사람 클래스
박스 점수: [0.5, 0.4]
threshold (person): 0.3
→ 결과: [0.5] → 1개만 통과
```

---

## 동영상 처리

### 동영상에서 프레임 추출 + 라벨 생성

`config_video.json`:
```json
{
  "video_config": {
    "video_source": "./videos",
    "fps": 1,
    "jpeg_dir": "./data/frames"
  },

  "label_dir": "./data/labels",

  "classes": {
    "person": 0,
    "helmet": 1,
    "car": 2
  },

  "detection_config": {
    "use_presence": false,
    "default_threshold": 0.3,
    "class_thresholds": {
      "helmet": 0.15
    }
  }
}
```

**실행**:
```bash
python sam3_offline.py --config config_video.json --gpu 0
```

**처리 과정**:
1. `./videos` 폴더의 모든 동영상에서 프레임 추출 (매 프레임마다 추출)
2. 추출된 프레임을 `./data/frames`에 저장
3. 각 프레임에 대해 라벨 생성 (helmet은 threshold 0.15)
4. 라벨을 `./data/labels`에 저장

### FPS 설정

**중요**: fps는 "N프레임마다 1번 추출"을 의미합니다.

| 값 | 의미 | 원본 30fps 기준 결과 |
|----|------|---------------------|
| `1` | 매 프레임 추출 | 30fps (전체) |
| `5` | 5프레임마다 1번 | 6fps |
| `30` | 30프레임마다 1번 | 1fps (1초당 1프레임) |
| `60` | 60프레임마다 1번 | 0.5fps (2초당 1프레임) |
| `0` 또는 `-1` | 모든 프레임 추출 | 30fps (전체) |

**예시**:
```json
{
  "video_config": {
    "fps": 30  // 30프레임마다 1번 = 1초당 1프레임 (30fps 영상 기준)
  }
}
```

### 특정 동영상 파일 처리

```json
{
  "video_config": {
    "video_source": "./videos/construction_site.mp4",
    "fps": 2  // 2프레임마다 1번 추출
  }
}
```

### 동영상 폴더 일괄 처리

```json
{
  "video_config": {
    "video_source": "./videos",
    "fps": 1  // 매 프레임 추출 (전체)
  }
}
```

→ `./videos` 폴더의 모든 `.mp4`, `.avi`, `.mov` 파일 처리

---

## 커맨드라인 인자와 함께 사용

커맨드라인 인자가 우선권을 가집니다:

```bash
# Config 파일: threshold 0.3
# 커맨드라인: threshold 0.5
python sam3_offline.py --config config.json --threshold 0.5

# 결과: 0.5 사용 (커맨드라인 우선)
```

단, `detection_config`는 커맨드라인으로 오버라이드 불가능:
- `use_presence`는 config 파일로만 설정
- `class_thresholds`는 config 파일로만 설정

---

## 전체 Config 예시

```json
{
  "model_dir": "./models",
  "image_dir": "./data/images",
  "label_dir": "./data/labels",
  "viz_dir": "./data/results",

  "classes": {
    "person": 0,
    "helmet": 1,
    "vest": 2,
    "gloves": 3,
    "car": 4,
    "truck": 5
  },

  "detection_config": {
    "use_presence": false,
    "default_threshold": 0.3,
    "class_thresholds": {
      "person": 0.3,
      "helmet": 0.12,
      "vest": 0.15,
      "gloves": 0.1,
      "car": 0.3,
      "truck": 0.3
    },
    "max_dets_per_img": 100
  },

  "inference": {
    "chunk_size": 4
  },

  "output": {
    "save_viz": false,
    "show": false
  }
}
```

---

## 권장 설정

### 공사장 안전 장비 감지

```json
{
  "classes": {
    "person": 0,
    "helmet": 1,
    "vest": 2
  },

  "detection_config": {
    "use_presence": false,
    "default_threshold": 0.3,
    "class_thresholds": {
      "person": 0.3,
      "helmet": 0.15,
      "vest": 0.2
    }
  }
}
```

**이유**:
- `use_presence: false`: 헬멧 전체 누락 방지
- `helmet: 0.15`: 작은 헬멧도 놓치지 않음
- `vest: 0.2`: 조끼는 중간 수준

---

### 교통 차량 감지

```json
{
  "classes": {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3
  },

  "detection_config": {
    "use_presence": true,
    "default_threshold": 0.3,
    "class_thresholds": {
      "motorcycle": 0.2
    }
  }
}
```

**이유**:
- `use_presence: true`: 차량은 크고 명확 → presence 사용 OK
- `motorcycle: 0.2`: 오토바이는 작아서 낮은 threshold

---

## 문제 해결

### Q1: use_presence를 false로 했는데도 헬멧이 누락됩니다

**A**: `class_thresholds`를 확인하세요. threshold가 너무 높을 수 있습니다.

```json
{
  "detection_config": {
    "use_presence": false,
    "class_thresholds": {
      "helmet": 0.1  // 더 낮춤
    }
  }
}
```

---

### Q2: False positive가 너무 많이 나옵니다

**A**: threshold를 높이거나 `use_presence`를 `true`로 설정:

```json
{
  "detection_config": {
    "use_presence": true,
    "default_threshold": 0.4  // 높임
  }
}
```

---

### Q3: 클래스별 threshold가 적용 안 됩니다

**A**: `classes` 매핑과 `class_thresholds` 이름이 일치하는지 확인:

```json
{
  "classes": {
    "helmet": 1  // 이름: "helmet"
  },

  "detection_config": {
    "class_thresholds": {
      "helmet": 0.15  // 동일하게 "helmet"
    }
  }
}
```

---

## 구현 세부사항

### 코드 위치

- **PostProcessor 확장**: `sam3/eval/postprocessors_classwise.py`
- **Config 적용**: `sam3_offline.py:327-395`
- **예시 파일**: `config_example.json`

### 클래스 구조

```python
# 일반 버전 (단일 threshold)
PostProcessImage(
    use_presence=True,
    detection_threshold=0.3
)

# 클래스별 버전
PostProcessImageWithClassThresholds(
    use_presence=False,
    detection_threshold=0.3,  # 기본값
    class_thresholds={'helmet': 0.15},
    class_to_id={'helmet': 1}
)
```

---

## 관련 문서

- **README_MISSING_LABELS.md** - 라벨 누락 문제 빠른 해결
- **HELMET_MISSING_ROOT_CAUSE.md** - 근본 원인 분석
- **PRESENCE_SCORE_EXPLAINED.md** - Presence score 상세 설명
