# SAM3 Model Package for Offline Server

## 📦 포함된 파일

- `sam3_checkpoint.pth` - SAM3 모델 체크포인트 (3.14 GB)
- `bpe_simple_vocab_16e6.txt.gz` - BPE Vocabulary
- `config.json` - 모델 설정 정보
- `README.md` - 이 파일

## 📥 다운로드 정보

- 다운로드 날짜: 2025-12-08 18:32:57
- PyTorch 버전: 2.7.0+cu126
- CUDA 사용 가능: True

## 🚀 오프라인 서버에서 사용 방법

### 1. 파일 복사
```bash
# 이 폴더를 서버로 복사
scp -r models/ user@server:/path/to/project/
```

### 2. 코드에서 로드
```python
from sam3 import build_sam3_image_model
import torch

# 로컬 체크포인트 로드
model = build_sam3_image_model(
    bpe_path="/path/to/models/bpe_simple_vocab_16e6.txt.gz"
)
model.load_state_dict(torch.load("/path/to/models/sam3_checkpoint.pth"))
model.eval()
model = model.cuda()
```

### 3. 환경 변수 설정 (오프라인 모드)
```bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

## ⚠️ 주의사항

1. **서버 환경 확인**
   - PyTorch 버전 일치 필요
   - CUDA 버전 호환성 확인
   - GPU 메모리 충분 (최소 16GB 권장)

2. **의존성 패키지**
   - requirements.txt 참조
   - sam3 패키지 설치 필요

3. **파일 크기**
   - 전체 약 3.14 GB
   - 네트워크 전송 시간 고려

## 🔧 트러블슈팅

### 모델 로드 실패
```python
# 방법 1: 체크포인트 직접 로드
state_dict = torch.load("sam3_checkpoint.pth", map_location='cpu')
model.load_state_dict(state_dict)

# 방법 2: strict=False로 시도
model.load_state_dict(state_dict, strict=False)
```

### CUDA 메모리 부족
```python
# Mixed precision 사용
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(batch)
```

## 📞 문의

문제 발생 시 config.json의 정보와 함께 문의하세요.
