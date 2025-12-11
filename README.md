# SAM3 YOLO Dataset Creator - Offline Server Package

오프라인 리눅스 서버에서 GPU별로 분산 실행 가능한 SAM3 기반 YOLO 데이터셋 생성 도구

## 📦 패키지 구성

```
SAM3_Offline_Package/
├── requirements.txt           # Python 패키지 의존성
├── download_sam3_model.py     # 모델 다운로드 스크립트 (온라인 PC용)
├── sam3_offline.py           # 메인 실행 코드 (오프라인 서버용)
├── run_multi_gpu.sh          # 멀티 GPU 분산 실행 스크립트
└── README.md                 # 이 파일
```

---

## 🚀 빠른 시작 가이드

### **1단계: 온라인 PC에서 모델 다운로드**

```bash
# SAM3 설치 (GitHub)
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .

# 패키지 설치
pip install -r requirements.txt

# 모델 다운로드 (./models/ 폴더 생성됨)
python download_sam3_model.py --output_dir ./models
```

**다운로드 결과:**
```
models/
├── sam3_checkpoint.pth         (약 3-4 GB)
├── bpe_simple_vocab_16e6.txt.gz
├── config.json
└── README.md
```

---

### **2단계: 서버로 파일 전송**

```bash
# 방법 1: SCP로 전송
scp -r models/ user@server:/path/to/project/
scp requirements.txt sam3_offline.py run_multi_gpu.sh user@server:/path/to/project/

# 방법 2: 압축 후 전송
tar -czf sam3_offline_package.tar.gz models/ requirements.txt sam3_offline.py run_multi_gpu.sh
scp sam3_offline_package.tar.gz user@server:/path/to/project/

# 서버에서 압축 해제
tar -xzf sam3_offline_package.tar.gz
```

---

### **3단계: 서버에서 환경 설정**

```bash
# 1. 오프라인 패키지 준비 (온라인 PC에서)
pip download -r requirements.txt -d packages/

# 2. packages/ 폴더를 서버로 복사

# 3. 서버에서 설치
pip install --no-index --find-links=packages/ -r requirements.txt

# 4. SAM3 설치 (GitHub 또는 wheel)
pip install --no-index --find-links=packages/ sam3-*.whl
```

---

### **4단계: 멀티 GPU 실행**

```bash
# 1. 실행 스크립트 수정
vi run_multi_gpu.sh

# 수정할 항목:
# - GPU_LIST=(0 1 2 3)  # 사용할 GPU
# - IMAGE_DIR="/path/to/images"
# - CLASSES="person:0,car:1,dog:2"

# 2. 실행 권한 부여
chmod +x run_multi_gpu.sh

# 3. 실행
./run_multi_gpu.sh

# 또는 백그라운드 실행
nohup ./run_multi_gpu.sh > run.log 2>&1 &
```

---

## 📘 상세 사용법

### **1. 단일 GPU 실행**

```bash
python sam3_offline.py \
    --gpu 0 \
    --model_dir ./models \
    --image_dir ./data/images \
    --label_dir ./data/labels \
    --classes "person:0,car:1,bicycle:2" \
    --threshold 0.5 \
    --chunk_size 4
```

### **2. Config 파일 사용**

```bash
# config.json 생성
cat > config.json << 'EOF'
{
  "model_dir": "./models",
  "image_dir": "./data/images",
  "label_dir": "./data/labels",
  "classes": {
    "person": 0,
    "car": 1,
    "bicycle": 2
  },
  "threshold": 0.5,
  "chunk_size": 4,
  "save_viz": true,
  "viz_dir": "./data/results"
}
EOF

# 실행
python sam3_offline.py --config config.json --gpu 0
```

### **3. 동영상 처리**

```bash
python sam3_offline.py \
    --gpu 0 \
    --model_dir ./models \
    --video_source ./videos \
    --fps 1 \
    --jpeg_dir ./data/JPEGImages \
    --label_dir ./data/labels
```

---

## 🎯 주요 기능

### ✅ **완전 오프라인 동작**
- HuggingFace 자동 다운로드 없음
- 로컬 체크포인트 로드
- 외부 네트워크 불필요

### ✅ **멀티 GPU 분산 처리**
- 자동 데이터 분할
- GPU별 독립 프로세스
- 실시간 모니터링

### ✅ **유연한 설정**
- argparse 커맨드라인 인자
- JSON config 파일 지원
- 클래스 매핑 자유 설정

### ✅ **배치 추론 최적화**
- 프롬프트 청크 처리
- GPU 메모리 절약
- 추론 속도 향상

---

## ⚙️ 파라미터 설명

### **sam3_offline.py 주요 옵션**

| 파라미터 | 설명 | 기본값 |
|----------|------|--------|
| `--gpu` | GPU 인덱스 (0, 1, 2...) | auto |
| `--model_dir` | 모델 디렉토리 | ./models |
| `--config` | JSON 설정 파일 | None |
| `--image_dir` | 이미지 폴더 | None |
| `--video_source` | 동영상 경로 | None |
| `--fps` | 프레임 추출 FPS | 1 |
| `--label_dir` | 라벨 출력 경로 | ./data/labels |
| `--classes` | 클래스 매핑 | person:0,car:1 |
| `--threshold` | 검출 임계값 | 0.3 |
| `--chunk_size` | 프롬프트 청크 크기 | 4 |
| `--show` | 실시간 표시 | False |
| `--save_viz` | 시각화 저장 | False |

### **run_multi_gpu.sh 설정 항목**

```bash
# GPU 설정
GPU_LIST=(0 1 2 3)           # 사용할 GPU 리스트

# 데이터 경로
IMAGE_DIR="/path/to/images"  # 이미지 폴더
LABEL_DIR="/path/to/labels"  # 라벨 출력
VIZ_DIR="/path/to/results"   # 시각화 출력

# 클래스 설정
CLASSES="person:0,car:1"     # 클래스 매핑

# 추론 설정
THRESHOLD=0.3                # 검출 임계값
CHUNK_SIZE=4                 # 청크 크기

# 표시 옵션
SHOW_REALTIME=false          # 실시간 표시
SAVE_VISUALIZATION=true      # 시각화 저장
```

---

## 📊 성능 가이드

### **GPU 메모리별 권장 설정**

| GPU | VRAM | chunk_size | 예상 속도 |
|-----|------|------------|-----------|
| RTX 4090 | 24GB | 6-8 | ~3초/이미지 |
| RTX 3090 | 24GB | 4-6 | ~4초/이미지 |
| RTX 3080 | 10GB | 2-4 | ~5초/이미지 |
| RTX 3070 | 8GB | 2-3 | ~6초/이미지 |

### **프롬프트 개수별 처리 시간**

- **3개 프롬프트**: 약 2-3초/이미지
- **6개 프롬프트**: 약 4-5초/이미지
- **10개 프롬프트**: 약 7-8초/이미지

---

## 🛠️ 트러블슈팅

### **1. 모델 로드 실패**

```python
# 오류: FileNotFoundError: sam3_checkpoint.pth
# 해결: 모델 경로 확인
ls -lh ./models/sam3_checkpoint.pth
```

### **2. CUDA Out of Memory**

```bash
# 해결 1: chunk_size 줄이기
python sam3_offline.py --chunk_size 2

# 해결 2: 프롬프트 개수 줄이기
--classes "person:0,car:1"  # 2개만 사용
```

### **3. GPU 인식 안됨**

```bash
# GPU 확인
nvidia-smi

# CUDA 버전 확인
nvcc --version

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### **4. 한글 경로 문제**

```python
# 코드 내부에서 자동 처리됨
# imread_unicode / imwrite_unicode 함수 사용
```

---

## 📁 출력 구조

```
프로젝트/
├── models/                    # 모델 파일
│   ├── sam3_checkpoint.pth
│   └── bpe_simple_vocab_16e6.txt.gz
├── data/
│   ├── images/               # 입력 이미지
│   ├── labels/               # YOLO 라벨 (출력)
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── results/              # 시각화 (옵션)
│       ├── image1_result.jpg
│       └── image2_result.jpg
└── logs/                     # 실행 로그
    ├── gpu_0.log
    ├── gpu_1.log
    └── ...
```

---

## 🔍 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/gpu_0.log

# 특정 GPU 로그
cat logs/gpu_1.log | grep "성공"

# 오류만 확인
cat logs/gpu_0.log | grep "✗"
```

---

## 📞 문의 및 지원

### **버그 리포트**
- 로그 파일 첨부: `logs/*.log`
- 환경 정보: `config.json`
- GPU 정보: `nvidia-smi` 출력

### **성능 최적화 문의**
- GPU 모델 및 VRAM
- 이미지 해상도
- 프롬프트 개수

---

## 📝 라이센스

이 도구는 SAM3 라이센스를 따릅니다.

---

## 🎉 시작하기

```bash
# 1. 온라인 PC에서
python download_sam3_model.py

# 2. 서버로 전송
scp -r models/ user@server:/project/

# 3. 서버에서 실행
./run_multi_gpu.sh
```

**준비 완료!** 🚀
