"""
SAM3 Model Downloader for Offline Server
온라인 환경에서 실행하여 SAM3 모델을 다운로드하고
오프라인 서버로 이동 가능한 형태로 패키징
"""

import os
import sys
import shutil
import torch
from pathlib import Path
import json
import time
os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6"

import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()


def print_header(text):
    """헤더 출력"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def download_sam3_model(output_dir="./models", hf_token=None):
    """
    SAM3 모델 다운로드 및 저장
    
    Args:
        output_dir: 모델 저장 디렉토리
        hf_token: HuggingFace 토큰 (필요시)
    """
    print_header("SAM3 Model Download - Offline Preparation")
    
    # HuggingFace 토큰 설정
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print(f"✓ HuggingFace 토큰 설정 완료")
    
    # SAM3 import 시도
    try:
        import sam3
        from sam3 import build_sam3_image_model
        print("✓ SAM3 패키지 import 성공")
    except ImportError as e:
        print("✗ SAM3 패키지를 찾을 수 없습니다!")
        print("   다음 명령으로 설치하세요:")
        print("   git clone https://github.com/facebookresearch/sam3.git")
        print("   cd sam3 && pip install -e .")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 출력 디렉토리: {output_dir.absolute()}")
    
    # ========== 1. BPE Vocab 파일 복사 ==========
    print_header("Step 1: BPE Vocab 파일 복사")
    
    sam3_root = Path(sam3.__file__).parent.parent
    bpe_source = sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    bpe_target = output_dir / "bpe_simple_vocab_16e6.txt.gz"
    
    if bpe_source.exists():
        shutil.copy2(bpe_source, bpe_target)
        print(f"✓ BPE Vocab 복사 완료")
        print(f"  From: {bpe_source}")
        print(f"  To:   {bpe_target}")
    else:
        print(f"⚠ BPE Vocab 파일을 찾을 수 없습니다: {bpe_source}")
        print("  수동으로 복사해주세요.")
    
    # ========== 2. SAM3 모델 다운로드 ==========
    print_header("Step 2: SAM3 모델 다운로드")
    
    print("모델 다운로드 중... (시간이 걸릴 수 있습니다)")
    print("HuggingFace에서 자동으로 다운로드됩니다.\n")
    
    start_time = time.time()
    
    try:
        # 모델 빌드 (자동 다운로드)
        model = build_sam3_image_model(bpe_path=str(bpe_target))
        
        download_time = time.time() - start_time
        print(f"✓ 모델 다운로드 완료 ({download_time:.2f}초)")
        
    except Exception as e:
        print(f"✗ 모델 다운로드 실패: {e}")
        print("\n가능한 원인:")
        print("  1. 인터넷 연결 문제")
        print("  2. HuggingFace 토큰 필요")
        print("  3. 저장 공간 부족")
        sys.exit(1)
    
    # ========== 3. 모델 State Dict 저장 ==========
    print_header("Step 3: 모델 State Dict 저장")
    
    checkpoint_path = output_dir / "sam3_checkpoint.pth"
    
    try:
        # State dict 저장
        torch.save(model.state_dict(), checkpoint_path)
        
        # 파일 크기 확인
        file_size = checkpoint_path.stat().st_size / (1024**3)  # GB
        print(f"✓ Checkpoint 저장 완료")
        print(f"  경로: {checkpoint_path}")
        print(f"  크기: {file_size:.2f} GB")
        
    except Exception as e:
        print(f"✗ Checkpoint 저장 실패: {e}")
        sys.exit(1)
    
    # ========== 4. HuggingFace 캐시 찾기 ==========
    print_header("Step 4: HuggingFace 캐시 분석")
    
    # HF 캐시 디렉토리
    hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface'))
    hub_cache = hf_home / 'hub'
    
    print(f"HuggingFace 캐시 위치: {hub_cache}")
    
    if hub_cache.exists():
        # SAM3 관련 캐시 찾기
        sam3_models = list(hub_cache.glob("models--*sam3*"))
        
        if sam3_models:
            print(f"✓ SAM3 캐시 발견: {len(sam3_models)}개")
            
            for model_cache in sam3_models:
                print(f"\n  📁 {model_cache.name}")
                
                # snapshots 폴더에서 실제 파일 찾기
                snapshots = model_cache / "snapshots"
                if snapshots.exists():
                    for snapshot in snapshots.iterdir():
                        if snapshot.is_dir():
                            files = list(snapshot.iterdir())
                            print(f"     └─ {snapshot.name[:12]}... ({len(files)} files)")
                            
                            # 주요 파일 표시
                            for f in files[:5]:  # 최대 5개만
                                size_mb = f.stat().st_size / (1024**2)
                                print(f"        - {f.name} ({size_mb:.1f} MB)")
                            
                            if len(files) > 5:
                                print(f"        ... and {len(files)-5} more files")
        else:
            print("⚠ SAM3 캐시를 찾을 수 없습니다.")
            print("  모델이 다른 위치에 다운로드되었을 수 있습니다.")
    else:
        print(f"⚠ HuggingFace 캐시 디렉토리가 없습니다: {hub_cache}")
    
    # ========== 5. Config 파일 생성 ==========
    print_header("Step 5: Config 파일 생성")
    
    config = {
        "model_type": "sam3",
        "architecture": "SAM3 Image Model",
        "checkpoint_file": "sam3_checkpoint.pth",
        "bpe_vocab_file": "bpe_simple_vocab_16e6.txt.gz",
        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        config["cuda_version"] = torch.version.cuda
        config["gpu_name"] = torch.cuda.get_device_name(0)
    
    config_path = output_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Config 파일 생성: {config_path}")
    
    # ========== 6. README 생성 ==========
    print_header("Step 6: README 생성")
    
    readme_content = f"""# SAM3 Model Package for Offline Server

## 📦 포함된 파일

- `sam3_checkpoint.pth` - SAM3 모델 체크포인트 ({file_size:.2f} GB)
- `bpe_simple_vocab_16e6.txt.gz` - BPE Vocabulary
- `config.json` - 모델 설정 정보
- `README.md` - 이 파일

## 📥 다운로드 정보

- 다운로드 날짜: {time.strftime("%Y-%m-%d %H:%M:%S")}
- PyTorch 버전: {torch.__version__}
- CUDA 사용 가능: {torch.cuda.is_available()}

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
   - 전체 약 {file_size:.2f} GB
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
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✓ README 생성: {readme_path}")
    
    # ========== 7. 최종 요약 ==========
    print_header("다운로드 완료!")
    
    print("\n📦 생성된 파일:")
    for file in output_dir.iterdir():
        if file.is_file():
            size = file.stat().st_size
            if size > 1024**3:  # GB
                size_str = f"{size/(1024**3):.2f} GB"
            elif size > 1024**2:  # MB
                size_str = f"{size/(1024**2):.2f} MB"
            else:  # KB
                size_str = f"{size/1024:.2f} KB"
            
            print(f"  ✓ {file.name} ({size_str})")
    
    print(f"\n📁 전체 저장 위치: {output_dir.absolute()}")
    
    print("\n🚀 다음 단계:")
    print("  1. models/ 폴더를 서버로 복사")
    print("  2. 서버에서 sam3_offline.py 실행")
    print("  3. GPU별로 run_multi_gpu.sh로 분산 실행")
    
    print("\n" + "=" * 70)
    print("  준비 완료! 서버로 이동하세요.")
    print("=" * 70 + "\n")
    
    return output_dir


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SAM3 모델 다운로드 (오프라인 서버 준비용)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="모델 저장 디렉토리 (default: ./models)"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace 토큰 (필요시)"
    )
    
    args = parser.parse_args()
    
    try:
        download_sam3_model(
            output_dir=args.output_dir,
            hf_token=args.hf_token
        )
    except KeyboardInterrupt:
        print("\n\n⚠ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
