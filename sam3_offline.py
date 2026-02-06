"""
SAM3 YOLO Dataset Creator - Offline Server Version
오프라인 리눅스 서버용 (GPU별 분산 실행)
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json

# ========== 오프라인 모드 설정 ==========
# HuggingFace 오프라인 모드 강제 설정
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ========== CUDA 경로 자동 설정 (OS 감지) ==========
if "CUDA_PATH" not in os.environ:
    import platform
    system = platform.system()
    
    if system == "Windows":
        # Windows CUDA 경로
        possible_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        ]
    else:  # Linux
        # Linux CUDA 경로
        possible_paths = [
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.1",
            "/usr/local/cuda-11.8",
            "/usr/local/cuda",
        ]
    
    # 존재하는 경로 찾기
    for cuda_path in possible_paths:
        if os.path.exists(cuda_path):
            os.environ["CUDA_PATH"] = cuda_path
            break
    else:
        # CUDA 없어도 진행 (Triton 오류 방지용 빈 문자열)
        os.environ["CUDA_PATH"] = ""

import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

# SAM3 imports
from sam3 import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata, FindQueryLoaded, 
    Image as SAMImage, Datapoint
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
)
from sam3.eval.postprocessors import PostProcessImage
from sam3.eval.postprocessors_classwise import (
    PostProcessImageWithClassThresholds,
    create_postprocessor_from_config
)

# Global counter for query IDs
GLOBAL_COUNTER = 1


def recursive_to_device(obj, device):
    """재귀적으로 모든 텐서를 device로 이동"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_device(item, device) for item in obj)
    elif hasattr(obj, '__dict__'):
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr = getattr(obj, attr_name)
                    if isinstance(attr, (torch.Tensor, dict, list, tuple)) or hasattr(attr, '__dict__'):
                        setattr(obj, attr_name, recursive_to_device(attr, device))
                except (AttributeError, TypeError):
                    pass
        return obj
    else:
        return obj


def setup_environment(gpu_id=None, device='auto'):
    """
    환경 설정
    
    Args:
        gpu_id: GPU 인덱스 (0, 1, 2, ...) - None이면 자동 선택
        device: 'auto', 'cuda', 'cpu'
    """
    print("=" * 60)
    print("환경 설정 중...")
    print("=" * 60)
    
    # GPU 인덱스 지정
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"✓ GPU 인덱스 지정: {gpu_id}")
        device = 'cuda'  # GPU 지정 시 자동으로 CUDA 모드
    
    # 디바이스 설정
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ CUDA 자동 감지: {torch.cuda.get_device_name(0)}")
            print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            print("⚠ CUDA 없음 - CPU 모드")
    elif device == 'cuda' or device.startswith('cuda:'):
        if torch.cuda.is_available():
            print(f"✓ CUDA 사용: {torch.cuda.get_device_name(0)}")
            print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            device = 'cuda'  # 'cuda:0' → 'cuda'로 통일
        else:
            print("✗ CUDA 불가 - CPU로 전환")
            device = 'cpu'
    elif device == 'cpu':
        device = 'cpu'
        print("✓ CPU 모드 선택")
    else:
        # 알 수 없는 device 값
        print(f"⚠ 알 수 없는 device: {device}, CPU로 전환")
        device = 'cpu'
    
    # CUDA 최적화 설정
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.inference_mode().__enter__()
        print("✓ CUDA 최적화 활성화 (TF32, bfloat16)")
    
    print()
    return device


def load_model(model_dir, device='cuda'):
    """
    SAM3 모델 로드 (오프라인 - 로컬 경로)
    
    Args:
        model_dir: 모델 디렉토리 경로 (checkpoint, bpe_vocab 포함)
        device: 'cuda' 또는 'cpu'
    """
    print("=" * 60)
    print("모델 로드 중... (오프라인 모드)")
    print("=" * 60)
    
    model_dir = Path(model_dir)
    
    # 필수 파일 확인
    checkpoint_path = model_dir / "sam3_checkpoint.pth"
    bpe_path = model_dir / "bpe_simple_vocab_16e6.txt.gz"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    if not bpe_path.exists():
        raise FileNotFoundError(f"BPE vocab을 찾을 수 없습니다: {bpe_path}")
    
    print(f"✓ 체크포인트: {checkpoint_path}")
    print(f"✓ BPE Vocab: {bpe_path}")
    
    # 모델 빌드
    start_time = time.time()
    print("\n모델 구조 초기화 중...")
    
    try:
        # 빈 모델 생성 (오프라인 모드)
        model = build_sam3_image_model(bpe_path=str(bpe_path))
        print("✓ 모델 구조 생성 완료")
        
        # 체크포인트 로드
        print("✓ 체크포인트 로드 중...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print("✓ 체크포인트 로드 완료")
        
        load_time = time.time() - start_time
        print(f"✓ 전체 로드 시간: {load_time:.2f}초")
        
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        raise
    
    # 디바이스로 이동
    if device.startswith('cuda'):
        model = model.cuda()
        print("✓ 모델을 GPU로 이동")
        
        # CUDA 캐시 워밍업
        print("✓ CUDA 캐시 워밍업 중...")
        try:
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
                decoder = model.transformer.decoder
                target_device = torch.device('cuda')
                
                # compilable_cord_cache를 CUDA로 강제 이동
                if hasattr(decoder, 'compilable_cord_cache') and decoder.compilable_cord_cache is not None:
                    coords_h, coords_w = decoder.compilable_cord_cache
                    decoder.compilable_cord_cache = (
                        coords_h.to(target_device),
                        coords_w.to(target_device)
                    )
                    print(f"  - compilable_cord_cache → {target_device}")
                
                # coord_cache 딕셔너리의 모든 엔트리를 CUDA로 이동
                if hasattr(decoder, 'coord_cache'):
                    for feat_size, (coords_h, coords_w) in list(decoder.coord_cache.items()):
                        decoder.coord_cache[feat_size] = (
                            coords_h.to(target_device),
                            coords_w.to(target_device)
                        )
                    if decoder.coord_cache:
                        print(f"  - coord_cache ({len(decoder.coord_cache)} entries) → {target_device}")
                
                # Monkey patch _get_rpb_matrix
                original_get_rpb_matrix = decoder._get_rpb_matrix
                
                def patched_get_rpb_matrix(reference_boxes, feat_size):
                    """Patched version that ensures coords are on same device as reference_boxes"""
                    from sam3.model.box_ops import box_cxcywh_to_xyxy
                    
                    H, W = feat_size
                    boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
                    bs, num_queries, _ = boxes_xyxy.shape
                    
                    target_dev = reference_boxes.device
                    
                    if decoder.compilable_cord_cache is None:
                        coords_h, coords_w = decoder._get_coords(H, W, target_dev)
                        decoder.compilable_cord_cache = (coords_h, coords_w)
                        decoder.compilable_stored_size = (H, W)
                    
                    if torch.compiler.is_dynamo_compiling() or decoder.compilable_stored_size == (H, W):
                        coords_h, coords_w = decoder.compilable_cord_cache
                        if coords_h.device != target_dev:
                            coords_h = coords_h.to(target_dev)
                            coords_w = coords_w.to(target_dev)
                            decoder.compilable_cord_cache = (coords_h, coords_w)
                    else:
                        if feat_size not in decoder.coord_cache:
                            decoder.coord_cache[feat_size] = decoder._get_coords(H, W, target_dev)
                        coords_h, coords_w = decoder.coord_cache[feat_size]
                        if coords_h.device != target_dev:
                            coords_h = coords_h.to(target_dev)
                            coords_w = coords_w.to(target_dev)
                            decoder.coord_cache[feat_size] = (coords_h, coords_w)
                    
                    deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
                    deltas_y = deltas_y.view(bs, num_queries, -1, 2)
                    deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
                    deltas_x = deltas_x.view(bs, num_queries, -1, 2)
                    
                    if decoder.boxRPB in ["log", "both"]:
                        deltas_x_log = deltas_x * 8
                        deltas_x_log = (
                            torch.sign(deltas_x_log)
                            * torch.log2(torch.abs(deltas_x_log) + 1.0)
                            / np.log2(8)
                        )
                        deltas_y_log = deltas_y * 8
                        deltas_y_log = (
                            torch.sign(deltas_y_log)
                            * torch.log2(torch.abs(deltas_y_log) + 1.0)
                            / np.log2(8)
                        )
                        if decoder.boxRPB == "log":
                            deltas_x = deltas_x_log
                            deltas_y = deltas_y_log
                        else:
                            deltas_x = torch.cat([deltas_x, deltas_x_log], dim=-1)
                            deltas_y = torch.cat([deltas_y, deltas_y_log], dim=-1)
                    
                    deltas_x = decoder.boxRPB_embed_x(deltas_x)
                    deltas_y = decoder.boxRPB_embed_y(deltas_y)
                    
                    B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
                    B = B.flatten(2, 3)
                    B = B.permute(0, 3, 1, 2)
                    B = B.contiguous()
                    return B
                
                decoder._get_rpb_matrix = patched_get_rpb_matrix
                print(f"  - _get_rpb_matrix patched for device consistency")
                
        except Exception as e:
            print(f"  ⚠ 캐시 워밍업 중 오류 (무시): {e}")
    else:
        model = model.cpu()
        print("✓ 모델을 CPU에 유지")
    
    print()
    return model


def create_transforms():
    """전처리 Transform 생성"""
    return ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_postprocessor(detection_config=None, class_mapping=None, device='cuda'):
    """
    후처리 PostProcessor 생성

    Args:
        detection_config: detection 설정 딕셔너리
            {
                "use_presence": false,
                "default_threshold": 0.3,
                "class_thresholds": {"helmet": 0.15, ...},
                "max_dets_per_img": 100
            }
        class_mapping: 클래스 이름 → ID 매핑
        device: 'cuda' or 'cpu'
    """
    # 기본값 설정
    if detection_config is None:
        detection_config = {
            "use_presence": True,
            "default_threshold": 0.3,
            "max_dets_per_img": 100
        }

    use_presence = detection_config.get('use_presence', True)
    default_threshold = detection_config.get('default_threshold', 0.3)
    class_thresholds = detection_config.get('class_thresholds', {})
    max_dets = detection_config.get('max_dets_per_img', 100)

    # CPU/GPU 설정
    to_cpu = (device == 'cpu')
    use_gpu_interpolate = (device != 'cpu')

    # 클래스별 threshold가 있으면 ClassWise 버전 사용
    if class_thresholds and class_mapping:
        print(f"📊 클래스별 threshold 사용:")
        print(f"   use_presence: {use_presence}")
        print(f"   default_threshold: {default_threshold}")
        for class_name, threshold in class_thresholds.items():
            print(f"   {class_name}: {threshold}")

        return PostProcessImageWithClassThresholds(
            max_dets_per_img=max_dets,
            class_thresholds=class_thresholds,
            class_to_id=class_mapping,
            use_presence=use_presence,
            detection_threshold=default_threshold,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            to_cpu=to_cpu,
            always_interpolate_masks_on_gpu=use_gpu_interpolate
        )
    else:
        print(f"📊 단일 threshold 사용:")
        print(f"   use_presence: {use_presence}")
        print(f"   threshold: {default_threshold}")

        return PostProcessImage(
            max_dets_per_img=max_dets,
            use_presence=use_presence,
            detection_threshold=default_threshold,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            to_cpu=to_cpu,
            always_interpolate_masks_on_gpu=use_gpu_interpolate
        )


def create_datapoint_with_prompts(pil_image, text_prompts):
    """이미지와 여러 프롬프트로 Datapoint 생성"""
    global GLOBAL_COUNTER
    
    datapoint = Datapoint(find_queries=[], images=[])
    
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
    
    prompt_ids = []
    for text_query in text_prompts:
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=GLOBAL_COUNTER,
                    original_image_id=GLOBAL_COUNTER,
                    original_category_id=1,
                    original_size=[h, w],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
        prompt_ids.append(GLOBAL_COUNTER)
        GLOBAL_COUNTER += 1
    
    return datapoint, prompt_ids


def parse_image_source(image_source):
    """이미지 소스 파싱"""
    image_paths = []
    
    if os.path.isdir(image_source):
        print(f"📁 폴더에서 이미지 검색: {image_source}")
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_paths_set = set()
        for ext in extensions:
            image_paths_set.update(Path(image_source).glob(f"*{ext}"))
            image_paths_set.update(Path(image_source).glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in sorted(image_paths_set)]
        print(f"  ✓ {len(image_paths)}개 이미지 발견")
    
    elif os.path.isfile(image_source):
        print(f"📄 리스트 파일에서 이미지 로드: {image_source}")
        
        with open(image_source, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line):
                    image_paths.append(line)
        
        print(f"  ✓ {len(image_paths)}개 이미지 로드")
    
    else:
        raise ValueError(f"잘못된 소스: {image_source}")
    
    if len(image_paths) == 0:
        raise ValueError("이미지를 찾을 수 없습니다")
    
    return sorted(image_paths)


def parse_video_source(video_source):
    """동영상 소스 파싱"""
    video_paths = []
    
    if os.path.isdir(video_source):
        print(f"📁 폴더에서 동영상 검색: {video_source}")
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        video_paths_set = set()
        for ext in extensions:
            video_paths_set.update(Path(video_source).glob(f"*{ext}"))
            video_paths_set.update(Path(video_source).glob(f"*{ext.upper()}"))
        
        video_paths = [str(p) for p in sorted(video_paths_set)]
        print(f"  ✓ {len(video_paths)}개 동영상 발견")
    
    elif os.path.isfile(video_source):
        print(f"📄 동영상 파일: {video_source}")
        video_paths = [video_source]
    
    else:
        raise ValueError(f"잘못된 동영상 소스: {video_source}")
    
    if len(video_paths) == 0:
        raise ValueError("동영상을 찾을 수 없습니다")
    
    return sorted(video_paths)


def parse_class_mapping(class_str):
    """클래스 매핑 파싱"""
    if isinstance(class_str, dict):
        return class_str
    
    mapping = {}
    pairs = class_str.split(',')
    
    for pair in pairs:
        pair = pair.strip()
        if ':' in pair:
            name, idx = pair.split(':')
            mapping[name.strip()] = int(idx.strip())
    
    return mapping


def compute_box_iou_matrix(boxes_a, boxes_b):
    """두 박스 집합 간의 IoU 행렬 계산

    Args:
        boxes_a: (N, 4) numpy array [x1, y1, x2, y2]
        boxes_b: (M, 4) numpy array [x1, y1, x2, y2]
    Returns:
        iou_matrix: (N, M) numpy array
    """
    x1_a, y1_a, x2_a, y2_a = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    x1_b, y1_b, x2_b, y2_b = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    inter_x1 = np.maximum(x1_a[:, None], x1_b[None, :])
    inter_y1 = np.maximum(y1_a[:, None], y1_b[None, :])
    inter_x2 = np.minimum(x2_a[:, None], x2_b[None, :])
    inter_y2 = np.minimum(y2_a[:, None], y2_b[None, :])

    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


def apply_nms_to_results(results_by_prompt, nms_config, class_mapping):
    """결과에 NMS 적용 (global 또는 per_class 모드)

    Args:
        results_by_prompt: {prompt_name: {'boxes': np.array, 'scores': np.array}}
        nms_config: {'enabled': bool, 'mode': 'global'|'per_class', 'iou_threshold': float}
        class_mapping: {prompt_name: class_id}
    Returns:
        results_by_prompt: NMS가 적용된 결과
    """
    if not nms_config.get('enabled', False):
        return results_by_prompt

    iou_threshold = nms_config.get('iou_threshold', 0.5)
    mode = nms_config.get('mode', 'global')

    if mode == 'per_class':
        # 클래스별 독립 NMS
        for prompt_name in results_by_prompt:
            result = results_by_prompt[prompt_name]
            boxes = result['boxes']
            scores = result['scores']

            if len(boxes) == 0:
                continue

            if boxes.ndim != 2:
                continue

            keep = _nms_boxes(boxes, scores, iou_threshold)
            results_by_prompt[prompt_name] = {
                'boxes': boxes[keep],
                'scores': scores[keep]
            }

    elif mode == 'global':
        # 전체 클래스 글로벌 NMS - 모든 클래스를 합쳐서 NMS 적용
        all_boxes = []
        all_scores = []
        all_labels = []
        all_indices = []  # (prompt_name, original_index)

        for prompt_name in results_by_prompt:
            result = results_by_prompt[prompt_name]
            boxes = result['boxes']
            scores = result['scores']

            if len(boxes) == 0:
                continue

            if boxes.ndim != 2:
                continue

            for i in range(len(boxes)):
                all_boxes.append(boxes[i])
                all_scores.append(scores[i])
                all_labels.append(prompt_name)
                all_indices.append((prompt_name, i))

        if len(all_boxes) == 0:
            return results_by_prompt

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)

        # 글로벌 NMS 적용 - 클래스 무관하게 스코어 순으로 억제
        keep = _nms_boxes(all_boxes, all_scores, iou_threshold)

        # 결과 재구성
        new_results = {name: {'boxes': [], 'scores': []} for name in results_by_prompt}
        for idx in keep:
            prompt_name = all_labels[idx]
            new_results[prompt_name]['boxes'].append(all_boxes[idx])
            new_results[prompt_name]['scores'].append(all_scores[idx])

        for prompt_name in new_results:
            boxes = new_results[prompt_name]['boxes']
            scores = new_results[prompt_name]['scores']
            if len(boxes) > 0:
                new_results[prompt_name]['boxes'] = np.array(boxes)
                new_results[prompt_name]['scores'] = np.array(scores)
            else:
                new_results[prompt_name]['boxes'] = np.array([])
                new_results[prompt_name]['scores'] = np.array([])

        results_by_prompt = new_results

    return results_by_prompt


def _nms_boxes(boxes, scores, iou_threshold):
    """단순 박스 NMS (greedy)

    Args:
        boxes: (N, 4) numpy array [x1, y1, x2, y2]
        scores: (N,) numpy array
        iou_threshold: float
    Returns:
        keep: list of indices to keep
    """
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        remaining = order[1:]
        ious = compute_box_iou_matrix(
            boxes[i:i+1], boxes[remaining]
        )[0]

        mask = ious <= iou_threshold
        order = remaining[mask]

    return keep


def bbox_to_yolo_format(box, img_width, img_height):
    """박스를 YOLO 형식으로 변환"""
    x1, y1, x2, y2 = box
    
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return x_center, y_center, width, height


def save_yolo_annotation(image_path, results_by_prompt, class_mapping, output_dir, img_width, img_height):
    """YOLO 형식 어노테이션 저장"""
    image_name = Path(image_path).stem
    txt_path = os.path.join(output_dir, f"{image_name}.txt")
    
    lines = []
    total_objects = 0
    
    for prompt_name, result in results_by_prompt.items():
        if result is None or len(result['boxes']) == 0:
            continue
        
        class_id = class_mapping.get(prompt_name, -1)
        if class_id < 0:
            continue
        
        boxes = result['boxes']
        scores = result['scores']
        
        for idx, (box, score) in enumerate(zip(boxes, scores)):
            x_center, y_center, width, height = bbox_to_yolo_format(
                box, img_width, img_height
            )
            
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            lines.append(line)
            total_objects += 1
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return total_objects


def imread_unicode(image_path):
    """한글 경로 지원 이미지 읽기"""
    import cv2
    import numpy as np
    
    try:
        stream = open(image_path, "rb")
        bytes_data = bytearray(stream.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        stream.close()
        return image
    except:
        return None


def imwrite_unicode(image_path, image):
    """한글 경로 지원 이미지 저장"""
    import cv2
    
    try:
        ext = os.path.splitext(image_path)[1]
        result, encoded_img = cv2.imencode(ext, image)
        if result:
            with open(image_path, mode='w+b') as f:
                encoded_img.tofile(f)
            return True
        return False
    except:
        return False


def save_visualization_result(image_path, results_by_prompt, class_mapping, output_dir):
    """시각화 결과 저장"""
    import cv2
    
    image = imread_unicode(image_path)
    if image is None:
        return None
    
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
    ]
    
    color_idx = 0
    total_objects = 0
    
    for prompt_name, result in results_by_prompt.items():
        if result is None or len(result['boxes']) == 0:
            continue
        
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        boxes = result['boxes']
        scores = result['scores']
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{prompt_name}: {score:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            total_objects += 1
    
    info_text = f"{Path(image_path).name} | Objects: {total_objects}"
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).stem
    save_path = os.path.join(output_dir, f"{image_name}_result.jpg")
    imwrite_unicode(save_path, image)
    
    return save_path


def show_realtime_result(image_path, results_by_prompt, class_mapping, window_name="SAM3 Detection"):
    """실시간 결과 표시"""
    import cv2
    
    image = imread_unicode(image_path)
    if image is None:
        return
    
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (0, 255, 255), (255, 0, 255), (255, 255, 0),
    ]
    
    color_idx = 0
    total_objects = 0
    
    for prompt_name, result in results_by_prompt.items():
        if result is None or len(result['boxes']) == 0:
            continue
        
        color = colors[color_idx % len(colors)]
        color_idx += 1
        
        boxes = result['boxes']
        scores = result['scores']
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{prompt_name}: {score:.2f}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )
            
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 0, 0),
                -1
            )
            
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            total_objects += 1
    
    info_text = f"{Path(image_path).name} | Objects: {total_objects}"
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.imshow(window_name, image)
    cv2.waitKey(200)


def extract_frames_from_videos(video_source, jpeg_output_dir, fps_extraction=1, verbose=True):
    """동영상에서 프레임 추출

    Args:
        video_source: 동영상 파일/폴더 경로
        jpeg_output_dir: JPEGImages 저장 경로
        fps_extraction: N프레임마다 1번 추출 (1=매 프레임, 30=30프레임마다 1번, 0/-1=원본 전체)
        verbose: 상세 출력 여부
    """
    import cv2

    print("\n" + "=" * 60)
    print("동영상 프레임 추출 시작")
    print("=" * 60)

    video_paths = parse_video_source(video_source)

    os.makedirs(jpeg_output_dir, exist_ok=True)
    print(f"📁 JPEGImages 저장 경로: {jpeg_output_dir}\n")

    total_extracted = 0
    global_frame_index = 1

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    for video_idx, video_path in enumerate(video_paths):
        video_name = Path(video_path).stem

        if verbose:
            print(f"\n[{video_idx+1}/{len(video_paths)}] {video_name}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  ✗ 동영상을 열 수 없습니다: {video_path}")
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        if verbose:
            print(f"  원본 FPS: {original_fps:.2f}")
            print(f"  총 프레임: {total_frames}")
            print(f"  길이: {duration:.2f}초")

        if fps_extraction <= 0:
            # 0 또는 음수면 원본 전체 추출
            frame_interval = 1
            if verbose:
                print(f"  추출 모드: 원본 전체 (매 1프레임)")
        else:
            # fps_extraction = N프레임마다 1번 추출
            frame_interval = fps_extraction
            if verbose:
                print(f"  추출 모드: 매 {frame_interval}프레임마다 1번")
                actual_fps = original_fps / frame_interval if frame_interval > 0 else original_fps
                print(f"  실제 추출 FPS: {actual_fps:.2f}fps")
        
        estimated_frames = total_frames // frame_interval
        if verbose:
            print(f"  예상 추출: {estimated_frames}프레임\n")
        
        frame_count = 0
        extracted_count = 0
        
        iterator = tqdm(total=total_frames, desc=f"  {video_name}") if use_tqdm else range(total_frames)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_filename = f"{video_name}_frame_{global_frame_index:06d}.jpg"
                frame_path = os.path.join(jpeg_output_dir, frame_filename)
                
                success = imwrite_unicode(frame_path, frame)
                
                if success:
                    extracted_count += 1
                    global_frame_index += 1
            
            frame_count += 1
            
            if use_tqdm:
                iterator.update(1)
        
        if use_tqdm:
            iterator.close()
        
        cap.release()
        
        if verbose:
            print(f"  ✓ 추출 완료: {extracted_count}프레임")
        
        total_extracted += extracted_count
    
    print("\n" + "=" * 60)
    print("프레임 추출 완료!")
    print("=" * 60)
    print(f"✓ 처리 동영상: {len(video_paths)}개")
    print(f"✓ 추출 프레임: {total_extracted}개")
    print(f"✓ 저장 경로: {jpeg_output_dir}")
    print("=" * 60 + "\n")
    
    return total_extracted


def process_single_image_batch(
    image_path, model, transform, postprocessor, prompts,
    class_mapping, output_dir, device='cuda',
    show_realtime=False, save_visualizations=False,
    visualization_dir=None, window_name="SAM3 Detection",
    prompt_chunk_size=4, nms_config=None
):
    """단일 이미지 배치 처리"""
    try:
        total_start = time.time()
        
        load_start = time.time()
        pil_image = Image.open(image_path)

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image = Image.fromarray(np.array(pil_image))

        img_width, img_height = pil_image.size
        load_time = time.time() - load_start
        
        results_by_prompt = {}
        
        total_prep_time = 0
        total_inference_time = 0
        total_post_time = 0
        
        num_chunks = (len(prompts) + prompt_chunk_size - 1) // prompt_chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * prompt_chunk_size
            end_idx = min(start_idx + prompt_chunk_size, len(prompts))
            chunk_prompts = prompts[start_idx:end_idx]
            
            prep_start = time.time()
            datapoint, prompt_ids = create_datapoint_with_prompts(pil_image, chunk_prompts)
            
            datapoint = transform(datapoint)
            
            batch = collate([datapoint], dict_key="dummy")["dummy"]
            
            target_device = torch.device(device if device.startswith('cuda') else 'cpu')
            
            batch = copy_data_to_device(batch, target_device, non_blocking=True)
            
            batch = recursive_to_device(batch, target_device)
            
            prep_time = time.time() - prep_start
            total_prep_time += prep_time
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            
            inference_start = time.time()
            output = model(batch)
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            post_start = time.time()
            processed_results = postprocessor.process_results(output, batch.find_metadatas)
            post_time = time.time() - post_start
            total_post_time += post_time
            
            if not isinstance(processed_results, list):
                if isinstance(processed_results, dict):
                    for prompt_name, prompt_id in zip(chunk_prompts, prompt_ids):
                        if prompt_id in processed_results:
                            result = processed_results[prompt_id]
                            boxes = result['boxes'].float().cpu().numpy() if hasattr(result['boxes'], 'cpu') else result['boxes']
                            scores = result['scores'].float().cpu().numpy() if hasattr(result['scores'], 'cpu') else result['scores']
                            
                            results_by_prompt[prompt_name] = {
                                'boxes': boxes,
                                'scores': scores
                            }
                        else:
                            if prompt_name not in results_by_prompt:
                                results_by_prompt[prompt_name] = {
                                    'boxes': np.array([]),
                                    'scores': np.array([])
                                }
                else:
                    for prompt_name in chunk_prompts:
                        if prompt_name not in results_by_prompt:
                            results_by_prompt[prompt_name] = {
                                'boxes': np.array([]),
                                'scores': np.array([])
                            }
            else:
                for result in processed_results:
                    if isinstance(result, dict) and 'query_id' in result:
                        query_id = result['query_id']
                        
                        for prompt_name, prompt_id in zip(chunk_prompts, prompt_ids):
                            if query_id == prompt_id:
                                boxes = result['boxes'].float().cpu().numpy() if hasattr(result['boxes'], 'cpu') else result['boxes']
                                scores = result['scores'].float().cpu().numpy() if hasattr(result['scores'], 'cpu') else result['scores']
                                results_by_prompt[prompt_name] = {
                                    'boxes': boxes,
                                    'scores': scores
                                }
                                break
                
                for prompt_name in chunk_prompts:
                    if prompt_name not in results_by_prompt:
                        results_by_prompt[prompt_name] = {
                            'boxes': np.array([]),
                            'scores': np.array([])
                        }
            
            if device.startswith('cuda'):
                del batch, output, processed_results
                torch.cuda.empty_cache()

        # NMS 적용 (global 또는 per_class)
        if nms_config and nms_config.get('enabled', False):
            results_by_prompt = apply_nms_to_results(
                results_by_prompt, nms_config, class_mapping
            )

        save_start = time.time()
        num_objects = save_yolo_annotation(
            image_path, results_by_prompt, class_mapping, 
            output_dir, img_width, img_height
        )
        save_time = time.time() - save_start
        
        if show_realtime:
            show_realtime_result(image_path, results_by_prompt, class_mapping, window_name)
        
        visualization_path = None
        if save_visualizations and visualization_dir:
            visualization_path = save_visualization_result(
                image_path, results_by_prompt, class_mapping, visualization_dir
            )
        
        total_time = time.time() - total_start
        
        return {
            'success': True,
            'num_objects': num_objects,
            'image_size': (img_width, img_height),
            'visualization_path': visualization_path,
            'timing': {
                'total': total_time,
                'load': load_time,
                'preprocess': total_prep_time,
                'inference': total_inference_time,
                'postprocess': total_post_time,
                'save': save_time
            },
            'num_chunks': num_chunks
        }
        
    except Exception as e:
        import traceback
        print(f"\n오류 발생: {image_path}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'num_objects': 0,
            'visualization_path': None
        }


def create_yolo_dataset(
    image_source,
    output_dir,
    class_mapping,
    model_dir,
    prompts=None,
    model=None,
    transform=None,
    postprocessor=None,
    detection_threshold=0.3,
    device='cuda',
    prompt_chunk_size=4,
    verbose=True,
    show_realtime=False,
    save_visualizations=False,
    visualization_dir=None,
    nms_config=None
):
    """YOLO 형식 데이터셋 생성"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "YOLO Dataset Creation Tool" + " " * 12 + "║")
    print("║" + " " * 15 + "(Batch Inference Mode)" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    start_time = time.time()
    
    print("=" * 60)
    print("설정 확인")
    print("=" * 60)
    
    class_mapping = parse_class_mapping(class_mapping)
    print(f"클래스 매핑: {class_mapping}")
    
    if prompts is None:
        prompts = list(class_mapping.keys())
    print(f"프롬프트: {prompts}")
    print(f"프롬프트 총 개수: {len(prompts)}개")
    print(f"프롬프트 청크 크기: {prompt_chunk_size}개")
    print(f"청크 수: {(len(prompts) + prompt_chunk_size - 1) // prompt_chunk_size}개")
    print(f"검출 임계값: {detection_threshold}")
    print(f"디바이스: {device}")
    if nms_config and nms_config.get('enabled', False):
        nms_mode = nms_config.get('mode', 'global')
        nms_iou = nms_config.get('iou_threshold', 0.5)
        print(f"NMS: {nms_mode} (IoU threshold: {nms_iou})")
    else:
        print(f"NMS: disabled")
    print(f"실시간 표시: {show_realtime}")
    print(f"시각화 저장: {save_visualizations}")
    if save_visualizations and visualization_dir:
        print(f"시각화 디렉토리: {visualization_dir}")
    print()
    
    print("=" * 60)
    print("이미지 로드")
    print("=" * 60)
    image_paths = parse_image_source(image_source)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 라벨 출력 디렉토리: {output_dir}")
    
    if save_visualizations and visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
        print(f"📁 시각화 출력 디렉토리: {visualization_dir}")
    print()
    
    if model is None:
        model = load_model(model_dir, device)
    
    if transform is None:
        transform = create_transforms()
        print("✓ Transform 생성 완료")
    
    if postprocessor is None:
        # detection_config 기본값 생성
        default_detection_config = {
            "use_presence": True,
            "default_threshold": detection_threshold,
            "max_dets_per_img": 100
        }
        postprocessor = create_postprocessor(
            detection_config=default_detection_config,
            class_mapping=class_mapping,
            device=device
        )
        print("✓ PostProcessor 생성 완료")
    
    print()
    
    window_name = "SAM3 Real-time Detection"
    if show_realtime:
        import cv2
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("🖥️  실시간 표시 윈도우 생성\n")
    
    print("=" * 60)
    print("이미지 처리 시작 (배치 추론 모드 + 청크 처리)")
    print("=" * 60)
    num_chunks = (len(prompts) + prompt_chunk_size - 1) // prompt_chunk_size
    print(f"✓ 프롬프트 {len(prompts)}개를 {num_chunks}개 청크로 나눠서 처리")
    print(f"✓ 청크당 {prompt_chunk_size}개 프롬프트 동시 처리")
    print(f"✓ 이미지당 총 {num_chunks}번 forward\n")
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    total_objects = 0
    success_count = 0
    fail_count = 0
    
    timing_stats = {
        'total': [],
        'load': [],
        'preprocess': [],
        'inference': [],
        'postprocess': [],
        'save': []
    }
    
    first_inference_time = None
    
    iterator = tqdm(image_paths, desc="Processing") if use_tqdm else image_paths
    
    for idx, image_path in enumerate(iterator):
        if not use_tqdm and verbose:
            print(f"[{idx+1}/{len(image_paths)}] {Path(image_path).name}", end=" ... ")
        
        result = process_single_image_batch(
            image_path, model, transform, postprocessor, prompts,
            class_mapping, output_dir, device,
            show_realtime=show_realtime,
            save_visualizations=save_visualizations,
            visualization_dir=visualization_dir,
            window_name=window_name,
            prompt_chunk_size=prompt_chunk_size,
            nms_config=nms_config
        )
        
        if result['success']:
            success_count += 1
            total_objects += result['num_objects']
            
            timing = result.get('timing', {})
            for key in timing_stats.keys():
                if key in timing:
                    timing_stats[key].append(timing[key])
            
            if first_inference_time is None and 'inference' in timing:
                first_inference_time = timing['inference']
            
            if not use_tqdm and verbose:
                print(f"✓ ({result['num_objects']} objects, {timing.get('inference', 0):.3f}s)")
        else:
            fail_count += 1
            if not use_tqdm and verbose:
                print(f"✗ {result['error']}")
    
    if show_realtime:
        import cv2
        cv2.destroyAllWindows()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"✓ 성공: {success_count}/{len(image_paths)} 이미지")
    print(f"✓ 총 객체 수: {total_objects}")
    print(f"✓ 평균 객체/이미지: {total_objects/max(success_count,1):.1f}")
    print(f"✓ 프롬프트 수: {len(prompts)}개")
    num_chunks = (len(prompts) + prompt_chunk_size - 1) // prompt_chunk_size
    print(f"✓ 청크 처리 모드: 이미지당 {num_chunks}번 forward ({prompt_chunk_size}개씩)")
    
    if fail_count > 0:
        print(f"✗ 실패: {fail_count} 이미지")
    
    print(f"\n⏱  총 소요 시간: {elapsed_time:.2f}초")
    print(f"⏱  이미지당 평균: {elapsed_time/len(image_paths):.2f}초")
    
    if timing_stats['inference']:
        print("\n" + "=" * 60)
        print("📊 상세 타이밍 통계 (평균)")
        print("=" * 60)
        
        if first_inference_time is not None:
            print(f"첫 추론 시간 (컴파일 포함): {first_inference_time:.3f}초")
        
        if len(timing_stats['inference']) > 1:
            avg_inference_without_compile = np.mean(timing_stats['inference'][1:])
            print(f"이후 추론 평균 (컴파일 제외): {avg_inference_without_compile:.3f}초")
        
        print(f"\n각 단계별 평균 시간:")
        print(f"  - 이미지 로드:     {np.mean(timing_stats['load']):.3f}초")
        print(f"  - 전처리:          {np.mean(timing_stats['preprocess']):.3f}초")
        print(f"  - 추론 (forward):  {np.mean(timing_stats['inference']):.3f}초")
        print(f"  - 후처리:          {np.mean(timing_stats['postprocess']):.3f}초")
        print(f"  - 저장:            {np.mean(timing_stats['save']):.3f}초")
        print(f"  - 전체:            {np.mean(timing_stats['total']):.3f}초")
        
        print(f"\n추론 성능 분석:")
        inference_times = timing_stats['inference']
        print(f"  - 최소: {np.min(inference_times):.3f}초")
        print(f"  - 최대: {np.max(inference_times):.3f}초")
        print(f"  - 평균: {np.mean(inference_times):.3f}초")
        print(f"  - 중앙값: {np.median(inference_times):.3f}초")
        
        print(f"\n디바이스: {device.upper()}")
        if device.startswith('cuda'):
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        avg_inference = np.mean(inference_times)
        time_per_prompt = avg_inference / len(prompts)
        print(f"\n효율성:")
        print(f"  - 프롬프트당 평균 시간: {time_per_prompt:.3f}초")
        print(f"  - 초당 처리 프롬프트: {1/time_per_prompt:.1f}개")
    
    print(f"\n📁 라벨 저장 위치: {output_dir}")
    if save_visualizations and visualization_dir:
        print(f"📁 시각화 저장 위치: {visualization_dir}")
    print("=" * 60)


def load_config_from_json(config_path):
    """JSON 파일에서 설정 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="SAM3 YOLO Dataset Creator (Offline Server)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # GPU 0번에서 실행
  python sam3_offline.py --gpu 0 --model_dir ./models --image_dir ./data/images

  # Config 파일 사용
  python sam3_offline.py --config config.json --gpu 1

  # 동영상 프레임 추출 + 라벨 생성
  python sam3_offline.py --gpu 0 --video_source ./videos --fps 1
        """
    )
    
    # 기본 설정
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU 인덱스 (0, 1, 2, ...) - None이면 자동')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='모델 디렉토리 경로')
    parser.add_argument('--config', type=str, default=None,
                        help='JSON 설정 파일 경로')
    
    # 동영상 프레임 추출
    parser.add_argument('--video_source', type=str, default=None,
                        help='동영상 파일/폴더 경로')
    parser.add_argument('--fps', type=int, default=None,
                        help='N프레임마다 1번 추출 (1=매프레임, 30=30프레임마다 1번, 0/-1=원본 전체)')
    
    # 이미지 소스
    parser.add_argument('--image_dir', type=str, default=None,
                        help='이미지 폴더 경로')
    parser.add_argument('--jpeg_dir', type=str, default='./data/JPEGImages',
                        help='JPEGImages 경로 (동영상 추출용)')
    
    # 출력 경로
    parser.add_argument('--label_dir', type=str, default=None,
                        help='라벨 출력 경로 (기본값: ./data/labels)')
    parser.add_argument('--viz_dir', type=str, default=None,
                        help='시각화 출력 경로 (기본값: ./data/results)')
    
    # 클래스 설정
    parser.add_argument('--classes', type=str, default=None,
                        help='클래스 매핑 (예: "person:0,car:1,dog:2")')
    
    # 추론 설정
    parser.add_argument('--threshold', type=float, default=None,
                        help='검출 임계값')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='프롬프트 청크 크기')

    # NMS 설정
    parser.add_argument('--nms_mode', type=str, default=None,
                        choices=['global', 'per_class', 'none'],
                        help='NMS 모드: global(전체 클래스), per_class(클래스별), none(비활성)')
    parser.add_argument('--nms_iou', type=float, default=None,
                        help='NMS IoU threshold')

    # 표시 옵션
    parser.add_argument('--show', action='store_true',
                        help='실시간 결과 표시')
    parser.add_argument('--save_viz', action='store_true',
                        help='시각화 이미지 저장')
    
    args = parser.parse_args()
    
    # Config 파일 우선 로드
    detection_config = None
    nms_config = None
    if args.config:
        print(f"📄 Config 파일 로드: {args.config}")
        config = load_config_from_json(args.config)

        # detection_config 추출
        detection_config = config.get('detection_config', None)

        # video_config 추출 및 적용
        video_config = config.get('video_config', None)
        if video_config:
            print("📹 video_config 발견:")
            print(f"  {video_config}")
            print("📹 video_config 적용 중...")

            if args.video_source is None:
                args.video_source = video_config.get('video_source', None)
                if args.video_source:
                    print(f"  ✓ video_source: {args.video_source}")

            if args.fps is None:
                config_fps = video_config.get('fps', None)
                if config_fps is not None:
                    args.fps = config_fps
                    print(f"  ✓ fps: {args.fps}")

            if args.jpeg_dir == './data/JPEGImages':  # 기본값이면
                config_jpeg_dir = video_config.get('jpeg_dir', None)
                if config_jpeg_dir:
                    args.jpeg_dir = config_jpeg_dir
                    print(f"  ✓ jpeg_dir: {args.jpeg_dir}")

        # NMS config 추출
        nms_config = config.get('nms', None)

        # Config 값으로 덮어쓰기 (커맨드라인 인자가 없는 경우만)
        for key, value in config.items():
            if key in ['detection_config', 'video_config', 'inference', 'output', 'nms']:
                continue  # 특수 config는 별도 처리
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

        # inference config 적용
        inference_config = config.get('inference', {})
        if 'chunk_size' in inference_config and args.chunk_size is None:
            args.chunk_size = inference_config['chunk_size']

        # output config 적용
        output_config = config.get('output', {})
        if 'save_viz' in output_config and not args.save_viz:  # 기본값이면
            args.save_viz = output_config['save_viz']
        if 'show' in output_config and not args.show:  # 기본값이면
            args.show = output_config['show']
    
    # 커맨드라인 NMS 인자로 오버라이드
    if args.nms_mode is not None:
        if args.nms_mode == 'none':
            nms_config = {'enabled': False}
        else:
            if nms_config is None:
                nms_config = {'enabled': True, 'mode': args.nms_mode, 'iou_threshold': 0.5}
            else:
                nms_config['enabled'] = True
                nms_config['mode'] = args.nms_mode
    if args.nms_iou is not None:
        if nms_config is None:
            nms_config = {'enabled': True, 'mode': 'global', 'iou_threshold': args.nms_iou}
        else:
            nms_config['iou_threshold'] = args.nms_iou

    # 숫자 인자 기본값 적용 (config에서도 설정되지 않은 경우)
    if args.fps is None:
        args.fps = 1
    if args.threshold is None:
        args.threshold = 0.3
    if args.chunk_size is None:
        args.chunk_size = 4

    # 기본 설정
    if args.classes is None:
        args.classes = {
            "person": 0,
            "car": 1,
            "bicycle": 2
        }
        print("⚠ 클래스 매핑이 없습니다. 기본값 사용:")
        print(f"   {args.classes}")

    if args.label_dir is None:
        args.label_dir = './data/labels'

    if args.viz_dir is None:
        args.viz_dir = './data/results'

    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 5 + "SAM3 YOLO Dataset Creator (Offline Server)" + " " * 5 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        # 환경 설정
        device = setup_environment(gpu_id=args.gpu, device='auto')
        
        # 1단계: 동영상 프레임 추출
        if args.video_source is not None:
            print("\n" + "🎬 " * 20)
            print("1단계: 동영상 프레임 추출")
            print("🎬 " * 20 + "\n")
            print(f"📊 프레임 추출 설정:")
            print(f"  - 동영상 소스: {args.video_source}")
            print(f"  - 출력 경로: {args.jpeg_dir}")
            print(f"  - FPS: {args.fps}\n")

            extract_frames_from_videos(
                video_source=args.video_source,
                jpeg_output_dir=args.jpeg_dir,
                fps_extraction=args.fps,
                verbose=True
            )
            
            print("\n" + "✓ " * 20)
            print("1단계 완료: 프레임 추출 성공!")
            print("✓ " * 20 + "\n")
            
            # image_dir 자동 설정
            if args.image_dir is None:
                args.image_dir = args.jpeg_dir
        
        # 2단계: YOLO 라벨 생성
        if args.image_dir is None:
            print("✗ 이미지 소스가 지정되지 않았습니다.")
            print("   --image_dir 또는 --video_source를 지정하세요.")
            sys.exit(1)
        
        print("\n" + "🏷️ " * 20)
        print("2단계: YOLO 라벨 생성")
        print("🏷️ " * 20 + "\n")
        
        # 모델 로드
        model = load_model(args.model_dir, device)
        
        # Transform & PostProcessor 생성
        transform = create_transforms()

        # detection_config가 없으면 기본값으로 생성
        if detection_config is None:
            detection_config = {
                "use_presence": True,
                "default_threshold": args.threshold,
                "max_dets_per_img": 100
            }

        postprocessor = create_postprocessor(
            detection_config=detection_config,
            class_mapping=args.classes,
            device=device
        )
        
        # YOLO 데이터셋 생성
        create_yolo_dataset(
            image_source=args.image_dir,
            output_dir=args.label_dir,
            class_mapping=args.classes,
            model_dir=args.model_dir,
            prompts=None,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            detection_threshold=args.threshold,
            device=device,
            prompt_chunk_size=args.chunk_size,
            verbose=True,
            show_realtime=args.show,
            save_visualizations=args.save_viz,
            visualization_dir=args.viz_dir if args.save_viz else None,
            nms_config=nms_config
        )
        
        print("\n" + "=" * 60)
        print("✓ 모든 작업 완료!")
        print("=" * 60)
        
        # 최종 요약
        print("\n📊 최종 요약:")
        if args.video_source is not None:
            print(f"  1. 동영상 소스: {args.video_source}")
            print(f"  2. 추출 FPS: {args.fps}")
            print(f"  3. JPEGImages: {args.jpeg_dir}")
        else:
            print(f"  1. 이미지 소스: {args.image_dir}")
        print(f"  2. YOLO 라벨: {args.label_dir}")
        if args.save_viz:
            print(f"  3. 시각화: {args.viz_dir}")
        print(f"  4. GPU: {args.gpu if args.gpu is not None else 'auto'}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()