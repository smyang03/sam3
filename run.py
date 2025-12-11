"""
SAM 3 YOLO Dataset Creator - Video Frame Extraction Support
동영상에서 프레임 추출 후 YOLO 데이터셋 생성
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path

os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6"

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

# Global counter for query IDs
GLOBAL_COUNTER = 1


def recursive_to_device(obj, device):
    """
    재귀적으로 모든 텐서를 device로 이동
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_device(item, device) for item in obj)
    elif hasattr(obj, '__dict__'):
        # 객체의 모든 속성을 재귀적으로 처리
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


def setup_environment(hf_token=None, device='auto'):
    """환경 설정"""
    print("=" * 60)
    print("환경 설정 중...")
    print("=" * 60)
    
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("✓ HuggingFace 토큰 설정 완료")
    
    # 디바이스 설정
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"✓ CUDA 자동 감지: {torch.cuda.get_device_name(0)}")
            print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            print("⚠ CUDA 없음 - CPU 모드")
    elif device == 'cuda':
        if torch.cuda.is_available():
            print(f"✓ CUDA 강제 사용: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA 불가 - CPU로 전환")
            device = 'cpu'
    else:
        device = 'cpu'
        print("✓ CPU 모드 선택")
    
    # CUDA 최적화 설정
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        torch.inference_mode().__enter__()
        print("✓ CUDA 최적화 활성화 (TF32, bfloat16)")
    
    print()
    return device


def load_model(bpe_path, device='cuda'):
    """SAM3 모델 로드"""
    print("=" * 60)
    print("모델 로드 중...")
    print("=" * 60)
    
    start_time = time.time()
    model = build_sam3_image_model(bpe_path=bpe_path)
    load_time = time.time() - start_time
    
    print(f"✓ 모델 로드 완료 ({load_time:.2f}초)")
    
    if device == 'cuda':
        model = model.cuda()
        print("✓ 모델을 GPU로 이동")
        
        # CUDA 캐시 워밍업 (매우 중요!)
        print("✓ CUDA 캐시 워밍업 중...")
        try:
            # Decoder의 coord cache를 CUDA에 생성
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
                
                # Monkey patch _get_rpb_matrix to ensure device consistency
                original_get_rpb_matrix = decoder._get_rpb_matrix
                
                def patched_get_rpb_matrix(reference_boxes, feat_size):
                    """Patched version that ensures coords are on same device as reference_boxes"""
                    H, W = feat_size
                    boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
                    bs, num_queries, _ = boxes_xyxy.shape
                    
                    # Get device from reference_boxes
                    target_dev = reference_boxes.device
                    
                    # Check cache first
                    if decoder.compilable_cord_cache is None:
                        coords_h, coords_w = decoder._get_coords(H, W, target_dev)
                        decoder.compilable_cord_cache = (coords_h, coords_w)
                        decoder.compilable_stored_size = (H, W)
                    
                    if torch.compiler.is_dynamo_compiling() or decoder.compilable_stored_size == (H, W):
                        coords_h, coords_w = decoder.compilable_cord_cache
                        # Ensure on correct device
                        if coords_h.device != target_dev:
                            coords_h = coords_h.to(target_dev)
                            coords_w = coords_w.to(target_dev)
                            decoder.compilable_cord_cache = (coords_h, coords_w)
                    else:
                        if feat_size not in decoder.coord_cache:
                            decoder.coord_cache[feat_size] = decoder._get_coords(H, W, target_dev)
                        coords_h, coords_w = decoder.coord_cache[feat_size]
                        # Ensure on correct device
                        if coords_h.device != target_dev:
                            coords_h = coords_h.to(target_dev)
                            coords_w = coords_w.to(target_dev)
                            decoder.coord_cache[feat_size] = (coords_h, coords_w)
                    
                    # Continue with original logic
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
                
                # Import box_cxcywh_to_xyxy for the patched function
                from sam3.model.box_ops import box_cxcywh_to_xyxy
                import numpy as np
                
                decoder._get_rpb_matrix = patched_get_rpb_matrix
                print(f"  - _get_rpb_matrix patched for device consistency")
                
        except Exception as e:
            print(f"  ⚠ 캐시 워밍업 중 오류 (무시): {e}")
            import traceback
            traceback.print_exc()
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


def create_postprocessor(detection_threshold=0.3, device='cuda'):
    """후처리 PostProcessor 생성"""
    # CPU 모드에서는 GPU 관련 설정 비활성화
    if device == 'cpu':
        return PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=True,
            always_interpolate_masks_on_gpu=False
        )
    else:
        return PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=False,
        )


def create_datapoint_with_prompts(pil_image, text_prompts):
    """
    이미지와 여러 프롬프트로 Datapoint 생성
    
    Args:
        pil_image: PIL Image
        text_prompts: 텍스트 프롬프트 리스트
        
    Returns:
        datapoint: Datapoint 객체
        prompt_ids: 각 프롬프트의 ID 리스트
    """
    global GLOBAL_COUNTER
    
    datapoint = Datapoint(find_queries=[], images=[])
    
    # 이미지 설정
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]
    
    # 여러 프롬프트 추가
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
                    original_size=[h, w],  # height, width 순서!
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
    """동영상 소스 파싱 (파일 또는 폴더)"""
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
            # YOLO 형식으로 변환
            x_center, y_center, width, height = bbox_to_yolo_format(
                box, img_width, img_height
            )
            
            # 5개 값만 저장
            line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            lines.append(line)
            total_objects += 1
    
    # 파일 저장
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
        (0, 255, 0),    # 녹색
        (255, 0, 0),    # 파란색
        (0, 0, 255),    # 빨간색
        (0, 255, 255),  # 노란색
        (255, 0, 255),  # 마젠타
        (255, 255, 0),  # 시안
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
    
    img_h, img_w = image.shape[:2]
    
    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
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
        
        for idx, (box, score) in enumerate(zip(boxes, scores)):
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
    """
    동영상에서 프레임 추출 및 JPEGImages 저장
    
    Args:
        video_source: 동영상 파일 또는 폴더 경로
        jpeg_output_dir: JPEGImages 저장 경로
        fps_extraction: 추출 FPS
                       - 1, 5, 30 등: 1초당 N프레임 추출
                       - 0 또는 -1: 원본 FPS 전체 프레임
        verbose: 로그 출력 여부
    
    Returns:
        extracted_count: 추출된 총 프레임 수
    """
    import cv2
    
    print("\n" + "=" * 60)
    print("동영상 프레임 추출 시작")
    print("=" * 60)
    
    # 동영상 파일 파싱
    video_paths = parse_video_source(video_source)
    
    # 출력 디렉토리 생성
    os.makedirs(jpeg_output_dir, exist_ok=True)
    print(f"📁 JPEGImages 저장 경로: {jpeg_output_dir}\n")
    
    total_extracted = 0
    global_frame_index = 1  # 전체 프레임 인덱스 (모든 동영상 통합)
    
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    for video_idx, video_path in enumerate(video_paths):
        video_name = Path(video_path).stem
        
        if verbose:
            print(f"\n[{video_idx+1}/{len(video_paths)}] {video_name}")
        
        # OpenCV로 동영상 열기
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  ✗ 동영상을 열 수 없습니다: {video_path}")
            continue
        
        # 동영상 정보
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0
        
        if verbose:
            print(f"  원본 FPS: {original_fps:.2f}")
            print(f"  총 프레임: {total_frames}")
            print(f"  길이: {duration:.2f}초")
        
        # FPS 추출 로직 결정
        if fps_extraction <= 0:
            # 원본 FPS 전체 추출
            frame_interval = 1
            extract_fps = original_fps
            if verbose:
                print(f"  추출 모드: 원본 FPS 전체 ({original_fps:.2f}fps)")
        else:
            # 지정된 FPS로 추출
            frame_interval = int(original_fps / fps_extraction)
            if frame_interval < 1:
                frame_interval = 1
            extract_fps = fps_extraction
            if verbose:
                print(f"  추출 모드: {fps_extraction}fps (매 {frame_interval}프레임)")
        
        # 예상 추출 프레임 수
        estimated_frames = total_frames // frame_interval
        if verbose:
            print(f"  예상 추출: {estimated_frames}프레임\n")
        
        # 프레임 추출
        frame_count = 0
        extracted_count = 0
        
        iterator = tqdm(total=total_frames, desc=f"  {video_name}") if use_tqdm else range(total_frames)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 프레임 간격에 맞춰 추출
            if frame_count % frame_interval == 0:
                # 파일명: video_name_frame_000001.jpg
                frame_filename = f"{video_name}_frame_{global_frame_index:06d}.jpg"
                frame_path = os.path.join(jpeg_output_dir, frame_filename)
                
                # 한글 경로 지원 저장
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
    prompt_chunk_size=4
):
    """단일 이미지 배치 처리"""
    try:
        total_start = time.time()
        
        # 이미지 로드
        load_start = time.time()
        pil_image = Image.open(image_path)

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # EXIF 완전 제거 (numpy 경유)
        pil_image = Image.fromarray(np.array(pil_image))

        img_width, img_height = pil_image.size
        load_time = time.time() - load_start
        
        # 전체 결과를 저장할 딕셔너리
        results_by_prompt = {}
        
        # 타이밍 누적
        total_prep_time = 0
        total_inference_time = 0
        total_post_time = 0
        
        # 프롬프트를 청크로 나눠서 처리
        num_chunks = (len(prompts) + prompt_chunk_size - 1) // prompt_chunk_size
        
        for chunk_idx in range(num_chunks):
            # 현재 청크의 프롬프트
            start_idx = chunk_idx * prompt_chunk_size
            end_idx = min(start_idx + prompt_chunk_size, len(prompts))
            chunk_prompts = prompts[start_idx:end_idx]
            
            # Datapoint 생성 (청크 단위 프롬프트)
            prep_start = time.time()
            datapoint, prompt_ids = create_datapoint_with_prompts(pil_image, chunk_prompts)
            
            # Transform 적용
            datapoint = transform(datapoint)
            
            # 배치로 collate
            batch = collate([datapoint], dict_key="dummy")["dummy"]
            
            # Device로 완전히 이동
            target_device = torch.device(device)
            
            # Method 1: 기본 이동
            batch = copy_data_to_device(batch, target_device, non_blocking=True)
            
            # Method 2: 재귀적으로 모든 텐서 이동 (안전장치)
            batch = recursive_to_device(batch, target_device)
            
            prep_time = time.time() - prep_start
            total_prep_time += prep_time
            
            # GPU 동기화 (정확한 측정을 위해)
            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 청크 추론
            inference_start = time.time()
            output = model(batch)
            
            # GPU 동기화 (추론 완료 대기)
            if device == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            total_inference_time += inference_time
            
            # 후처리
            post_start = time.time()
            processed_results = postprocessor.process_results(output, batch.find_metadatas)
            post_time = time.time() - post_start
            total_post_time += post_time
            
            # 결과 정리 (prompt_id별로 분류)
            if not isinstance(processed_results, list):
                # 딕셔너리인 경우
                if isinstance(processed_results, dict):
                    # 각 query_id별로 결과 추출
                    for prompt_name, prompt_id in zip(chunk_prompts, prompt_ids):
                        if prompt_id in processed_results:
                            result = processed_results[prompt_id]
                            # bfloat16 → float32 → numpy
                            boxes = result['boxes'].float().cpu().numpy() if hasattr(result['boxes'], 'cpu') else result['boxes']
                            scores = result['scores'].float().cpu().numpy() if hasattr(result['scores'], 'cpu') else result['scores']
                            
                            results_by_prompt[prompt_name] = {
                                'boxes': boxes,
                                'scores': scores
                            }
                        else:
                            # 검출 결과 없음
                            if prompt_name not in results_by_prompt:
                                results_by_prompt[prompt_name] = {
                                    'boxes': np.array([]),
                                    'scores': np.array([])
                                }
                else:
                    # 알 수 없는 형식 - 빈 결과 반환
                    for prompt_name in chunk_prompts:
                        if prompt_name not in results_by_prompt:
                            results_by_prompt[prompt_name] = {
                                'boxes': np.array([]),
                                'scores': np.array([])
                            }
            else:
                # 리스트인 경우 (기존 로직)
                for result in processed_results:
                    # result가 딕셔너리인지 확인
                    if isinstance(result, dict) and 'query_id' in result:
                        query_id = result['query_id']
                        
                        # query_id로 프롬프트 찾기
                        for prompt_name, prompt_id in zip(chunk_prompts, prompt_ids):
                            if query_id == prompt_id:
                                # bfloat16 → float32 → numpy
                                boxes = result['boxes'].float().cpu().numpy() if hasattr(result['boxes'], 'cpu') else result['boxes']
                                scores = result['scores'].float().cpu().numpy() if hasattr(result['scores'], 'cpu') else result['scores']
                                results_by_prompt[prompt_name] = {
                                    'boxes': boxes,
                                    'scores': scores
                                }
                                break
                
                # 현재 청크에서 결과 없는 프롬프트는 빈 배열
                for prompt_name in chunk_prompts:
                    if prompt_name not in results_by_prompt:
                        results_by_prompt[prompt_name] = {
                            'boxes': np.array([]),
                            'scores': np.array([])
                        }
            
            # GPU 메모리 정리
            if device == 'cuda':
                del batch, output, processed_results
                torch.cuda.empty_cache()
        
        # YOLO 어노테이션 저장
        save_start = time.time()
        num_objects = save_yolo_annotation(
            image_path, results_by_prompt, class_mapping, 
            output_dir, img_width, img_height
        )
        save_time = time.time() - save_start
        
        # 실시간 결과 표시
        if show_realtime:
            show_realtime_result(image_path, results_by_prompt, class_mapping, window_name)
        
        # 시각화 이미지 저장
        visualization_path = None
        if save_visualizations and visualization_dir:
            visualization_path = save_visualization_result(
                image_path, results_by_prompt, class_mapping, visualization_dir
            )
        
        # 전체 처리 시간
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
    bpe_path,
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
    visualization_dir=None
):
    """
    YOLO 형식 데이터셋 생성 (배치 추론 최적화 + 청크 처리)
    
    Args:
        prompt_chunk_size: 한번에 처리할 프롬프트 개수 (GPU 메모리에 따라 조절)
                          - RTX 4090 24GB: 6-8 추천
                          - RTX 3090 24GB: 4-6 추천
                          - RTX 3080 10GB: 2-4 추천
    """
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "YOLO Dataset Creation Tool" + " " * 12 + "║")
    print("║" + " " * 15 + "(Batch Inference Mode)" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    start_time = time.time()
    
    # 클래스 매핑 파싱
    print("=" * 60)
    print("설정 확인")
    print("=" * 60)
    
    class_mapping = parse_class_mapping(class_mapping)
    print(f"클래스 매핑: {class_mapping}")
    
    if prompts is None:
        prompts = list(class_mapping.keys())
    print(f"프롬프트: {prompts}")
    print(f"프롬프트 총 개수: {len(prompts)}개")
    print(f"프롬프트 청크 크기: {prompt_chunk_size}개 (GPU 메모리 절약)")
    print(f"청크 수: {(len(prompts) + prompt_chunk_size - 1) // prompt_chunk_size}개")
    print(f"검출 임계값: {detection_threshold}")
    print(f"디바이스: {device}")
    print(f"실시간 표시: {show_realtime}")
    print(f"시각화 저장: {save_visualizations}")
    if save_visualizations and visualization_dir:
        print(f"시각화 디렉토리: {visualization_dir}")
    print()
    
    # 이미지 소스 파싱
    print("=" * 60)
    print("이미지 로드")
    print("=" * 60)
    image_paths = parse_image_source(image_source)
    print()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 라벨 출력 디렉토리: {output_dir}")
    
    if save_visualizations and visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
        print(f"📁 시각화 출력 디렉토리: {visualization_dir}")
    print()
    
    # 모델 로드
    if model is None:
        model = load_model(bpe_path, device)
    
    if transform is None:
        transform = create_transforms()
        print("✓ Transform 생성 완료")
    
    if postprocessor is None:
        postprocessor = create_postprocessor(detection_threshold, device)
        print("✓ PostProcessor 생성 완료")
    
    print()
    
    # 실시간 표시용 윈도우 생성
    window_name = "SAM3 Real-time Detection"
    if show_realtime:
        import cv2
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("🖥️  실시간 표시 윈도우 생성\n")
    
    # 이미지 처리
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
    
    # 타이밍 통계
    timing_stats = {
        'total': [],
        'load': [],
        'preprocess': [],
        'inference': [],
        'postprocess': [],
        'save': []
    }
    
    # 첫 추론 시간 (컴파일 포함)
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
            prompt_chunk_size=prompt_chunk_size
        )
        
        if result['success']:
            success_count += 1
            total_objects += result['num_objects']
            
            # 타이밍 통계 수집
            timing = result.get('timing', {})
            for key in timing_stats.keys():
                if key in timing:
                    timing_stats[key].append(timing[key])
            
            # 첫 추론 시간 기록 (컴파일 포함)
            if first_inference_time is None and 'inference' in timing:
                first_inference_time = timing['inference']
            
            if not use_tqdm and verbose:
                print(f"✓ ({result['num_objects']} objects, {timing.get('inference', 0):.3f}s)")
        else:
            fail_count += 1
            if not use_tqdm and verbose:
                print(f"✗ {result['error']}")
    
    # 윈도우 정리
    if show_realtime:
        import cv2
        cv2.destroyAllWindows()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 결과 출력
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
    
    # 타이밍 통계 출력
    if timing_stats['inference']:
        print("\n" + "=" * 60)
        print("📊 상세 타이밍 통계 (평균)")
        print("=" * 60)
        
        # 첫 추론 시간 (컴파일 포함)
        if first_inference_time is not None:
            print(f"첫 추론 시간 (컴파일 포함): {first_inference_time:.3f}초")
        
        # 두 번째 이후 추론 평균 (컴파일 제외)
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
        
        # 추론 속도 분석
        print(f"\n추론 성능 분석:")
        inference_times = timing_stats['inference']
        print(f"  - 최소: {np.min(inference_times):.3f}초")
        print(f"  - 최대: {np.max(inference_times):.3f}초")
        print(f"  - 평균: {np.mean(inference_times):.3f}초")
        print(f"  - 중앙값: {np.median(inference_times):.3f}초")
        
        # GPU vs CPU 표시
        print(f"\n디바이스: {device.upper()}")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # 프롬프트당 추론 시간
        avg_inference = np.mean(inference_times)
        time_per_prompt = avg_inference / len(prompts)
        print(f"\n효율성:")
        print(f"  - 프롬프트당 평균 시간: {time_per_prompt:.3f}초")
        print(f"  - 초당 처리 프롬프트: {1/time_per_prompt:.1f}개")
    
    print(f"\n📁 라벨 저장 위치: {output_dir}")
    if save_visualizations and visualization_dir:
        print(f"📁 시각화 저장 위치: {visualization_dir}")
    print("=" * 60)


def main():
    """메인 실행 함수"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 5 + "SAM 3 YOLO Dataset Creator (Video Support)" + " " * 5 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # HuggingFace 토큰
    #HF_TOKEN = ""
    
    # SAM3 루트 경로
    import sam3
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    
    # 데이터셋 생성 설정
    DATASET_CONFIG = {
        # 동영상 관련 설정 (NEW!)
        "video_source": None,  # 동영상 파일/폴더 경로 (None이면 스킵)
        "jpeg_output_dir": "X:/박창현/pipe_lower_part/data/JPEGImages",  # 프레임 저장 경로
        "fps_extraction": 1,  # 추출 FPS (1=1fps, 5=5fps, 0/-1=원본 전체)
        
        # 기존 설정
        "image_source": "X:/박창현/pipe_lower_part/data/JPEGImages",
        "label_dir": "X:/박창현/pipe_lower_part/data/labels",
        "class_mapping": {
            "industrial hose": 0,
            "Chemical Hose": 1,
            "flexible duct": 2
        },
        "detection_threshold": 0.5,
        "prompt_chunk_size": 4,
        "show_realtime": True,
        "save_visualizations": True,
        "visualization_dir": "X:/박창현/pipe_lower_part/data/result",
        "device": "auto",
    }
    
    try:
        # 환경 설정
        device = setup_environment(HF_TOKEN, DATASET_CONFIG["device"])
        
        # ========== 1단계: 동영상 → JPEGImages 추출 ==========
        if DATASET_CONFIG["video_source"] is not None:
            print("\n" + "🎬 " * 20)
            print("1단계: 동영상 프레임 추출")
            print("🎬 " * 20 + "\n")
            
            extract_frames_from_videos(
                video_source=DATASET_CONFIG["video_source"],
                jpeg_output_dir=DATASET_CONFIG["jpeg_output_dir"],
                fps_extraction=DATASET_CONFIG["fps_extraction"],
                verbose=True
            )
            
            print("\n" + "✓ " * 20)
            print("1단계 완료: 프레임 추출 성공!")
            print("✓ " * 20 + "\n")
            
            # image_source를 jpeg_output_dir로 자동 설정
            DATASET_CONFIG["image_source"] = DATASET_CONFIG["jpeg_output_dir"]
        
        # ========== 2단계: JPEGImages → YOLO 라벨 생성 ==========
        print("\n" + "🏷️ " * 20)
        print("2단계: YOLO 라벨 생성")
        print("🏷️ " * 20 + "\n")
        
        # 모델 로드
        model = load_model(bpe_path, device)
        
        # Transform & PostProcessor 생성
        transform = create_transforms()
        postprocessor = create_postprocessor(DATASET_CONFIG["detection_threshold"], device)
        
        # YOLO 데이터셋 생성
        create_yolo_dataset(
            image_source=DATASET_CONFIG["image_source"],
            output_dir=DATASET_CONFIG["label_dir"],
            class_mapping=DATASET_CONFIG["class_mapping"],
            bpe_path=bpe_path,
            prompts=None,
            model=model,
            transform=transform,
            postprocessor=postprocessor,
            detection_threshold=DATASET_CONFIG["detection_threshold"],
            device=device,
            prompt_chunk_size=DATASET_CONFIG["prompt_chunk_size"],
            verbose=True,
            show_realtime=DATASET_CONFIG["show_realtime"],
            save_visualizations=DATASET_CONFIG["save_visualizations"],
            visualization_dir=DATASET_CONFIG["visualization_dir"]
        )
        
        print("\n" + "=" * 60)
        print("✓ 모든 작업 완료!")
        print("=" * 60)
        
        # 최종 요약
        if DATASET_CONFIG["video_source"] is not None:
            print("\n📊 최종 요약:")
            print(f"  1. 동영상 소스: {DATASET_CONFIG['video_source']}")
            print(f"  2. 추출 FPS: {DATASET_CONFIG['fps_extraction']}")
            print(f"  3. JPEGImages: {DATASET_CONFIG['jpeg_output_dir']}")
            print(f"  4. YOLO 라벨: {DATASET_CONFIG['label_dir']}")
            if DATASET_CONFIG["save_visualizations"]:
                print(f"  5. 시각화: {DATASET_CONFIG['visualization_dir']}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()