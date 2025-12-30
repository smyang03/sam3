# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Class-wise threshold support for PostProcessImage
클래스별로 다른 detection threshold를 적용할 수 있는 postprocessor
"""

import logging
from typing import Dict, Optional
import torch

from sam3.eval.postprocessors import PostProcessImage
from sam3.model import box_ops


class PostProcessImageWithClassThresholds(PostProcessImage):
    """
    PostProcessImage with per-class detection thresholds

    클래스별로 다른 detection threshold를 적용할 수 있도록 확장한 버전
    작은 객체(헬멧 등)는 낮은 threshold를, 큰 객체는 높은 threshold를 적용 가능
    """

    def __init__(
        self,
        max_dets_per_img: int,
        class_thresholds: Optional[Dict[str, float]] = None,
        class_to_id: Optional[Dict[str, int]] = None,
        iou_type="bbox",
        to_cpu: bool = True,
        use_original_ids: bool = False,
        use_original_sizes_box: bool = False,
        use_original_sizes_mask: bool = False,
        convert_mask_to_rle: bool = False,
        always_interpolate_masks_on_gpu: bool = True,
        use_presence: bool = True,
        detection_threshold: float = -1.0,
    ) -> None:
        """
        Args:
            class_thresholds: 클래스별 threshold 딕셔너리
                예: {"person": 0.3, "helmet": 0.15, "car": 0.3}
            class_to_id: 클래스 이름 → ID 매핑
                예: {"person": 0, "helmet": 1, "car": 2}
            detection_threshold: 기본 threshold (class_thresholds에 없는 클래스에 적용)
        """
        super().__init__(
            max_dets_per_img=max_dets_per_img,
            iou_type=iou_type,
            to_cpu=to_cpu,
            use_original_ids=use_original_ids,
            use_original_sizes_box=use_original_sizes_box,
            use_original_sizes_mask=use_original_sizes_mask,
            convert_mask_to_rle=convert_mask_to_rle,
            always_interpolate_masks_on_gpu=always_interpolate_masks_on_gpu,
            use_presence=use_presence,
            detection_threshold=detection_threshold,
        )

        self.class_thresholds = class_thresholds or {}
        self.class_to_id = class_to_id or {}

        # ID → threshold 매핑 생성
        self.id_to_threshold = {}
        for class_name, threshold in self.class_thresholds.items():
            if class_name in self.class_to_id:
                class_id = self.class_to_id[class_name]
                self.id_to_threshold[class_id] = threshold

        # 로깅
        if self.class_thresholds:
            logging.info(f"클래스별 threshold 설정: {self.class_thresholds}")
            logging.info(f"ID 매핑: {self.id_to_threshold}")

    def _process_boxes_and_labels(
        self, target_sizes, forced_labels, out_bbox, out_probs
    ):
        """
        박스와 라벨 처리 (클래스별 threshold 적용)
        """
        if out_bbox is None:
            return None, None, None, None

        assert len(out_probs) == len(target_sizes)

        if self.to_cpu:
            out_probs = out_probs.cpu()

        scores, labels = out_probs.max(-1)

        if forced_labels is None:
            labels = torch.ones_like(labels)
        else:
            labels = forced_labels[:, None].expand_as(labels)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.to_cpu:
            boxes = boxes.cpu()

        keep = None

        # 클래스별 threshold 적용
        if self.class_thresholds and forced_labels is not None:
            # 클래스별로 다른 threshold 적용
            keep = self._apply_class_wise_threshold(scores, labels, forced_labels)

            boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
            scores = [s[k.to(s.device)] for s, k in zip(scores, keep)]
            labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]

        elif self.detection_threshold > 0:
            # 기본 threshold 적용 (모든 클래스 동일)
            keep = scores > self.detection_threshold
            assert len(keep) == len(boxes) == len(scores) == len(labels)

            boxes = [b[k.to(b.device)] for b, k in zip(boxes, keep)]
            scores = [s[k.to(s.device)] for s, k in zip(scores, keep)]
            labels = [l[k.to(l.device)] for l, k in zip(labels, keep)]

        return boxes, scores, labels, keep

    def _apply_class_wise_threshold(self, scores, labels, forced_labels):
        """
        클래스별로 다른 threshold를 적용

        Args:
            scores: [batch_size, num_queries] 점수
            labels: [batch_size, num_queries] 라벨
            forced_labels: [batch_size] 강제 라벨 (query된 클래스 ID)

        Returns:
            keep: [batch_size, num_queries] bool 텐서
        """
        batch_size = scores.shape[0]
        num_queries = scores.shape[1]

        keep_list = []

        for batch_idx in range(batch_size):
            # 이 배치의 클래스 ID
            class_id = forced_labels[batch_idx].item()

            # 이 클래스의 threshold (없으면 기본값)
            threshold = self.id_to_threshold.get(
                class_id,
                self.detection_threshold if self.detection_threshold > 0 else 0.0
            )

            # 이 배치의 모든 query에 대해 threshold 적용
            batch_keep = scores[batch_idx] > threshold
            keep_list.append(batch_keep)

        # 스택
        keep = torch.stack(keep_list, dim=0)

        return keep


def create_postprocessor_from_config(config: Dict, class_mapping: Dict[str, int]):
    """
    Config에서 postprocessor 생성

    Args:
        config: detection_config 딕셔너리
            {
                "use_presence": false,
                "default_threshold": 0.3,
                "class_thresholds": {"helmet": 0.15, ...}
            }
        class_mapping: 클래스 이름 → ID 매핑

    Returns:
        PostProcessor 인스턴스
    """
    use_presence = config.get('use_presence', True)
    default_threshold = config.get('default_threshold', 0.3)
    class_thresholds = config.get('class_thresholds', {})
    max_dets = config.get('max_dets_per_img', 100)

    # 클래스별 threshold가 있으면 ClassWise 버전 사용
    if class_thresholds:
        logging.info("📊 클래스별 threshold 적용")
        postprocessor = PostProcessImageWithClassThresholds(
            max_dets_per_img=max_dets,
            class_thresholds=class_thresholds,
            class_to_id=class_mapping,
            use_presence=use_presence,
            detection_threshold=default_threshold,
            iou_type="bbox",
        )
    else:
        logging.info("📊 단일 threshold 적용")
        postprocessor = PostProcessImage(
            max_dets_per_img=max_dets,
            use_presence=use_presence,
            detection_threshold=default_threshold,
            iou_type="bbox",
        )

    # 설정 출력
    logging.info(f"  use_presence: {use_presence}")
    logging.info(f"  default_threshold: {default_threshold}")
    if class_thresholds:
        for class_name, threshold in class_thresholds.items():
            logging.info(f"  {class_name}: {threshold}")

    return postprocessor
