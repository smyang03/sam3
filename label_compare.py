import os
import argparse
import glob
import numpy as np
from collections import defaultdict

def load_yolo_labels(label_file):
    """YOLO txt 형식 라벨 읽기"""
    boxes = []
    if not os.path.exists(label_file):
        return boxes
    with open(label_file, "r") as f:
        for line in f.readlines():
            cls, xc, yc, w, h = map(float, line.strip().split())
            boxes.append([int(cls), xc, yc, w, h])
    return boxes


def yolo_to_xyxy(box):
    """YOLO 형식을 x1,y1,x2,y2 픽셀 비율 좌표로 변환"""
    cls, xc, yc, w, h = box
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return cls, np.array([x1, y1, x2, y2])


def compute_iou(box1, box2):
    """IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union


def match_predictions(gt_boxes, pred_boxes, iou_thresh):
    """GT와 Pred 매칭 결과 반환"""
    matches = []
    used_pred = set()

    for g in gt_boxes:
        g_cls, g_xy = yolo_to_xyxy(g)
        best_iou = 0
        best_pred_idx = -1

        for i, p in enumerate(pred_boxes):
            if i in used_pred:
                continue
            p_cls, p_xy = yolo_to_xyxy(p)

            if p_cls != g_cls:
                continue

            iou = compute_iou(g_xy, p_xy)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = i

        if best_iou >= iou_thresh:
            matches.append((g_cls, True))
            used_pred.add(best_pred_idx)
        else:
            matches.append((g_cls, False))

    # 나머지 Pred 중 GT와 매칭되지 않은 것은 FP
    unmatched_pred = []
    for i, p in enumerate(pred_boxes):
        if i not in used_pred:
            unmatched_pred.append(p)

    return matches, unmatched_pred


def evaluate_yolo(gt_folder, pred_folder, iou_thresh=0.5):
    gt_files = sorted(glob.glob(os.path.join(gt_folder, "*.txt")))
    pred_files = sorted(glob.glob(os.path.join(pred_folder, "*.txt")))

    per_class = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for gt_file in gt_files:
        fname = os.path.basename(gt_file)

        gt_labels = load_yolo_labels(gt_file)
        pred_labels = load_yolo_labels(os.path.join(pred_folder, fname))

        matches, unmatched_pred = match_predictions(gt_labels, pred_labels, iou_thresh)

        # 매칭 결과 처리
        for cls, is_tp in matches:
            if is_tp:
                per_class[cls]["TP"] += 1
            else:
                per_class[cls]["FN"] += 1

        # 매칭되지 않은 prediction = FP
        for p in unmatched_pred:
            cls = int(p[0])
            per_class[cls]["FP"] += 1

    # 전체 결과
    results = {}
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for cls, stats in per_class.items():
        TP = stats["TP"]
        FP = stats["FP"]
        FN = stats["FN"]

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # ← 추가

        results[cls] = {
            "Precision": precision,
            "Recall": recall,
            "F1": f1_score,  # ← 추가
            "TP": TP,
            "FP": FP,
            "FN": FN
        }

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1_score)  # ← 추가

    mean_precision = np.mean(all_precisions) if all_precisions else 0
    mean_recall = np.mean(all_recalls) if all_recalls else 0
    mean_f1 = np.mean(all_f1s) if all_f1s else 0  # ← 추가
    mAP = mean_precision

    return results, mean_precision, mean_recall, mean_f1, mAP  # ← mean_f1 추가


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="정답지(YOLO labels) 폴더 경로")
    parser.add_argument("--pred", required=True, help="비교해야 할 라벨 폴더 경로")
    parser.add_argument("--iou", default=0.5, type=float, help="IoU threshold")
    args = parser.parse_args()

    results, mp, mr, mf1, mAP = evaluate_yolo(args.gt, args.pred, args.iou)  # ← mf1 추가

    print("\n========== 📊 YOLO Evaluation Results ==========")
    for cls, r in results.items():
        print(f"[Class {cls}]  P={r['Precision']:.4f}  R={r['Recall']:.4f}  F1={r['F1']:.4f}  TP={r['TP']}  FP={r['FP']}  FN={r['FN']}")  # ← F1 추가

    print("------------------------------------------------")
    print(f"Mean Precision: {mp:.4f}")
    print(f"Mean Recall   : {mr:.4f}")
    print(f"Mean F1       : {mf1:.4f}")  # ← 추가
    print(f"mAP@{args.iou}: {mAP:.4f}")
