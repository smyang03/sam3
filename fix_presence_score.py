#!/usr/bin/env python3
"""
Presence Score 디버깅 및 수정 유틸리티

사용법:
1. 디버그 모드: presence score 로깅
   python fix_presence_score.py --mode debug --input debug_output.pt

2. Presence score 비활성화 테스트
   (sam3_offline.py 실행 시 --use-presence 인자 추가 필요)
"""

import argparse
import torch
import numpy as np


def analyze_presence_scores(debug_file):
    """저장된 디버그 출력 분석"""
    data = torch.load(debug_file, map_location='cpu')

    print("="*70)
    print("Presence Score 분석")
    print("="*70)

    frame_idx = data.get('frame', 'unknown')
    prompts = data.get('prompts', [])

    print(f"\n프레임: {frame_idx}")
    print(f"프롬프트 수: {len(prompts)}")

    # Presence scores
    if 'presence_logit_dec' in data:
        presence_logits = data['presence_logit_dec']
        presence_scores = torch.sigmoid(presence_logits)

        print(f"\nPresence Scores:")
        for prompt, score in zip(prompts, presence_scores):
            score_val = score.item() if hasattr(score, 'item') else score
            status = "⚠️ 낮음" if score_val < 0.5 else "✓"
            print(f"  {prompt:15s}: {score_val:.4f} {status}")

    # 개별 객체 점수
    if 'pred_logits' in data:
        pred_logits = data['pred_logits']
        pred_probs = torch.sigmoid(pred_logits)

        print(f"\n개별 객체 점수 (sigmoid 후):")
        for i, prompt in enumerate(prompts):
            if i < len(pred_probs):
                obj_scores = pred_probs[i]
                max_score = obj_scores.max().item()
                mean_score = obj_scores.mean().item()
                num_objs = len(obj_scores)

                print(f"  {prompt:15s}: {num_objs}개 객체, "
                      f"최대={max_score:.4f}, 평균={mean_score:.4f}")

                # Presence 곱셈 후 점수
                if 'presence_logit_dec' in data:
                    p_score = presence_scores[i].item()
                    after_scores = obj_scores * p_score

                    print(f"    → Presence 곱셈 후: "
                          f"최대={after_scores.max().item():.4f}, "
                          f"평균={after_scores.mean().item():.4f}")

                    # Threshold 필터링 시뮬레이션
                    for threshold in [0.1, 0.2, 0.3, 0.5]:
                        survived = (after_scores > threshold).sum().item()
                        print(f"    → Threshold {threshold}: {survived}/{num_objs}개 통과")

    print("="*70)


def calculate_optimal_threshold(scores_with_labels):
    """최적 threshold 계산 (F1-score 기준)"""
    # scores_with_labels: list of (score, is_correct_label)
    scores = [s for s, _ in scores_with_labels]
    labels = [l for _, l in scores_with_labels]

    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        predictions = [s > threshold for s in scores]

        tp = sum(1 for p, l in zip(predictions, labels) if p and l)
        fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
        fn = sum(1 for p, l in zip(predictions, labels) if not p and l)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def create_patched_postprocessor_code():
    """Presence score 문제 해결 패치 코드 생성"""

    patch_code = '''
# postprocessors.py에 추가할 패치

# 방법 1: Presence score 하한선 설정
if self.use_presence:
    presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)

    # 패치: 최소값 보장 (너무 낮게 떨어지는 것 방지)
    MIN_PRESENCE_SCORE = 0.3  # 조정 가능
    presence_score = torch.clamp(presence_score, min=MIN_PRESENCE_SCORE)

    out_probs = out_probs * presence_score

# 방법 2: Presence score 로깅 (디버깅용)
if self.use_presence and hasattr(self, 'log_presence_scores'):
    import logging
    logger = logging.getLogger(__name__)

    presence_score = outputs["presence_logit_dec"].sigmoid()
    for i, score in enumerate(presence_score):
        score_val = score.item()
        if score_val < 0.5:
            logger.warning(f"낮은 presence score 감지: query {i}, score={score_val:.4f}")
'''

    print(patch_code)
    return patch_code


def main():
    parser = argparse.ArgumentParser(
        description='Presence Score 디버깅 도구'
    )

    parser.add_argument('--mode', choices=['debug', 'patch'],
                        default='debug',
                        help='실행 모드')
    parser.add_argument('--input', type=str,
                        help='디버그 파일 경로 (.pt)')

    args = parser.parse_args()

    if args.mode == 'debug':
        if not args.input:
            print("오류: --input 인자 필요")
            return 1

        analyze_presence_scores(args.input)

    elif args.mode == 'patch':
        print("Presence Score 문제 해결 패치 코드:")
        print("="*70)
        create_patched_postprocessor_code()

    return 0


if __name__ == '__main__':
    exit(main())
