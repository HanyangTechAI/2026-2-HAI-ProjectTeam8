"""공통 평가 함수. 모든 모델은 여기 함수만 사용한다 (직접 구현 금지).

양성 클래스 = 불법(1).
  - 주 지표: PR-AUC, F1(불법)
  - 임계값 규칙 A: 0.5 고정 / 규칙 B: 검증 구간에서 F1 최대인 임계값(best_threshold)을 테스트에 그대로 적용
  - 테스트 구간에서는 임계값을 다시 고르지 않는다.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """불법 클래스 기준 지표. 한 클래스만 있으면 AUC 계열은 NaN."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    both = len(np.unique(y_true)) == 2
    return dict(
        pr_auc=float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        threshold=float(threshold),
    )


def best_threshold(y_true, y_prob) -> float:
    """F1(불법)이 최대가 되는 임계값. 반드시 검증 구간에서만 호출한다."""
    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() == 0:
        return 0.5
    p, r, t = precision_recall_curve(y_true, np.asarray(y_prob, dtype=float))
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[int(np.argmax(f1))])


def evaluate_by_timestep(y_true, y_prob, steps, threshold: float) -> dict:
    """시간 단계별 F1(불법). 불법 노드가 없는 단계는 건너뛴다."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    steps = np.asarray(steps)
    out = {}
    for s in np.unique(steps):
        m = steps == s
        if y_true[m].sum() > 0:
            out[int(s)] = evaluate(y_true[m], y_prob[m], threshold)["f1"]
    return out
