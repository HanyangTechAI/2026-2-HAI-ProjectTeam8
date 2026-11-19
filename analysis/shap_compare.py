"""SHAP 비교 (계획서 5.2): XGBoost 베이스라인 vs 하이브리드의 피처 기여도.

비교 모델 (모두 같은 XGBoost 설정·절차: 베이스라인 튜닝값 고정, Train 조기 종료 → Train+Val 재학습)
  xgb_raw          : 원본 피처 165개만
  hybrid_oofemb    : 원본 + OOF 임베딩 64차원 (누수 처리, 팀 방식)
  hybrid_oofprob   : 원본 + OOF 불법 확률 1열
  hybrid_leaky     : 원본 + 기존 임베딩 64차원 (누수 있음, 비교용)
설명 대상: 테스트 노드(35~49) 전체. SHAP = TreeExplainer (로그오즈 단위).

피처 그룹 (Weber et al. 2019, 시간 단계 열 제외):
  local : 0~92번  — 거래 자체 정보 93개
  agg   : 93~164번 — 1-hop 이웃 거래를 요약(집계)한 72개
  gnn   : 165번~  — GNN 출력 (임베딩 64개 또는 확률 1개). 개별 차원은 의미가 없으므로 합산해서 본다.

실행:
  python -m analysis.shap_compare --gnn sage --oof-dir results/oof
  (OOF 임베딩은 hybrid/skip_gcn.py가 results/oof/oofemb_{gnn}_seed{n}.npy로 생성)
저장: results/shap/{gnn}/  group_share.csv, feature_importance.csv, top20.csv, *.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xgboost  # noqa: F401  (macOS libomp 로드 순서)
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.xgb_baseline import make_clf, pos_weight  # noqa: E402
from common.data import load_data  # noqa: E402
from common.utils import RESULTS_DIR, set_seed  # noqa: E402
from hybrid.skip_gcn import baseline_xgb_params  # noqa: E402

N_LOCAL, N_AGG = 93, 72
GROUPS = ["local", "agg", "gnn"]
GROUP_LABEL = {"local": "Local tx features (93)", "agg": "Neighbor-aggregated features (72)",
               "gnn": "GNN output"}
# 차트 색 (dataviz 기본 팔레트 slot 1~3, 검증 통과. 대비 보완용으로 범례·CSV 제공)
GROUP_COLOR = {"local": "#2a78d6", "agg": "#eb6834", "gnn": "#1baf7a"}
MODEL_LABEL = {"xgb_raw": "XGBoost (raw only)", "hybrid_oofemb": "Hybrid: OOF embedding",
               "hybrid_oofprob": "Hybrid: OOF probability", "hybrid_leaky": "Hybrid: leaky embedding"}


def feature_names(n_extra: int) -> list[str]:
    names = [f"local_{i}" for i in range(N_LOCAL)] + [f"agg_{i}" for i in range(N_AGG)]
    return names + ([f"gnn_prob"] if n_extra == 1 else [f"gnn_emb_{i}" for i in range(n_extra)])


def group_of(name: str) -> str:
    return name.split("_")[0]


def train_final(X, data, params, seed):
    """베이스라인과 같은 절차로 최종 모델을 만든다 (Train 조기 종료 → 같은 트리 수로 Train+Val 재학습)."""
    set_seed(seed)
    y, tr, va = data.y, data.train_mask, data.val_mask
    clf = make_clf(params, pos_weight(y[tr]), seed, -1, early_stopping=True)
    clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    n_trees = int(clf.best_iteration) + 1
    trval = tr | va
    final = make_clf({**params, "n_estimators": n_trees}, pos_weight(y[trval]), seed, -1, early_stopping=False)
    final.fit(X[trval], y[trval], verbose=False)
    return final, n_trees


def plot_group_share(share: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = share.groupby("model")[GROUPS].mean().reindex([k for k in MODEL_LABEL if k in share.model.unique()])
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(m) + 1.4), dpi=150)
    left = np.zeros(len(m))
    for g in GROUPS:
        vals = m[g].to_numpy() * 100
        ax.barh(range(len(m)), vals, left=left, height=0.55, color=GROUP_COLOR[g],
                edgecolor="white", linewidth=2, label=GROUP_LABEL[g])
        for i, (l, v) in enumerate(zip(left, vals)):
            if v >= 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += vals
    ax.set_yticks(range(len(m)), [MODEL_LABEL[k] for k in m.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total mean |SHAP| on test nodes (%)", fontsize=9, color="#555")
    ax.set_title("Where does each model's decision come from?", fontsize=11, loc="left")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", colors="#777", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_top20(imp: pd.DataFrame, models: list[str], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 6.2), dpi=150)
    for ax, mname in zip(np.atleast_1d(axes), models):
        top = imp[imp.model == mname].nlargest(20, "mean_abs_shap").iloc[::-1]
        ax.barh(range(len(top)), top.mean_abs_shap, height=0.6,
                color=[GROUP_COLOR[group_of(n)] for n in top.feature])
        ax.set_yticks(range(len(top)), top.feature, fontsize=7)
        ax.set_title(MODEL_LABEL[mname], fontsize=10, loc="left")
        ax.set_xlabel("mean |SHAP| (log-odds)", fontsize=8, color="#555")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", colors="#777", labelsize=7)
        ax.tick_params(axis="y", length=0)
    fig.legend(handles=[Patch(color=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in GROUPS],
               ncol=3, fontsize=8, frameon=False, loc="lower center")
    fig.suptitle("Top 20 features by mean |SHAP| (test nodes, averaged over seeds)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default="sage")
    ap.add_argument("--data-root", default="data/elliptic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--oof-dir", default=None, help="oofemb/oofprob 파일 위치 (기본 results/oof)")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)

    src = Path(RESULTS_DIR)
    oof_dir = Path(args.oof_dir) if args.oof_dir else src / "oof"
    out = Path(args.results_dir or src) / "shap" / args.gnn
    out.mkdir(parents=True, exist_ok=True)
    G = args.gnn

    data = load_data(args.data_root, standardize=False)
    params = baseline_xgb_params(src)
    te = data.test_mask
    extras = {
        "xgb_raw": lambda s: None,
        "hybrid_oofemb": lambda s: np.load(oof_dir / f"oofemb_{G}_seed{s}.npy"),
        "hybrid_oofprob": lambda s: np.load(oof_dir / f"oofprob_{G}_seed{s}.npy")[:, None],
        "hybrid_leaky": lambda s: np.load(src / "emb" / f"emb_{G}_seed{s}.npy"),
    }

    imp_rows, share_rows = [], []
    for mname, get_extra in extras.items():
        for seed in args.seeds:
            extra = get_extra(seed)
            X = data.x if extra is None else np.hstack([data.x, extra]).astype(np.float32)
            names = feature_names(0 if extra is None else extra.shape[1])[: X.shape[1]]
            model, n_trees = train_final(X, data, params, seed)
            sv = shap.TreeExplainer(model).shap_values(X[te])
            mabs = np.abs(sv).mean(0)
            imp_rows += [dict(model=mname, seed=seed, feature=n, mean_abs_shap=float(v))
                         for n, v in zip(names, mabs)]
            tot = mabs.sum()
            grp = {g: float(sum(v for n, v in zip(names, mabs) if group_of(n) == g) / tot) for g in GROUPS}
            share_rows.append(dict(model=mname, seed=seed, trees=n_trees, **grp))
            print(f"[{mname} seed {seed}] trees={n_trees} 비중 local {grp['local']:.1%} | "
                  f"agg {grp['agg']:.1%} | gnn {grp['gnn']:.1%}", flush=True)

    imp = pd.DataFrame(imp_rows)
    share = pd.DataFrame(share_rows)
    share.to_csv(out / "group_share.csv", index=False)
    imp_mean = imp.groupby(["model", "feature"], as_index=False).mean_abs_shap.mean()
    imp_mean["group"] = imp_mean.feature.map(group_of)
    imp_mean["rank"] = imp_mean.groupby("model").mean_abs_shap.rank(ascending=False, method="first").astype(int)
    imp_mean.sort_values(["model", "rank"]).to_csv(out / "feature_importance.csv", index=False)
    top20 = imp_mean[imp_mean["rank"] <= 20].sort_values(["model", "rank"])
    top20.to_csv(out / "top20.csv", index=False)

    plot_group_share(share, out / "group_share.png")
    plot_top20(imp_mean, ["xgb_raw", "hybrid_oofemb", "hybrid_leaky"], out / "top20.png")

    print("\n=== 그룹별 기여도 비중 (시드 평균 ± 표준편차) ===")
    s = share.groupby("model")[GROUPS + ["trees"]].agg(["mean", "std"])
    for mname in extras:
        r = s.loc[mname]
        print(f"{MODEL_LABEL[mname]:28s} " + " | ".join(
            f"{g} {r[(g, 'mean')]:.1%} ± {r[(g, 'std')]:.1%}" for g in GROUPS) + f" | trees {r[('trees', 'mean')]:.0f}")
    print("\n=== 상위 20개 피처의 그룹 구성 ===")
    print(top20.groupby(["model", "group"]).size().unstack(fill_value=0).to_string())
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
