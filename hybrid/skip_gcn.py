"""Skip-GCN 하이브리드: [원본 피처 | GNN 임베딩] → XGBoost  (계획서 5.1)

GNN이 만든 64차원 임베딩을 원본 피처 옆에 붙여 XGBoost를 다시 학습한다.
GCN, SAGE, GAT 어느 임베딩이든 같은 코드로 돌리고, 임베딩 파일 경로만 바꾸면 된다.

XGBoost 쪽은 베이스라인과 **완전히 같은 프로토콜**이다 (baselines/xgb_baseline.py의 함수를 그대로 가져다 쓴다).
  - 하이퍼파라미터: 베이스라인이 검증 PR-AUC로 고른 최적 설정을 그대로 고정 (results/xgb_tuning.csv의 1등)
  - ① Train(1~29)으로 조기 종료 학습(검증 PR-AUC) → best_iteration, 검증 기반 임계값
  - ② Train+Val(1~34)을 best_iteration+1 트리로 재학습 → 테스트 1회 평가
  - 임베딩 시드 n ↔ XGBoost 시드 n 으로 짝을 짓는다 (시드 0~4)

실행 예 (저장소 루트에서):
  # 1) 지금 저장된 임베딩(Train+Val 정답을 본 GNN에서 나온 것, 누수 있음)으로 하이브리드  → hybrid_gcn_naive
  python -m hybrid.skip_gcn --emb-models gcn --seeds 0 1 2 3 4
  # 2) OOF 임베딩(results/emb_oof/emb_oof_gcn_seed{n}.npy)으로 하이브리드                → hybrid_gcn
  python -m hybrid.skip_gcn --emb-models gcn --emb-kind oof
  # 3) 임베딩만 입력한 대조군                                                            → emb_only_gcn_naive
  python -m hybrid.skip_gcn --emb-models gcn --mode only
  # 4) 세 모델 임베딩을 전부 이어 붙인 선택 실험 (165 + 64×3)                            → hybrid_all_naive
  python -m hybrid.skip_gcn --emb-models gcn sage gat
  # 5) 파이프라인 점검: 원본 피처만으로 돌려서 xgb 베이스라인 숫자가 그대로 재현되는지 확인 → xgb_check
  python -m hybrid.skip_gcn --mode raw
  # 임베딩 폴더가 다를 때 (예: 드라이브에서 받은 폴더)
  python -m hybrid.skip_gcn --emb-models gcn --emb-dir ~/Downloads/emb

저장: results/metrics/{name}.csv (split = test / val / train), results/pred/pred_{name}_seed{n}.npy,
      results/timestep_f1_{name}.csv, results/hybrid_importance_{name}.csv (그룹별 gain 비중)

누수 주의 (README '하이브리드 누수 처리' 참고):
  --emb-kind naive 는 Train+Val을 다 쓴 GNN의 임베딩이라 학습·검증 노드의 임베딩에 정답이 새겨져 있다.
  그래서 학습·검증 점수가 부풀 수 있으니 테스트 점수를 기준으로 보고, train / val / test 점수 격차를 같이 본다.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.xgb_baseline import BASE_PARAMS, SPACE, make_clf, pos_weight  # noqa: E402
from common.data import EllipticData, load_data, split_summary  # noqa: E402
from common.metrics import best_threshold, evaluate, evaluate_by_timestep  # noqa: E402
from common.utils import RESULTS_DIR, Timer, save_array, save_rows, set_seed  # noqa: E402

EMB_PREFIX = {"naive": "emb", "oof": "emb_oof"}
EMB_DEFAULT_DIR = {"naive": ROOT / "results" / "emb", "oof": ROOT / "results" / "emb_oof"}
ALL_GNNS = ("gcn", "sage", "gat")
INT_PARAMS = ("max_depth", "min_child_weight")


# ---------------------------------------------------------------- 설정·이름
def default_name(mode: str, models: list[str], kind: str, retuned: bool) -> str:
    """결과 이름 규칙 (README 실험 구성표와 같다).
    hybrid_gcn = OOF 임베딩 본 실험, hybrid_gcn_naive = 누수 있는 임베딩, emb_only_* = 임베딩만, xgb_check = 파이프라인 점검."""
    if mode == "raw":
        return "xgb_check"
    tag = "all" if sorted(models) == sorted(ALL_GNNS) else "_".join(models)
    base = "hybrid" if mode == "hybrid" else "emb_only"
    return f"{base}_{tag}" + ("_naive" if kind == "naive" else "") + ("_retuned" if retuned else "")


def load_baseline_params(tuning_csv: str | Path) -> dict:
    """XGBoost 베이스라인이 검증 PR-AUC로 고른 최적 설정을 읽는다 (baselines.xgb_baseline.tune의 결과 파일)."""
    path = Path(tuning_csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 XGBoost 베이스라인을 돌리거나 --tuning-csv 로 경로를 지정하세요.")
    df = pd.read_csv(path)
    missing = [k for k in (*SPACE, "val_pr_auc") if k not in df.columns]
    if missing:
        raise ValueError(f"{path} 에 필요한 열이 없습니다: {missing}")
    best = df.loc[df["val_pr_auc"].idxmax()]
    params = dict(BASE_PARAMS)
    for k in SPACE:
        params[k] = int(best[k]) if k in INT_PARAMS else float(best[k])
    return params


# ---------------------------------------------------------------- 임베딩 읽기
def emb_path(emb_dir: str | Path, kind: str, model: str, seed: int) -> Path:
    return Path(emb_dir) / f"{EMB_PREFIX[kind]}_{model}_seed{seed}.npy"


def check_embeddings_exist(models, kind, emb_dir, seeds) -> None:
    """학습을 시작하기 전에 필요한 임베딩 파일이 다 있는지 확인한다."""
    missing = [emb_path(emb_dir, kind, m, s) for m in models for s in seeds if not emb_path(emb_dir, kind, m, s).exists()]
    if missing:
        hint = ("OOF 임베딩은 hybrid/make_oof_emb.py(구현 예정)로 만들거나, 지금은 --emb-kind naive 로 실행하세요."
                if kind == "oof" else "공유 드라이브에서 받아 results/emb/ 에 넣거나 --emb-dir 로 폴더를 지정하세요.")
        raise FileNotFoundError("임베딩 파일을 찾지 못했습니다:\n  " + "\n  ".join(str(p) for p in missing) + "\n" + hint)


def load_embedding(path: Path, n_nodes: int) -> np.ndarray:
    with open(path, "rb") as f:
        head = f.read(64)
    if head.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{path.name} 은 Git LFS 포인터 파일입니다 (실제 데이터가 아님). "
                           "`git lfs install` 후 `git lfs pull` 로 받으세요.")
    emb = np.load(path, mmap_mode="r")
    if emb.ndim != 2 or emb.shape[0] != n_nodes:
        raise ValueError(f"{path.name}: 모양이 {emb.shape} 입니다. 노드 수 {n_nodes}개인 (N, 차원) 배열이어야 합니다 "
                         "(노드 순서 = load_data() 순서 = txId 오름차순).")
    if not np.isfinite(emb).all():
        raise ValueError(f"{path.name}: NaN 또는 inf 가 들어 있습니다.")
    return np.asarray(emb, dtype=np.float32)


def build_features(data: EllipticData, embs: list[np.ndarray], mode: str) -> np.ndarray:
    """hybrid = [원본 | 임베딩...], only = [임베딩...], raw = 원본만."""
    if mode == "raw":
        return data.x
    blocks = ([data.x] if mode == "hybrid" else []) + list(embs)
    return np.ascontiguousarray(np.hstack(blocks), dtype=np.float32)


def feature_groups(n_raw: int, models: list[str], dims: list[int], mode: str) -> dict[str, tuple[int, int]]:
    """열 범위: {'raw': (0, 165), 'gcn': (165, 229), ...}  (중요도를 그룹별로 합산할 때 사용)"""
    groups, start = {}, 0
    if mode in ("hybrid", "raw"):
        groups["raw"] = (0, n_raw)
        start = n_raw
    if mode != "raw":
        for m, d in zip(models, dims):
            groups[m] = (start, start + d)
            start += d
    return groups


# ---------------------------------------------------------------- 보조 기록
def gnn_cost(models: list[str], seed: int, results_dirs) -> tuple[float, float] | None:
    """GNN 쪽 학습·추론 시간 (비용 비교용). 모든 모델의 기록이 있을 때만 합산해서 돌려준다."""
    tot_train = tot_infer = 0.0
    for m in models:
        got = None
        for d in results_dirs:
            p = Path(d) / "metrics" / f"{m}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, dtype={"threshold_type": str})
            r = df[(df["seed"] == seed) & (df["split"] == "test") & (df["threshold_type"] == "fixed_0.5")]
            if len(r):
                got = (float(r["train_sec"].iloc[0]), float(r["infer_sec"].iloc[0]))
                break
        if got is None:
            return None
        tot_train += got[0]
        tot_infer += got[1]
    return tot_train, tot_infer


def group_importance(booster, groups: dict, name: str, seed: int) -> list[dict]:
    """그룹(원본 / 각 GNN 임베딩)별 gain 비중. 계획서 5.2의 '임베딩 전체 기여도 비율'의 간이 버전 (SHAP은 analysis/ 에서)."""
    gain = booster.get_score(importance_type="gain")  # {'f12': gain, ...} 사용된 피처만
    total = float(sum(gain.values()))
    rows = []
    for g, (a, b) in groups.items():
        used = [v for k, v in gain.items() if a <= int(k[1:]) < b]
        rows.append(dict(model=name, seed=seed, group=g, n_features=b - a, n_used=len(used),
                         gain_share=float(sum(used) / total) if total > 0 else 0.0))
    return rows


def save_replace_seed(df_new: pd.DataFrame, path: Path, seed: int) -> None:
    """같은 시드의 이전 기록은 교체하고 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        df_new = pd.concat([old[old["seed"] != seed], df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


# ---------------------------------------------------------------- 재튜닝 (선택)
def retune(X: np.ndarray, data: EllipticData, trials: int, n_jobs: int, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """하이브리드 입력으로 검증 PR-AUC 기준 랜덤 탐색 (선택 실험). 후보 목록은 베이스라인 tune()과 같은 방식으로 뽑는다."""
    Xtr, ytr = X[data.train_mask], data.y[data.train_mask]
    Xva, yva = X[data.val_mask], data.y[data.val_mask]
    spw = pos_weight(ytr)
    grid = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(grid), size=min(trials, len(grid)), replace=False)
    log = []
    for i, gi in enumerate(picks):
        params = {**BASE_PARAMS, **grid[gi]}
        clf = make_clf(params, spw, seed, n_jobs, early_stopping=True)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        score = float(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))
        log.append({**grid[gi], "val_pr_auc": score, "best_iteration": int(clf.best_iteration)})
        print(f"[retune {i + 1}/{len(picks)}] {grid[gi]} val PR-AUC={score:.4f}")
    log_df = pd.DataFrame(log).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best = {k: (int(log_df.loc[0, k]) if k in INT_PARAMS else float(log_df.loc[0, k])) for k in SPACE}
    return {**BASE_PARAMS, **best}, log_df


# ---------------------------------------------------------------- 시드별 실행
def run_seed(data: EllipticData, X: np.ndarray, params: dict, name: str, seed: int, n_jobs: int,
             results_dir, groups: dict, notes_extra: str = "") -> list[dict]:
    set_seed(seed)
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    ytr, yva, yte = data.y[tr], data.y[va], data.y[te]
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    spw = pos_weight(ytr)

    # ① Train으로 조기 종료 학습 → best_iteration, 검증 기반 임계값 (베이스라인과 같은 절차)
    clf = make_clf(params, spw, seed, n_jobs, early_stopping=True)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    n_trees = int(clf.best_iteration) + 1
    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_val = clf.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, p_val)

    # ② Train+Val 재학습 (고정 트리 수), 테스트 1회 평가
    Xall, yall = X[tr | va], data.y[tr | va]
    final = make_clf({**params, "n_estimators": n_trees}, pos_weight(yall), seed, n_jobs, early_stopping=False)
    with Timer() as t_train:
        final.fit(Xall, yall, verbose=False)
    times = []
    for _ in range(5):
        with Timer() as t_inf:
            p_test = final.predict_proba(Xte)[:, 1]
        times.append(t_inf.sec)
    infer_sec = float(np.median(times))
    n_params = int(len(final.get_booster().trees_to_dataframe()))  # 트리 노드 수

    notes = f"trees={n_trees}" + (f";{notes_extra}" if notes_extra else "")
    common = dict(model=name, seed=seed, train_sec=t_train.sec, infer_sec=infer_sec,
                  n_params=n_params, device="cpu", notes=notes)
    rows = []
    for ttype, th in (("fixed_0.5", 0.5), ("val_opt", thr)):
        rows.append({**common, "split": "test", "threshold_type": ttype, **evaluate(yte, p_test, th)})
    # 검증·학습 행: ①(Train만 학습)의 모델 기준. 누수 진단용 (train/val/test 격차)
    light = {k: v for k, v in common.items() if k not in ("train_sec", "infer_sec", "n_params")}
    rows.append({**light, "split": "val", "threshold_type": "val_opt", **evaluate(yva, p_val, thr)})
    rows.append({**light, "split": "train", "threshold_type": "val_opt", **evaluate(ytr, p_tr, thr)})

    save_array(final.predict_proba(X)[:, 1], "pred", name, seed, results_dir)

    # 시간 단계별 F1 (테스트)
    by_step = evaluate_by_timestep(yte, p_test, data.time_step[te], thr)
    step_df = pd.DataFrame([dict(model=name, seed=seed, time_step=s, f1=f) for s, f in by_step.items()])
    save_replace_seed(step_df, Path(results_dir) / f"timestep_f1_{name}.csv", seed)
    # 그룹별 중요도
    imp = pd.DataFrame(group_importance(final.get_booster(), groups, name, seed))
    save_replace_seed(imp, Path(results_dir) / f"hybrid_importance_{name}.csv", seed)

    share = {r["group"]: r["gain_share"] for r in imp.to_dict("records")}
    share_txt = " ".join(f"{g}={v:.0%}" for g, v in share.items())
    print(f"[seed {seed}] PR-AUC train={rows[3]['pr_auc']:.4f} val={rows[2]['pr_auc']:.4f} test={rows[0]['pr_auc']:.4f} | "
          f"F1@0.5={rows[0]['f1']:.4f} F1@val_opt={rows[1]['f1']:.4f} (thr={thr:.3f}) | "
          f"트리 {n_trees}개, gain 비중 {share_txt} | train={t_train.sec:.1f}s")
    return rows


# ---------------------------------------------------------------- 요약
def _mean_std(s: pd.Series) -> str:
    return f"{s.mean():.4f} ± {(s.std(ddof=1) if len(s) > 1 else 0.0):.4f}"


def summarize(rows: list[dict], name: str, seeds: list[int], results_dirs) -> None:
    df = pd.DataFrame(rows)
    pick = lambda split, tt: df[(df["split"] == split) & (df["threshold_type"] == tt)]
    print(f"\n===== {name}  (시드 {len(seeds)}개 평균 ± 표준편차) =====")
    for label, split in (("학습(train, ①모델)", "train"), ("검증(val, ①모델)", "val"), ("테스트(test, ②모델)", "test")):
        d = pick(split, "val_opt" if split != "test" else "fixed_0.5")
        print(f"PR-AUC  {label:<18} {_mean_std(d['pr_auc'])}")
    t05, tvo = pick("test", "fixed_0.5"), pick("test", "val_opt")
    print(f"F1      테스트 임계값 0.5      {_mean_std(t05['f1'])}")
    print(f"F1      테스트 검증 최적 임계값 {_mean_std(tvo['f1'])}")
    tr_pr, va_pr, te_pr = (pick(s, "val_opt" if s != "test" else "fixed_0.5")["pr_auc"].mean() for s in ("train", "val", "test"))
    print(f"격차    train − test = {tr_pr - te_pr:+.4f}   val − test = {va_pr - te_pr:+.4f}   "
          "(해석: 같은 방식으로 --mode raw 를 돌린 xgb_check의 격차와 비교하세요. XGBoost는 원래 train 점수가 1에 가깝고, "
          "시간이 지나며 분포가 변해 val − test 도 원래 큽니다)")
    for d in results_dirs:  # 베이스라인(xgb)과 비교: 같은 시드끼리, 같은 임계값 규칙끼리
        p = Path(d) / "metrics" / "xgb.csv"
        if not p.exists():
            continue
        x = pd.read_csv(p, dtype={"threshold_type": str})
        x = x[(x["split"] == "test") & (x["seed"].isin(seeds))]
        x05, xvo = x[x["threshold_type"] == "fixed_0.5"], x[x["threshold_type"] == "val_opt"]
        if len(x05) and len(xvo):
            print(f"xgb 대비  ΔPR-AUC {t05['pr_auc'].mean() - x05['pr_auc'].mean():+.4f} "
                  f"(xgb {x05['pr_auc'].mean():.4f})   ΔF1@0.5 {t05['f1'].mean() - x05['f1'].mean():+.4f}   "
                  f"ΔF1@val_opt {tvo['f1'].mean() - xvo['f1'].mean():+.4f}   [비교 파일: {p}]")
            break


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Skip-GCN 하이브리드 / 대조군 (XGBoost 입력에 GNN 임베딩을 붙인다)")
    ap.add_argument("--data-root", default="data/elliptic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--emb-models", nargs="+", default=["gcn"], choices=list(ALL_GNNS),
                    help="사용할 임베딩의 GNN. 여러 개면 전부 이어 붙인다 (예: gcn sage gat)")
    ap.add_argument("--emb-kind", choices=list(EMB_PREFIX), default="naive",
                    help="naive: results/emb/emb_{model}_seed{n}.npy (누수 있음)  /  oof: results/emb_oof/emb_oof_{model}_seed{n}.npy")
    ap.add_argument("--emb-dir", default=None, help="임베딩 폴더 (기본: kind에 따라 results/emb 또는 results/emb_oof)")
    ap.add_argument("--mode", choices=["hybrid", "only", "raw"], default="hybrid",
                    help="hybrid: 원본+임베딩 / only: 임베딩만 / raw: 원본만 (베이스라인 재현 점검용)")
    ap.add_argument("--tuning-csv", default=str(ROOT / "results" / "xgb_tuning.csv"),
                    help="베이스라인 튜닝 기록. 1등 설정을 그대로 고정해서 쓴다")
    ap.add_argument("--retune-trials", type=int, default=0,
                    help="0보다 크면 하이브리드 입력으로 재튜닝 (선택 실험, 결과 이름 뒤에 _retuned)")
    ap.add_argument("--name", default=None, help="결과 이름을 직접 지정 (기본: 자동)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)

    rdir = Path(args.results_dir or RESULTS_DIR)
    rdir.mkdir(parents=True, exist_ok=True)
    models = args.emb_models
    name = args.name or default_name(args.mode, models, args.emb_kind, args.retune_trials > 0)
    emb_dir = Path(args.emb_dir) if args.emb_dir else EMB_DEFAULT_DIR[args.emb_kind]
    if args.mode != "raw":
        check_embeddings_exist(models, args.emb_kind, emb_dir, args.seeds)

    print(f"결과 이름: {name} | 모드: {args.mode} | 임베딩: {'+'.join(models) if args.mode != 'raw' else '-'} ({args.emb_kind}) | 시드: {args.seeds}")
    if args.mode != "raw" and args.emb_kind == "naive":
        print("※ naive 임베딩은 Train+Val 정답을 본 GNN에서 나온 것입니다. 학습·검증 점수는 부풀 수 있으니 테스트 점수를 기준으로 보세요.")

    data = load_data(args.data_root, standardize=False)
    print(split_summary(data).to_string(index=False))

    def embs_for(seed):
        return [load_embedding(emb_path(emb_dir, args.emb_kind, m, seed), data.num_nodes) for m in models] if args.mode != "raw" else []

    params = load_baseline_params(args.tuning_csv)
    print("XGBoost 설정(베이스라인 최적):", {k: params[k] for k in SPACE})
    if args.retune_trials > 0:
        t0 = time.time()
        X0 = build_features(data, embs_for(args.seeds[0]), args.mode)
        params, log_df = retune(X0, data, args.retune_trials, args.n_jobs)
        log_df.to_csv(rdir / f"xgb_tuning_{name}.csv", index=False)
        print("재튜닝 선택:", {k: params[k] for k in SPACE}, f"({time.time() - t0:.0f}s, 임베딩은 시드 {args.seeds[0]} 기준)")
        del X0

    all_rows = []
    for seed in args.seeds:
        embs = embs_for(seed)
        X = build_features(data, embs, args.mode)
        groups = feature_groups(data.num_features, models, [e.shape[1] for e in embs], args.mode)
        cost = gnn_cost(models, seed, [rdir, ROOT / "results"]) if args.mode != "raw" else None
        extra = f"mode={args.mode};emb={'+'.join(models) if embs else '-'};kind={args.emb_kind};dim={X.shape[1]}"
        if cost:
            extra += f";gnn_train_sec={cost[0]:.1f};gnn_infer_sec={cost[1]:.3f}"
        all_rows += run_seed(data, X, params, name, seed, args.n_jobs, rdir, groups, extra)
        del X, embs
    path = save_rows(name, all_rows, rdir)
    print("저장:", path)
    summarize(all_rows, name, args.seeds, [rdir, ROOT / "results"])


if __name__ == "__main__":
    main()
