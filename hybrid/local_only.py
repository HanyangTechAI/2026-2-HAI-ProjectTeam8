"""local 피처(93개)만 쓰는 조건으로 기존 스크립트를 그대로 실행하는 래퍼.

Elliptic 피처 165개 = local 93개(0~92열, 거래 자체 정보) + aggregated 72개(93~164열, 1-hop 이웃 집계).
aggregated 피처는 사람이 미리 만든 '이웃 정보'라서, 165개를 다 쓰면 GNN 임베딩이 더할 정보가 적다.
이 래퍼는 load_data()가 돌려주는 피처를 앞 93개로 자른 뒤 지정한 모듈을 실행한다.
GNN 학습(models.*), OOF 임베딩(hybrid.make_oof_emb), 하이브리드(hybrid.run_hybrid) 코드는 수정하지 않는다.

  python -m hybrid.local_only models.gcn --data-root data/elliptic --results-dir results_local --lr 0.01 ...
  python -m hybrid.local_only hybrid.make_oof_emb --model gcn --ref-emb-dir results_local/emb ...
  python -m hybrid.local_only hybrid.run_hybrid --mode raw --name xgb_local --results-dir results_local

결과 이름·폴더는 기존 결과와 섞이지 않게 실행할 때 따로 지정한다 (--results-dir, --name, --out-dir).
"""
from __future__ import annotations

import dataclasses
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common.data as cdata  # noqa: E402

N_LOCAL = 93
_orig_load_data = cdata.load_data


def load_data_local(*args, **kwargs):
    """common.data.load_data와 같고, x와 x_std를 local 93열로 자른다 (표준화 통계는 열별이라 그대로 유효)."""
    d = _orig_load_data(*args, **kwargs)
    if d.x.shape[1] != cdata.EXPECTED["n_features"]:
        raise ValueError(f"피처 수가 {d.x.shape[1]}개입니다. 165개(원본)여야 local 93개를 정확히 고를 수 있습니다.")
    x_std = None if d.x_std is None else d.x_std[:, :N_LOCAL].copy()
    return dataclasses.replace(d, x=d.x[:, :N_LOCAL].copy(), x_std=x_std)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        raise SystemExit("사용법: python -m hybrid.local_only <모듈> [그 모듈의 인자...]  (예: models.gcn, hybrid.run_hybrid)")
    module, rest = argv[0], argv[1:]
    # 대상 모듈이 `from common.data import load_data`로 가져가기 전에 바꿔 둔다
    cdata.load_data = load_data_local
    for name, mod in list(sys.modules.items()):
        if name != "common.data" and getattr(mod, "load_data", None) is _orig_load_data:
            mod.load_data = load_data_local
    print(f"[local_only] 피처를 local {N_LOCAL}개로 제한해서 실행: {module} {' '.join(rest)}", flush=True)
    sys.argv = [module, *rest]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
