"""실제 Elliptic 데이터 검증 (11/1 완료 기준 2번).

  python scripts/verify_data.py data/elliptic

원본 데이터셋의 알려진 값과 대조해 불일치가 있으면 FAIL을 출력하고 종료 코드 1로 끝난다.
--lenient 를 주면 규모 검사를 건너뛰고 구조 검사만 한다 (가짜 데이터용).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import EXPECTED, N_STEPS, load_data, split_summary  # noqa: E402


def verify(root, lenient: bool = False) -> bool:
    d = load_data(root, standardize=True)
    ok = True

    def check(name, got, want=None, cond=None):
        nonlocal ok
        passed = (got == want) if cond is None else cond
        print(f"[{'OK' if passed else 'FAIL'}] {name}: {got}" + ("" if want is None or cond is not None else f" (기대 {want})"))
        ok &= bool(passed)

    n_ill, n_lic, n_unk = int((d.y == 1).sum()), int((d.y == 0).sum()), int((d.y == 2).sum())
    if not lenient:
        check("노드 수", d.num_nodes, EXPECTED["n_nodes"])
        check("원본 방향 간선 수", d.n_edges_raw, EXPECTED["n_edges"])
        check("피처 차원(시간 단계 제외)", d.num_features, EXPECTED["n_features"])
        check("불법 수", n_ill, EXPECTED["n_illicit"])
        check("정상 수", n_lic, EXPECTED["n_licit"])
        check("미분류 수", n_unk, EXPECTED["n_unknown"])
    else:
        print(f"노드 {d.num_nodes}, 원본 간선 {d.n_edges_raw}, 피처 {d.num_features}, 불법 {n_ill} / 정상 {n_lic} / 미분류 {n_unk}")

    check("시간 단계 범위", (int(d.time_step.min()), int(d.time_step.max())), (1, N_STEPS))
    check("노드 순서 = txId 오름차순", bool(np.all(np.diff(d.node_ids) > 0)), cond=bool(np.all(np.diff(d.node_ids) > 0)))
    tr, va, te = d.train_mask, d.val_mask, d.test_mask
    check("마스크 상호 배타", int((tr & va).sum() + (tr & te).sum() + (va & te).sum()), 0)
    check("마스크에 미분류 없음", int((d.y[tr | va | te] == 2).sum()), 0)
    check("train 시간 단계 <= 29", int(d.time_step[tr].max()), cond=int(d.time_step[tr].max()) <= 29)
    check("val 시간 단계 30~34", (int(d.time_step[va].min()), int(d.time_step[va].max())),
          cond=int(d.time_step[va].min()) >= 30 and int(d.time_step[va].max()) <= 34)
    check("test 시간 단계 >= 35", int(d.time_step[te].min()), cond=int(d.time_step[te].min()) >= 35)

    ei = d.edge_index
    same_step = float((d.time_step[ei[0]] == d.time_step[ei[1]]).mean())
    check("간선이 같은 시간 단계 안에서만 존재하는 비율", round(same_step, 4), cond=same_step > 0.999)
    n = np.int64(d.num_nodes)
    fwd, rev = ei[0] * n + ei[1], ei[1] * n + ei[0]
    check("간선 무방향 대칭·중복 없음", True,
          cond=bool(np.array_equal(np.sort(fwd), np.sort(rev)) and len(np.unique(fwd)) == len(fwd)))

    print("\n분할별 노드 수 (README에 기록):")
    print(split_summary(d).to_string(index=False))
    print("\n결과:", "모든 검사 통과" if ok else "FAIL이 있습니다. 위 항목을 확인하세요.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/elliptic")
    ap.add_argument("--lenient", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if verify(a.root, a.lenient) else 1)
