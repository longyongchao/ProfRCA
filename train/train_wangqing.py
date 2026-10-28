#!/usr/bin/env python3
"""wangqing 指标工具（wangqing_metric_ens_v1.pkl）：通用训练器 multiclass 模式（v1 + shape_v1，与 DBPA 同配方）。

数据：--data 下带 metrics.csv 的 101 个 case（CockroachDB 单主机，全部为双根因组合；
宿主项目为 benchmarks-data/closed_world_rca/wangqing），按 splits/wangqing_cases.txt 的顺序读取。
标签说明按 mechanisms.NATIVE 映射到 30 机制自动生成。

  logo:labels         主口径：留一组合（14 折），测试组合整体未见过
  logo:derived.round  消融：留一轮（同一组合按注入时间的第 r 次重复为一折），同分布上界
  另外输出 prior 基线（各留一组合折按训练集标签频次排序的常数预测）到 runs/wangqing/prior.json。

  DYLD_LIBRARY_PATH=... python -m profrca.train.train_wangqing --data benchmarks-data/closed_world_rca/wangqing
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from . import common as T
from . import train_benchmark

HERE = Path(__file__).resolve().parent
DEFAULTS = ["--name", "wangqing", "--model-file", "wangqing_metric_ens_v1.pkl",
            "--cases", str(HERE / "splits" / "wangqing_cases.txt")]


def add_rounds(items):
    """轮次 = 该组合按 scoring_only.phases.inject_start 排序的第几次重复（只用于分组，不进特征）。"""
    seen = Counter()
    for it in sorted(items, key=lambda it: datetime.fromisoformat(
            it["gold"]["scoring_only"]["phases"]["inject_start"].replace("Z", "+00:00")).timestamp()):
        combo = T.field(it["gold"], "labels")
        it["gold"].setdefault("derived", {})["round"] = seen[combo]
        seen[combo] += 1


def prior(items):
    rows = []
    for _, test in T.folds(items, "logo:labels"):
        freq = Counter(l for i, it in enumerate(items) if i not in test for l in it["labels"])
        ranked = [l for l, _ in freq.most_common()]
        rows += [T.score_ranked(items[i]["labels"], ranked) for i in test]
    return T.mean_rows(rows, ("acc", "top1"))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--cv" not in argv:
        argv += ["--cv", "logo:labels", "logo:derived.round"]
    kept = []

    def prepare(items):
        add_rounds(items)
        kept.extend(items)

    res = train_benchmark.main(DEFAULTS + argv, prepare=prepare)
    res["prior"] = prior(kept)
    print(f"[wangqing] prior: {res['prior']}", flush=True)
    out = Path(res["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "prior.json").write_text(json.dumps(res["prior"], indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
