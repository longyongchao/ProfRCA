#!/usr/bin/env python3
"""RCAEval 指标工具（rcaeval_metric_clf_v1.pkl）：通用训练器 service 模式的固定参数版本。

数据：--data 下 732 个 case（RE1 373 + RE2/RE3 359；宿主项目为 benchmarks-data/closed_world_rca/rcaeval），
按 splits/rcaeval_cases.txt 的顺序读取（与宿主项目 manifest 一致，LightGBM 结果与读取顺序有关）。
splits/rcaeval_heldout.txt 的 35 个 case（宿主项目 val90 的 30 个 + 开发集 5 个）不进入任何训练集。

默认口径 logo:benchmark_attrs.system（留一系统 OB/SS/TT，主口径）；可追加 group5:benchmark_attrs.system,location,labels。
其余参数原样传给 train_benchmark（如 --k-per-label 20 --seeds 5 --no-save）。

  DYLD_LIBRARY_PATH=... python -m profrca.train.train_rcaeval --data benchmarks-data/closed_world_rca/rcaeval
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import train_benchmark

HERE = Path(__file__).resolve().parent
DEFAULTS = ["--name", "rcaeval", "--mode", "service", "--model-file", "rcaeval_metric_clf_v1.pkl",
            "--cases", str(HERE / "splits" / "rcaeval_cases.txt"),
            "--exclude", str(HERE / "splits" / "rcaeval_heldout.txt"),
            "--label-info", str(HERE / "label_info" / "rcaeval.json")]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--cv" not in argv:
        argv += ["--cv", "logo:benchmark_attrs.system"]
    return train_benchmark.main(DEFAULTS + argv)


if __name__ == "__main__":
    main()
