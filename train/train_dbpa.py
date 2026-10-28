#!/usr/bin/env python3
"""DBPA 指标工具（dbpa_metric_ens_v1.pkl）的训练与评测，复现宿主项目 train_metric_classifier.py +
eval_metric_clf_ensemble.py 的配方。

训练数据：官方 DBPA_dataset 全量记录（--full 解压目录，含 32_128/compound/*.pickle 等），**排除** 10% 评测子集
（--dbpa 下 manifest/sources.csv 的 selected_original_indices，即 2,155 个评测 case），不含 normal。
测试数据：--cases 下 2,155 个 case 目录（metrics.csv + gold.json；宿主项目为 benchmarks-data/closed_world_rca/dbpa）。
特征只用整段指标，不读注入窗口；复合故障记录按每个标签各复制一份。

  random   训练 = 全量 − 子集，测试 = 子集
  loho     留一硬件配置（主口径）；--k-per-label K 时每折每标签只抽 K 条训练记录，重复 --seeds 个种子

  DYLD_LIBRARY_PATH=... python -m profrca.train.train_dbpa --full /path/DBPA_dataset \\
      --dbpa benchmarks-data/DBPA --cases benchmarks-data/closed_world_rca/dbpa
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import common as T
from ..metric_classifier import featurize
from ..pipeline import MODELS_DIR

META = {"sample_id", "source_id", "original_index", "row_index"}
CATEGORY = {"io_saturation": "disk_io", "small_shared_buffer": "memory"}   # 其余 7 个标签为 db_internal
LABEL_INFO = {
    "io_saturation": ("磁盘 I/O 饱和（其他进程争用磁盘 I/O）", "外部进程大量读写磁盘，数据库 I/O 等待升高、吞吐下降"),
    "small_shared_buffer": ("共享缓冲区 / 缓冲池配置过小", "缓冲命中率下降，频繁从磁盘读页"),
    "lock_waits": ("数据库锁等待", "事务互相等待行/表锁，活跃连接堆积"),
    "missing_indexes": ("缺少索引导致全表扫描", "查询走顺序扫描，读放大、CPU 与读 I/O 升高"),
    "too_many_indexes": ("索引过多导致写入维护开销大", "每次写入需维护大量索引，写放大、DML 变慢"),
    "vacuum": ("清理（vacuum）负载 / 死元组膨胀", "大量更新删除后 vacuum 清理占用 I/O 与 CPU"),
    "concurrent_commits": ("高并发小事务频繁提交（WAL 写锁争用）", "大量短事务提交，WAL 刷盘与 WALWriteLock 争用"),
    "concurrent_inserts": ("高并发插入", "大量并发 INSERT 争用同一表/页，插入延迟升高"),
    "heavy_workload": ("整体业务负载过重", "并发业务请求量整体上升，资源普遍吃紧"),
}
NAMES = sorted(LABEL_INFO)


def label_info():
    return {l: {"category": CATEGORY.get(l, "db_internal"), "headline": h, "description": d}
            for l, (h, d) in LABEL_INFO.items()}


def sources(dbpa):
    with open(Path(dbpa) / "manifest" / "sources.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def header(dbpa, src):
    with open(Path(dbpa) / src["viewable_csv"], encoding="utf-8-sig", newline="") as f:
        cols = [c for c in next(csv.reader(f)) if c not in META]
    return ["time"] + [f"{src['database']}|{c}" for c in cols[1:]]


def full_records(full, dbpa, src, sets):
    """一个来源的全量 pickle 中未被抽进评测子集的记录 → [{特征集: 特征}]（pickle 只从可信来源读）。"""
    cols = header(dbpa, src)
    sel = set(json.loads(src["selected_original_indices"]))
    with open(Path(full) / src["source_pickle"], "rb") as f:
        data = pickle.load(f)
    out = []
    for i, d in enumerate(data):
        if i in sel:
            continue
        w = pd.DataFrame(list(d[0]), columns=cols)
        w["time"] = 0.0
        out.append({fs: featurize(w, feature_set=fs) for fs in sets})
    return out


def score(P, tests):
    rows = []
    for it, p in zip(tests, P):
        ranked = [NAMES[i] for i in np.argsort(-p)]
        a = it["gold"]["benchmark_attrs"]
        rows.append({"case_id": it["cid"], "db": a["database"], "hw": a["hardware"], "gold": it["labels"],
                     "pred": ranked[:4], **T.score_ranked(it["labels"], ranked)})
    by = {}
    for key in ("db", "hw"):
        by[key] = {v: T.mean_rows([r for r in rows if r[key] == v], ("acc", "top1"))
                   for v in sorted({r[key] for r in rows})}
    single = {l: T.mean_rows([r for r in rows if r["gold"] == [l]], ("acc",))["acc"]
              for l in NAMES if any(r["gold"] == [l] for r in rows)}
    return {"all": T.mean_rows(rows, ("acc", "top1", "hit4")), **by, "single_label": single}, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="官方 DBPA_dataset 解压目录")
    ap.add_argument("--dbpa", required=True, help="DBPA 子集目录（含 manifest/sources.csv 与 viewable_csv/）")
    ap.add_argument("--cases", required=True, help="2,155 个评测 case 目录的根")
    ap.add_argument("--sets", nargs="+", default=["v1", "shape_v1"])
    ap.add_argument("--k-per-label", type=int, help="loho 每折每标签训练记录数（论文主口径 20）")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/dbpa")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--model-file", default="dbpa_metric_ens_v1.pkl")
    ap.add_argument("--save-members", action="store_true", help="另存各特征集单模型（role=ablation）")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    srcs = sources(a.dbpa)
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(full_records, [a.full] * len(srcs), [a.dbpa] * len(srcs), srcs, [a.sets] * len(srcs)))
        tests = list(ex.map(T.multiclass_item, T.list_cases(a.cases), [a.sets] * 9999, chunksize=32))
    train = [{"feats": f, "labels": s["ground_truth"].split("|"), "hw": s["hardware"]}
             for s, p in zip(srcs, parts) for f in p]
    print(f"train records={len(train)}（全量 − 子集，不含 normal） test cases={len(tests)}", flush=True)
    if not train:
        sys.exit("全量记录为空：--full 目录不对，或全部记录都在评测子集里")

    res = {"sets": a.sets, "train_records": len(train)}
    members = T.fit_ensemble(train, NAMES, a.sets)
    res["random"], rows = score(T.ensemble_proba(members, tests, NAMES), tests)
    hws = sorted({t["hw"] for t in train})
    for k, seeds in ((None, [0]),) + (((a.k_per_label, range(a.seeds)),) if a.k_per_label else ()):
        accs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            P = np.zeros((len(tests), len(NAMES)))
            for hw in hws:
                pool = [i for i, t in enumerate(train) if t["hw"] != hw]
                sub = [train[i] for i in T.sample_k(pool, lambda i: train[i]["labels"], k, rng)]
                idx = [i for i, t in enumerate(tests) if t["gold"]["benchmark_attrs"]["hardware"] == hw]
                P[idx] = T.ensemble_proba(T.fit_ensemble(sub, NAMES, a.sets), [tests[i] for i in idx], NAMES)
            r, _ = score(P, tests)
            accs.append(r)
        key = "loho" if k is None else f"loho_k{k}"
        res[key] = accs[0] if len(accs) == 1 else {
            "acc_mean": round(float(np.mean([r["all"]["acc"] for r in accs])), 4),
            "acc_std": round(float(np.std([r["all"]["acc"] for r in accs])), 4),
            "top1_mean": round(float(np.mean([r["all"]["top1"] for r in accs])), 4), "seeds": len(accs)}
    for k, r in res.items():
        if isinstance(r, dict):
            print(k, r.get("all", r), r.get("db", ""), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with open(out / "rows_random.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.no_save:
        return res
    info = label_info()
    evaluation = {k: (v["all"] if "all" in v else v) for k, v in res.items() if isinstance(v, dict)}
    entry = {"mode": "multiclass", "labels": NAMES, "train_cases": len(train),
             "training": {"script": "profrca.train.train_dbpa", "data": "DBPA 官方全量 − 10% 评测子集",
                          "sets": a.sets, "k_per_label": a.k_per_label, "seeds": a.seeds}}
    if a.save_members:
        for m in members:
            m.label_info, fs = info, m.meta["feature_set"]
            m.meta = {"train_records": len(train), "feature_set": fs, "features": len(m.features)}
            m.save(Path(a.models_dir) / f"dbpa_metric_clf_{fs}.pkl")
            T.register(a.models_dir, f"dbpa_{fs}", f"dbpa_metric_clf_{fs}.pkl",
                       entry | {"role": "ablation", "class": "MetricClassifier", "feature_sets": [fs]})
    ens = T.save_ensemble(members, info, {"train_records": len(train)})
    ens.save(Path(a.models_dir) / a.model_file)
    T.register(a.models_dir, "dbpa", a.model_file, entry | {"role": "production", "class": type(ens).__name__,
                                                           "feature_sets": a.sets, "evaluation": evaluation})
    print(f"saved {Path(a.models_dir) / a.model_file}", flush=True)
    return res


if __name__ == "__main__":
    main()
