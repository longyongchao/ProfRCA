#!/usr/bin/env python3
"""通用 benchmark 的有监督指标工具训练：交叉验证评测 + 全量训练产物 + 登记到 models/MANIFEST.json。

新 benchmark 只要整理成 case 目录（见 common.py 文件头），即可：

  python -m profrca.train.train_benchmark --data <case 根目录> --name mybench \\
      --mode multiclass --cv logo:benchmark_attrs.hardware --k-per-label 20 --seeds 5

两种模式：
  multiclass  整段指标统计 → LightGBM 多分类（DBPA / wangqing 工具的配方）。适合单实体或实体固定的系统；
              复合根因按标签复制；--sets 指定特征集，多个特征集时等权概率平均（默认 v1 + shape_v1）。
  service     两段式（RCAEval 工具的配方）：逐实体二分类给出根因实体，再在该实体上做故障类型多分类。
              适合微服务等实体名随系统变化的场景；特征不含实体名，可跨系统；每个 case 只取第一个根因。

交叉验证（--cv，可给多个）：logo:<键> 留一组、group<K>:<键> 分组 K 折、random<K> 随机 K 折（只作上界参考）。
键是 gold.json 的点号路径（如 benchmark_attrs.system），或 labels（标签组合）、location。
--exclude 列出的 case 不进入任何训练集（交叉验证各折与最终产物），但仍参与交叉验证测试。
--k-per-label K 时每折训练集为每个标签只抽 K 个 case，重复 --seeds 个种子（论文主口径 K=20、5 种子）。

输出：--out 下 summary.json（各口径均值）与 rows_<口径>.jsonl（逐 case 预测，不含指标数据）；
     --models-dir 下 <name> 模型文件与 MANIFEST.json 条目（--no-save 时跳过）。
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import common as T
from ..pipeline import MODELS_DIR


def load_items(a):
    dirs = T.list_cases(a.data, [l.strip() for l in open(a.cases, encoding="utf-8") if l.strip()] if a.cases else None)
    if not dirs:
        sys.exit(f"{a.data} 下没有带 metrics.csv 的 case 目录")
    with ProcessPoolExecutor(a.workers) as ex:
        if a.mode == "service":
            kinds = sorted(set().union(*ex.map(T.header_kinds, dirs, chunksize=16)))
            items = list(ex.map(T.service_item, dirs, [kinds] * len(dirs), chunksize=8))
        else:
            kinds = None
            items = list(ex.map(T.multiclass_item, dirs, [a.sets] * len(dirs), chunksize=16))
    return items, kinds


class Trainer:
    """把两种模式统一成 fit(训练 items) → predict(测试 items) → 每个 case 的候选 [{label, location, probability}]。"""

    def __init__(self, a, names, kinds, info):
        self.a, self.names, self.kinds, self.info = a, names, kinds, info

    def fit(self, items):
        if self.a.mode == "service":
            return T.fit_service(items, self.kinds, self.names, self.info)
        return T.fit_ensemble(items, self.names, self.a.sets)

    def predict(self, model, items):
        if self.a.mode == "service":
            return [model.candidates(it["table"], it["ctx"]) for it in items]
        P = T.ensemble_proba(model, items, self.names)
        return [[{"label": self.names[i], "location": None, "probability": round(float(p[i]), 4)}
                 for i in np.argsort(-p)[:4]] for p in P]

    def save(self, model, meta):
        if self.a.mode == "service":
            model.meta = meta | {"kinds": self.kinds}
            return model
        return T.save_ensemble(model, self.info, meta)


def row(it, cands, info):
    gold = it["labels"]
    ranked = list(dict.fromkeys(c["label"] for c in cands))
    r = {"case_id": it["cid"], "gold": gold, "pred": ranked, **T.score_ranked(gold, ranked)}
    mid = {l: (info.get(l) or {}).get("mechanism_id") or T.NATIVE.get(l) or (l if l in T.CATEGORY else None)
           for l in set(gold) | set(ranked)}
    if all(mid.values()):   # 标签可映射到 30 机制时同时给机制级 Acc（封闭世界口径）
        r["acc_mech"] = T.dbot(list(dict.fromkeys(mid[l] for l in ranked)), list(dict.fromkeys(mid[l] for l in gold)))
    if it.get("location") is not None:
        r["loc1"] = int(bool(cands) and cands[0].get("location") == it["location"])
        r["joint1"] = int(bool(cands) and cands[0].get("location") == it["location"] and cands[0]["label"] in gold)
    return r


def cross_validate(tr, items, spec, excluded, k, seeds):
    rows = {}
    for s in seeds:
        rng = np.random.default_rng(s)
        for fold, test in T.folds(items, spec):
            pool = [i for i in range(len(items)) if i not in test and items[i]["cid"] not in excluded]
            train = [items[i] for i in T.sample_k(pool, lambda i: items[i]["labels"], k, rng)]
            if not train:
                continue
            te = sorted(test)
            for i, c in zip(te, tr.predict(tr.fit(train), [items[i] for i in te])):
                rows.setdefault(s, []).append(row(items[i], c, tr.info) | {"fold": fold, "seed": s})
    return rows


def summarize(rows_by_seed):
    keys = [k for k in ("acc", "top1", "hit4", "acc_mech", "loc1", "joint1")
            if all(k in r for rs in rows_by_seed.values() for r in rs)]
    per = [T.mean_rows(rs, keys) for rs in rows_by_seed.values()]
    out = {k: round(float(np.mean([p[k] for p in per])), 4) for k in keys}
    if len(per) > 1:
        out |= {f"{k}_std": round(float(np.std([p[k] for p in per])), 4) for k in keys}
    return out | {"n": per[0]["n"], "seeds": len(per)}


def options(a):
    """登记到 MANIFEST 的训练选项（只记文件名，不记本机绝对路径）。"""
    return {"mode": a.mode, "sets": a.sets if a.mode == "multiclass" else None, "cv": a.cv,
            "k_per_label": a.k_per_label, "seeds": a.seeds,
            **{k: Path(v).name for k in ("exclude", "cases", "label_info") if (v := getattr(a, k))}}


def main(argv=None, prepare=None):
    """prepare(items)：可选钩子，在交叉验证前给 item 的 gold 补派生分组字段（如 train_wangqing 的轮次）。"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="case 根目录（每个子目录一个 case）")
    ap.add_argument("--name", required=True, help="工具名（路由器里的名字，也是默认文件名前缀）")
    ap.add_argument("--mode", choices=("multiclass", "service"), default="multiclass")
    ap.add_argument("--sets", nargs="+", default=["v1", "shape_v1"], help="multiclass 的特征集（v1 / rel_v1 / shape_v1）")
    ap.add_argument("--cv", nargs="*", default=[], help="交叉验证口径，如 logo:benchmark_attrs.hardware group5:labels")
    ap.add_argument("--k-per-label", type=int, help="每折每标签训练 case 数（不给则用全部）")
    ap.add_argument("--seeds", type=int, default=1, help="--k-per-label 时的随机种子个数")
    ap.add_argument("--exclude", help="不进入任何训练集的 case 列表（每行一个或 JSON 列表）")
    ap.add_argument("--cases", help="只用并按此顺序读取这些 case（每行一个）")
    ap.add_argument("--label-info", help="标签说明 JSON：{标签: {category, headline, description[, mechanism_id]}}")
    ap.add_argument("--out", help="评测输出目录（默认 runs/<name>）")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument("--model-file", help="模型文件名（默认 <name>_metric_tool.pkl）")
    ap.add_argument("--role", default="production", help="MANIFEST 中的角色：production 会被 load_tools 默认加载")
    ap.add_argument("--no-save", action="store_true", help="只评测，不保存模型")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    items, kinds = load_items(a)
    if prepare:
        prepare(items)
    excluded = T.read_ids(a.exclude)
    names = sorted({l for it in items for l in it["labels"]})
    info = T.label_info_for(names, a.label_info)
    tr = Trainer(a, names, kinds, info)
    print(f"[{a.name}] cases={len(items)} excluded={len(excluded & {it['cid'] for it in items})} labels={names}"
          + (f" kinds={kinds}" if kinds else ""), flush=True)

    out = Path(a.out or Path("runs") / a.name)
    res = {"name": a.name, "out": str(out), "mode": a.mode, "sets": a.sets if a.mode == "multiclass" else None,
           "k_per_label": a.k_per_label, "cases": len(items), "labels": names, "protocols": {}}
    seeds = range(a.seeds) if a.k_per_label else [0]
    for spec in a.cv:
        rows = cross_validate(tr, items, spec, excluded, a.k_per_label, seeds)
        res["protocols"][spec] = summarize(rows)
        print(f"[{a.name}] {spec}: {res['protocols'][spec]}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"rows_{spec.replace(':', '_').replace(',', '+')}.jsonl", "w", encoding="utf-8") as f:
            for rs in rows.values():
                for r in rs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if a.cv:
        (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    if a.no_save:
        return res
    train = [it for it in items if it["cid"] not in excluded]
    model = tr.save(tr.fit(train), {"train_cases": len(train), "excluded_cases": len(excluded),
                                    "model": f"LightGBM {a.mode}", "name": a.name})
    file = a.model_file or f"{a.name}_metric_tool.pkl"
    Path(a.models_dir).mkdir(parents=True, exist_ok=True)
    model.save(Path(a.models_dir) / file)
    T.register(a.models_dir, a.name, file, {
        "role": a.role, "class": type(model).__name__, "mode": a.mode, "feature_sets": res["sets"],
        "labels": names, "train_cases": len(train), "training": {"script": "profrca.train.train_benchmark",
                                                                   "data": Path(a.data).name, **options(a)},
        "evaluation": res["protocols"]})
    print(f"[{a.name}] saved {Path(a.models_dir) / file} train_cases={len(train)}", flush=True)
    return res


if __name__ == "__main__":
    main()
