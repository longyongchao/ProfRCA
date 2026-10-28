"""训练公共部分：通用 case 目录读取、LightGBM 拟合配方、交叉验证、小样本抽样、计分与模型清单登记。

通用 case 目录格式（新 benchmark 按此准备即可，宿主项目的 closed_world_rca/<bench>/ 已是此格式）：

    <data_dir>/<case_id>/metrics.csv   指标宽表：time 列 + 「实体|指标」列（没有 "|" 时按最后一个下划线切实体）
    <data_dir>/<case_id>/gold.json     标签：{"root_causes": [{"native_label" 或 "label": ..., "location": ...}], ...}
                                       （也可以叫 label.json；其余字段可作分组键，如 benchmark_attrs.hardware）

训练配方（三个已发布工具相同，不调参）：lgb.LGBMClassifier(random_state=42, verbose=-1)；
复合根因 case 按每个标签各复制一行做多分类；服务级定位模型额外用 class_weight="balanced"。
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..mechanisms import CATEGORY, MECHANISMS, NATIVE
from ..metric_classifier import (MetricClassifier, MetricEnsemble, ServiceFaultClassifier, case_context, featurize,
                                 metric_kind, service_table)

LGB_PARAMS = {"random_state": 42, "verbose": -1}


# ---------------------------------------------------------------- 数据
def read_gold(d):
    d = Path(d)
    for name in ("gold.json", "label.json"):
        if (d / name).exists():
            return json.loads((d / name).read_text(encoding="utf-8"))
    raise FileNotFoundError(f"{d} 下没有 gold.json / label.json")


def labels_of(gold):
    """金标准 native 标签，按顺序去重。"""
    return list(dict.fromkeys(r.get("native_label") or r["label"] for r in gold["root_causes"]))


def field(gold, path):
    """分组键取值：点号路径（benchmark_attrs.hardware；列表值排序后用 + 连接）；
    labels = 标签组合；location = 首个根因位置。"""
    if path == "labels":
        return "+".join(sorted(labels_of(gold)))
    if path == "location":
        return gold["root_causes"][0].get("location")
    v = gold
    for k in path.split("."):
        v = v[k]
    return "+".join(sorted(map(str, v))) if isinstance(v, list) else v


def list_cases(data_dir, case_ids=None):
    """有 metrics.csv 的 case 目录（按名称排序；case_ids 给定时按给定顺序）。"""
    root = Path(data_dir)
    ids = case_ids if case_ids is not None else sorted(p.name for p in root.iterdir() if p.is_dir())
    return [root / c for c in ids if (root / c / "metrics.csv").exists()]


def read_ids(path):
    """case 列表文件：每行一个 case_id，或 JSON 列表。"""
    if not path:
        return set()
    text = Path(path).read_text(encoding="utf-8")
    return set(json.loads(text)) if text.lstrip().startswith("[") else {l.strip() for l in text.splitlines() if l.strip()}


def multiclass_item(d, sets):
    wide = pd.read_csv(Path(d) / "metrics.csv")
    g = read_gold(d)
    return {"cid": Path(d).name, "gold": g, "labels": labels_of(g),
            "feats": {fs: featurize(wide, feature_set=fs) for fs in sets}}


def header_kinds(d):
    cols = pd.read_csv(Path(d) / "metrics.csv", nrows=0).columns
    return {metric_kind(c) for c in cols if c != "time"}


def service_item(d, kinds):
    wide = pd.read_csv(Path(d) / "metrics.csv")
    g = read_gold(d)
    table = service_table(wide, kinds)
    return {"cid": Path(d).name, "gold": g, "labels": labels_of(g)[:1], "table": table,
            "ctx": case_context(table, kinds), "location": g["root_causes"][0].get("location")}


# ---------------------------------------------------------------- 拟合
def fit_multiclass(feats, labs, names):
    """特征字典列表 + 每条的标签列表 → MetricClassifier（复合按标签复制）。"""
    import lightgbm as lgb
    X, y = [], []
    for f, ls in zip(feats, labs):
        for l in ls:
            X.append(f)
            y.append(names.index(l))
    features = sorted({k for f in feats for k in f})
    clf = MetricClassifier(None, features, names, {})
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(clf.matrix(X), np.array(y))
    clf.model = model
    clf.labels = [names[i] for i in model.classes_]   # 小样本折里可能缺类
    return clf


def fit_ensemble(items, names, sets):
    members = []
    for fs in sets:
        m = fit_multiclass([it["feats"][fs] for it in items], [it["labels"] for it in items], names)
        m.meta = {"feature_set": fs, "features": len(m.features)}
        members.append(m)
    return members


def ensemble_proba(members, items, names):
    """等权平均概率，列按 names 对齐（缺类列为 0）。"""
    P = np.zeros((len(items), len(names)))
    for m in members:
        p = m.model.predict_proba(m.matrix([it["feats"][m.meta["feature_set"]] for it in items]))
        P[:, [names.index(l) for l in m.labels]] += p / len(members)
    return P


def fit_service(items, kinds, labels, label_info):
    """逐实体根因二分类（balanced）+ 根因实体上的故障类型多分类 → ServiceFaultClassifier。"""
    import lightgbm as lgb
    loc_feats = sorted({k for it in items for d in it["table"].values() for k in d})
    X, y = [], []
    for it in items:
        for e, d in it["table"].items():
            X.append([d.get(f, np.nan) for f in loc_feats])
            y.append(int(e == it["location"]))
    loc = lgb.LGBMClassifier(**LGB_PARAMS, class_weight="balanced").fit(np.array(X, float), y)
    rows = [ServiceFaultClassifier.fault_row(it["table"], it["location"], it["ctx"]) for it in items]
    fault_feats = sorted({k for r in rows for k in r})
    yf = [labels.index(it["labels"][0]) for it in items]
    fm = lgb.LGBMClassifier(**LGB_PARAMS).fit(np.array([[r.get(f, np.nan) for f in fault_feats] for r in rows], float), yf)
    present = [labels[i] for i in fm.classes_]   # fault 模型只见过训练里出现的类别
    return ServiceFaultClassifier(kinds, loc, loc_feats, fm, fault_feats, present, label_info)


# ---------------------------------------------------------------- 标签说明
def label_info_for(names, path=None):
    """标签 → {category, headline, description[, mechanism_id]}。

    给了 JSON 文件就用文件（缺的标签报错）；否则按 mechanisms.NATIVE 映射到 30 机制取说明，
    映射不到的标签用标签名本身、类别记为 unknown（封闭世界评测会计为无效 ID，建议提供文件）。"""
    if path:
        info = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = [n for n in names if n not in info]
        if missing:
            raise KeyError(f"{path} 缺少标签说明：{missing}")
        return {n: info[n] for n in names}
    desc = {m: d for m, _, d in MECHANISMS}
    out = {}
    for n in names:
        mid = NATIVE.get(n) or (n if n in CATEGORY else None)
        out[n] = ({"category": CATEGORY[mid], "headline": desc[mid], "description": desc[mid], "mechanism_id": mid}
                  if mid else {"category": "unknown", "headline": n, "description": n})
    return out


# ---------------------------------------------------------------- 评测
def dbot(pred, gold):
    """Top-K_GT D-Bot Acc：K = 金标准个数，max(0, (Ac - 0.1*Aw) / K)。"""
    k = len(gold)
    top = pred[:k]
    ac = len(set(top) & set(gold))
    return max(0.0, (ac - 0.1 * (len(top) - ac)) / k) if k else 0.0


def score_ranked(gold_labels, ranked):
    return {"acc": dbot(ranked, gold_labels), "top1": int(bool(ranked) and ranked[0] in gold_labels),
            "hit4": int(bool(set(ranked[:4]) & set(gold_labels)))}


def mean_rows(rows, keys):
    return {"n": len(rows), **{k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys}} if rows else None


def sample_k(pool, labels_fn, k, rng):
    """pool 内下标 → 每个标签各随机抽 k 个含该标签的 case，取并集；k=None 返回全部。"""
    if k is None:
        return list(pool)
    chosen = set()
    for lab in sorted({l for i in pool for l in labels_fn(i)}):
        cand = [i for i in pool if lab in labels_fn(i)]
        chosen |= set(rng.choice(cand, min(k, len(cand)), replace=False).tolist())
    return sorted(chosen)


def folds(items, spec, seed=42):
    """交叉验证切分 → [(折名, 测试下标集合)]。

    spec: "logo:<键>"   留一组（键可用逗号连接多个字段）
          "group<K>:<键>" 分组 K 折（同组整体落在同一折）
          "random<K>"  随机 K 折（同场重复可能分在两侧，偏乐观，只作上界参考）"""
    rng = np.random.default_rng(seed)
    if spec.startswith("random"):
        k = int(spec[6:] or 5)
        f = rng.permutation(len(items)) % k
        return [(f"fold{j}", {i for i in range(len(items)) if f[i] == j}) for j in range(k)]
    kind, key = spec.split(":", 1)
    gkey = [tuple(str(field(it["gold"], p)) for p in key.split(",")) for it in items]
    groups = sorted(set(gkey))
    if kind == "logo":
        return [("/".join(g), {i for i in range(len(items)) if gkey[i] == g}) for g in groups]
    k = int(kind[5:] or 5)
    gf = {groups[j]: n % k for n, j in enumerate(rng.permutation(len(groups)))}
    return [(f"fold{j}", {i for i in range(len(items)) if gf[gkey[i]] == j}) for j in range(k)]


# ---------------------------------------------------------------- 产物登记
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lib_versions():
    import lightgbm
    import sklearn
    return {"python": platform.python_version(), "lightgbm": lightgbm.__version__, "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


def register(models_dir, name, file, entry):
    """把模型写进 models/MANIFEST.json（同名覆盖条目；文件本身由调用方保存）。"""
    d = Path(models_dir)
    mf = d / "MANIFEST.json"
    man = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"tools": {}}
    man["tools"][name] = {"file": file, "sha256": sha256(d / file), "bytes": (d / file).stat().st_size,
                          "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "libs": lib_versions(), **entry}
    mf.write_text(json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def save_ensemble(members, label_info, meta):
    for m in members:
        m.label_info = label_info
    ens = MetricEnsemble(members) if len(members) > 1 else members[0]
    ens.label_info = label_info
    ens.meta |= meta
    return ens
