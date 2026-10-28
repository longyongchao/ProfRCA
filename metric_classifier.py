"""指标-only 的有监督根因分类分支（DBPA 官方基线思路：多指标窗口 → 梯度提升树多分类）。

与 LLM 零样本链路不同，这里需要带标签的历史故障记录训练；模型只在输入指标的 schema 与训练 schema
足够重合时启用（能力域门控），否则返回 None 由调用方回落到 LLM 链路。特征不依赖注入窗口：
每个指标取整段统计量，再加一个数据估计的变点（单序列 CUSUM 最大分割）前后的稳健水平。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

STATS = ("mean", "std", "min", "max", "p10", "p50", "p90", "first", "last",
         "pre_med", "post_med", "shift", "rel_shift", "onset_frac")
# 尺度无关统计：以变点前水平/离散度归一化，硬件规格不同导致的绝对量级差异被消掉
REL_STATS = ("z_shift", "log_ratio", "max_rel", "cv", "range_rel")
SHAPE_STATS = ("rel_shift", "onset_frac") + REL_STATS
FEATURE_SETS = {"v1": STATS, "rel_v1": STATS + REL_STATS, "shape_v1": SHAPE_STATS}
MIN_COVERAGE = 0.9
MAX_CANDIDATES = 4
MIN_PROB = 0.01


def metric_name(col):
    """'entity|metric' 取 metric；同名指标在不同实体（数据库）间共享特征。"""
    return col.split("|", 1)[1] if "|" in col else col


def _series_stats(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [np.nan] * len(STATS + REL_STATS)
    n = len(x)
    p10, p50, p90 = np.percentile(x, (10, 50, 90))
    # 单序列变点：最大化 |前后均值差|·sqrt(n1·n2/n)，两段各至少 2 点
    k = n // 2
    if n >= 4:
        c = np.cumsum(x)
        i = np.arange(2, n - 1)
        gain = np.abs(c[i - 1] / i - (c[-1] - c[i - 1]) / (n - i)) * np.sqrt(i * (n - i) / n)
        k = int(i[np.argmax(gain)])
    pre, post = np.median(x[:k]), np.median(x[k:])
    mad = np.median(np.abs(x[:k] - pre)) * 1.4826
    scale = mad + 0.1 * x.std() + 1e-9
    return [x.mean(), x.std(), x.min(), x.max(), p10, p50, p90, x[0], x[-1],
            pre, post, post - pre, (post - pre) / (abs(pre) + 1e-9), k / n,
            (post - pre) / scale, np.log1p(abs(post)) - np.log1p(abs(pre)),
            (x.max() - pre) / (abs(pre) + 1e-9), x.std() / (abs(x.mean()) + 1e-9),
            (p90 - p10) / (abs(p50) + 1e-9)]


def featurize(wide, time_col="time", feature_set="v1"):
    """宽表（time + 'entity|metric' 列）→ {f'{metric}::{stat}': value}。"""
    keep = set(FEATURE_SETS[feature_set])
    out = {}
    for c in wide.columns:
        if c == time_col:
            continue
        x = pd.to_numeric(wide[c], errors="coerce").to_numpy(dtype=float)
        for s, v in zip(STATS + REL_STATS, _series_stats(x)):
            if s in keep:
                out[f"{metric_name(c)}::{s}"] = v
    return out


class MetricClassifier:
    """训练产物：特征列、标签表（标签 → 类别/中文机制描述）与 LightGBM 多分类模型。"""

    def __init__(self, model, features, labels, label_info, meta=None):
        self.model, self.features, self.labels = model, list(features), list(labels)
        self.label_info, self.meta = label_info, meta or {}
        self.metrics = {f.split("::", 1)[0] for f in self.features}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        # 只加载本地训练生成的产物
        with open(path, "rb") as f:
            return pickle.load(f)

    def coverage(self, wide, time_col="time"):
        cols = {metric_name(c) for c in wide.columns if c != time_col}
        return len(cols & self.metrics) / len(cols) if cols else 0.0

    def matrix(self, feats):
        return np.array([[d.get(f, np.nan) for f in self.features] for d in feats], dtype=float)

    def predict(self, wide, time_col="time"):
        """→ (各标签概率, 各标签贡献最大的 3 个指标名)。"""
        X = self.matrix([featurize(wide, time_col, self.meta.get("feature_set", "v1"))])
        prob = self.model.predict_proba(X)[0]
        contrib = self.model.predict(X, pred_contrib=True)
        contrib = np.asarray(contrib).reshape(-1, len(self.features) + 1)
        if len(contrib) == 1 < len(self.labels):  # 二分类只给正类贡献
            contrib = np.vstack([-contrib, contrib])
        ev = [[self.features[j].split("::", 1)[0] for j in np.argsort(-c[:-1])[:3]] for c in contrib]
        return prob, ev

    def diagnose(self, wide, time_col="time"):
        """返回 fast_rca 同格式的 root_causes；schema 覆盖不足时返回 None。"""
        cov = self.coverage(wide, time_col)
        if cov < MIN_COVERAGE:
            return None
        prob, ev = self.predict(wide, time_col)
        return _root_causes(self.labels, self.label_info, prob, ev, cov)


class MetricEnsemble:
    """多个特征集分类器的等权概率平均（原始统计偏同环境精度，尺度无关统计偏跨硬件泛化）。"""

    def __init__(self, members):
        assert all(m.labels == members[0].labels for m in members)
        self.members = list(members)
        self.labels, self.label_info = members[0].labels, members[0].label_info
        self.meta = {"ensemble": [m.meta.get("feature_set", "v1") for m in members]}

    save = MetricClassifier.save

    def coverage(self, wide, time_col="time"):
        return min(m.coverage(wide, time_col) for m in self.members)

    def diagnose(self, wide, time_col="time"):
        cov = self.coverage(wide, time_col)
        if cov < MIN_COVERAGE:
            return None
        preds = [m.predict(wide, time_col) for m in self.members]
        prob = np.mean([p for p, _ in preds], axis=0)
        ev = [sum((e[i] for _, e in preds), []) for i in range(len(self.labels))]
        return _root_causes(self.labels, self.label_info, prob, ev, cov)


KIND_ALIAS = {"latency": "latency-90", "load": "workload"}  # 同一指标在不同导出版本里的命名
SERVICE_STATS = ("z_shift", "log_ratio", "max_rel", "rel_shift", "cv")


def metric_kind(col):
    k = metric_name(col)
    return KIND_ALIAS.get(k, k)


def service_table(wide, kinds, time_col="time"):
    """宽表（'实体|指标'）→ {实体: {kind::stat: v}}。

    除各实体自身的变化统计外，再加该统计在本 case 所有实体间的分位秩与相对最大值：
    特征只描述"这个实体相对同伴变得多异常"，不含实体名，可跨系统迁移。"""
    pos = {s: i for i, s in enumerate(STATS + REL_STATS)}
    ent = {}
    for c in wide.columns:
        if c == time_col or "|" not in c or metric_kind(c) not in kinds:
            continue
        st = _series_stats(pd.to_numeric(wide[c], errors="coerce").to_numpy(dtype=float))
        d = ent.setdefault(c.split("|", 1)[0], {})
        for s in SERVICE_STATS:
            d[f"{metric_kind(c)}::{s}"] = st[pos[s]]
    for k in kinds:
        for s in ("z_shift", "log_ratio"):
            key = f"{k}::{s}"
            vals = {e: abs(d[key]) for e, d in ent.items() if np.isfinite(d.get(key, np.nan))}
            if not vals:
                continue
            mx = max(vals.values()) + 1e-9
            for r, e in enumerate(sorted(vals, key=vals.get)):
                ent[e][key + "_rank"] = (r + 1) / len(vals)
                ent[e][key + "_rel"] = vals[e] / mx
    return ent


def case_context(table, kinds):
    """case 级上下文：每种指标在所有实体上的最大 |z| 与 |z|>3 的实体比例。"""
    out = {}
    for k in kinds:
        z = np.array([abs(d.get(f"{k}::z_shift", np.nan)) for d in table.values()], dtype=float)
        z = z[np.isfinite(z)]
        out[f"ctx::{k}::max_z"] = z.max() if len(z) else np.nan
        out[f"ctx::{k}::frac_z3"] = (z > 3).mean() if len(z) else np.nan
    return out


class ServiceFaultClassifier:
    """微服务指标的两段式有监督诊断：①逐实体二分类给出根因实体排序；②以该实体的变化统计 + case 上下文
    做故障类型多分类。实体特征不含实体名，跨系统可用；schema 门控同 MetricClassifier。"""

    def __init__(self, kinds, loc_model, loc_features, fault_model, fault_features, labels, label_info, meta=None):
        self.kinds, self.loc_model, self.loc_features = set(kinds), loc_model, list(loc_features)
        self.fault_model, self.fault_features = fault_model, list(fault_features)
        self.labels, self.label_info, self.meta = list(labels), label_info, meta or {}

    save, load = MetricClassifier.save, staticmethod(MetricClassifier.load)

    def coverage(self, wide, time_col="time"):
        cols = {metric_kind(c) for c in wide.columns if c != time_col}
        return len(cols & self.kinds) / len(cols) if cols else 0.0

    @staticmethod
    def fault_row(table, ent, ctx):
        return {**{f"svc::{k}": v for k, v in table[ent].items()}, **ctx}

    def rank(self, table):
        """各实体特征表 → 实体按根因概率降序 [(实体, p)]。"""
        ents = sorted(table)
        X = np.array([[table[e].get(f, np.nan) for f in self.loc_features] for e in ents], dtype=float)
        p = self.loc_model.predict_proba(X)[:, 1]
        return sorted(zip(ents, p), key=lambda t: -t[1])

    def fault_proba(self, table, ent, ctx):
        row = self.fault_row(table, ent, ctx)
        return self.fault_model.predict_proba(np.array([[row.get(f, np.nan) for f in self.fault_features]],
                                                       dtype=float))[0]

    def diagnose(self, wide, time_col="time"):
        cov = self.coverage(wide, time_col)
        if cov < MIN_COVERAGE:
            return None
        table = service_table(wide, self.kinds, time_col)
        return {"root_causes": self.candidates(table, case_context(table, self.kinds)), "coverage": round(cov, 3)}

    def candidates(self, table, ctx):
        ranked = self.rank(table)
        # 候选：前两个实体各自最可能的故障交替排列，再补首位实体的第二故障类型
        cands = []
        for ent, ps in ranked[:2]:
            fp = self.fault_proba(table, ent, ctx)
            cands.append([(ent, ps, int(i), fp[i]) for i in np.argsort(-fp)[:2]])
        order = [c[0] for c in cands] + [c[1] for c in cands]
        out = []
        for ent, ps, i, pf in order[:MAX_CANDIDATES]:
            info = self.label_info[self.labels[i]]
            out.append({"type": info["category"], "headline": f"服务 {ent}：{info['headline']}",
                        "description": f"{info['description']}（有监督指标分类器：根因实体置信度 {ps:.2f}，"
                                       f"故障类型置信度 {pf:.2f}）",
                        "location": ent, "evidence_templates": [], "evidence_refs": [],
                        "label": self.labels[i], "probability": round(float(ps * pf), 4)})
        return out


class MetricToolRouter:
    """在多个有监督指标工具间路由：按输入指标 schema 与各工具训练 schema 的重合度选最匹配者；
    都不够匹配时返回 None，由调用方回落到 LLM 零样本链路。路由只看指标名，0 次 LLM 调用。"""

    def __init__(self, tools):
        self.tools = dict(tools)  # 名称 → 工具（需提供 coverage / diagnose）

    def route(self, wide, time_col="time"):
        cov = {n: t.coverage(wide, time_col) for n, t in self.tools.items()}
        best = max(cov, key=cov.get) if cov else None
        return (best if best is not None and cov[best] >= MIN_COVERAGE else None), cov

    def diagnose(self, wide, time_col="time"):
        name, cov = self.route(wide, time_col)
        if name is None:
            return None
        out = self.tools[name].diagnose(wide, time_col)
        if out is not None:
            out["tool"] = name
        return out


def _root_causes(labels, label_info, prob, ev, cov):
    order = [i for i in np.argsort(-prob) if prob[i] >= MIN_PROB][:MAX_CANDIDATES] or [int(np.argmax(prob))]
    out = []
    for i in order:
        lab = labels[i]
        info = label_info[lab]
        out.append({"type": info["category"], "headline": info["headline"],
                    "description": f"{info['description']}（有监督指标分类器置信度 {prob[i]:.2f}；"
                                   f"主要依据指标：{'、'.join(dict.fromkeys(ev[i]))}）",
                    "location": None, "evidence_templates": [], "evidence_refs": [],
                    "label": lab, "probability": round(float(prob[i]), 4)})
    return {"root_causes": out, "coverage": round(cov, 3)}
