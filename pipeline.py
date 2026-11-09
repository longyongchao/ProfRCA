"""一次诊断的完整组合：有监督指标工具优先，schema 未命中时回落到 日志画像 + LLM 两轮推理。

与宿主项目 benchmark/unified 的评测口径一致：指标列名形如 "实体|指标"（无 "|" 时按最后一个下划线切），
封闭世界答案空间为 mechanisms.MECHANISMS 的 30 个机制 ID；工具输出的 native 标签经 mechanisms.NATIVE
（或训练时写入 label_info 的 mechanism_id）映射到机制 ID。
"""
from __future__ import annotations

import json
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


def load_tools(models_dir=None, names=None, include_ablation=False):
    """按 models/MANIFEST.json 加载有监督指标工具 → MetricToolRouter。

    names 限定工具名（如 ["dbpa"]）；include_ablation 为 True 时也加载消融用的单模型。
    pickle 只能加载本地可信来源的文件。"""
    from .metric_classifier import MetricClassifier, MetricToolRouter
    d = Path(models_dir) if models_dir else MODELS_DIR
    manifest = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    tools = {}
    for name, ent in manifest["tools"].items():
        if (names and name not in names) or (ent.get("role") != "production" and not include_ablation):
            continue
        tools[name] = MetricClassifier.load(d / ent["file"])
    return MetricToolRouter(tools)


def closed_label_space():
    """封闭世界 30 机制答案空间（论文主口径）。"""
    from .mechanisms import CATEGORY, prompt_block
    return {"prompt": prompt_block(), "category": CATEGORY}


def split_metric_col(col):
    """"实体|指标" → (实体, 指标)；没有 "|" 时退回 metric_events.split_col（最后一个下划线）。"""
    from .metric_events import split_col
    return tuple(col.split("|", 1)) if "|" in col else split_col(col)


def _read_metrics(metrics):
    import pandas as pd
    return metrics if isinstance(metrics, pd.DataFrame) else pd.read_csv(metrics)


def _tool_root_causes(rcs, label_info, closed):
    """工具候选的 native 标签 → 机制 ID，按机制去重（复合标签可能映射到同一机制）。"""
    from .mechanisms import CATEGORY, NATIVE
    if not closed:
        return list(rcs)
    out, seen = [], set()
    for rc in rcs:
        mid = (label_info.get(rc["label"]) or {}).get("mechanism_id") or NATIVE.get(rc["label"])
        if mid in seen:
            continue
        seen.add(mid)
        out.append({**rc, "mechanism_id": mid, "type": CATEGORY.get(mid, rc["type"])})
    return out


def diagnose(log=None, metrics=None, *, tools=None, label_space="mechanisms", time_col="time",
             use_tools=True, line_refs=0, **llm_kwargs):
    """诊断一个故障样例。

    log:         日志 TXT 路径（可选；没有日志时走仅指标路径）
    metrics:     指标宽表 CSV 路径或 DataFrame（可选；首列 time，其余列 "实体|指标"）
    tools:       MetricToolRouter / {名称: 工具} / None（None 时不用有监督工具）
    label_space: "mechanisms"（封闭 30 机制，默认）、None（宿主项目的一级类别白名单）或自定义 dict
    line_refs:   > 0 时每个候选附 evidence_lines {T#: [行号, ...]}（每个模板最多 line_refs 行，1 起算），
                 不改变 prompt；默认 0 不输出
    其余关键字参数原样传给 fast_rca.diagnose_profile（world_mode、metrics_only_mode、request_timeout 等）。

    返回 {"root_causes": [...], "route": "metric_tool"|"llm", "llm_calls", "output_tokens", ...}；
    每个候选含 mechanism_id（封闭世界时）、type、headline、description、location、evidence_*。
    """
    from .metric_classifier import MetricToolRouter
    ls = closed_label_space() if label_space == "mechanisms" else label_space
    wide = _read_metrics(metrics) if metrics is not None else None
    if wide is not None and tools is not None and use_tools:
        router = tools if isinstance(tools, MetricToolRouter) else MetricToolRouter(tools)
        hit = router.diagnose(wide, time_col)
        if hit is not None:
            info = router.tools[hit["tool"]].label_info
            return {"root_causes": _tool_root_causes(hit["root_causes"], info, ls is not None),
                    "route": "metric_tool", "tool": hit["tool"], "coverage": hit["coverage"],
                    "usage": [], "llm_calls": 0, "output_tokens": 0, "error": None}
    from .fast_rca import build_profile, diagnose_profile, parse_log
    from .metric_events import metrics_text
    mt, mode = metrics_text(wide, time_col, split=split_metric_col) if wide is not None else (None, None)
    prof = build_profile(parse_log(str(log)), line_refs=line_refs) if log else None
    out = diagnose_profile(prof, mt, mode == "raw", label_space=ls, **llm_kwargs)
    usage = out.get("usage") or []
    out.update(route="llm", metrics_mode=mode, llm_calls=len(usage),
               output_tokens=sum(int(u.get("completion_tokens") or 0) for u in usage))
    return out
