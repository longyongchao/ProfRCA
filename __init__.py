"""ProfRCA：基于异常画像的可信高效根因诊断。

常用入口：
    from profrca import diagnose, load_tools
    out = diagnose(log="app.log", metrics="metrics.csv", tools=load_tools())

子模块：
    fast_rca           日志结构化（Drain3）→ 异常画像 → 两轮 LLM 推理
    metric_events      指标宽表 → 原始表 / 降采样表 / 变化事件（M#）
    metric_classifier  有监督指标工具（LightGBM）与 schema 路由
    mechanisms         封闭世界 30 机制集与 Top-K_GT 评分
    pipeline           以上模块的组合（工具优先，未命中回落 LLM）
"""

__version__ = "1.0.0"

from .pipeline import diagnose, load_tools  # noqa: E402

__all__ = ["diagnose", "load_tools", "__version__"]
