"""Full-text, single-shot LLM baseline ported from the pinned host implementation.
No Drain3, summarization, resampling or silent truncation is applied.
Source and adaptation notes: docs/SOURCE_PROVENANCE.md.
"""
import json
import re
from ...llm import chat

CONTEXT_TOKENS = 1_048_576
SKIP_FACTOR = 1.25
MAX_TOKENS = 800

SYSTEM = """你是资深数据库/分布式系统 SRE。用户在收到告警后上传了这段时间的全部原始日志和监控指标，这段时间内系统确实发生了故障（可能同时有 1~3 个根因）。
请直接阅读原始日志和指标，判断根因。

- 根因类别只能取：cpu | memory | disk_io | disk_space | network | process | syscall | connection | db_internal | code
  （process=进程/容器被杀、暂停、停止、重启；db_internal=锁争用/事务重试风暴/Schema 变更/检查点风暴等数据库内部负载；
   connection=连接池/会话/连接数耗尽；syscall=系统调用注入错误或延迟；code=应用代码缺陷）
- mechanism 写具体机制（如"网络延迟""网络丢包""磁盘写 IO 压力""磁盘空间占满""CPU 满载""进程被 SIGKILL"），不要泛泛写"性能下降"。
- 输出 4 个候选根因，类别互不相同，按可能性从高到低排序（第 1 个最可信，后面是鉴别诊断的备选）。

只输出 JSON：{"root_causes":[{"category":"...","mechanism":"...","location":"组件/节点/服务","evidence":["日志片段或指标列名，每条不超过60字，最多2条"],"why":"不超过20字"}]}"""

NO_LOG = "（本次没有可用的日志，只有监控指标。请只依据指标判断。）"


def to_predictions(text):
    m = re.search(r"\{.*\}", text, re.S)
    obj = json.loads(m.group(0)) if m else {}
    out = []
    for rc in (obj.get("root_causes") or [])[:4]:
        cat, mech, loc = (str(rc.get(k, "")).strip() for k in ("category", "mechanism", "location"))
        out.append({"type": cat, "headline": mech, "description": f"{mech}（类别 {cat}，位置 {loc}）。{rc.get('why', '')}",
                    "location": loc, "evidence_templates": [str(e)[:120] for e in rc.get("evidence") or []]})
    return out



def diagnose(log=None, metrics=None, label_space=None):
    if label_space is not None:
        from ...errors import ConfigurationError
        raise ConfigurationError('The full-text baseline preserves the upstream open-category prompt; label_space must be None')
    parts = [("以下是日志全文：\n\n" + log.read_text(encoding="utf-8-sig")) if log else NO_LOG]
    if metrics is not None:
        parts.append("以下是同一时段的监控指标全文（CSV，time 为秒，其余列为实体|指标）：\n\n"
                     + metrics.to_csv(index=False))
    user = "\n\n".join(parts)
    estimate = int(sum(len(x) for x in parts) / 1.85)
    if estimate > CONTEXT_TOKENS * SKIP_FACTOR:
        raise OverflowError("Full input exceeds the baseline context budget; input was not truncated")
    usage, preds, error = [], [], None
    for _ in range(2):
        try:
            content, used = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                 max_tokens=MAX_TOKENS)
            usage.append(used)
            preds = to_predictions(content)
            if preds:
                break
        except json.JSONDecodeError as exc:
            error = type(exc).__name__ + ": invalid prediction JSON"
    return {"root_causes": preds, "usage": usage, "error": None if preds else error or "no predictions",
            "route": "llm_full_text", "prompt_chars": len(user)}
