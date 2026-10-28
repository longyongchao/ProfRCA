"""快速日志根因诊断：日志 TXT → Drain3 模板 → 时序异常画像（纯统计）→ 少量 LLM 调用。

与 Tree Search 链路的区别：不再从十几万行里抽 60 条 AEU，而是把全量日志压成
「模板 × 时间桶」计数矩阵，用变点检测估异常起点，按模板前后速率变化（泊松似然比）
给模板排序，只把排名靠前的模板、错误级模板、新出现模板和十类信号词对照表交给 LLM。

用法：
    from profrca.fast_rca import diagnose_log_file
    out = diagnose_log_file("xxx.txt")   # {"root_causes": [...], "usage": [...], "profile": {...}}
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone

import requests

N_BINS = 40
TOP_ANOMALOUS = 30
TOP_ERROR = 15
TOP_NEW = 15
TOP_COMPONENTS = 12
TOP_PER_SIGNAL = 2   # 每个信号类别额外补入的代表模板数
TOP_PARAMS = 12      # 数值参数变化表的行数
MAX_TEMPLATE_CHARS = 180
MAX_EXAMPLE_CHARS = 260
CATEGORIES = ("cpu", "memory", "disk_io", "disk_space", "network", "process",
              "syscall", "connection", "db_internal", "code")

# 通用运维词汇（不针对任何数据集的标签），只做计数交给 LLM，不做规则判定
SIGNAL_WORDS = {
    "cpu": r"\bcpu\b|throttl|load average|high load|runnable|scheduler latency",
    "memory": r"out of memory|\boom\b|memory|heap|\bgc\b|garbage collect|malloc|\bswap|\brss\b",
    "disk_io": r"\bdisk\b|i/o|\bio\b|fsync|sync took|slow disk|write stall|iops|flush|\bwal\b",
    "disk_space": r"no space|disk full|enospc|quota|space left|capacity|disk usage|disk space|free space|"
                  r"space (available|remaining|exhausted)|filesystem .*full|out of disk",
    "network": r"timeout|timed out|connection reset|unreachable|network|packet|retransmi|\brpc\b|"
               r"heartbeat|deadline exceeded|unavailable|\bdial\b|broken pipe|clock offset|latency",
    "process": r"\bkill|signal|sigterm|sigkill|terminat|exited|restart|shutdown|stopped|crash|"
               r"panic|core dump|\bdied\b|not running|process .*(start|stop)",
    "syscall": r"syscall|errno|\beio\b|eintr|operation not permitted|permission denied|strace",
    "connection": r"too many (connections|clients|sessions)|connection pool|max_connections|"
                  r"session limit|pool exhaust|connection refused|accept",
    "db_internal": r"\block|deadlock|retry|restart transaction|contention|checkpoint|schema|"
                   r"compaction|\btxn\b|transaction|rollback|abort",
    "code": r"exception|traceback|stack trace|nullpointer|nil pointer|assert|invalid argument|"
            r"unexpected|failed to (parse|convert|decode)|undefined|not implemented",
}
_SIGNAL_RE = {k: re.compile(v, re.I) for k, v in SIGNAL_WORDS.items()}
_SEVERE = {"ERROR", "ERR", "FATAL", "CRITICAL", "CRIT", "PANIC", "SEVERE", "EMERG", "ALERT"}
_WARN = {"WARN", "WARNING"}
_LEVEL_IN_TEXT = re.compile(r"\b(FATAL|PANIC|CRITICAL|ERROR|ERR|WARNING|WARN)\b")
_MASK = [(re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}"), "<UUID>"),
         (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
         (re.compile(r"\d+(\.\d+)?"), "<N>")]


# ---------------------------------------------------------------------------
# 解析与画像（不调 LLM）
# ---------------------------------------------------------------------------
def _parse_ts(s):
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00")
    s = re.sub(r"(\.\d{6})\d+", r"\1", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _mask(text):
    for pat, rep in _MASK:
        text = pat.sub(rep, text)
    return text


_PARTIAL_VAR = re.compile(r"(?:\d{1,4}|<\*>)(?:[-:](?:\d{2}|<\*>))+[\d<>*.,TZ+-]*"   # 正文里再嵌一个时间/日期
                          r"|[^\s'\",\[\]{}()=:]*(?:<\*>[^\s'\",\[\]{}()=:]*|[0-9a-fA-F]{12,}[^\s'\",\[\]{}()=:]*)"
                          r"|\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z\d]{8,}\b"   # 字母数字混排的长 ID
                          r"|(?<=\[)[^\[\]\s]*-\d+(?=\])")   # 方括号里带序号的线程名（如 [p-nio-80-exec-3]）


def _template_key(entry):
    """Drain3 模板；模板退化成纯通配（如整行 JSON）时退回数字掩码后的正文前缀。
    含通配的词（如 "65ac2a<*>"）和长十六进制 ID 整体折成 <*>：Drain3 会按 ID 前缀把同一语句裂成几十个模板。"""
    t = entry.template.template_str if entry.template else ""
    if len(re.sub(r"<\*>|\W", "", t)) < 4:
        t = _mask(entry.message or entry.raw_text)[:120]
    return re.sub(r"(?:<\*>\s*){2,}", "<*> ", _PARTIAL_VAR.sub("<*>", t)).strip()


def _level(entry):
    lv = (entry.level or "").upper()
    if not lv:
        m = _LEVEL_IN_TEXT.search((entry.message or "")[:200])
        lv = m.group(1) if m else ""
    if lv in _SEVERE:
        return "ERROR"
    if lv in _WARN:
        return "WARN"
    return "INFO"


# 结构化 v2（默认）：直接 Drain3 + 通用掩码 + 最终模板后处理，不走 LILAC 缓存/静态快捷层。
# 规则全部通用（日期、主机名、key=value、枚举词），不含任何数据集专用正则；Loghub-2.0 上的对比见
# 宿主项目 benchmark/eval_loghub_profrca.py。FAST_RCA_PARSER=v1 退回旧的 LILAC Drain3 链路。
PARSER = os.environ.get("FAST_RCA_PARSER", "v2")
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
_DAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[a-z]*"
_V2_MASKS = [
    # JSON 字段拆成单独的词，Drain3 才能按字段对齐（整段 JSON 是一个词时只要任一字段不同就整体变成 <*>）
    (re.compile(r'([,{])(?=")'), r"\1 "),
    # 正文里的日期时间（行首时间戳已由预处理剥离）
    (re.compile(rf"\b(?:{_DAY},?\s+)?{_MON}\s+\d{{1,2}},?\s+(?:\d{{4}}\s+)?\d{{1,2}}:\d{{2}}(?::\d{{2}})?(?:[.,]\d+)?"
                rf"(?:\s+[A-Z]{{3,4}})?(?:\s+\d{{4}})?\b"), "<*>"),
    (re.compile(rf"\b\d{{1,2}}/{_MON}/\d{{4}}(?::\d{{2}}){{3}}(?:\s+[+-]\d{{4}})?"), "<*>"),
    (re.compile(r"\b\d{4}[-/.]\d{2}[-/.]\d{2}(?:[T\s-]+\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b"), "<*>"),
    (re.compile(r"(?<![\w:.])\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?(?![\w:])"), "<*>"),
    # 主机名（带端口 / 常见顶级域）与 IP 掩码后残留的端口
    (re.compile(r"(?<![\w.-])(?:[A-Za-z0-9-]+\.)+[A-Za-z][A-Za-z0-9-]*:\d{2,5}\b"), "<*>:<*>"),
    (re.compile(r"(?<![\w.-])(?:[A-Za-z0-9-]+\.)+(?:com|net|org|edu|gov|io|cn|hk|uk|de|jp|local|internal)\b(?![\w-])"), "<*>"),
    (re.compile(r"(?<=<\*>):\d{2,5}\b"), ":<*>"),
]
_JSON_PARTIAL = re.compile(r'("[\w.$-]{1,40}"\s*:\s*)"[^"]*<\*>[^"]*"')
_KV_PREFIX = re.compile(r"^([^\s=]{1,40}[=:])\S")
_UNIT = re.compile(r"^(?:[KMGTPE]i?B|[KMGTPE]|B|bytes?|ms|us|ns|s|sec|secs|min|mins)[,;.]?$", re.I)
ENUM_K = 4   # 变量位上取值至多 ENUM_K 种、且都是不含数字的词（单位除外）时视为枚举常量，如状态名、事件类型


_WRAPS = {"[]", "()", "{}", "<>", '""', "''"}
MAX_TOKENS = int(os.environ.get("FAST_RCA_MAX_TOKENS", "40"))


def _is_enum_value(v):
    return ("<*>" not in v and not re.search(r"\d", v) and len(v) <= 80
            and not _UNIT.match(v) and re.search(r"[A-Za-z]{2}", v) is not None)


def _v2_preprocessor():
    from .preprocessor import TIMESTAMP_PATTERNS, LogPreprocessor

    class _Prep(LogPreprocessor):
        def _find_timestamp(self, line):
            # 推断行头时，时间戳前至多 4 个词；否则是正文里的日期（如 "connection from x at Fri Jun 17 ..."）
            best, kind = None, ""
            for k, pat in TIMESTAMP_PATTERNS:
                m = pat.search(line[:160])
                if m and len(line[:m.start()].split()) <= 4 and (best is None or m.start() < best.start()):
                    best, kind = m, k
            return best, kind

        def _apply_masks(self, text):
            for pat, rep in _V2_MASKS:
                text = pat.sub(rep, text)
            # 已含变量的 JSON 字符串值整体视为变量：值里常嵌着其他字段（如 "email":"<用户名>@<*>"），
            # 逐词比对时同一语句会裂成上千个单例模板；不含变量的值（如 "msg":"Connection ended"）保留
            return _JSON_PARTIAL.sub(r'\1"<*>"', super()._apply_masks(text))

    return _Prep()


def _parse_log_v2(path):
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig
    from .entries import GenericLogEntry, LogTemplate

    prep = _v2_preprocessor()
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read().splitlines()
    rows, i = [], 0   # (行号, 行, 预处理结果)；只有行头的行与下一行拼接（同 LILAC _assemble_multiline）
    while i < len(raw):
        line = raw[i]
        if not line.strip():
            i += 1
            continue
        p = prep.preprocess(line)
        if (p.header_format not in ("unknown", "json_log") and not p.body
                and i + 1 < len(raw) and raw[i + 1].strip()):
            line = line + " " + raw[i + 1].strip()
            rows.append((i + 1, line, prep.preprocess(line)))
            i += 2
        else:
            rows.append((i + 1, line, p))
            i += 1

    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_sim_th = 0.5
    miner = TemplateMiner(config=cfg)
    cid, cut = {}, set()   # masked body -> cluster id；相同正文重复送入 Drain3 不改变聚类，只送一次
    for _, line, p in rows:
        m = p.masked_body or p.body or line
        if m not in cid:
            # Drain3 按词数分组：变长列表（如整段 Response 数据）每种长度都会自成模板，聚类只看前 MAX_TOKENS 个词
            tk = m.split()
            cid[m] = miner.add_log_message(" ".join(tk[:MAX_TOKENS]))["cluster_id"]
            if len(tk) > MAX_TOKENS:
                cut.add(cid[m])
    tpl = {c.cluster_id: c.get_template().replace("<:*:>", "<*>").split() + (["<*>"] if c.cluster_id in cut else [])
           for c in miner.drain.clusters}
    # 每个变量位上出现过的原词：所有成员共享 "key=" 前缀则保留前缀，共享外层括号/引号则保留括号；
    # 取值是少数几个纯字母词则视为枚举常量
    vals = defaultdict(set)
    for m, c in cid.items():
        tk = m.split()
        for j, w in enumerate(tpl[c]):
            if w == "<*>" and j < min(len(tk), MAX_TOKENS):   # 截断补的末尾 <*> 代表剩余整段，不做后处理
                vals[(c, j)].add(tk[j])
    enum = defaultdict(list)
    for (c, j), s in vals.items():
        pre = {(_KV_PREFIX.match(v) or [None, None])[1] for v in s}
        wrap = {(v[0], v[-1]) if len(v) > 2 and v[0] + v[-1] in _WRAPS else None for v in s}
        if len(pre) == 1 and None not in pre:
            tpl[c][j] = pre.pop() + "<*>"
        elif len(s) <= ENUM_K and all(_is_enum_value(v) for v in s):
            enum[c].append(j)
        elif len(wrap) == 1 and None not in wrap:
            a, b = wrap.pop()
            tpl[c][j] = a + "<*>" + b
    cache, entries = {}, []
    for ln, line, p in rows:
        m = p.masked_body or p.body or line
        c = cid[m]
        t = list(tpl[c])
        tk = m.split()
        for j in enum.get(c, ()):
            t[j] = tk[j]
        s = " ".join(t)
        if s not in cache:
            cache[s] = LogTemplate.from_template_str(s, source="drain3")
        f = p.header_fields
        entries.append(GenericLogEntry(timestamp=f.get("timestamp", ""), level=f.get("level", ""),
                                       message=p.body or line, template=cache[s], source_file=path,
                                       line_number=ln, raw_text=line, metadata=f))
    return entries


def parse_log(path):
    if PARSER == "v2":
        return _parse_log_v2(path)
    # v1 是宿主项目（sundb-ai-ops）的旧 LILAC 链路，只为复现历史结果保留；独立使用时只有 v2
    try:
        from server.diagnose.lilac.config import LilacConfig
        from server.diagnose.lilac.models import LogTemplate
        from server.diagnose.lilac.parser import LilacParser
    except ImportError as e:
        raise ImportError("FAST_RCA_PARSER=v1 依赖宿主项目的 server.diagnose.lilac；独立使用请用默认 v2") from e
    # 每次用独立的内存模板缓存：不读写共享的 data/lilac_cache.db（并发写会 disk I/O error，
    # 且其中混有历史 LLM 模板），保证同一份日志每次解析结果一致、纯 Drain3
    parser = LilacParser(LilacConfig(cache_db_path=":memory:", enable_llm=False))
    entries = parser.parse_file(path, parse_mode="drain3").entries
    # Drain3 的模板随后续日志不断泛化，但每行拿到的是解析当时的模板快照：同一语句早期行是具体模板、
    # 后期行是泛化模板，会被拆成「只在前段 / 只在后段」的两个模板，伪造出变点前后的速率变化。
    # 解析完后把每个快照模板重新匹配到最终聚类，取最终模板。
    miner = parser._drain3._miner if parser._drain3 and parser._drain3.available else None
    final = {}
    for e in entries:
        t = e.template
        if miner is None or t is None or t.source != "drain3":
            continue
        s = t.template_str
        if s not in final:
            c = miner.match(parser._preprocessor.preprocess(e.raw_text).masked_body, full_search_strategy="fallback")
            final[s] = t if c is None else parser._normalize_template(
                LogTemplate.from_template_str(c.get_template().replace("<:*:>", "<*>"), source="drain3"))
        e.template = final[s]
    return entries


def _poisson_llr(n1, n2, f1):
    """两段计数 n1/n2、前段时长占比 f1 下，「速率变化」相对「速率不变」的对数似然比。"""
    n = n1 + n2
    if n == 0:
        return 0.0
    e1, e2 = n * f1, n * (1 - f1)
    return sum(k * math.log(k / e) for k, e in ((n1, e1), (n2, e2)) if k > 0)


_DUR_SCALE = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0}
_TS_PREFIX = re.compile(r"^\S*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s*")
_PROG = re.compile(r"^([A-Za-z][\w.-]*)\[\d+\]:?$")


def _component(raw):
    """行首时间戳后的第一个词（主机/服务名）；syslog 再拼上程序名。纯启发式，取不到返回 ''。"""
    rest = _TS_PREFIX.sub("", raw or "", count=1).split()
    if not rest:
        return ""
    comp = re.sub(r"[\[\]:,]", "", rest[0])[:40]
    if len(rest) > 1:
        m = _PROG.match(rest[1])
        if m:
            comp += "/" + m.group(1)
    return comp if re.search(r"[A-Za-z]", comp) else ""


# 正文里的数值参数（资源用量、连接数、队列长度、耗时等）：按「模板 + 第几个数」对齐，比较变点前后中位数。
# 内嵌的日期/时间先抹掉；带时间单位的数换算成秒，免得同一位置 ms/s 混用
_EMBED_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]?|\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?|\d{6} ")
_NUM = re.compile(r"(?<![\w.#:/-])(\d+(?:\.\d+)?)(ns|µs|us|ms|s|m)?(?![\w.:-])")


def _numbers(text):
    """返回（数值, 上下文片段）列表；上下文取数前 24 字、数后 16 字，供 LLM 知道这是什么量。
    不带单位的 5 位以上整数多是 ID/序号/时间戳，跳过；返回的是否带时间单位用于后面过滤微小变化。"""
    text = _EMBED_TIME.sub(" ", text or "")
    out = []
    for m in _NUM.finditer(text):
        if not m.group(2) and "." not in m.group(1) and len(m.group(1)) >= 5:
            continue
        v = float(m.group(1)) * (_DUR_SCALE[m.group(2)] if m.group(2) else 1.0)
        out.append((v, bool(m.group(2)), text[max(0, m.start() - 24):m.end() + 16].strip()))
    return out


def _monotonic(vals):
    """计数器/序号/运行时长这类单调递增的数，前后中位数必然不同，但不是异常信号。"""
    ups = sum(b >= a for a, b in zip(vals, vals[1:]))
    return len(vals) > 3 and ups >= 0.9 * (len(vals) - 1)


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


# 异常栈帧行（Java "at x.y(...)"、"... N more"、Python 'File "...", line N'）：每一帧都会裂成独立模板、
# 计数与异常头相同，会挤占排序前列；异常头（Exception/Caused by）保留，帧行和空正文行不参与画像
_FRAME = re.compile(r'^\s*(at [\w$.<>/]+[.(]|\.\.\. \d+ (more|common frames omitted)|File ".*", line \d+)')


def _is_filler(raw):
    rest = _TS_PREFIX.sub("", raw or "", count=1).split(None, 1)
    body = rest[1] if len(rest) > 1 else ""
    return not body.strip() or bool(_FRAME.match(body))


def build_profile(entries):
    rows, last_ts = [], None
    for e in entries:
        ts = _parse_ts(e.timestamp) or last_ts   # 多行日志的续行继承上一行时间
        if ts is None:
            continue
        last_ts = ts
        if _is_filler(e.raw_text):
            continue
        rows.append((ts, _template_key(e), _level(e), e))
    if not rows:
        return {"error": "no timestamps"}
    t0 = min(r[0] for r in rows)
    t1 = max(r[0] for r in rows)
    span = max(t1 - t0, 1e-6)
    nb = N_BINS

    counts = defaultdict(lambda: [0] * nb)
    comp_counts = defaultdict(lambda: [0] * nb)
    info = {}
    for ts, key, lv, e in rows:
        b = min(int((ts - t0) / span * nb), nb - 1)
        counts[key][b] += 1
        comp = _component(e.raw_text)
        if comp:
            comp_counts[comp][b] += 1
        d = info.get(key)
        if d is None:
            info[key] = d = {"level": lv, "first": ts, "last": ts, "example": e.raw_text, "comps": Counter()}
        d["last"] = ts
        d["comps"][comp] += 1
        if lv == "ERROR" or (lv == "WARN" and d["level"] == "INFO"):
            d["level"] = lv

    # 变点：让所有模板「前后速率变化」似然比之和最大的切分点。每个模板的似然比取 log1p，
    # 避免一两个高频模板（如按请求打的业务日志）独自决定变点；只出现在窗口开头两桶的模板（启动/窗口截断）不参与
    lo, hi = max(2, nb // 10), nb - max(2, nb // 10)
    cp_counts = [c for c in counts.values() if sum(c[2:])]
    best_k, best_s = lo, -1.0
    for k in range(lo, hi):
        f1 = k / nb
        # 速率不变时 LLR 期望约 0.5，先扣掉 1 的噪声底，免得几百个低频模板的随机波动合起来决定变点
        s = sum(math.log1p(max(0.0, _poisson_llr(sum(c[:k]), sum(c[k:]), f1) - 1)) for c in cp_counts)
        if s > best_s:
            best_k, best_s = k, s
    f1 = best_k / nb
    onset = t0 + span * f1

    # 变点后的示例（比首次出现更能反映故障期的参数值）与数值参数的前后序列
    series = defaultdict(lambda: ([], [], []))   # (key, 第几个数, 数的个数) → (前, 后, 全序列)
    ctx = {}
    for ts, key, lv, e in rows:
        after = ts >= onset
        if after and "example_after" not in info[key]:
            info[key]["example_after"] = e.raw_text
        nums = _numbers(e.message)
        for i, (v, unit, snip) in enumerate(nums):
            s = series[(key, i, len(nums), unit)]
            s[1 if after else 0].append(v)
            s[2].append(v)
            if after:
                ctx.setdefault((key, i, len(nums), unit), snip)

    params = []
    for sk, (vb, va, allv) in series.items():
        if len(vb) < 3 or len(va) < 3 or _monotonic(allv):
            continue
        mb, ma = _median(vb), _median(va)
        # 太小的绝对变化（0.1% → 0%、亚毫秒耗时）没有诊断意义
        if abs(ma - mb) < (1e-3 if sk[3] else 1):
            continue
        r = (ma + 1e-9) / (mb + 1e-9)
        if not 0.67 < r < 1.5:
            params.append({"key": sk[0], "ctx": ctx.get(sk, ""), "before": mb, "after": ma, "ratio": r,
                           "n": (len(vb), len(va))})

    tmpl = []
    for key, c in counts.items():
        n1, n2 = sum(c[:best_k]), sum(c[best_k:])
        r1, r2 = n1 / f1, n2 / (1 - f1)
        d = info[key]
        t = {"key": key, "n": n1 + n2, "before": n1, "after": n2,
             "ratio": round((r2 + 0.5) / (r1 + 0.5), 2),
             "llr": round(_poisson_llr(n1, n2, f1), 1), "bins": c,
             "level": d["level"], "first": d["first"], "last": d["last"],
             "example": d.get("example_after") or d["example"],
             "comps": [k for k, _ in d["comps"].most_common(3) if k]}
        tmpl.append(t)

    comps = []
    for comp, c in comp_counts.items():
        n1, n2 = sum(c[:best_k]), sum(c[best_k:])
        comps.append({"comp": comp, "before": n1, "after": n2,
                      "ratio": round((n2 / (1 - f1) + 0.5) / (n1 / f1 + 0.5), 2),
                      "llr": round(_poisson_llr(n1, n2, f1), 1), "bins": c})

    sig = {}
    for cat, pat in _SIGNAL_RE.items():
        hit = [t for t in tmpl if pat.search(t["key"]) or pat.search(t["example"][:300])]
        sig[cat] = {"templates": len(hit), "before": sum(t["before"] for t in hit),
                    "after": sum(t["after"] for t in hit)}

    levels = Counter(r[2] for r in rows)
    return {"lines": len(entries), "timed_lines": len(rows), "templates": len(tmpl),
            "t0": t0, "t1": t1, "onset": onset, "onset_bin": best_k, "bin_seconds": span / nb,
            "levels": dict(levels), "volume": [sum(c[b] for c in counts.values()) for b in range(nb)],
            "templates_stats": tmpl, "signals": sig, "components": comps, "params": params}


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spark(bins):
    """把 N_BINS 个计数压成 20 格字符曲线（0-9，. 表示 0）。"""
    step = max(1, len(bins) // 20)
    agg = [sum(bins[i:i + step]) for i in range(0, len(bins), step)]
    m = max(agg) or 1
    return "".join("." if v == 0 else str(min(9, int(9 * v / m) or 1)) for v in agg)


def render_profile(p):
    """画像 → 给 LLM 的紧凑文本；每个模板带编号 T#，LLM 用编号引用证据。"""
    ts = p["templates_stats"]
    chosen, seen = [], set()

    def take(items, tag):
        for t in items:
            if t["key"] not in seen:
                seen.add(t["key"])
                chosen.append((tag, t))

    take(sorted(ts, key=lambda t: -t["llr"])[:TOP_ANOMALOUS], "变化")
    take(sorted([t for t in ts if t["level"] == "ERROR"], key=lambda t: -t["n"])[:TOP_ERROR], "错误")
    take(sorted([t for t in ts if t["before"] == 0], key=lambda t: -t["after"])[:TOP_NEW], "新增")
    take(sorted([t for t in ts if t["level"] == "WARN"], key=lambda t: -t["llr"])[:TOP_ERROR], "告警")
    # 每个信号类别补入窗内出现、前面没选上的代表模板：只出现一两次的关键提示（如磁盘空间不足）不会因计数小被漏掉
    for cat, pat in _SIGNAL_RE.items():
        hit = [t for t in ts if t["after"] and t["key"] not in seen and (pat.search(t["key"]) or pat.search(t["example"][:300]))]
        take(sorted(hit, key=lambda t: (-t["llr"], -t["after"]))[:TOP_PER_SIGNAL], f"信号:{cat}")

    lines = [f"日志 {p['lines']} 行，{p['templates']} 个模板；时间 {_iso(p['t0'])} ~ {_iso(p['t1'])}，"
             f"均分 {N_BINS} 个时间桶（每桶 {p['bin_seconds']:.1f}s）",
             f"级别分布：{p['levels']}",
             f"统计变点（疑似异常起点）：{_iso(p['onset'])}（第 {p['onset_bin']} 桶），"
             "以下 before/after 指该变点前/后的行数，ratio 为后/前速率比",
             f"总日志量曲线：{_spark(p['volume'])}", "",
             "## 十类信号词命中（模板数 / 变点前行数 / 变点后行数）"]
    for cat, s in p["signals"].items():
        if s["templates"]:
            lines.append(f"- {cat}: {s['templates']} / {s['before']} / {s['after']}")
    comps = sorted(p.get("components") or [], key=lambda c: -c["llr"])[:TOP_COMPONENTS]
    if comps:
        lines += ["", "## 组件（主机/服务）日志量变化，按变化显著性排序（before / after / ratio / 曲线）"]
        for c in comps:
            lines.append(f"- {c['comp']}: {c['before']} / {c['after']} / {c['ratio']} / {_spark(c['bins'])}")
    # 同一模板只列变化最大的 2 个数值，免得一条多参数日志（如运行时统计）占满
    per, prm, dup = Counter(), [], set()
    for q in sorted(p.get("params") or [], key=lambda q: -abs(math.log(q["ratio"]))):
        sig_q = (q["key"], round(q["before"], 6), round(q["after"], 6))
        if per[q["key"]] < 2 and sig_q not in dup:
            per[q["key"]] += 1
            dup.add(sig_q)
            prm.append(q)
    if prm:
        lines += ["", "## 数值参数变化（同一模板同一位置的数，变点前 → 后中位数；时间单位已换算成秒）"]
        for q in prm[:TOP_PARAMS]:
            lines.append(f"- {q['before']:.4g} → {q['after']:.4g}（×{min(99, q['ratio']):.2g}）「{q['ctx']}」  {q['key'][:100]}")
    lines += ["", "## 模板（按入选原因分组；曲线为 20 格时间分布；示例优先取变点后）"]
    for i, (tag, t) in enumerate(chosen):
        ex = t["example"].strip()[:MAX_EXAMPLE_CHARS]
        cs = f" 组件={','.join(t['comps'])}" if t.get("comps") else ""
        lines.append(f"T{i} [{tag}|{t['level']}] n={t['n']} before={t['before']} after={t['after']} "
                     f"ratio={t['ratio']} 曲线={_spark(t['bins'])} 首次={_iso(t['first'])[11:19]} "
                     f"末次={_iso(t['last'])[11:19]}{cs}\n   模板: {t['key'][:MAX_TEMPLATE_CHARS]}\n   示例: {ex}")
    return "\n".join(lines), chosen


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
_SYSTEM_PREFIX = """你是资深数据库/分布式系统 SRE。用户在收到告警后上传了一段日志，这段时间内系统确实发生了故障（可能同时有 1~3 个根因）。
给你的是全量日志的统计画像：模板按「统计变点前后的速率变化」排序，另有错误级、新增模板和十类信号词计数。

判断原则：
- 变点前就一直存在、前后速率无明显变化的模板（如周期性后台任务、无关服务重启）是背景噪声，不是根因。
- 区分根因与症状：请求超时、副本不可用、会话断开等往往是下游症状，要追问是什么资源/机制导致的。
- 组件日志量变化表用于定位受影响的组件，但日志量下降本身不说明机制：CPU/内存/磁盘/网络压力都会让组件处理变慢、日志变少，
  上游随之报错；只有出现进程退出/信号/启动日志、或日志完全中断后重新启动时，才判为 process。
  机制要看该组件变点后的具体日志内容和耗时类数值（心跳、同步、RPC 耗时变大分别指向对应资源变慢）。
"""

_CLOSED_WORLD = """- 根因类别只能取：cpu | memory | disk_io | disk_space | network | process | syscall | connection | db_internal | code
  （process=进程/容器被杀、暂停、停止、重启；db_internal=锁争用/事务重试风暴/Schema 变更/检查点风暴等数据库内部负载；
   connection=连接池/会话/连接数耗尽；syscall=系统调用注入错误或延迟；code=应用代码缺陷）
"""

_SYSTEM_SUFFIX = """- mechanism 写具体机制（如"网络延迟""网络丢包""磁盘写 IO 压力""磁盘空间占满""CPU 满载""进程被 SIGKILL"），不要泛泛写"性能下降"。
- 资源类故障（CPU 满载、内存压力、网络延迟/丢包、磁盘 IO 或空间）常常没有直接的报错词，只表现为处理变慢、超时、心跳/租约异常；
  没有直接证据时也要把它们作为鉴别诊断候选考虑，不要只盯着日志里最显眼的业务/后台任务模板。
"""

_ORDER_EMPHASIZED = "- 输出 4 个候选根因，类别互不相同，按可能性从高到低排序（第 1 个最可信，后面是鉴别诊断的备选）。\n"
_ORDER_NONE = "- 输出 4 个候选根因，类别互不相同。\n"
_CANDIDATE_SUFFIX = "  备选也必须是可能造成本次异常的故障；你判断为正常行为、不是故障的现象不要列为候选。\n\n只输出 JSON：{\"root_causes\":[{\"category\":\"...\",\"mechanism\":\"...\",\"location\":\"组件/节点/服务\",\"evidence\":[\"T3\",\"T7\"],\"why\":\"不超过20字\"}]}"
# 候选个数消融：FAST_RCA_FREE_K=1 时不固定 4 个，由模型按证据自行决定个数。
FREE_K = os.environ.get("FAST_RCA_FREE_K", "0") == "1"
_METRICS_SYSTEM_PREFIX = """你是资深数据库/分布式系统 SRE。当前输入只有同一时段的监控指标，没有可引用的日志模板。
指标文本可能是带 M# 编号的变化事件，也可能是原始 CSV；所有判断必须来自下方指标文本，证据只能引用实际出现的 M# 行，不要编造证据。

判断原则：
- 区分根因与症状：延迟、错误率、流量下降等通常是结果，不要直接当成根因。
- 只根据指标表现概括可检验的资源或系统机制；原始 CSV 中没有 M# 行时不要填写指标 evidence。
"""
_METRICS_SYSTEM_SUFFIX = """- mechanism 写具体、可检验的指标支持机制，不要泛泛写“性能下降”。
- 资源类故障（CPU、内存、网络、磁盘 IO 或空间）可由指标水平变化支持；指标变化也可能只是正常波动。
"""
# review_v2：通用 DBA 知识的数据库机制鉴别（不含任何数据集标签/样本信息），只在 metrics-only 显式启用
_METRICS_DB_DIFFERENTIAL = """- 如果指标里有数据库引擎自身的计数（事务提交/回滚、增删改、行或块读取、锁、连接/会话、缓冲/缓存、配置参数等），
  先判断数据库层面的工作负载或引擎机制能否解释操作系统资源的变化：CPU、磁盘读写、内存、上下文切换的升高常常是数据库机制的结果，
  这时根因应写成具体的数据库机制，而不是只写资源类别。常见的数据库机制（仅供鉴别，必须有指标支持，也可以是清单之外的机制）：
  业务负载/查询量整体上升；高并发小事务频繁提交（提交率↑、日志刷盘写↑）；大批量或高并发插入（插入率↑、数据量增长）；
  缺少索引导致全表扫描（行/块读取↑而写入变化不大、CPU 或读 IO↑、长查询/长事务）；索引过多导致写放大（同样的增删改带来更多磁盘写、写入变慢）；
  清理/回收（vacuum、purge）或检查点刷脏；锁等待/死锁（锁数、阻塞进程、等待会话↑）；缓冲池/共享缓冲配置过小（命中率↓、物理读↑）；
  连接数耗尽；事务重试风暴；Schema 变更；磁盘 IO 饱和；数据库所在机器的资源争用。
- 常量配置参数也是证据：把缓冲/内存、连接上限、检查点/清理相关配置与机器内存和实际负载对比，明显不合理的配置可能就是根因。
"""
_METRICS_DB_REVIEW = ("- 检查第一轮是否把数据库机制造成的结果（CPU、IO、内存、连接数变化）当成了根因；"
                      "如果数据库计数或配置能解释这些变化，把首位候选改写为具体的数据库机制，并引用对应 M# 证据；\n")
# review_v3：把数据库机制鉴别限定在确有引擎内部计数的输入上（review_v2 在只有服务级 CPU/内存/延迟的微服务指标上会过度改判 db_internal）
_METRICS_DB_SCOPE = ("- 上一条只适用于指标里确实有数据库引擎内部计数（事务、锁、缓冲、行读写等）的情况；只有服务/容器级 CPU、内存、"
                     "延迟、错误率等指标时不适用：某个服务自身的资源指标率先发生水平突变，根因就是该服务上的这种资源故障，"
                     "不要凭空推测数据库机制。\n")
_METRICS_DB_REVIEW_SCOPE = ("- 没有数据库引擎内部计数时，不要把服务自身的 CPU/内存等资源突变改写为数据库机制；"
                            "首位保留率先突变的资源指标对应的服务与资源类别；\n")

# review_v4：提交/事务数上升几乎是所有数据库负载变化的共同表现，按“每事务/每行操作的代价”做鉴别排序，避免首位全部塌缩成提交风暴
_METRICS_DB_RANKING = """- 有数据库引擎内部计数时的排序方法：事务数/提交率、活动连接、CPU、写 IO 同时上升是绝大多数数据库负载变化的共同表现，
  单凭它们不足以把“高并发小事务频繁提交”排在首位。先比较“每个事务/每次增删改”的代价是否变化，再选最特异的机制：
  每事务读取行/块、CPU 或读 IO 明显变多而写入不多→缺少索引/全表扫描；同样的增删改带来不成比例的写 IO、写入变慢→索引过多/写放大；
  插入占增删改的比例明显上升、表或数据量持续增长→大批量/高并发插入；增删改、查询与提交按相近比例整体放大→业务负载整体上升；
  死元组/清理计数变化、长事务存在、后台写入而前台事务不多→清理/vacuum；锁数或等待会话上升而吞吐下降→锁等待；
  物理读上升、命中率下降、缓冲配置相对内存很小→缓冲过小；IO 等待高且读写吞吐触顶→磁盘 IO 饱和。
  只有每事务代价基本不变、事务本身很小且提交率是唯一突出的变化时，才把高并发小事务提交排在首位。
- 有数据库引擎内部计数时，备选候选也应优先写成其他可被指标区分的数据库机制（按上面的鉴别排序），不要用 CPU、内存、连接数等症状占位。
"""
_METRICS_DB_RANKING_REVIEW = ("- 逐一核对首位机制：说明它的哪个特异指标信号排除了其他数据库机制；如果首位只是“提交率/事务数上升”"
                              "而每事务代价（读行、写 IO、CPU、插入占比）也有明显变化，按该变化改选更特异的机制；\n")

# review_v5：v3 + 负载归一化证据（metric_events mechanism_v3 的 workload_reference / load_rank 行）的读法。
# 只在指标文本里确有这些行时加入；v8 中 LLM 把 per_load≈1 的随负载同比变化改写成“负载整体升降”或配置问题。
_METRICS_LOAD_GUIDE = """- 指标里有 workload_reference / load_rank 行（负载归一化证据）时：per_load≈1 的指标只是随工作量同比变化，单独不支持任何特定机制；
  per_load 远离 1（每单位工作量的读行、写 IO、CPU、数据量增长成倍变化）才是“每单位代价变了”的特异证据，优先据此选机制。
  如果工作量参考指标本身显著变化而其余指标大多 per_load≈1，根因是负载变化本身：按占主导的操作写成具体的负载机制
  （提交/事务数激增而增删改未同比→高并发小事务提交；插入与数据量激增→大批量/高并发插入；增删改查与提交同比放大→业务负载整体上升），
  不要改写成与这些证据无关的配置或资源问题。
"""
_METRICS_LOAD_REVIEW = ("- 核对首位机制引用的证据：若只引用 per_load≈1 的随负载同比变化，按工作量参考指标与占主导的操作改写为具体负载机制；"
                        "若有 per_load 远离 1 的特异指标，首位应解释这条证据；\n")

# review_v6：v3 + 工作量回归残差证据（metric_events mechanism_v4 的 workload_reference / residual_rank 行）的读法。
_METRICS_RESID_GUIDE = """- 指标里有 workload_reference / residual_rank 行（工作量回归残差证据）时：每个指标先按变点前与工作量的关系拟合，
  expected_post 是按变点后工作量“应有”的值；residual_z 大（偏离工作量可解释部分）的指标才是特异证据，优先据此选机制，
  并结合 direction（above/below_expected）说明每单位工作量的读行、写 IO、CPU、锁、缓存、数据量等代价如何变化。
  explained_by_workload≈1 的变化只是工作量变化的结果，单独不支持特定机制。若工作量参考指标本身显著变化而残差很少或很小，
  根因是负载变化本身：按占主导的操作写成具体的负载机制，不要改写成与这些证据无关的配置或资源问题。
"""
_METRICS_RESID_REVIEW = ("- 核对首位机制引用的证据：若有 residual_z 大的指标，首位应解释它们（方向与幅度一致）；"
                         "若只引用 explained_by_workload≈1 的变化，按工作量参考指标与占主导的操作改写为具体负载机制；\n")
# review_v7：v3 + 复核轮“预测—核对”（Panda 式 verify）：每个候选机制先推出它成立时应出现的关键指标变化，
# 与已给 M# 核对，预测缺失或方向相反的候选降级；只改复核轮提示，不改证据和第一轮。
_METRICS_VERIFY_REVIEW = ("- 逐个候选做预测核对：先推出该机制若成立应出现的 2~3 个关键指标变化（指标与方向），再与上文 M# 证据对照；"
                          "关键预测缺失或方向相反的候选降级或删除；首位应是预测与观测吻合最多、且其他候选不能同样解释这些观测的机制。"
                          "核对过程不输出，只输出调整后的候选；\n")

_LOAD_GUIDES = {"review_v5": (_METRICS_LOAD_GUIDE, _METRICS_LOAD_REVIEW),
                "review_v6": (_METRICS_RESID_GUIDE, _METRICS_RESID_REVIEW)}

_REVIEW_MODES = ("review_v1", "review_v2", "review_v3", "review_v4", "review_v5", "review_v6", "review_v7")


def _validate_modes(world_mode, ordering_mode, metrics_only_mode="baseline", candidate_diversity="distinct"):
    if world_mode not in ("closed", "open"):
        raise ValueError(f"invalid world_mode: {world_mode}")
    if ordering_mode not in ("emphasized", "none"):
        raise ValueError(f"invalid ordering_mode: {ordering_mode}")
    if metrics_only_mode not in ("baseline",) + _REVIEW_MODES:
        raise ValueError(f"invalid metrics_only_mode: {metrics_only_mode}")
    if candidate_diversity not in ("distinct", "auto"):
        raise ValueError(f"invalid candidate_diversity: {candidate_diversity}")


def _build_system(world_mode="closed", ordering_mode="emphasized", label_space=None):
    _validate_modes(world_mode, ordering_mode)
    world = _CLOSED_WORLD if world_mode == "closed" else "- 根因类别和细粒度机制由你根据日志与指标证据自由概括，不限于预先给定的类别名称；不要把正常现象列为根因。\n"
    if FREE_K:
        order = ("- 候选根因的个数由你根据证据自行决定，不必凑数；类别互不相同，"
                 "按可能性从高到低排序（排在前面的可能性更大，第 1 个最可信）。\n"
                 if ordering_mode == "emphasized" else
                 "- 候选根因的个数由你根据证据自行决定，不必凑数；类别互不相同。\n")
    else:
        order = _ORDER_EMPHASIZED if ordering_mode == "emphasized" else _ORDER_NONE
    if label_space:
        return _label_system(_SYSTEM_PREFIX + label_space["prompt"] + _SYSTEM_SUFFIX + order + _CANDIDATE_SUFFIX)
    return _SYSTEM_PREFIX + world + _SYSTEM_SUFFIX + order + _CANDIDATE_SUFFIX


# 封闭机制集（label_space={"prompt": 机制清单文本, "category": {mechanism_id: 一级类别}}）：
# 候选直接给 mechanism_id，同一类别可以重复，ID 互不相同；其余判断原则与类别模式相同。
_LABEL_MECHANISM_HINT = '- mechanism 写具体机制（如"网络延迟""网络丢包""磁盘写 IO 压力""磁盘空间占满""CPU 满载""进程被 SIGKILL"），不要泛泛写"性能下降"。\n'
_METRICS_MECHANISM_HINT = "- mechanism 写具体、可检验的指标支持机制，不要泛泛写“性能下降”。\n"


def _label_system(text):
    return (text.replace(_LABEL_MECHANISM_HINT, "").replace(_METRICS_MECHANISM_HINT, "")
            .replace("类别互不相同", "mechanism_id 互不相同")
            .replace('{"category":"...","mechanism":"...",', '{"mechanism_id":"...",'))


def _label_text(text, label_space):
    """复核轮、初筛后缀等用户侧提示在封闭机制集下的改写。"""
    if not label_space:
        return text
    return (text.replace("个类别互不相同", "个 mechanism_id 互不相同").replace("类别互不相同", "mechanism_id 互不相同")
            .replace("category、mechanism、location 三个字段", "mechanism_id、location 两个字段"))


_AUTO_DIVERSITY = ("同一类别只有在具体机制不同、每个候选都有实际定义的 M# 证据，且后续机制引用至少一个"
                   "前面同类候选未引用的独立 M# 时才允许重复（合计至少两个不同的 M#）；不要改写同一机制来凑数；最多输出 4 个候选。")


def _build_metrics_system(world_mode="closed", ordering_mode="emphasized", candidate_diversity="distinct",
                          db_differential=False, db_scope=False, db_ranking=False, load_guide="", label_space=None):
    """Metrics-only prompt. The log prompt is deliberately not changed."""
    _validate_modes(world_mode, ordering_mode, candidate_diversity=candidate_diversity)
    base = _build_system("closed" if label_space else world_mode, ordering_mode)
    if label_space:   # 封闭机制集取代类别白名单（world_mode 不再起作用）
        base = base.replace(_CLOSED_WORLD, label_space["prompt"])
    base = base.replace(_SYSTEM_PREFIX, _METRICS_SYSTEM_PREFIX)
    base = base.replace(_SYSTEM_SUFFIX, _METRICS_SYSTEM_SUFFIX + (_METRICS_DB_DIFFERENTIAL if db_differential else "")
                        + (_METRICS_DB_SCOPE if db_scope else "") + (_METRICS_DB_RANKING if db_ranking else "")
                        + load_guide)
    base = base.replace('["T3","T7"]', '["M0","M1"]')
    if candidate_diversity == "auto":
        base = base.replace("类别互不相同", _AUTO_DIVERSITY)
    return _label_system(base) if label_space else base


_METRICS_REVIEW = """这是指标-only 的第二轮复核。只依据上文已经提供的指标文本和第一轮候选：
- 严格沿用系统提示中的候选数量、类别多样性和 evidence 约束，不因复核而改变候选政策；
- 保留有明确 M# 指标证据支持的机制，删除把延迟、错误率、流量等症状直接当成根因的候选；
- 必要时把候选改写为更具体的机制或重新排序；不要引入日志、故障标签、样本信息或未出现在指标中的事实。
按系统 JSON 格式只输出最终候选。"""


SYSTEM = _build_system()
THINKING = os.environ.get("FAST_RCA_THINKING", "0") == "1"


def _chat(messages, max_tokens=None, temperature=0.0, timeout=None):
    # 不用 response_format=json_object：实测 DeepSeek-V4-Flash 在该模式下偶发吐满 max_tokens 却返回空 content。
    # 思考模式的 reasoning token 计入 completion_tokens，默认关闭；GLM/Kimi 思考常超过 4000 token，上限放到 32000。DeepSeek 认 thinking，GLM 认 enable_thinking。
    # FAST_RCA_MODEL 只换诊断模型（换模型消融），判官仍用 DEEPSEEK_MODEL。
    base = os.environ.get("DEEPSEEK_API_BASE", "").rstrip("/")
    model = os.environ.get("FAST_RCA_MODEL") or os.environ.get("DEEPSEEK_MODEL", "").split(",")[0].strip()
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens or (32000 if THINKING else 800),
            "chat_template_kwargs": {"thinking": THINKING, "enable_thinking": THINKING}}
    r = requests.post(f"{base}/chat/completions", timeout=timeout or (3600 if THINKING else 300), proxies={"http": None, "https": None},
                      headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}, json=body)
    r.raise_for_status()
    js = r.json()
    return js["choices"][0]["message"]["content"] or "", js.get("usage") or {}


def _json_obj(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


_METRIC_ID_LINE = re.compile(r"(?m)^\s*(M\d+)\b")


def _metric_ids(text):
    """Return only M# identifiers defined at the start of an evidence line."""
    return set(_METRIC_ID_LINE.findall(text or ""))


def _to_predictions(obj, chosen, *, valid_metric_refs=None, candidate_diversity=None, label_space=None):
    """Convert model JSON, optionally applying the metrics-only evidence policy.

    ``candidate_diversity=None`` is intentional: it preserves the historical log
    path exactly.  Metrics-only callers pass the effective mode explicitly.
    With ``label_space`` each candidate carries a ``mechanism_id`` (unknown IDs are
    kept and marked category ``invalid``); repeated IDs are dropped.
    """
    roots = obj.get("root_causes") or []
    limit = 4 if candidate_diversity == "auto" else (10 if FREE_K else 4)
    out, seen_ids = [], set()
    for rc in roots:
        if len(out) >= limit:
            break
        mid = None
        if label_space:
            mid = str(rc.get("mechanism_id", "")).strip()
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)
        cat = label_space["category"].get(mid, "invalid") if label_space else str(rc.get("category", "")).strip()
        mech = mid if label_space else str(rc.get("mechanism", "")).strip()
        loc = str(rc.get("location", "")).strip()
        ev, refs = [], []
        for ref in rc.get("evidence") or []:
            ref = str(ref).strip()
            m = re.match(r"T(\d+)$", ref)
            if m and int(m.group(1)) < len(chosen):
                index = int(m.group(1))
                refs.append(f"T{index}")
                ev.append(chosen[index][1]["key"][:120])
            elif re.match(r"M\d+$", ref) and (valid_metric_refs is None or ref in valid_metric_refs):
                refs.append(ref)
                ev.append(ref)
        out.append({"type": cat, "headline": mech,
                    "description": f"{mech}（类别 {cat}，位置 {loc}）。{rc.get('why', '')}",
                    "location": loc, "evidence_templates": ev, "evidence_refs": refs,
                    **({"mechanism_id": mid} if label_space else {})})

    if candidate_diversity is None or label_space:   # 机制集模式只要求 mechanism_id 互不相同（上面已去重）
        return out
    accepted, categories, mechanisms = [], set(), defaultdict(set)
    for pred in out:
        category = pred["type"]
        mechanism = re.sub(r"\s+", " ", pred["headline"].casefold()).strip()
        if candidate_diversity == "distinct":
            if category in categories:
                continue
        elif category in categories:
            metric_refs = {r for r in pred["evidence_refs"] if r.startswith("M")}
            prior_refs = set().union(*(set(p["evidence_refs"]) for p in accepted
                                       if p["type"] == category))
            if (not prior_refs or len(prior_refs | metric_refs) < 2
                    or not (metric_refs - prior_refs) or mechanism in mechanisms[category]):
                continue
        categories.add(category)
        mechanisms[category].add(mechanism)
        accepted.append(pred)
    return accepted


ZOOM_COMPONENTS = 2   # 第二轮放大的组件数
ZOOM_TEMPLATES = 12   # 每个组件放大展示的模板数

_REVIEW_PREFIX = """下面是你第一轮给出的候选根因，以及第一轮候选所指组件的全部模板（按变化显著性排序，编号接着上文的 T#）。
请结合这些组件内的细节复核：保留有证据的候选、修正机制或位置、用更有证据的候选替换证据不足的候选。
"""
def _build_review(world_mode="closed", ordering_mode="emphasized", label_space=None):
    _validate_modes(world_mode, ordering_mode)
    count = "候选个数自行决定，类别互不相同的候选" if FREE_K else "4 个类别互不相同的候选"
    if ordering_mode == "emphasized":
        tail = f"按系统提示的完整格式输出（{count}，按可能性排序，每个带 why），只输出 JSON。"
    else:
        tail = f"按系统提示的完整格式输出（{count}，每个带 why），只输出 JSON。"
    return _label_text(_REVIEW_PREFIX + tail, label_space)


REVIEW = _build_review()

# 第一轮只用于决定放大哪些组件，省掉 why、少列证据以减少输出 token；完整说明留给复核轮
FIRST_ROUND = "\n\n（本轮为初筛：每个候选只输出 category、mechanism、location 三个字段，省略 evidence 和 why。）"


def _zoom(prof, preds, chosen):
    """第一轮候选的 location 里提到的组件 → 该组件内变化最显著、上文未展示的模板。"""
    comps = [c["comp"] for c in sorted(prof.get("components") or [], key=lambda c: -c["llr"])]
    picked = []
    for p in preds:
        loc = p["location"].lower()
        for c in comps:
            if c not in picked and c.split("/")[0].lower() in loc:
                picked.append(c)
        if len(picked) >= ZOOM_COMPONENTS:
            break
    picked = picked[:ZOOM_COMPONENTS]
    # 再补日志量变化最显著的组件（候选没指向它时，它的细节也可能改变判断）
    if comps and comps[0] not in picked:
        picked.append(comps[0])
    seen = {t["key"] for _, t in chosen}
    lines, extra = [], []
    for c in picked:
        # 组件内按 ID 前缀裂开的同一语句（如 "task: 65ac2b<*>"）只留变化最显著的一条
        ts, coarse = [], set()
        for t in sorted(prof["templates_stats"], key=lambda t: -t["llr"]):
            ck = re.sub(r"[\w<>*]*[<\d][\w<>*]*", "#", t["key"])[:80]   # 含数字/通配的词整体折叠；键可能被截断，只比前缀
            if c in t["comps"] and t["key"] not in seen and ck not in coarse:
                coarse.add(ck)
                ts.append(t)
        ts = ts[:ZOOM_TEMPLATES]
        if not ts:
            continue
        lines.append(f"## 组件 {c}")
        for t in ts:
            seen.add(t["key"])
            i = len(chosen) + len(extra)
            extra.append(("放大", t))
            lines.append(f"T{i} [{t['level']}] n={t['n']} before={t['before']} after={t['after']} ratio={t['ratio']} "
                         f"曲线={_spark(t['bins'])}\n   模板: {t['key'][:MAX_TEMPLATE_CHARS]}\n"
                         f"   示例: {t['example'].strip()[:MAX_EXAMPLE_CHARS]}")
    return "\n".join(lines), extra


def diagnose_log_file(path, metrics_text=None, *, world_mode="closed", ordering_mode="emphasized"):
    entries = parse_log(path)
    return diagnose_profile(build_profile(entries), metrics_text,
                            world_mode=world_mode, ordering_mode=ordering_mode)


# 有指标事件时附在画像后面；指标与日志时间戳同为 UTC，可按时间对照
METRICS_NOTE = ("\n\n以下是同一时段监控指标的显著变化事件（统计检测得到，编号 M#，evidence 里可以引用）。"
                "资源类故障（CPU、内存、磁盘 IO、网络）往往直接体现在对应指标上，请与日志一起判断根因的机制和位置；"
                "注意区分源头与被波及的下游（延迟、错误率、流量类指标多是症状），也有指标变化是正常波动。\n\n")
# 指标原始表足够短时直接给表（metric_events.metrics_text 的 raw 模式），不做事件抽象
METRICS_RAW_NOTE = ("\n\n以下是同一时段的监控指标原始数据（CSV，时间为 UTC，可与日志时间对照；evidence 里可引用 \"实体|指标\" 列名）。"
                    "资源类故障（CPU、内存、磁盘 IO、网络）往往直接体现在对应指标上，请先找出在某一时刻前后发生明显水平变化的列，"
                    "再与日志一起判断根因的机制和位置；注意区分源头与被波及的下游（延迟、错误率、流量类指标多是症状），"
                    "也有指标变化是正常波动。\n\n")
NO_LOG_NOTE = "（本次没有可用的日志，只有监控指标。请只依据指标判断。）"


def diagnose_profile(prof, metrics_text=None, metrics_raw=False, *, progress_cb=None,
                     cancel_check=None, include_details=False, request_timeout=None,
                     world_mode="closed", ordering_mode="emphasized",
                     metrics_only_mode="baseline", candidate_diversity="distinct", label_space=None):
    """Diagnose a statistical profile, or metrics alone when ``prof is None``.

    ``label_space`` (opt-in) replaces the category whitelist with a closed list of
    mechanism IDs; candidates then carry ``mechanism_id``.

    ``metrics_only_mode`` and ``candidate_diversity`` affect only the metrics-only
    path.  For a real log profile, the historical prompts, evidence, and call
    sequence remain unchanged.
    """
    _validate_modes(world_mode, ordering_mode, metrics_only_mode, candidate_diversity)
    metrics_only = prof is None
    enhanced_metrics = metrics_only and (metrics_only_mode in _REVIEW_MODES or candidate_diversity == "auto")
    db_differential = metrics_only_mode in ("review_v2", "review_v3", "review_v4", "review_v5", "review_v6", "review_v7")
    db_scope = metrics_only_mode in ("review_v3", "review_v4", "review_v5", "review_v6", "review_v7")
    verify_review = _METRICS_VERIFY_REVIEW if metrics_only_mode == "review_v7" else ""
    db_ranking = metrics_only_mode == "review_v4"
    # 只在指标文本里确有工作量参考行时加入对应读法
    load_guide, load_review = (_LOAD_GUIDES.get(metrics_only_mode, ("", ""))
                               if metrics_only and metrics_text and "workload_reference" in metrics_text else ("", ""))
    defined_metric_refs = _metric_ids(metrics_text) if enhanced_metrics else set()
    effective_diversity = ("auto" if candidate_diversity == "auto" and len(defined_metric_refs) >= 2
                           else "distinct")
    system_prompt = (_build_metrics_system(world_mode, ordering_mode, effective_diversity, db_differential, db_scope,
                                           db_ranking, load_guide, label_space)
                     if enhanced_metrics else _build_system(world_mode, ordering_mode, label_space))
    review_prompt = _build_review(world_mode, ordering_mode, label_space)
    if metrics_only:   # 只有指标：不做日志画像和组件放大复核
        if not metrics_text:
            return {"root_causes": [], "usage": [], "error": "no logs and no metrics",
                    "metrics_review": {"attempted": False, "completed": False, "error": None},
                    "candidate_diversity": effective_diversity}
        prof, text, chosen = None, NO_LOG_NOTE, []
    elif "error" in prof:
        return {"root_causes": [], "usage": [], "error": prof["error"]}
    else:
        text, chosen = render_profile(prof)
    if metrics_text:
        text += (METRICS_RAW_NOTE if metrics_raw else METRICS_NOTE) + metrics_text
    valid_metric_refs = _metric_ids(text) if enhanced_metrics else None
    first_suffix = "" if enhanced_metrics else _label_text(FIRST_ROUND, label_space)
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text + first_suffix}]
    usage, preds, err, content = [], [], None, ""
    metrics_review = {"attempted": False, "completed": False, "error": None}
    if progress_cb:
        progress_cb("first_round_started", {"candidate_count": 0})
    for _ in range(2):
        if cancel_check and cancel_check():
            result = {"root_causes": preds, "usage": usage, "error": "cancelled",
                      "profile": prof, "prompt_chars": len(text)}
            if metrics_only:
                result.update(metrics_review=metrics_review, candidate_diversity=effective_diversity)
            return result
        try:
            content, u = _chat(msgs, timeout=request_timeout)
            usage.append(u)
            preds = _to_predictions(
                _json_obj(content), chosen,
                valid_metric_refs=valid_metric_refs,
                candidate_diversity=effective_diversity if enhanced_metrics else None,
                label_space=label_space,
            )
            if preds:
                err = None  # a successful retry must not retain the first error
                if progress_cb:
                    progress_cb("first_round_completed", {"candidate_count": len(preds)})
                break
            err = "no predictions"
        except Exception as e:  # 解析失败重试一次
            err = repr(e)
    zoom, extra = _zoom(prof, preds, chosen) if preds and prof else ("", [])
    if metrics_only and metrics_only_mode in _REVIEW_MODES and preds:
        metrics_review["attempted"] = True
        if progress_cb:
            progress_cb("metrics_review_started", {"candidate_count": len(preds)})
        try:
            if cancel_check and cancel_check():
                metrics_review["error"] = "cancelled"
                result = {"root_causes": preds, "usage": usage, "error": "cancelled",
                          "profile": prof, "prompt_chars": len(text)}
                result.update(metrics_review=metrics_review, candidate_diversity=effective_diversity)
                return result
            # The first user message already contains all metric evidence.  Reuse
            # that conversation instead of appending a second copy of ``text``.
            db_review = (_METRICS_DB_REVIEW + (_METRICS_DB_RANKING_REVIEW if db_ranking else "")
                         + (_METRICS_DB_REVIEW_SCOPE if db_scope else "") + load_review + verify_review)
            review_text = (_METRICS_REVIEW.replace("- 必要时", db_review + "- 必要时")
                           if db_differential else _METRICS_REVIEW)
            content2, u = _chat(msgs + [{"role": "assistant", "content": content},
                                        {"role": "user", "content": review_text}],
                               timeout=request_timeout)
            usage.append(u)
            reviewed = _to_predictions(
                _json_obj(content2), chosen,
                valid_metric_refs=valid_metric_refs,
                candidate_diversity=effective_diversity,
                label_space=label_space,
            )
            if reviewed:
                preds = reviewed
                metrics_review["completed"] = True
                if progress_cb:
                    progress_cb("metrics_review_completed", {"candidate_count": len(preds)})
            else:
                metrics_review["error"] = "no predictions"
                if progress_cb:
                    progress_cb("metrics_review_failed", {"error": metrics_review["error"]})
        except Exception as e:  # 复核失败时保留第一轮结果
            metrics_review["error"] = repr(e)
            if progress_cb:
                progress_cb("metrics_review_failed", {"error": metrics_review["error"]})
    elif progress_cb and zoom:
        progress_cb("zoom_started", {"template_count": len(extra)})
    if zoom:
        try:
            if cancel_check and cancel_check():
                return {"root_causes": preds, "usage": usage, "error": "cancelled",
                        "profile": prof, "prompt_chars": len(text) + len(zoom)}
            content2, u = _chat(msgs + [{"role": "assistant", "content": content},
                                        {"role": "user", "content": review_prompt + "\n\n" + zoom}],
                               timeout=request_timeout)
            usage.append(u)
            preds = _to_predictions(_json_obj(content2), chosen + extra, label_space=label_space) or preds
            if progress_cb:
                progress_cb("review_completed", {"candidate_count": len(preds)})
        except Exception as e:  # 复核失败时保留第一轮结果
            err = repr(e)
            if progress_cb:
                progress_cb("review_failed", {"error": repr(e)})
    # Keep the complete statistical profile for production consumers.  The benchmark
    # callers can ignore the extra fields, while the UI needs template/component
    # evidence to explain the ranked candidates without inventing AEU spans.
    result = {"root_causes": preds, "usage": usage, "error": None if preds else err,
              "profile": prof if prof else None,
              "prompt_chars": len(text) + len(zoom)}
    if metrics_only:
        result.update(metrics_review=metrics_review, candidate_diversity=effective_diversity)
    if include_details:
        result["evidence_templates"] = [
            {"id": f"T{i}", **item[1]} for i, item in enumerate(chosen + extra)
        ]
    return result
