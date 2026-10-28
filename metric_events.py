"""通用指标事件管线：多维时序指标 → 少量「显著变化事件」文本，供 LLM 与日志时间线一起阅读。

不依赖领域知识（数据库、微服务均可），只假设输入是长表 (ts, entity, metric, value)，
或「time 列（epoch 秒）+ 每列一条序列」的宽表。不需要、也不能给故障注入时间。步骤：
  1. 重采样到统一时间网格（≤MAX_POINTS 个桶，桶内取中位数）
  2. 自动判型：常量丢弃；单调计数器差分成速率（计数下降记为「重置」事件）；
     ≤5 个取值的状态量只报少数几次切换；其余为量值；采集中断/停止上报本身也是事件
  3. 量值按差分噪声尺度归一化后跑 PELT（L2 代价、PEN_FACTOR·log n 惩罚）找水平突变，
     score = |后段中位数 − 前段中位数| / 前段稳健波动（有下限，含同名指标跨实体量级）；
     后段很短又回落的记为短时尖峰；同一序列反复跳变时按次数降分
  4. 同一实体在容差内同时发生的变化合并：保留最强一条，附「+N 同步」
  5. 排序分 = score ×（1 + 0.2·同步指标数），取 top-K（固定 token 预算）
  6. 渲染成带绝对时间戳（UTC）的文本行，编号 M#

    from profrca.metric_events import extract_events, render_events
    evs, stats = extract_events(wide_df)         # 列名 "<entity>_<metric>"，或传 split=
    text = render_events(evs, stats)
"""

import math
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MAX_POINTS = 300
MIN_SEG = 3            # PELT 最短段长（桶）
PEN_FACTOR = 3.0       # 惩罚 = PEN_FACTOR · log(n)，序列已按噪声尺度归一化
MIN_SCORE = 3.0
MAX_CP_PER_SERIES = 3
MAX_STATE_SWITCHES = 3  # 状态量切换超过这个次数视为正常抖动，不报
REL_FLOOR = 0.05       # 前段波动下限：前段中位数绝对值的 5%，免得平滑序列的微小漂移得高分
RANGE_FLOOR = 0.02     # 以及整条序列 p1~p99 跨度的 2%
SCORE_CAP = 99.0
XENT_FLOOR = 0.05      # 波动下限还不低于同名指标跨实体典型量级的 5%
RECUR_FREE = 2         # 一条序列超过这么多个显著变点时按比例降分
SYNC_BOOST = 0.2       # 每个同步变化的指标给排序分加 20%（最多 5 个）
TOP_K = 15

# 可选：按指标名词根打大类标签，只作阅读提示，不参与打分
KIND = [("cpu", r"cpu|load|throttl"), ("memory", r"mem|memory|rss|heap|swap|oom|gc"),
        ("disk", r"disk\w*|io|iops|fsync|wal|blk\w*"),
        ("network", r"net\w*|tcp|rtt|packets?|retrans\w*|sockets?|rx|tx|recv|send"),
        ("connection", r"conn\w*|sessions?|pool|clients?"),
        ("latency", r"latency|duration|elapsed|lat|p50|p90|p95|p99"),
        ("error", r"errors?|fail\w*|err|5xx|exceptions?"),
        ("db", r"lock\w*|txn|transactions?|retry|retries|deadlock|checkpoint|compaction")]
_KIND = [(k, re.compile(rf"\b({p})\b")) for k, p in KIND]


def metric_kind(name):
    s = re.sub(r"[^a-z0-9]+", " ", name.lower())
    for k, rx in _KIND:
        if rx.search(s):
            return k
    return ""


def split_col(col):
    """默认列名切分："<entity>_<metric>"，按最后一个下划线切。"""
    return tuple(col.rsplit("_", 1)) if "_" in col else ("", col)


def long_to_wide(df, ts="ts", entity="entity", metric="metric", value="value"):
    """长表 → 宽表（time 列 + "<entity>\\x1f<metric>" 列），配合 split=split_long 使用。"""
    d = df.assign(_c=df[entity].astype(str) + "\x1f" + df[metric].astype(str),
                  _v=pd.to_numeric(df[value], errors="coerce"))
    w = d.pivot_table(index=ts, columns="_c", values="_v", aggfunc="median")
    return w.reset_index().rename(columns={ts: "time"})


def split_long(col):
    return tuple(col.split("\x1f", 1))


def _resample(wide, time_col):
    t = pd.to_numeric(wide[time_col], errors="coerce").to_numpy(float)
    ok = ~np.isnan(t)
    wide, t = wide[ok], t[ok]
    u = np.unique(t)
    dt = float(np.median(np.diff(u))) if len(u) > 1 else 1.0
    t0 = float(u[0])
    step = max(dt, (float(u[-1]) - t0) / MAX_POINTS)
    b = np.floor((t - t0) / step).astype(int)
    vals = wide.drop(columns=[time_col]).apply(pd.to_numeric, errors="coerce")
    g = vals.groupby(b).median().reindex(range(int(b.max()) + 1))
    return g, t0, step


def _pelt(z, pen, min_size=MIN_SEG):
    """L2 代价（均值变化）的 PELT，返回变点位置（新段起点）列表。"""
    n = len(z)
    c1 = np.concatenate([[0.0], np.cumsum(z)])
    c2 = np.concatenate([[0.0], np.cumsum(z * z)])
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    last = np.zeros(n + 1, dtype=int)
    R = np.array([0])
    for t in range(min_size, n + 1):
        ok = R <= t - min_size
        s = R[ok]
        cost = (c2[t] - c2[s]) - (c1[t] - c1[s]) ** 2 / (t - s)
        v = F[s] + cost + pen
        i = int(np.argmin(v))
        F[t], last[t] = v[i], s[i]
        keep = np.ones(len(R), dtype=bool)
        keep[ok] = F[s] + cost <= F[t]
        R = np.append(R[keep], t)
    cps, t = [], n
    while t > 0:
        t = last[t]
        if t > 0:
            cps.append(t)
    return sorted(cps)


def _mad(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if len(x) else 0.0


def _fmt(v):
    a = abs(v)
    for unit, s in (("G", 1e9), ("M", 1e6), ("K", 1e4)):
        if a >= s:
            return f"{v / (s if unit != 'K' else 1e3):.3g}{unit if unit != 'K' else 'K'}"
    return f"{v:.3g}"


def _series_events(x, mag_floor=0.0, infer_counter=True):
    """单条序列 → (类型, [(桶号, score, 描述)])；类型 None 表示丢弃。
    mag_floor：同名指标在各实体上的典型量级 × XENT_FLOOR，免得某个实体上接近 0 的序列的微小抖动得高分。"""
    n = len(x)
    valid = ~np.isnan(x)
    idx = np.flatnonzero(valid)
    if len(idx) < max(10, 0.2 * n):
        return None, []
    ev = []
    lim = max(3, int(0.05 * n))
    for j in np.flatnonzero(np.diff(idx) > lim):
        ev.append((idx[j] + 1, 6.0, f"采集中断 {idx[j + 1] - idx[j] - 1} 个桶"))
    if n - 1 - idx[-1] >= lim:
        ev.append((idx[-1] + 1, 6.0, "停止上报"))
    if idx[0] >= lim:
        ev.append((idx[0], 4.0, "开始上报"))
    v = x[idx]
    if len(np.unique(v)) <= 1:
        return ("const" if not ev else "gap"), ev
    dv = np.diff(v)
    kind = "gauge"
    if infer_counter and (dv < 0).mean() <= 0.02 and (dv > 0).mean() >= 0.5:   # 单调计数器 → 速率；下降是重置
        kind = "counter"
        for j in np.flatnonzero(dv < 0):
            ev.append((idx[j + 1], 6.0, f"计数器重置 {_fmt(v[j])}→{_fmt(v[j + 1])}"))
        rate = dv / np.diff(idx)
        rate[dv < 0] = np.nan
        x = np.full(n, np.nan)
        x[idx[1:]] = rate
        idx = np.flatnonzero(~np.isnan(x))
        if len(idx) < 10:
            return kind, ev
        v = x[idx]
    elif len(np.unique(v)) <= 5:                               # 状态量：只报少数几次切换
        sw = np.flatnonzero(v[1:] != v[:-1])
        if 1 <= len(sw) <= MAX_STATE_SWITCHES:
            ev += [(idx[j + 1], 5.0, f"状态切换 {_fmt(v[j])}→{_fmt(v[j + 1])}") for j in sw]
        return "state", ev
    xi = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    noise = _mad(np.diff(xi)) / math.sqrt(2)
    lo, hi = np.percentile(xi, [1, 99])
    floor = max(RANGE_FLOOR * (hi - lo), mag_floor, 1e-12)
    sd = max(noise, _mad(xi) * 0.1, floor)
    cps = _pelt((xi - np.median(xi)) / sd, PEN_FACTOR * math.log(n))
    bounds = [0] + cps + [n]
    found = []
    for k, c in enumerate(cps):
        a, e = bounds[k], bounds[k + 2]
        left, right = xi[a:c], xi[c:e]
        ml, mr = float(np.median(left)), float(np.median(right))
        scale = max(_mad(left), noise, REL_FLOOR * abs(ml), floor)
        s = min(SCORE_CAP, abs(mr - ml) / scale)
        if s < MIN_SCORE:
            continue
        arrow = "↑" if mr > ml else "↓"
        nxt = xi[e:bounds[k + 3]] if k + 3 < len(bounds) else xi[e:e]
        if e - c <= 2 * MIN_SEG and len(nxt) and abs(float(np.median(nxt)) - ml) < 0.3 * abs(mr - ml):
            desc = f"短时尖峰{arrow} {_fmt(ml)}→{_fmt(mr)}→{_fmt(float(np.median(nxt)))}（{e - c} 个桶）"
        else:
            desc = f"水平突变{arrow} {_fmt(ml)}→{_fmt(mr)}"
        found.append((c, s, desc))
    # 同一条序列反复跳变（周期性刷盘、突发流量）说明这种幅度对它是常态：显著变点越多，每个的分数越低
    if len(found) > RECUR_FREE:
        found = [(c, s * RECUR_FREE / len(found), d + f"（该序列共 {len(found)} 次跳变）") for c, s, d in found]
    found.sort(key=lambda f: -f[1])
    return kind, ev + found[:MAX_CP_PER_SERIES]


def extract_events(wide, time_col="time", split=split_col, top_k=TOP_K, infer_counters=True):
    """宽表 → (事件列表, 统计)。事件：{t, entity, metric, kind, score, desc, sync}。"""
    g, t0, step = _resample(wide, time_col)
    n = len(g)
    stats = {"series": 0, "const": 0, "counter": 0, "state": 0, "entities": set(), "step": step,
             "t0": t0, "t1": t0 + n * step}
    # 同名指标跨实体的典型量级（各序列 |x| 的 p95 取中位数）；计数器按差分后的量级算
    names = {col: split(str(col)) for col in g.columns}
    p95 = {}
    for col in g.columns:
        x = g[col].to_numpy(float)
        x = x[~np.isnan(x)]
        if len(x) > 1:
            d = np.diff(x)
            if infer_counters and (d < 0).mean() <= 0.02 and (d > 0).mean() >= 0.5:
                x = d
            p95.setdefault(names[col][1], []).append(float(np.percentile(np.abs(x), 95)))
    typical = {m: float(np.median(v)) for m, v in p95.items() if len(v) >= 3}
    raw = []
    for col in g.columns:
        ent, met = names[col]
        kind, evs = _series_events(g[col].to_numpy(float), XENT_FLOOR * typical.get(met, 0.0),
                                   infer_counter=infer_counters)
        if kind is None:
            continue
        stats["series"] += 1
        if kind in ("const", "counter", "state"):
            stats[kind] += 1
        stats["entities"].add(ent)
        for b, s, desc in evs:
            raw.append({"t": t0 + b * step, "entity": ent, "metric": met + ("（速率）" if kind == "counter" else ""),
                        "kind": metric_kind(met), "score": round(s, 1), "desc": desc, "sync": []})
    stats["entities"] = len(stats["entities"])
    stats["raw_events"] = len(raw)
    # 同一实体、容差内同时发生的变化合并到最强的一条
    tol = max(2 * step, 0.02 * n * step)
    raw.sort(key=lambda e: -e["score"])
    merged, used = [], set()
    for i, e in enumerate(raw):
        if i in used:
            continue
        for j in range(i + 1, len(raw)):
            o = raw[j]
            if j not in used and o["entity"] == e["entity"] and abs(o["t"] - e["t"]) <= tol:
                used.add(j)
                e["sync"].append(f"{o['metric']}{'↑' if '↑' in o['desc'] else '↓' if '↓' in o['desc'] else ''}")
        # 同一实体多个指标同时变化，比单条序列孤立的跳变更可能是真实故障
        e["rank"] = e["score"] * (1 + SYNC_BOOST * min(len(e["sync"]), 5))
        merged.append(e)
    merged.sort(key=lambda e: -e["rank"])
    stats["events"] = len(merged)
    return merged[:top_k], stats


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


MECHANISM_MODES = ("mechanism_v1", "mechanism_v2", "mechanism_v3", "mechanism_v4", "mechanism_v5")


def _validate_relation_mode(relation_mode):
    if relation_mode not in ("baseline",) + MECHANISM_MODES:
        raise ValueError(f"invalid relation_mode: {relation_mode}")


def _evidence_frame(wide, time_col):
    """Sanitize only opt-in inputs; timestamps are numeric epoch seconds."""
    t = pd.to_numeric(wide.get(time_col, pd.Series(np.nan, index=wide.index)), errors="coerce")
    # Python datetime's representable UTC range; excludes NaN and infinities too.
    valid = t.between(-62135596800, 253402300799)
    clean = wide.drop(columns=[time_col], errors="ignore").apply(pd.to_numeric, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean[time_col] = t
    clean = clean.loc[valid].sort_values(time_col, kind="stable")
    duplicates = int(clean[time_col].duplicated().sum())
    if duplicates:
        clean = clean.groupby(time_col, as_index=False, sort=True).median()
    clean = clean.reset_index(drop=True)
    note = (f"rows_input={len(wide)} valid_time={int(valid.sum())} invalid_time={int((~valid).sum())} "
            f"duplicate_times_merged={duplicates} (median); series={len(clean.columns) - 1}")
    return clean, note


def _evidence_series(clean, time_col, split):
    """Observed endpoints/windows, never a healthy baseline or a causal relation.

    The existing monotonicity heuristic is applied BEFORE resampling; inferred
    counters are converted with actual elapsed seconds, never bucket indexes.
    """
    transformed = clean.copy()
    times = clean[time_col].to_numpy(float)
    settings, static, observations, counters = [], [], [], set()
    for col in clean.columns:
        if col == time_col:
            continue
        ent, met = split(str(col))
        x = clean[col].to_numpy(float)
        idx = np.flatnonzero(np.isfinite(x))
        v, ts = x[idx], times[idx]
        body = f"entity={str(ent)!r} metric={str(met)!r} observed={len(v)} missing={len(x) - len(v)}"
        if not len(v):
            observations.append(("unavailable", body + " unavailable", None))
            continue
        if np.all(v == v[0]):
            item = ("static", body + f" observed_constant={v[0]:.12g}", None)
            tokens = re.findall(r"[a-z]+", str(col).lower())
            target = settings if set(tokens) & {"config", "configuration", "setting", "settings", "parameter", "parameters"} else static
            target.append(item)
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dv = np.diff(v)
            counter = len(v) >= 3 and (dv < 0).mean() <= 0.02 and (dv > 0).mean() >= 0.5
            if counter:
                counters.add(col)
                rates = dv / np.diff(ts)
                rates[(dv < 0) | ~np.isfinite(rates)] = np.nan
                finite = np.isfinite(rates)
                # Floating subtraction/division noise is not a rate change.
                if finite.any() and np.allclose(rates[finite], rates[finite][0], rtol=1e-12, atol=0):
                    rates[finite] = rates[finite][0]
                transformed[col] = np.nan
                transformed.loc[idx[1:], col] = rates
                body += (f" basis=heuristic_counter_rate_per_second resets_omitted={int((dv < 0).sum())}"
                         f" rate_points={int(finite.sum())}")
                v, ts = rates[finite], ts[1:][finite]
        if not counter:
            body += " basis=observed_value"
        if len(v):
            window = max(1, len(v) // 3)
            body += (f" first_t={ts[0]:.17g} last_t={ts[-1]:.17g} first={v[0]:.12g} last={v[-1]:.12g}"
                     f" min={v.min():.12g} max={v.max():.12g} window_n={window}"
                     f" first_window_median={np.median(v[:window]):.12g}"
                     f" last_window_median={np.median(v[-window:]):.12g}")
        else:
            body += " no_valid_rates"
        observations.append(("observations", body, None))
    return transformed, counters, settings, static, observations


DEVIATION_TOP = 20   # mechanism_v2：列出稳健偏离最大的序列数
MIN_SEG_PTS = 3


def _deviation_entries(transformed, time_col, split, top=DEVIATION_TOP):
    """mechanism_v2：BARO 式稳健偏离排序（变点前中位数/IQR 作基线，变点后的最大偏离 / IQR）。

    变点由数据估计：所有序列稳健标准化后，取两段均值差加权和最大的切分点（每条序列贡献封顶，避免单条主导）；
    不使用任何外部提供的异常窗口。计数器已在 transformed 里换成每秒速率。
    """
    times = transformed[time_col].to_numpy(float)
    series = _varying_series(transformed, time_col)
    n = len(times)
    if not series or n < 2 * MIN_SEG_PTS:
        return []
    k = _onset_split(series, n)
    rows = []
    for col, x in series.items():
        pre, post = x[:k][np.isfinite(x[:k])], x[k:][np.isfinite(x[k:])]
        if len(pre) < 2 or not len(post):
            continue
        med = float(np.median(pre))
        iqr = float(np.subtract(*np.percentile(pre, [75, 25])))
        scale = max(iqr, 1e-3 * abs(med), 1e-9)
        dev = post - med
        peak = float(post[int(np.argmax(np.abs(dev)))])
        score = float(np.max(np.abs(dev)) / scale)
        rows.append((score, col, med, float(np.median(post)), peak, iqr))
    rows.sort(key=lambda r: -r[0])
    out = []
    for rank, (score, col, med, post_med, peak, iqr) in enumerate(rows[:top], 1):
        ent, met = split(str(col))
        ratio = f"{post_med / med:.3g}" if med else "n/a"
        out.append(("deviation", f"deviation_rank={rank} entity={str(ent)!r} metric={str(met)!r}"
                    f" direction={'up' if peak > med else 'down'} pre_median={med:.4g} pre_iqr={iqr:.4g}"
                    f" post_median={post_med:.4g} post_peak={peak:.4g} post_over_pre_median={ratio}"
                    f" robust_score={min(score, 1e6):.3g}", None))
    if out:
        out.insert(0, ("deviation_onset", f"estimated_onset time_epoch={times[k]:.17g} (data-driven split; "
                       f"pre={k} points, post={n - k} points; deviation = max |post - pre_median| / pre_IQR; "
                       "counters as per-second rates)", None))
    return out


def _onset_split(series, n):
    """所有序列稳健标准化后，两段均值差加权和最大的切分点。"""
    gain = np.zeros(n)
    for x in series.values():
        f = np.isfinite(x)
        med = np.nanmedian(x)
        sd = _mad(x[f]) or float(np.nanstd(x)) or 1.0
        z = np.where(f, (x - med) / sd, 0.0)
        cs = np.cumsum(z)
        g = np.zeros(n)
        for k in range(MIN_SEG_PTS, n - MIN_SEG_PTS + 1):
            m1, m2 = cs[k - 1] / k, (cs[-1] - cs[k - 1]) / (n - k)
            g[k] = abs(m2 - m1) * math.sqrt(k * (n - k) / n)
        # 每条序列的切分曲线按自身最大值归一后再按强度（封顶）加权：保留形状，避免单条序列主导
        if g.max() > 0:
            gain += g / g.max() * min(g.max(), 10.0)
    return int(np.argmax(gain))


def _varying_series(transformed, time_col):
    out = {}
    for col in transformed.columns:
        if col == time_col:
            continue
        x = transformed[col].to_numpy(float)
        if np.isfinite(x).sum() >= 2 * MIN_SEG_PTS and np.nanmax(x) > np.nanmin(x):
            out[col] = x
    return out


# mechanism_v3：负载归一化。吞吐/请求量指标按名词根识别（通用的“工作量”概念，不含任何数据集标签）
_WORKLOAD_TOKENS = {"comm", "commit", "commits", "xact", "xacts", "transactions", "tps", "qps", "request",
                    "requests", "req", "reqs", "throughput", "ops", "queries", "query", "calls"}
_NOT_WORKLOAD_TOKENS = {"slow", "error", "errors", "fail", "failed", "failure", "latency", "active", "duration",
                        "time", "wait", "retry", "retries", "rollback", "roll"}
LOAD_TOP = 16
LOAD_MIN_CHANGE = math.log(1.2)   # 前后中位数变化不足 20% 的指标不列


def _pick_load(series, counters, split):
    """按指标名词根挑工作量参照（提交/事务/请求类），计数器优先、量级大者优先；没有返回 None。"""
    def tokens(col):
        return set(re.findall(r"[a-z]+", str(split(str(col))[1]).lower()))

    loads = [c for c in series if tokens(c) & _WORKLOAD_TOKENS and not tokens(c) & _NOT_WORKLOAD_TOKENS]
    if not loads:
        return None
    return max(loads, key=lambda c: (c in counters, float(np.nanmedian(np.abs(series[c])))))


def _workload_entries(transformed, counters, time_col, split, top=LOAD_TOP):
    """每个变化指标的前后比值 ÷ 工作量指标的前后比值（per_load），以及与工作量的秩相关。

    per_load≈1 说明随负载同比例变化（多是负载本身的结果）；明显偏离 1 说明每单位工作量的代价/构成变了。
    切分点与 mechanism_v2 相同由数据估计，不使用外部异常窗口。没有可识别的工作量指标时不输出。
    """
    series = _varying_series(transformed, time_col)
    n = len(transformed)
    if not series or n < 2 * MIN_SEG_PTS:
        return []

    load = _pick_load(series, counters, split)
    if load is None:
        return []
    k = _onset_split(series, n)
    times = transformed[time_col].to_numpy(float)

    def medians(x):
        pre, post = x[:k][np.isfinite(x[:k])], x[k:][np.isfinite(x[k:])]
        return (float(np.median(pre)), float(np.median(post))) if len(pre) and len(post) else (np.nan, np.nan)

    def ratio(pre, post):
        return post / pre if pre > 0 and post >= 0 else np.nan

    lpre, lpost = medians(series[load])
    lr = ratio(lpre, lpost)
    if not np.isfinite(lr) or lr <= 0:
        return []
    lx = series[load]
    rows = []
    for col, x in series.items():
        if col == load:
            continue
        pre, post = medians(x)
        r = ratio(pre, post)
        if not np.isfinite(r):
            continue
        f = np.isfinite(x) & np.isfinite(lx)
        corr = (float(pd.Series(x[f]).corr(pd.Series(lx[f]), method="spearman"))
                if f.sum() >= 2 * MIN_SEG_PTS else np.nan)
        per = r / lr
        change = abs(math.log(r)) if r > 0 else 50.0
        if change >= LOAD_MIN_CHANGE:   # 按指标自身变化幅度排序，per_load 用来判断它是否只是随负载同比变化
            rows.append((change, col, pre, post, r, per, corr))
    rows.sort(key=lambda r: -r[0])
    ent, met = split(str(load))
    out = [("load_reference",
            f"workload_reference entity={str(ent)!r} metric={str(met)!r}"
            f"{' [heuristic rate/s]' if load in counters else ''} split_time_epoch={times[k]:.17g}"
            f" pre_median={lpre:.4g} post_median={lpost:.4g} post_over_pre={lr:.3g} (data-driven split; "
            "per_load = metric post_over_pre / workload post_over_pre: ~1 scales with workload, "
            ">>1 or <<1 means cost or mix per unit of work changed; corr = Spearman with workload)", None)]
    for rank, (_, col, pre, post, r, per, corr) in enumerate(rows[:top], 1):
        ent, met = split(str(col))
        out.append(("load_normalized",
                    f"load_rank={rank} entity={str(ent)!r} metric={str(met)!r} pre_median={pre:.4g}"
                    f" post_median={post:.4g} post_over_pre={r:.3g} per_load={per:.3g}"
                    f" corr_with_workload={'n/a' if not np.isfinite(corr) else f'{corr:.2f}'}", None))
    return out


RESID_MIN_Z = 3.0  # 残差不足 3 个稳健离散度的指标视为“工作量可解释”，不列


def _residual_entries(transformed, counters, time_col, split, top=LOAD_TOP, excess_only=False):
    """CIRCA 式回归残差：在变点前用工作量线性拟合每个指标，看变点后偏离“按工作量应有值”多少。

    斜率向比例斜率（pre 中位数之比）收缩，避免前段点少时外推失真；残差以前段残差 MAD 加
    全段 0.1·std 为尺度。explained_by_workload≈1 表示变化几乎全由工作量变化解释。
    excess_only：只列高于工作量预测的残差（无法由工作量解释的额外消耗）；低于预测多是吞吐下降的结果。
    """
    series = _varying_series(transformed, time_col)
    n = len(transformed)
    if not series or n < 2 * MIN_SEG_PTS:
        return []
    load = _pick_load(series, counters, split)
    if load is None:
        return []
    k = _onset_split(series, n)
    times = transformed[time_col].to_numpy(float)
    lx = series[load]
    idx = np.arange(n)
    rows = []
    for col, y in series.items():
        if col == load:
            continue
        f = np.isfinite(y) & np.isfinite(lx)
        pre, post = f & (idx < k), f & (idx >= k)
        if pre.sum() < MIN_SEG_PTS or post.sum() < MIN_SEG_PTS:
            continue
        xp, yp = lx[pre], y[pre]
        mx, my = xp.mean(), yp.mean()
        sxx, sxy = float(((xp - mx) ** 2).sum()), float(((xp - mx) * (yp - my)).sum())
        mxp = float(np.median(xp))
        b0 = float(np.median(yp)) / mxp if mxp > 0 else 0.0
        lam = max(sxx, 1e-12)  # 先验与数据等权
        b = (sxy + lam * b0) / (sxx + lam)
        a = my - b * mx
        res_pre = yp - (a + b * xp)
        expected = a + b * lx[post]
        scale = (1.4826 * float(np.median(np.abs(res_pre - np.median(res_pre))))
                 + 0.1 * float(np.std(y[f])) + 1e-9)
        z = float(np.median(y[post] - expected)) / scale
        if abs(z) < RESID_MIN_Z or (excess_only and z < 0):
            continue
        pre_med, post_med, exp_med = float(np.median(yp)), float(np.median(y[post])), float(np.median(expected))
        delta = post_med - pre_med
        explained = (exp_med - pre_med) / delta if abs(delta) > 1e-12 else np.nan
        rows.append((abs(z), col, pre_med, post_med, exp_med, z, explained))
    rows.sort(key=lambda r: -r[0])
    ent, met = split(str(load))
    lpre, lpost = float(np.nanmedian(lx[:k])), float(np.nanmedian(lx[k:]))
    out = [("load_reference",
            f"workload_reference entity={str(ent)!r} metric={str(met)!r}"
            f"{' [heuristic rate/s]' if load in counters else ''} split_time_epoch={times[k]:.17g}"
            f" pre_median={lpre:.4g} post_median={lpost:.4g} (data-driven split; each metric is regressed on "
            "workload before the split; expected_post = value predicted from post-split workload; residual_z = "
            f"(observed - expected) / robust pre-split residual spread; only {'residual_z>=3 (excess over workload)' if excess_only else '|residual_z|>=3'} listed; "
            "explained_by_workload = share of the metric's change predicted by workload)", None)]
    for rank, (_, col, pre, post, exp, z, ex) in enumerate(rows[:top], 1):
        ent, met = split(str(col))
        out.append(("load_residual",
                    f"residual_rank={rank} entity={str(ent)!r} metric={str(met)!r} pre_median={pre:.4g}"
                    f" post_median={post:.4g} expected_post_median={exp:.4g} residual_z={z:.3g}"
                    f" direction={'above' if z > 0 else 'below'}_expected"
                    f" explained_by_workload={'n/a' if not np.isfinite(ex) else f'{ex:.2f}'}", None))
    return out


_LOAD_EVIDENCE = {"ratio": _workload_entries, "residual": _residual_entries,
                  "excess": lambda *a: _residual_entries(*a, excess_only=True)}
_RELATION_LOAD = {"mechanism_v3": "ratio", "mechanism_v4": "residual", "mechanism_v5": "excess"}


def _interleave(*groups):
    """Give static observations and events a turn instead of alphabetical starvation."""
    return [group[i] for i in range(max(map(len, groups), default=0)) for group in groups if i < len(group)]


def _pack_evidence(note, entries, max_chars=None, prefix="M", totals=None):
    """One shared ID allocator and a strict whole-line budget, including coverage."""
    cap = max(0, int(RAW_BUDGET_TOKENS * CHARS_PER_TOKEN if max_chars is None else max_chars))
    counts = dict(totals or {})
    for kind in {item[0] for item in entries}:
        counts[kind] = max(counts.get(kind, 0), sum(item[0] == kind for item in entries))
    counts = dict(sorted(counts.items()))
    shown = dict.fromkeys(counts, 0)

    def coverage():
        # Fixed widths reserve enough room even when shown/omitted digit counts change.
        return "coverage: " + "; ".join(
            f"{kind}={shown[kind]:{len(str(total))}d}/{total} omitted={total - shown[kind]:{len(str(total))}d}"
            for kind, total in counts.items())

    header = ("## 指标机制证据摘要（observations only; no healthy-baseline or causal assumption）\n"
              "Counter inference: >=3 points, >=50% positive increments, <=2% decreases; "
              "rates use actual elapsed seconds; first/last windows are not change times.\n"
              f"{note}\nbudget_chars={cap}; whole evidence lines only")
    used = len(header) + 1 + len(coverage())
    if used > cap:
        return "", []  # No partial header or misleading, silently truncated evidence.
    lines, kept = [], []
    for item in entries:
        kind, body, _ = item
        if kind == "raw_rows" and not shown.get("raw_schema", 0):
            continue
        line = f"{prefix}{len(kept)} {body}"
        if used + 1 + len(line) > cap:
            continue
        lines.append(line)
        kept.append(item)
        shown[kind] += 1
        used += 1 + len(line)
    return "\n".join([header, coverage()] + lines), kept


def _mechanism_raw(wide, time_col, split, max_rows=None, max_chars=None, deviation=False, workload=None):
    clean, note = _evidence_frame(wide, time_col)
    transformed, counters, settings, static, observations = _evidence_series(clean, time_col, split)
    entries = ((_deviation_entries(transformed, time_col, split) if deviation else [])
               + (_LOAD_EVIDENCE[workload](transformed, counters, time_col, split) if workload else [])
               + settings + _interleave(observations, static))
    cols = [col for col in clean if col != time_col and clean[col].nunique() > 1]
    totals = {"raw_rows": len(clean) if cols else 0}
    if cols:
        names = [(str(entity), str(metric)) for entity, metric in (split(str(col)) for col in cols)]
        entries.append(("raw_schema", f"raw_columns={names!r}; raw_values are original cumulative/gauge samples", None))
        sampled = clean
        if max_rows is not None and len(clean) > max(0, int(max_rows)):
            positions = np.linspace(0, len(clean) - 1, max(0, int(max_rows)), dtype=int)
            sampled = clean.iloc[positions]
        for row in sampled[[time_col] + cols].itertuples(index=False, name=None):
            cells = ",".join("" if not np.isfinite(v) else f"{v:.12g}" for v in row[1:])
            entries.append(("raw_rows", f"raw time_epoch={row[0]:.17g} values={cells}", None))
    text, kept = _pack_evidence(note, entries, max_chars, totals=totals)
    return text, sum(item[0] == "raw_rows" for item in kept), totals["raw_rows"]


def _mechanism_events(events, stats, wide, time_col, split, prefix, max_chars, deviation=False, workload=None):
    settings, static, observations, deviations = [], [], [], []
    note = "source_wide=unavailable; supplied event units/heuristics not independently verified"
    if wide is not None:
        clean, note = _evidence_frame(wide, time_col)
        transformed, counters, settings, static, observations = _evidence_series(clean, time_col, split)
        if deviation:
            deviations = _deviation_entries(transformed, time_col, split)
        if workload:
            deviations += _LOAD_EVIDENCE[workload](transformed, counters, time_col, split)

        def event_split(col):
            entity, metric = split(col)
            return entity, str(metric) + (" [heuristic rate/s]" if col in counters else "")

        # Supplied legacy events may contain bucket-based counter artifacts; when
        # source data is available, recompute from sanitized, per-second inputs.
        if len(clean) and len(clean.columns) > 1:
            events, stats = extract_events(transformed, time_col, event_split, infer_counters=False)
        else:
            events, stats = [], {"events": 0}
    ordered = sorted(events, key=lambda e: e["t"])
    event_lines = []
    for event in ordered:
        body = (f"event time_epoch={event['t']:.17g} entity={str(event['entity'])!r} metric={str(event['metric'])!r}"
                f" description={event['desc']!r} score={event['score']:g}")
        if event.get("sync"):
            # This tolerance was actually computed by extract_events; no lag,
            # direction-of-causation or midpoint-derived synchronization claim.
            body += f" merged_metrics={event['sync']!r}"
            if all(key in stats for key in ("step", "t0", "t1")):
                tol = max(2 * stats["step"], 0.02 * (stats["t1"] - stats["t0"]))
                body += f" merge_tolerance_seconds={tol:.12g}"
        event_lines.append(("events", body, event))
    note += f"; event_groups_detected={stats.get('events', len(ordered))}"
    entries = deviations + settings + _interleave(event_lines, observations, static)
    text, kept = _pack_evidence(note, entries, max_chars, prefix,
                                totals={"events": stats.get("events", len(ordered))})
    return text, [item[2] for item in kept if item[0] == "events"]


def render_events(events, stats, prefix="M", relation_mode="baseline", wide=None, split=split_col,
                  time_col="time", max_chars=None):
    """事件 → 文本（按时间排序，行首编号 M# 供 LLM 引用）。"""
    _validate_relation_mode(relation_mode)
    if relation_mode in MECHANISM_MODES:
        return _mechanism_events(events, stats, wide, time_col, split, prefix, max_chars,
                                 deviation=relation_mode == "mechanism_v2",
                                 workload=_RELATION_LOAD.get(relation_mode))
    head = (f"## 指标变化事件（{stats['entities']} 个实体、{stats['series']} 条序列，其中常量 {stats['const']}、"
            f"计数器 {stats['counter']}（已差分为速率）、状态量 {stats['state']}；重采样步长 {stats['step']:.0f}s；"
            f"时间 {_iso(stats['t0'])} ~ {_iso(stats['t1'])}；共检出 {stats['events']} 个显著变化，列出最强的 "
            f"{len(events)} 个，按时间排序）\n"
            "score = 变化量 / 变化前的稳健波动；[ ] 内是按指标名词根推断的大类，仅供参考")
    lines = [head]
    for i, e in enumerate(sorted(events, key=lambda e: e["t"])):
        sync = f"；+{len(e['sync'])} 同步: {' '.join(e['sync'][:6])}" if e["sync"] else ""
        kind = f"[{e['kind']}]" if e["kind"] else ""
        lines.append(f"{prefix}{i} {_iso(e['t'])} {e['entity']} {e['metric']}{kind} {e['desc']}"
                     f"（score {e['score']:g}{sync}）")
    return "\n".join(lines), sorted(events, key=lambda e: e["t"])


RAW_BUDGET_TOKENS = 12000  # 原始表（必要时按时间降采样）估计不超过这个 token 数时直接给 LLM 读，不做事件抽象
CHARS_PER_TOKEN = 1.5      # 数字密集 CSV 文本，DeepSeek-V4 实测约 1.5 字符/token
MIN_RAW_ROWS = 40          # 降采样后少于这么多行（时间分辨率太粗）就改用事件抽取


def render_raw(wide, time_col="time", max_rows=None, relation_mode="baseline", split=split_col, max_chars=None):
    """宽表 → 紧凑 CSV 文本：去掉全空/常量列，数值保留 4 位有效数字，时间只写 HH:MM:SS（日期在表头）。
    max_rows：行数超过时按等长时间桶取中位数降采样。"""
    _validate_relation_mode(relation_mode)
    if relation_mode in MECHANISM_MODES:
        return _mechanism_raw(wide, time_col, split, max_rows, max_chars, relation_mode == "mechanism_v2",
                              _RELATION_LOAD.get(relation_mode))[0]
    t = pd.to_numeric(wide[time_col], errors="coerce")
    vals = wide.drop(columns=[time_col]).apply(pd.to_numeric, errors="coerce")
    keep = [c for c in vals.columns if vals[c].nunique(dropna=True) > 1]
    vals, step = vals[keep], None
    if max_rows and len(vals) > max_rows:
        b = np.minimum(((t - t.min()) / (t.max() - t.min() + 1e-9) * max_rows).astype(int), max_rows - 1)
        vals, t = vals.groupby(b.values).median(), t.groupby(b.values).min()
        step = (t.max() - t.min()) / max(len(t) - 1, 1)
    head = (f"## 监控指标原始数据（{len(vals)} 行 × {len(keep)} 列，已省略 {wide.shape[1] - 1 - len(keep)} 个常量/全空列；"
            f"时间 {_iso(t.min())} ~ {_iso(t.max())}，行首为 UTC 时分秒"
            + (f"，已按约 {step:.0f}s 时间桶取中位数降采样" if step else "") + "；列名为 实体|指标，计数器类为累计值）")
    lines = [head, "time," + ",".join(map(str, keep))]
    for ts, row in zip(t, vals.itertuples(index=False)):
        cells = ("" if v != v else f"{v:.4g}" for v in row)
        lines.append(datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S") + "," + ",".join(cells))
    return "\n".join(lines)


def metrics_text(wide, time_col="time", split=split_col, budget=RAW_BUDGET_TOKENS, relation_mode="baseline"):
    """指标给 LLM 的文本：原始表（按需降采样）能在 budget 内保留 ≥MIN_RAW_ROWS 行就原样给（mode=raw），
    否则抽成 top-K 变化事件（mode=events）。"""
    _validate_relation_mode(relation_mode)
    if relation_mode in MECHANISM_MODES:
        cap = max(0, int(budget * CHARS_PER_TOKEN))
        text, shown, total = _mechanism_raw(wide, time_col, split, max_chars=cap,
                                            deviation=relation_mode == "mechanism_v2",
                                            workload=_RELATION_LOAD.get(relation_mode))
        if shown >= min(MIN_RAW_ROWS, total):
            return text, "raw"
        return render_events([], {}, relation_mode=relation_mode, wide=wide, split=split,
                             time_col=time_col, max_chars=cap)[0], "events"
    full = render_raw(wide, time_col)
    rows = max(full.count("\n") - 1, 1)
    per_row = len(full) / rows
    fit = int(budget * CHARS_PER_TOKEN / per_row)
    if fit >= rows:
        return full, "raw"
    if fit >= MIN_RAW_ROWS:
        return render_raw(wide, time_col, fit), "raw"
    evs, st = extract_events(wide, time_col, split)
    return render_events(evs, st)[0], "events"
