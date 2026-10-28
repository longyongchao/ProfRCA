"""封闭世界统一机制标签集：wangqing / RCAEval / DBPA 三个 benchmark 的全部 native 标签合并为 30 个机制 ID。

所有方法在提示里看到同一份清单（不随数据集变化，不提示根因个数），输出至多 4 个按可能性排序的 mechanism_id；
评分为 Top-K_GT 精确匹配（无判官）：K_GT = 去重后的金标准机制数，D-Bot 公式 max(0, (Ac - 0.1*Aw) / Aa)。
大类 Acc 用同一公式比较机制所属的一级类别。位置不计分。
"""
from __future__ import annotations

# (id, 一级类别, 提示里给出的说明)。说明只用通用运维语言，不含任何数据集名、样本或参数。
MECHANISMS = [
    ("cpu_saturation", "cpu", "CPU 满载：进程或机器 CPU 被打满"),
    ("cpu_limit", "cpu", "CPU 配额或可用核数被下调/限流"),
    ("memory_pressure", "memory", "内存压力：内存或页缓存占用过高"),
    ("memory_limit", "memory", "容器/进程内存上限被下调"),
    ("buffer_pool_small", "memory", "数据库共享缓冲/缓冲池配置过小"),
    ("disk_io_pressure", "disk_io", "磁盘 IO 压力或饱和（读/写/读写）"),
    ("disk_full", "disk_space", "磁盘空间被写满"),
    ("network_delay", "network", "网络延迟"),
    ("network_loss", "network", "网络丢包"),
    ("network_corrupt", "network", "网络包损坏"),
    ("network_duplicate", "network", "网络包重复"),
    ("port_block", "network", "端口或节点间连接被阻断"),
    ("socket_exhaustion", "network", "socket/端口资源耗尽"),
    ("process_kill", "process", "进程被杀死（信号终止）"),
    ("process_pause", "process", "进程被暂停/挂起（SIGSTOP 等）"),
    ("container_restart", "process", "容器/虚机被停止、关机或重启"),
    ("syscall_error", "syscall", "系统调用返回错误"),
    ("syscall_delay", "syscall", "系统调用延迟"),
    ("connection_exhaustion", "connection", "数据库连接、会话或连接池耗尽"),
    ("lock_contention", "db_internal", "锁争用/锁等待"),
    ("txn_retry_storm", "db_internal", "事务冲突重试风暴"),
    ("schema_change_storm", "db_internal", "频繁 Schema 变更（DDL）"),
    ("checkpoint_storm", "db_internal", "检查点风暴/频繁刷脏"),
    ("missing_index", "db_internal", "缺少索引导致全表扫描"),
    ("too_many_indexes", "db_internal", "索引过多导致写放大"),
    ("vacuum", "db_internal", "清理/回收任务（vacuum、purge）"),
    ("concurrent_commits", "db_internal", "高并发小事务频繁提交"),
    ("concurrent_inserts", "db_internal", "大批量或高并发插入"),
    ("workload_surge", "db_internal", "业务负载/查询量整体上升"),
    ("code_defect", "code", "应用代码缺陷"),
]
CATEGORY = {m: c for m, c, _ in MECHANISMS}

# native 标签 → 机制 ID（三个 benchmark 的标签名互不冲突，可用一张平表）
NATIVE = {
    # wangqing
    "cpu": "cpu_saturation", "cpu_fullload": "cpu_saturation",
    "gserver_cpu_throttle": "cpu_limit", "lxc_cpu_core_reduce": "cpu_limit",
    "memory": "memory_pressure", "mem_ram_load": "memory_pressure", "mem_cache_load": "memory_pressure",
    "lxc_memory_limit_reduce": "memory_limit",
    "disk_io": "disk_io_pressure", "disk_burn_read": "disk_io_pressure", "disk_burn_write": "disk_io_pressure",
    "disk_burn_rw": "disk_io_pressure",
    "disk_fill": "disk_full", "disk_fill_root": "disk_full",
    "network_delay": "network_delay", "network_client_delay": "network_delay",
    "network_loss": "network_loss", "network_corrupt": "network_corrupt", "network_duplicate": "network_duplicate",
    "block_group_port": "port_block",
    "network_drop": "port_block",   # chaosblade network drop：按目的端口丢弃全部入流量（参数 destination-port），即端口阻断
    "kill_gbalancer": "process_kill", "kill_gdispatcher": "process_kill", "kill_glsnr": "process_kill",
    "kill_gmaster": "process_kill", "process_kill_by_port": "process_kill", "process_kill_sigkill": "process_kill",
    "process_kill_sigterm": "process_kill",
    "process_pause_cserver": "process_pause", "process_pause_gserver": "process_pause",
    "process_stop": "process_pause",   # chaosblade process stop 发 SIGSTOP，即暂停
    "lxc_stop": "container_restart", "lxc_shutdown": "container_restart", "lxc_reboot": "container_restart",
    "strace_error": "syscall_error", "strace_delay": "syscall_delay",
    "connection_exhaust": "connection_exhaustion", "pool_exhaust": "connection_exhaustion",
    "session_exhaust": "connection_exhaustion",
    "lock_contention": "lock_contention", "txn_retry_storm": "txn_retry_storm",
    "schema_change_storm": "schema_change_storm", "checkpoint_storm": "checkpoint_storm",
    # RCAEval
    "mem": "memory_pressure", "disk": "disk_io_pressure", "delay": "network_delay", "loss": "network_loss",
    "socket": "socket_exhaustion", "f1": "code_defect", "f2": "code_defect", "f3": "code_defect",
    "f4": "code_defect", "f5": "code_defect",
    # DBPA
    "io_saturation": "disk_io_pressure", "small_shared_buffer": "buffer_pool_small", "lock_waits": "lock_contention",
    "missing_indexes": "missing_index", "too_many_indexes": "too_many_indexes", "vacuum": "vacuum",
    "concurrent_commits": "concurrent_commits", "concurrent_inserts": "concurrent_inserts",
    "heavy_workload": "workload_surge",
}
assert set(NATIVE.values()) <= set(CATEGORY)


def prompt_block():
    """提示里的答案空间：按一级类别分组列出全部机制 ID。所有方法、所有数据集共用。"""
    lines, last = [f"- 根因只能从下列 {len(MECHANISMS)} 个机制中选择，用 mechanism_id 作答（同一类别可以出现多个机制）："], None
    for mid, cat, desc in MECHANISMS:
        if cat != last:
            lines.append(f"  [{cat}]")
            last = cat
        lines.append(f"    {mid}: {desc}")
    return "\n".join(lines) + "\n"


def gold_ids(gold):
    """金标准机制 ID，按金标准顺序去重（复合根因映射到同一机制时只算一个）。"""
    out = []
    for r in gold["root_causes"]:
        m = r.get("mechanism_id") or NATIVE[r["label"]]
        if m not in out:
            out.append(m)
    return out


def pred_ids(root_causes):
    """候选 → 机制 ID 序列：优先取 mechanism_id 字段，没有则由 native label 映射；按顺序去重，无效 ID 保留（计为错误）。"""
    out = []
    for rc in root_causes or []:
        m = str(rc.get("mechanism_id") or NATIVE.get(rc.get("label") or "") or "<none>").strip()
        if m not in out:
            out.append(m)
    return out


def dbot(pred, gold):
    k = len(gold)
    top = pred[:k]
    ac = len(set(top) & set(gold))
    return max(0.0, (ac - 0.1 * (len(top) - ac)) / k)


def score(gold, root_causes):
    """→ 细粒度（机制）与大类的 Top-K_GT D-Bot Acc，以及 Top-1 / Hit@4。"""
    g, p = gold_ids(gold), pred_ids(root_causes)
    gc = list(dict.fromkeys(CATEGORY[m] for m in g))
    pc = list(dict.fromkeys(CATEGORY.get(m, "invalid:" + m) for m in p))
    return {"acc": dbot(p, g), "acc_cat": dbot(pc, gc),
            "top1": int(bool(p) and p[0] in g), "hit4": int(bool(set(p[:4]) & set(g))),
            "invalid": sum(m not in CATEGORY for m in p), "k_gt": len(g), "n_pred": len(p)}
