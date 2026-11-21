# Adapted for the standalone unified method API; evaluation harness removed.
# Source: longyongchao/sundb-ai-ops @ 6ee89780d0d9cba70787a582debeb975e1550a1c
#!/usr/bin/env python3
"""D-Bot 基线（Zhou et al., VLDB 2024；官方仓库 TsinghuaDatabaseGroup/DB-GPT @73cb329）的离线复现，不用知识库。

照搬原仓库的流程、提示词（英文原文）与超参，只替换数据来源：
  1. Chief DBA 把告警写成异常描述（ANOMALY_DESC_PROMPT）；role assigner 从 9 个专家里选（max_hired_experts=2）。
  2. 每个专家做 UCT 树搜索（simulation_count=2, epsilon_new_node=0.3, vote_candidates=2, single_chain_max_step=24），
     Thought/Action/Action Input 格式调工具，Speak 且含 "solution" 为终止；未终止的链做 reflexion，终止节点两两投票回传价值。
  3. 取价值最高的终止节点提炼 root cause；Chief DBA 逐专家做诊断摘要；每个专家做一轮 cross review；Chief DBA 按评审精炼 root cause。
与原仓库的差异（均为离线化所必需，或修掉原仓库的 bug）：
  - 知识库检索（search_docs）去掉；optimize_index_selection 需要真实 PostgreSQL，去掉（提示词里相应示例一并删去）。
  - 数据：原仓库从 anomaly 文件的 exceptions（按 cpu/io/memory/network 分组的 Prometheus 指标）读指标，
    这里把指标 CSV 按列名关键词分到同样四组（其余列归 cpu 组，即原仓库的默认组）；
    原仓库 slow_queries / workload 两段文本在这里换成日志里 error/warn 级行的去重计数（日志是唯一的文本证据）。
  - 原 detect_anomalies 的两条 CDF 都由同一份数据构造，只能检出重复值；改成前后半窗的两样本 KS（显著性仍为 0.2），
    同样按统计量取 top-5；whether_is_abnormal_metric 原实现恒返回 "Cannot decide"，改为对 cpu/memory 组做同一检验。
  - 告警：数据集没有 Alertmanager 告警，给一条不含故障信息的通用告警（时间取输入数据自身的首末时间戳，不用金标准窗口）。
  - ${start_time}/${end_time} 原仓库未替换（字面量进提示词），这里替换成真实值；工具观察用 user 角色回传（原为 assistant）。
  - 模型换成 DeepSeek-V4-Flash（原为 gpt-4），温度等沿用原配置（专家 / Chief DBA 0.7，role assigner 0）。
  - 最后多一步格式化：把精炼后的 root cause 报告整理成最多 4 个候选（类别取统一评测的 10 类，不许新增报告里没有的根因），
    以便用同一判官与 D-Bot Acc 口径评分；这一步的输出 token 计入。
    与根因无关的步骤（标题、PostgreSQL 标签、solutions 提炼与精炼、引用标注）不跑，因此 token 为下界。

  venv/bin/python benchmark/unified/dbot_full.py --name dbot_all --cases wq_0001 rcae_0002   # 试跑
  venv/bin/python benchmark/unified/dbot_full.py --name dbot_all
  venv/bin/python benchmark/unified/full_eval.py --name dbot_all --all --metrics
"""

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

from ...llm import chat as _chat

EXPERTS = ["ConfigurationExpert", "CpuExpert", "DiskExpert", "IndexExpert", "IoExpert", "MemoryExpert",
           "QueryExpert", "RecoveryExpert", "WorkloadExpert"]   # knowledge_base/ 下全部 *Expert 目录
MAX_HIRED_EXPERTS = 2
SIMULATIONS, EPSILON_NEW_NODE, VOTE_CANDIDATES, MAX_DEPTH = 2, 0.3, 2, 24
EXPERT_T, REPORTER_T, ASSIGNER_T = 0.7, 0.7, 0.0
MAX_TOKENS, ASSIGNER_MAX_TOKENS = 2048, 256
SIGNIFICANCE = 0.2
TOP_LOG_LINES = 20

# ---------------------------------------------------------------------------
# 原仓库提示词（multiagents/agent_conf/config.yaml、prompt_templates/*.py、agents/reporter.py、agents/solver.py）
# ---------------------------------------------------------------------------
CHIEF_DBA = """You are a Chief DBA with much database diagnosis experience. Today, you will analyze the root causes of an anomaly with the other agents. Here is the outline of the procedure:
1. Chat and ask how to diagnose the root causes of the anomaly with other agents.
2. Summarize the root causes and detailed solutions spoken by other agents.

Your answer needs to be concise and accurate."""

ASSIGNER_ROLE = "You are a role assigner. Your task is to select experts with diverse identity to correctly analyze the root causes of the given alerts."

ASSIGNER = """# Role Description
You are the leader of a group of diagnosis experts. Currently, the database in your company may meet problems. The anomaly alert is:
{alert_info}

Now you need to select experts with diverse identity to correctly analyze the root causes of the given alert. The names of available experts are:
{experts}

Which experts will you recruit to generate an accurate solution? Note you can only select one to three experts!!!

# Response Format Guidance
You should respond with a list of expert names (one expert name can only occur for one time). For example:
1. CpuExpert
2. QueryExpert
...

Only respond with the names of selected experts (separated by spaces). Do not include your reason."""

ANOMALY_DESC = """Please describe the following anomaly event in natural language:
{anomaly_str}

Note do not miss any important information in each line of the event. And the description templates should be like:

1. "During the inspection, it was found that from xx to xx time, the database's CPU usage was relatively high, approximately 80% to 90%. It is a *critical problem*, please solve it as soon as possible."

2. "During the inspection, it was found that from xx to xx time, the database's CPU usage was relatively high, approximately 80% to 90%. It is just a *warning*, please take time to solve it carefully." """

EXPERT_ROLE = ("You are a {name} agent that can use the tool apis to check resource usage (whether_is_abnormal_metric), "
               "analyze the root causes of high resource usage using the metrics, queries and knowledge gained from "
               "(match_diagnose_knowledge), and give optimization solutions (e.g., optimize_index_selection, "
               "enable_or_disable_nestloop_operator). Note the response should be in markdown format.")

# config.yaml expert_agent_format_prompt；删去 optimize_index_selection 的两段示例（该工具需要真实 PostgreSQL）
EXPERT = """You are in a company whose databases meet an anomaly. The anomaly's start_time is {start_time} and end_time is {end_time}. The anomaly alert is:
{alert_info}

# Rules and Format Instructions for Response

- During diagnosis, you have access to the following tools:
{tools}

===============

- You can respond as follows to use tool:
Thought: (your thought)
Action: (an action name, it can be one of [whether_is_abnormal_metric, match_diagnose_knowledge, Speak], pay attention to the capitalization)
Action Input: (argument for the action)

You can first determine abnormal metrics by using the tools, and use the following format:
Thought: Now that I have obtained the start and end time of the anomaly, check whether the CPU usage is abnormal during that time period.
Action: whether_is_abnormal_metric
Action Input: {{"start_time": {start_time}, "end_time": {end_time}, "metric_name": "cpu_usage"}}

Next you must diagnose root causes by using the tools, and must use the following format (any other choice is not allowed):
Thought: No matter cpu usage is abnormal or normal, I must to diagnose the cause of the anomaly using the metrics, queries and knowledge gained from match_diagnose_knowledge.
Action: match_diagnose_knowledge
Action Input: {{"start_time": {start_time}, "end_time": {end_time}, "metric_name": "cpu"}}

After you have got the observation from match_diagnose_knowledge, analyze the root causes and announce it to other experts, and use the following format:
Thought: I now know the root cause of the anomaly.
Action: Speak
Action Input: ({{"diagnose": the root causes you found, "solution": the optimization solutions for the root causes split by '\\n', "knowledge": the diagnosis knowledge you used}})

===============

Here is the conversation history


Here is the execution log of tools


Remember to pay attention to the response format instructions, and strictly follow the rules specified above!
Based on the above history, what will you, {name}, do next?"""

TOOLS = ("### whether_is_abnormal_metric\n\nwhether_is_abnormal_metric: detect if there is an abnormal metric. "
         "Parameters：['start_time', 'end_time', 'metric_name', 'diag_id', 'enable_prometheus']\n"
         "### match_diagnose_knowledge\n\nmatch_diagnose_knowledge: search the relevant diagnosis knowledge from "
         "diagnosis knowledge base. Parameters：['start_time', 'end_time', 'metric_name', 'alert_metric', 'diag_id', "
         "'enable_prometheus']")

DIVERSITY = """This is not the first time you try this task, all previous trails failed.
Before you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are some previous actions candidates:
{previous_candidate}
Remember you are now in the intermediate state of a trail, you will first analyze the now state and previous action candidates, then make actions that is different from all the previous."""

REFLEXION = """Remember that you are performing a mento-carlo search. Now That you have done an unsatisfied attempt in the former try, you were imperfect in completing the task. You will make a reflection of the former attempt.
The reflection information can be inherit to later trails:
1.reflections is short, at most 5 sentence.
2.reflections give knowledge of the task you are performing.
3.reflections give tool APIs that you can use but not used in the former try.
Begin!"""

VOTE_SYSTEM = """You are mento-carlo-choice-GPT. Some agents are performing some task by mento-carlo search method and already made some trails. Your task is to decide which trail is the best.
1.I will first show you some trail candidate
2.Then you first analyze all the candidates, one candidate one sentence. Then give your choice of the best candidate.
3.Your choice will increase the "value" of the target trail, and we will further explore that trail later, so your choice is important.
Let's Begin!
Task description: """

VOTE_USER = """
Here are some former trails:
********************************
{candidate_description}
When you give your vote, You must follow the following principles:
1.Better candidate is staying in the state which is more close to success.
2.If you want to restart the task with one of the candidates' trice, and make some different choice in the middle, you may want to follow the best candidates' trice.
3.Remember, longer action chain doesn't mean is closer to success, because the process can not rollback, you may refer to the example as "what is  a success".
Begin:
1.it's your turn to first analyze each candidates. One candidate one sentence.
2.Then you can tell me who do you think is most close to success, and the reason.
3.Finally you must tell which is the best in one Word.
Your output should with the following format:
- analyze candidate 0: xxx
- analyze candidate 1: xxx
...
- analyze candidate n: xxx
- most close to success: xxx
-  "<candidate_x>"
Begin!"""

ROOT_CAUSE = ("Analyze the diagnosed root causes based on above discussions in details. Note the analysis should be only "
              "about root causes in markdown format!!! And do not mention anything about the solutions!!!")

SUMMARY = """Please provide a searchable summary of the input diagnosis procedures or recommended solutions without losing important information. The input is as follows:
{diagnosis_messages}

=============
Note
1. you are an expert and do not miss any important information!!!
2. response with a list of key points of the input diagnosis procedures or recommended solutions (e.g. "1. ...<br>2. ...<br>3. ..."). Do not add any additional content!!!
3. The key points should be strictly separated by <br> ("1. ...<br>2. ...<br>3. ...")!!!! Do not use markdown format and # headings!!!
4. Do not mention anything about what are the charts like!! Ignore charts!!!
5. Do not say anything like "As an AI"!!"""

ROUND_TABLE = """Please review the above diagnosis results, and give necessary advice to correct the unclear diagnosis and proposed solutions.
1. The reply must be short, no more than 5 sentences.
2. No need to call tools, no fixed template, but please note that the reply must be in markdown format."""

UPDATE_DIAGNOSIS = ("You are writing a report. Please give the refined root cause analysis based on the above review advice. "
                    "The root cause analysis is as follows:\n{root_cause}\n ===== \n Note 1. the output should be in markdown "
                    "format.\n2. do not any additional content like 'Sure' and 'I will refine the anomaly diagnosis "
                    "description based on the above advice. 3. Do not add anything about solutions!!!'\n")

# 评测适配（非 D-Bot 原有）：把最终报告整理成统一评测的候选格式
FORMAT = """Below is the final root cause analysis report written by a database diagnosis system.
Convert it into structured root-cause candidates for evaluation. Only use root causes stated in the report; do not add new ones.

- category must be one of: cpu | memory | disk_io | disk_space | network | process | syscall | connection | db_internal | code
  (process = process/container killed, paused, stopped or restarted; db_internal = lock contention / transaction retry storm / schema change / checkpoint storm and other database-internal load; connection = connection pool / session / connection count exhausted; syscall = injected system call errors or delays; code = application code defect)
- At most 4 candidates with distinct categories, ordered as the report ranks them (the primary root cause first).
- mechanism: the specific mechanism stated in the report (e.g. "network delay", "CPU saturation", "disk write IO pressure").

Only output JSON: {{"root_causes":[{{"category":"...","mechanism":"...","location":"component/node/service","why":"at most 20 words"}}]}}

Report:
{report}"""


# ---------------------------------------------------------------------------
# 数据：指标 CSV → 原仓库 exceptions 的四组；日志 → 替代 slow_queries/workload 的文本
# ---------------------------------------------------------------------------
def metric_group(col):
    m = col.split("|", 1)[-1].lower()
    if "mem" in m:
        return "memory"
    if "cpu" in m or "load" in m or "proc" in m:
        return "cpu"
    if any(k in m for k in ("disk", "io", "fs", "rocksdb", "capacity", "write", "read")):
        return "io"
    if any(k in m for k in ("net", "socket", "tcp", "recv", "sent")):
        return "network"
    return "cpu"


def ks_score(v):
    """两样本 KS（前半窗 vs 后半窗），返回 (统计量, 是否异常)。替代原仓库只能检出重复值的 detect_anomalies。"""
    h = len(v) // 2
    if h < 5 or np.std(v) == 0:
        return 0.0, False
    r = ks_2samp(v[:h], v[h:])
    return float(r.statistic), bool(r.pvalue < SIGNIFICANCE)


def processed_values(v):   # 原仓库 tools/metrics.py
    step = max(len(v) // 10, 1)
    return (f"the max value is {round(float(np.max(v)), 2)}, the min value is {round(float(np.min(v)), 2)}, "
            f"the mean value is {round(float(np.mean(v)), 2)}, the deviation value is {round(float(np.std(v)), 2)}, "
            f"and the evenly_sampled_values are {[round(float(x), 2) for x in v[::step]]}.")


_LEVEL = re.compile(r"\b(FATAL|PANIC|CRITICAL|SEVERE|ERROR|ERR|WARN|WARNING|EXCEPTION)\b|^[EWF]\d{6} ", re.I)
_TS = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
_VAR = [(re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}"), "<UUID>"), (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),
        (re.compile(r"\d+"), "<N>")]


def log_view(path):
    """日志里 error/warn 级行按掩码去重计数（前 TOP_LOG_LINES 条，带一条原文示例），以及首末时间戳。"""
    cnt, ex, first, last, n = Counter(), {}, None, None, 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            m = _TS.search(line[:80])
            if m:
                first = first or m.group(0)
                last = m.group(0)
            if not _LEVEL.search(line):
                continue
            k = line[m.end():] if m else line
            for p, r in _VAR:
                k = p.sub(r, k)
            k = k.strip()[:200]
            cnt[k] += 1
            ex.setdefault(k, line.strip()[:300])
    rows = [f"{i + 1}. [count={c}] {ex[k]}" for i, (k, c) in enumerate(cnt.most_common(TOP_LOG_LINES))]
    text = (f"The abnormal (error/warning-level) log lines, deduplicated with counts, out of {n} log lines are:\n"
            + "\n".join(rows) + "\n\n") if rows else f"No error/warning-level lines were found in the {n} log lines.\n\n"
    return text, first, last


class Case:
    def __init__(self, log, metrics, seed=0):
        self.groups = {k: {} for k in ("cpu", "io", "memory", "network")}
        self.rng = random.Random(seed)
        first = last = None
        if metrics is not None:
            w = metrics
            t = w["time"]
            first, last = (datetime.fromtimestamp(float(x), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                           for x in (t.min(), t.max()))
            self.start_ts, self.end_ts = float(t.min()), float(t.max())
            for c in w.columns:
                if c == "time":
                    continue
                v = pd.to_numeric(w[c], errors="coerce").dropna().to_numpy(dtype=float)
                if len(v) >= 10:
                    self.groups[metric_group(c)][c] = v
        self.log_text = ""
        if log is not None:
            self.log_text, lf, ll = log_view(log)
            first, last = first or lf, last or ll
        self.start, self.end = first or "unknown", last or "unknown"
        if metrics is None:
            self.start_ts, self.end_ts = self.start, self.end
        self.alert = (f"1. Alert Starts At: {self.start}\nAlert Ends At: {self.end}\nAlert Status: firing\n"
                      "Alert Description: An anomaly (service degradation) was reported during this period; the root cause is unknown.\n"
                      "Alert Level: WARN")

    def abnormal(self, group):
        """top-5 异常指标（按 KS 统计量），同原仓库 metric_analysis_results。"""
        sc = [(ks_score(v), c) for c, v in self.groups[group].items()]
        top = sorted(((s, c) for (s, a), c in sc if a), reverse=True)[:5]
        return [(c, processed_values(self.groups[group][c])) for _, c in top]

    def whether_is_abnormal_metric(self, expert):
        name = expert.lower().replace("expert", "")
        group = name if name in ("cpu", "memory") else self.rng.choice(["cpu", "memory"])
        metric = f"{group}_usage"
        return (f"The metric {metric} is abnormal \n " if self.abnormal(group)
                else f"Cannot decide whether the metric {metric} is abnormal.\n")

    def match_diagnose_knowledge(self, expert):
        a = expert.lower()
        group = ("cpu" if any(k in a for k in ("cpu", "workload", "index", "query")) else
                 "io" if ("write" in a or "io" in a) else "memory" if "mem" in a else "cpu")
        top = self.abnormal(group)
        s = ("The abnormal metrics are:\n" + "".join(f"{i + 1}. {c} contains abnormal patterns: {p} \n"
                                                     for i, (c, p) in enumerate(top)) + "\n") if top else ""
        return s + self.log_text


# ---------------------------------------------------------------------------
# LLM 与树搜索（原仓库 reasoning_algorithms/tree_of_thought/UCT_vote_function.py）
# ---------------------------------------------------------------------------
class Budget:
    def __init__(self, llm_config=None, label_space=None):
        self.usage = []
        self.config = llm_config
        self.label_space = label_space

    def chat(self, system, msgs, temperature, max_tokens=MAX_TOKENS):
        if self.label_space is not None:
            system += ('\nShared candidate IDs and definitions. They are not case-specific evidence:\n'
                       + self.label_space.get('prompt', '\n'.join(self.label_space['category'])))
        for k in range(3):   # 端点偶发连接超时：重试，避免整条多轮链作废
            try:
                content, u = _chat([{"role": "system", "content": system}] + msgs, max_tokens=max_tokens,
                                   temperature=temperature, config=self.config)
                break
            except requests.RequestException:
                if k == 2:
                    raise
                time.sleep(30 * (k + 1))
        self.usage.append(u)
        return content


def parse_action(text):
    """原仓库 DBDiag 解析器：第 1/2/3 行分别以 Thought / Action / Action Input 开头。"""
    t = re.sub(r"\n+", "\n", text.strip())
    parts = t.split("\n", 2)
    if len(parts) < 3 or not (parts[0].lower().startswith("thought") and parts[1].lower().startswith("action")
                              and parts[2].lower().startswith("action input")):
        return None
    action = re.split(r"(?i)^action:", parts[1], maxsplit=1)[-1].strip()
    return action, re.split(r"(?i)^action input:", parts[2], maxsplit=1)[-1].strip()


class Node:
    def __init__(self, parent, messages, action=None, action_input=None, obs=None):
        self.parent, self.messages, self.children, self.values = parent, messages, [], []
        self.action, self.action_input, self.obs = action, action_input, obs
        self.depth = parent.depth + 3 if parent else 0   # 原实现每步 Thought/Action/Action Input 三个节点
        self.terminal = self.pruned = False

    def weight(self):
        return sum(self.values) / len(self.values) if self.values else 0.0

    def trace(self):
        out, n = [], self
        while n.parent:
            out.append({"action": n.action, "action_input": n.action_input, "observation": n.obs})
            n = n.parent
        return out[::-1]


def expert_search(case, name, llm, rng):
    role = EXPERT_ROLE.format(name=name)
    root = Node(None, [{"role": "user", "content": EXPERT.format(start_time=case.start_ts, end_time=case.end_ts,
                                                                 alert_info=case.alert, tools=TOOLS, name=name)}])
    terminals = []
    for _ in range(SIMULATIONS):
        node = root   # 选择：按子节点平均价值 softmax 采样，「新节点」权重 epsilon
        while node.children and not node.terminal:
            kids = [c for c in node.children if not c.pruned]
            w = np.array([c.weight() for c in kids] + [EPSILON_NEW_NODE])
            p = np.exp(w - w.max())
            i = rng.choice(len(w), p=p / p.sum())
            if i == len(kids):
                break
            node = kids[i]
        start, first = node, True
        while node.depth < MAX_DEPTH and not node.terminal:
            msgs = node.messages
            if first and node.children:
                prev = [{"name": c.action, "arguments": c.action_input, "function_output": c.obs,
                         "mento-carlo-action-value": c.weight()} for c in node.children]
                msgs = msgs + [{"role": "user", "content": DIVERSITY.format(previous_candidate=json.dumps(prev, ensure_ascii=False))}]
            first = False
            parsed = None
            for _ in range(3):
                content = llm.chat(role, msgs, EXPERT_T)
                parsed = parse_action(content)
                if parsed:
                    break
            if not parsed:
                break
            action, arg = parsed
            child = Node(node, node.messages + [{"role": "assistant", "content": content}], action, arg)
            node.children.append(child)
            if action == "Speak":
                child.terminal = '"solution"' in content.lower()
                node = child
                break
            tool = getattr(case, action, None) if action in ("whether_is_abnormal_metric", "match_diagnose_knowledge") else None
            if tool is None:   # 未注册的工具：剪枝并结束本链
                child.pruned = True
                break
            child.obs = tool(name)
            child.messages = child.messages + [{"role": "user", "content": f"Observation: {child.obs}"}]
            node = child
        if node.terminal:
            terminals.append(node)
        else:   # 反思，结果挂到起点与根节点，供后续模拟继承
            refl = llm.chat(role, node.messages + [{"role": "user", "content": REFLEXION}], EXPERT_T)
            for n in {id(start): start, id(root): root}.values():
                n.messages = n.messages + [{"role": "assistant", "content": refl}]
        if len(terminals) >= VOTE_CANDIDATES:   # 投票：随机取 2 个终止节点，被选中者价值 +，另一个 -，沿路径回传
            cands = rng.choice(len(terminals), VOTE_CANDIDATES, replace=False)
            desc = "\n".join(f"<candidate_{k}>\n{json.dumps(terminals[c].trace(), ensure_ascii=False)}\n</candidate_{k}>"
                             for k, c in enumerate(cands))
            reply = llm.chat(VOTE_SYSTEM, [{"role": "user", "content": VOTE_USER.format(candidate_description=desc)}], EXPERT_T)
            hits = re.findall(r"[\"<]?candidate[ _](\d+)[\">]?", reply)
            if hits and int(hits[-1]) < VOTE_CANDIDATES:
                best = int(hits[-1])
                for k, c in enumerate(cands):
                    v = (int(k == best) - 1 / VOTE_CANDIDATES)
                    n = terminals[c]
                    while n:
                        n.values.append(v)
                        n = n.parent
    alive = [t for t in terminals if not t.pruned]
    if not alive:
        return None
    best = max(alive, key=lambda t: sum(t.values))
    rc = llm.chat("You are a database expert", [best.messages[-1], {"role": "user", "content": ROOT_CAUSE}], EXPERT_T)
    return {"sender": name, "root cause": rc, "messages": best.messages, "role": role}


def diagnose(log=None, metrics=None, seed=0, llm_config=None, label_space=None):
    from .label_space import resolve, instruction, candidates
    label_space = resolve(label_space)
    rng = np.random.default_rng(seed)
    case, llm = Case(log, metrics, seed), Budget(llm_config, label_space)
    desc = llm.chat(CHIEF_DBA, [{"role": "user", "content": ANOMALY_DESC.format(anomaly_str=case.alert)}], REPORTER_T)
    hired = []
    for _ in range(3):
        reply = llm.chat(ASSIGNER_ROLE, [{"role": "user", "content": ASSIGNER.format(alert_info=desc, experts=str(EXPERTS))}],
                         ASSIGNER_T, ASSIGNER_MAX_TOKENS)
        hired = [e for e in EXPERTS if e.lower() in reply.lower()]
        if hired:
            break
    hired = hired[:MAX_HIRED_EXPERTS]
    with ThreadPoolExecutor(len(hired) or 1) as ex:   # 专家并行诊断（原仓库 asyncio.gather）
        results = [r for r in ex.map(lambda e: expert_search(case, e, llm, np.random.default_rng(rng.integers(1 << 30))), hired) if r]
    if not results:
        return {"root_causes": [], "usage": llm.usage, "error": "no expert reached a terminal Speak",
                "experts": hired, "alert_desc": desc}
    report = "".join(f"\nThe root causes identified by {r['sender']}:\n{r['root cause']}" for r in results)
    for r in results:   # 诊断摘要（Chief DBA），追加到每个专家的上下文
        proc = "\n".join(m["content"] for m in r["messages"][1:])
        summ = llm.chat(CHIEF_DBA, [{"role": "user", "content": SUMMARY.format(diagnosis_messages=proc)}], REPORTER_T)
        r["messages"] = r["messages"] + [{"role": "assistant", "content": summ}]
    reviews = []
    for r in results:   # 一轮交叉评审，顺序进行，后评审者能看到前面的评审
        rev = llm.chat(r["role"], r["messages"] + [{"role": "user", "content": ROUND_TABLE}], EXPERT_T)
        msg = {"role": "user", "content": f"{r['sender']}: {rev}"}
        for x in results:
            x["messages"] = x["messages"] + [msg]
        reviews.append(msg)
    refined = llm.chat(CHIEF_DBA, reviews + [{"role": "user", "content": UPDATE_DIAGNOSIS.format(root_cause=report)}], REPORTER_T)
    formatter = FORMAT.format(report=refined) if label_space is None else (
        'Convert only the causes actually stated in the following diagnosis report. Do not add new causes.\n'
        + instruction(label_space) + '\nReport:\n' + refined)
    fmt = llm.chat("You are a helpful assistant.", [{"role": "user", "content": formatter}], 0.0, 800)
    m = re.search(r"\{.*\}", fmt, re.S)
    preds = []
    for rc in ((json.loads(m.group(0)) if m else {}).get("root_causes") or [])[:4]:
        if label_space is not None:
            preds.extend(candidates([rc], label_space))
            continue
        cat, mech, loc = (str(rc.get(k, "")).strip() for k in ("category", "mechanism", "location"))
        preds.append({"type": cat, "headline": mech, "description": f"{mech}（类别 {cat}，位置 {loc}）。{rc.get('why', '')}",
                      "location": loc, "evidence_templates": []})
    return {"root_causes": preds, "usage": llm.usage, "error": None if preds else "format step returned no candidates",
            "experts": hired, "alert_desc": desc, "expert_root_causes": {r["sender"]: r["root cause"] for r in results},
            "reviews": [x["content"] for x in reviews], "refined_report": refined}
