# Adapted for the standalone unified method API; dataset harness and macOS-only runner removed.
# Source: longyongchao/sundb-ai-ops @ 6ee89780d0d9cba70787a582debeb975e1550a1c
#!/usr/bin/env python3
"""RCA-Agent 基线（OpenRCA, Xu et al., ICLR 2025；官方仓库 microsoft/OpenRCA @c1bd4af，rca/baseline/rca_agent）的移植。

照搬原仓库的控制器 / 执行器两级循环、提示词（英文原文）与超参：
  - Controller 每步输出 {analysis, completed, instruction}，把原子化的自然语言指令交给 Executor；
  - Executor 写 Python 代码在有状态的 IPython 内核里执行（失败时带报错重写，最多 2 次），
    执行成功后再调一次 LLM 用自然语言概括结果，概括 + 原始输出回传给 Controller；
  - completed=True 或达到 controller_max_step=25 步后，按 summary 提示词从候选列表里选根因（组件 + 原因）。
与原仓库的差异（均为适配本数据集或离线运行所必需）：
  - basic_prompt（数据目录结构、schema、候选组件 / 原因）按本数据集重写：遥测只有 telemetry/log.txt 与
    telemetry/metric.csv（宽表，time + 「实体|指标」列），没有 trace；时区为 UTC（原为 UTC+8，执行器规则 10 相应改为 UTC）。
    候选原因用统一评测 10 个类别下的通用机制（与系统、case 无关）；候选组件取指标列名里的实体，
    没有指标时说明组件为日志里的主机 / 服务名。agent_prompt（诊断规则）原样保留。
  - 告警（objective）：不给故障时间，时间范围取输入数据自身的首末时间戳，故障数为 unknown。
  - 代码在 sandbox-exec 里执行：禁网，只能写 case 临时目录，读不到仓库与数据集（金标准）；输入文件复制到临时目录。
    单个 cell 超过 300s 视为执行失败并重启内核（变量丢失）。原仓库每个 case 600s 总时限（含 LLM 时间）不沿用，
    因为端点时快时慢；改为只靠 25 步上限终止。
  - 模型换成 DeepSeek-V4-Flash（原为 GPT-4o / Claude 等），温度 0（原配置默认），max_tokens 8000；
    completed 字段兼容 JSON 布尔 true（原代码只认字符串 "True"）。
  - 端点网络错误重试 3 次后记 diagnose_error（续跑时重跑），不像原代码那样吞掉异常继续推理。
  - 输出：最终 JSON 里的每个根因（按顺序，最多 4 个）转成统一候选，类别由所选原因直接映射。
    最终回复解析不出 JSON 时（约 13%：沿用控制器格式作答或重复生成到上限），仿照原控制器对格式错误的处理追问一次
    "Please provide your final answer in the requested JSON format."；仍不行则取文中逐字出现的候选原因（记 parse=text_fallback）。
    原方法按故障数给答案，不强制 4 个候选。

  venv/bin/python benchmark/unified/rca_agent_full.py --name rca_agent_dev --cases wq_0001   # 试跑
  venv/bin/python benchmark/unified/rca_agent_full.py --name rca_agent_all
  venv/bin/python benchmark/unified/full_eval.py --name rca_agent_all --all --metrics
"""

import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import pandas as pd
import requests
import tiktoken

from contextvars import ContextVar
from ...llm import chat, current_config
from ...errors import ConfigurationError
from ..kernel import DockerKernel

MAX_STEP = 25        # run_agent_standard.py --controller_max_step 默认值
CELL_TIMEOUT = 300
MAX_TOKENS = 8000
CALL_TIMEOUT = 600   # 单次 LLM 调用总时长上限（requests 的 timeout 管不住挂住的连接）
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# 提示词：controller.py / executor.py / agent_prompt.py 原文
# ---------------------------------------------------------------------------
CTRL_SYSTEM = """You are the Administrator of a DevOps Assistant system for failure diagnosis. To solve each given issue, you should iteratively instruct an Executor to write and execute Python code for data analysis on telemetry files of target system. By analyzing the execution results, you should approximate the answer step-by-step.

There is some domain knowledge for you:

{background}

{agent}

The issue you are going to solve is:

{objective}

Solve the issue step-by-step. In each step, your response should follow the JSON format below:

{format}

Let's begin."""

CTRL_FORMAT = """{
    "analysis": (Your analysis of the code execution result from Executor in the last step, with detailed reasoning of 'what have been done' and 'what can be derived'. Respond 'None' if it is the first step.),
    "completed": ("True" if you believe the issue is resolved, and an answer can be derived in the 'instruction' field. Otherwise "False"),
    "instruction": (Your instruction for the Executor to perform via code execution in the next step. Do not involve complex multi-step instruction. Keep your instruction atomic, with clear request of 'what to do' and 'how to do'. Respond a summary by yourself if you believe the issue is resolved. Respond a summary by yourself if you believe the issue is resolved. Respond a summary by yourself if you believe the issue is resolved.)
}
(DO NOT contain "```json" and "```" tags. DO contain the JSON object with the brackets "{}" only. Use '\\n' instead of an actual newline character to ensure JSON compatibility when you want to insert a line break within a string.)"""

CTRL_SUMMARY = """Now, you have decided to finish your reasoning process. You should now provide the final answer to the issue. The candidates of possible root cause components and reasons are provided to you. The root cause components and reasons must be selected from the provided candidates.

{cand}

Recall the issue is: {objective}

Please first review your previous reasoning process to infer an exact answer of the issue. Then, summarize your final answer of the root causes using the following JSON format at the end of your response:

```json
{{
    "1": {{
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise ommited),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise ommited),
    }}, (mandatory)
    "2": {{
        "root cause occurrence datetime": (if asked by the issue, format: '%Y-%m-%d %H:%M:%S', otherwise ommited),
        "root cause component": (if asked by the issue, one selected from the possible root cause component list, otherwise ommited),
        "root cause reason": (if asked by the issue, one selected from the possible root cause reason list, otherwise ommited),
    }}, (only if the failure number is "unknown" or "more than one" in the issue)
    ... (only if the failure number is "unknown" or "more than one" in the issue)
}}
```
(Please use "```json" and "```" tags to wrap the JSON object. You only need to provide the elements asked by the issue, and ommited the other fields in the JSON.)
Note that all the root cause components and reasons must be selected from the provided candidates. Do not reply 'unknown' or 'null' or 'not found' in the JSON. Do not be too conservative in selecting the root cause components and reasons. Be decisive to infer a possible answer based on your current observation."""

EXEC_SYSTEM = """You are a DevOps assistant for writing Python code to answer DevOps questions. For each question, you need to write Python code to solve it by retrieving and processing telemetry data of the target system. Your generated Python code will be automatically submitted to a IPython Kernel. The execution result output in IPython Kernel will be used as the answer to the question.

{rule}

There is some domain knowledge for you:

{background}

Your response should follow the Python block format below:

{format}"""

EXEC_FORMAT = """```python
(YOUR CODE HERE)
```"""

EXEC_SUMMARY = """The code execution is successful. The execution result is shown below:

{result}

Please summarize a straightforward answer to the question based on the execution results. Use plain English."""

EXEC_CONCLUSION = """{answer}

The original code execution output of IPython Kernel is also provided below for reference:

{result}"""

# 规则 10 由 UTC+8 改为 UTC（本数据集时间戳均为 UTC），其余原文
EXEC_RULE = """## RULES OF PYTHON CODE WRITING:

1. Reuse variables as much as possible for execution efficiency since the IPython Kernel is stateful, i.e., variables define in previous steps can be used in subsequent steps.
2. Use variable name rather than `print()` to display the execution results since your Python environment is IPython Kernel rather than Python.exe. If you want to display multiple variables, use commas to separate them, e.g. `var1, var2`.
3. Use pandas Dataframe to process and display tabular data for efficiency and briefness. Avoid transforming Dataframe to list or dict type for display.
4. If you encounter an error or unexpected result, rewrite the code by referring to the given IPython Kernel error message.
5. Do not simulate any virtual situation or assume anything unknown. Solve the real problem.
6. Do not store any data as files in the disk. Only cache the data as variables in the memory.
7. Do not visualize the data or draw pictures or graphs via Python. You can only provide text-based results. Never include the `matplotlib` or `seaborn` library in the code.
8. Do not generate anything else except the Python code block except the instruction tell you to 'Use plain English'. If you find the input instruction is a summarization task (which is typically happening in the last step), you should comprehensively summarize the conclusion as a string in your code and display it directly.
9. Do not calculate threshold AFTER filtering data within the given time duration. Always calculate global thresholds using the entire KPI series of a specific component within a metric file BEFORE filtering data within the given time duration.
10. All issues use **UTC** time. However, the local machine's default timezone is unknown. Please use `pytz.UTC` to explicityly set the timezone to UTC.
"""

AGENT_RULES = """## RULES OF FAILURE DIAGNOSIS:

What you SHOULD do:

1. **Follow the workflow of `preprocess -> anomaly detection -> fault identification -> root cause localization` for failure diagnosis.**
    1.1. Preprocess:
        - Aggregate each KPI of each components that are possible to be the root cause component to obtain multiple time series classified by 'component-KPI' (e.g., service_A-cpu_usage_pct).
        - Then, calculate global thresholds (e.g., global P95, where 'global' means the threshold of all 'component-KPI' time series within a whole metric file) for each 'component-KPI' time series. - Finally, filter data within the given time duration for all time series to perform further analysis.
        - Since the root cause component must be selected from the provided possible root cause components, all other level's components (e.g., service mesh components, middleware components, etc.) should be ignored.
    1.2. Anomaly detection:
        - An anomaly is typically a data point that exceeds the global threshold.
        - Look for anomalies below a certain threshold (e.g., <=P95, <=P15, or <=P5) in traffic KPIs or business KPIs (e.g., success rate (ss)) since some network failures can cause a sudden drop on them due to packet loss.
        - Loose the global threshold (e.g., from >=P95 to >=P90, or from <=P95 to <=P15, <=P5) if you really cannot find any anomalies.
    1.3. Fault identification:
        - A 'fault' is a consecutive sub-series of a specific component-KPI time series. Thus, fault identification is the process of identifying which components experienced faults, on which resources, and at what occurrence time points.
        - Filter out isolated noise spikes to locate faults.
        - Faults where the maximum (or minimum) value in the sub-series only slightly exceeds (or falls below) the threshold (e.g., threshold breach <= 50% of the extremal), it’s likely a false positive caused by random KPI fluctuations, and should be excluded.
    1.4. Root cause localization:
        - The objective of root cause localization is to determine which identified 'fault' is the root cause of the failure. The root cause occurrence time, component, and reason can be derived from the first piece of data point of that fault.
        - If multiple faulty components are identified at **different levels** (e.g., some being containers and others nodes), and all of them are potential root cause candidates, while the issue itself describes a **single failure**, the root cause level should be determined by the fault that shows the most significant deviation from the threshold (i.e., >> 50%). However, this method is only applicable to identify the root cause level, not the root cause component. If there are multiple faulty components at the same level, you should use traces and logs to identify the root cause component.
        - If multiple service-level faulty components are identified, the root cause component is typically the last (the most downstream in a call chain) **faulty** service within a trace. Use traces to identify the root cause component among multiple faulty services.
        - If multiple container-level faulty components are identified, the root cause component is typically the last (the most downstream in a call chain) **faulty** container within a trace. Use traces to identify the root cause component among multiple faulty container.
        - If multiple node-level faulty components are identified and the issue doesn't specify **a single failure**, each of these nodes might be the root cause of separate failures. Otherwise, the predominant nodes with the most faults is the root cause component. The node-level failure do not propagate, and trace only captures communication between all containers or all services.
        - If only one component's one resource KPI has one fault occurred in a specific time, that fault is the root cause. Otherwise, you should use traces and logs to identify the root cause component and reason.
2. **Follow the order of `threshold calculation -> data extraction -> metric analyis -> trace analysis -> log analysis` for failure diagnosis.**
    2.0. Before analysis: You should extract and filter the data to include those within the failure duration only after the global threshold has been calculated. After these two steps, you can perform metric analysis, trace analysis, and log analysis.
    2.1. Metric analysis: Use metrics to calculate whether each KPIs of each component has consecutive anomalies beyond the global threshold is the fastest way to find the faults. Since there are a large number of traces and logs, metrics analysis should first be used to narrow down the search space of duration and components.
    2.2. Trace analysis: Use traces can further localize which container-level or service-level faulty component is the root cause components when there are multiple faulty components at the same level (container or service) identified by metrics analysis.
    2.3. Log analysis: Use logs can further localize which resource is the root cause reason when there are multiple faulty resource KPIs of a component identified by metrics analysis. Logs can also help to identify the root cause component among multiple faulty components at the same level.
    2.4. Always confirm whether the target key or field is valid (e.g., component's name, KPI's name, trace ID, log ID, etc.) when Executor's retrieval result is empty.

What you SHOULD NOT do:

1. **DO NOT include any programming language (Python) in your response.** Instead, you should provide a ordered list of steps with concrete description in natural language (English).
2. **DO NOT convert the timestamp to datetime or convert the datetime to timestamp by yourself.** These detailed process will be handled by the Executor.
3. **DO NOT use the local data (filtered/cached series in specific time duration) to calculate the global threshold of aggregated 'component-KPI' time series.** Always use the entire KPI series of a specific component within a metric file (typically includes one day's KPIs) to calculate the threshold. To obtain global threshold, you can first aggregate each component's each KPI to calculate their threshold, and then retrieve the objective time duration of aggregated 'component-KPI' to perform anomaly detection and spike filtering.
4. **DO NOT visualize the data or draw pictures or graphs via Python.** The Executor can only provide text-based results. Never include the `matplotlib` or `seaborn` library in the code.
5. **DO NOT save anything in the local file system.** Cache the intermediate results in the IPython Kernel. Never use the bash command in the code cell.
6. **DO NOT calculate threshold AFTER filtering data within the given time duration.** Always calculate global thresholds using the entire KPI series of a specific component within a metric file BEFORE filtering data within the given time duration.
7. **DO NOT query a specific KPI without knowing which KPIs are available.** Different systems may have completely different KPI naming conventions. If you want to query a specific KPI, first ensure that you are aware of all the available KPIs.
8. **DO NOT mistakenly identify a healthy (non-faulty) service at the downstream end of a trace that includes faulty components as the root cause.** The root cause component should be the most downstream **faulty** service to appear within the trace call chain, which must first and foremost be a FAULTY component identified by metrics analysis.
9. **DO NOT focus solely on warning or error logs during log analysis. Many info logs contain critical information about service operations and interactions between services, which can be valuable for root cause analysis.**"""

INIT_CODE = ("import pandas as pd\n"
             "pd.set_option('display.width', 427)\n"
             "pd.set_option('display.max_columns', 10)\n")

# ---------------------------------------------------------------------------
# basic_prompt（按本数据集重写）：候选原因与统一评测类别的映射与 case 无关
# ---------------------------------------------------------------------------
REASONS = [
    ("high CPU usage", "cpu"),
    ("high memory usage", "memory"),
    ("high disk I/O read/write usage", "disk_io"),
    ("high disk space usage", "disk_space"),
    ("network latency", "network"),
    ("network packet loss", "network"),
    ("network socket / port fault", "network"),
    ("process or container killed, paused, stopped or restarted", "process"),
    ("system call errors or delays", "syscall"),
    ("connection pool / session / connection count exhausted", "connection"),
    ("database internal contention (lock contention, transaction retry storm, schema change, checkpoint storm)",
     "db_internal"),
    ("application code defect", "code"),
]
_KEYWORDS = [("cpu", "cpu"), ("memory", "memory"), ("space", "disk_space"), ("disk", "disk_io"), ("i/o", "disk_io"),
             ("network", "network"), ("latency", "network"), ("packet", "network"), ("socket", "network"),
             ("process", "process"), ("container", "process"), ("system call", "syscall"), ("syscall", "syscall"),
             ("connection", "connection"), ("session", "connection"), ("lock", "db_internal"),
             ("transaction", "db_internal"), ("database", "db_internal"), ("code", "code")]
REASON_DESC = {}   # 封闭世界模式：候选原因 = 统一机制集的 mechanism_id，这里放每个 ID 的说明


_SPACE = ContextVar("rca_agent_label_space", default=None)

def _reason_pairs():
    space = _SPACE.get()
    return [(mid, cat) for mid, cat in space["category"].items()] if space else REASONS

def _reason_desc():
    space = _SPACE.get()
    return {mid: space.get('descriptions', {}).get(mid, mid) for mid in space["category"]} if space else {}


def reason_of(reason):
    """回复里的原因 → 候选原因：精确匹配，否则（封闭世界）取以 mechanism_id 开头的写法，如 "cpu_saturation: ..."。"""
    low = reason.lower()
    return next((r for r, _ in _reason_pairs() if r.lower() == low), None) or \
        next((r for r, _ in _reason_pairs() if _reason_desc() and low.startswith(r)), None)


def entities(cols):
    seen = []
    for c in cols:
        e = c.split("|", 1)[0]
        if "|" in c and e not in seen:
            seen.append(e)
    return seen


def basic_prompt(g, files):
    """返回 (schema, cand)：对应原仓库 basic_prompt_*.py 的 schema 与 cand。"""
    comps = []
    if "metric" in files:
        cols = list(pd.read_csv(files["metric"], nrows=0).columns)
        comps = entities(cols[1:])
    cand = "## POSSIBLE ROOT CAUSE REASONS:\n\n" + "\n".join(
        f"- {r}" + (f" ({_reason_desc()[r]})" if r in _reason_desc() else "") for r, _ in _reason_pairs()) + \
        "\n\n## POSSIBLE ROOT CAUSE COMPONENTS:\n\n"
    if comps:
        cand += "\n".join(f"- {c}" for c in comps)
    else:
        cand += "- any host / node / service name that appears in the log lines (e.g., the field after the timestamp)"

    sch = ["## TELEMETRY DIRECTORY STRUCTURE:", "",
           "- You can access the telemetry directory of the target system: `telemetry/`.", ""]
    sch.append("- This directory contains: " + ", ".join(f"`telemetry/{p.name}`" for p in files.values())
               + ". There is no trace data for this system.")
    sch += ["", "## DATA SCHEMA", ""]
    k = 1
    if "log" in files:
        head = [ln.rstrip("\n")[:300] for _, ln in zip(range(3), open(files["log"], encoding="utf-8", errors="replace"))]
        sch += [f"{k}.  **Log File** `log.txt` (plain text, one log record per line, collected from all nodes / services; "
                "most lines start with a timestamp followed by the host or service name; formats differ by component):", "",
                "    ```text", *("    " + h for h in head), "    ```", ""]
        k += 1
    if "metric" in files:
        cols = list(pd.read_csv(files["metric"], nrows=0).columns)
        row = open(files["metric"], encoding="utf-8").readlines()[1].strip().split(",")
        sch += [f"{k}.  **Metric File** `metric.csv` (wide table: column `time` is a Unix timestamp in seconds; each other "
                f"column is one KPI named `component|kpi_name`; {len(cols) - 1} KPI columns in total):", "",
                "    ```csv", "    " + ",".join(cols[:6]) + ",...", "    " + ",".join(row[:6]) + ",...", "    ```", ""]
    sch += [cand, "", "## CLARIFICATION OF TELEMETRY DATA:", "",
            "1. The target system is a distributed system (a database cluster or a microservice application).", "",
            "2. All timestamps are in UTC. Metric timestamps are in seconds; log timestamps are ISO-8601 strings "
            "(some components use their own formats).", "",
            "3. Only the telemetry around the alert is provided; the whole files cover the time range given in the issue."]
    return "\n".join(sch), cand

# ---------------------------------------------------------------------------
# LLM 与沙箱内核
# ---------------------------------------------------------------------------
class APIError(Exception):
    pass


class Overflow(Exception):
    pass


class LLM:
    def __init__(self):
        self.usage = []
        self.config = current_config()

    def _post(self, messages):
        try:
            return chat(messages, max_tokens=MAX_TOKENS, config=self.config)
        except OverflowError as exc:
            raise Overflow(str(exc)) from exc

    def __call__(self, messages):
        for k in range(3):
            ex = ThreadPoolExecutor(1)
            try:
                content, u = ex.submit(self._post, messages).result(timeout=CALL_TIMEOUT)
                self.usage.append(u)
                return content
            except (requests.RequestException, KeyError, FuturesTimeout) as e:
                if k == 2:
                    raise APIError(repr(e))
                time.sleep(30 * (k + 1))
            finally:
                ex.shutdown(wait=False)


# ---------------------------------------------------------------------------
# executor.py / controller.py 的移植
# ---------------------------------------------------------------------------
def execute_act(instruction, background, history, kernel, llm, log):
    if not history:
        history = [{"role": "system", "content": EXEC_SYSTEM.format(rule=EXEC_RULE, background=background, format=EXEC_FORMAT)}]
    code_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    code, result, retry_flag = "", "", False
    history.extend([{"role": "user", "content": instruction}])
    prompt = history.copy()
    note = [{"role": "user", "content": f"Continue your code writing process following the rules:\n\n{EXEC_RULE}\n\nResponse format:\n\n{EXEC_FORMAT}"}]
    for _ in range(2):
        if not retry_flag:
            response = llm(prompt + note)
        else:
            response = llm(prompt)
            retry_flag = False
        m = re.search(code_pattern, response)
        code = m.group(1).strip() if m else response.strip()
        if "import matplotlib" in code or "import seaborn" in code:
            prompt.append({"role": "assistant", "content": code})
            prompt.append({"role": "user", "content": "You are not permitted to generate visualizations. If the instruction requires visualization, please provide the text-based results."})
            continue
        ok, result, error = kernel.run(code)
        log["cells"] += 1
        if ok:
            result = result.strip()
            if len(TOKENIZER.encode(result, disallowed_special=())) > 16384:
                log["oversize"] += 1
                continue
            mm = re.search(r"\[(\d+)\s+rows\s+x\s+\d+\s+columns\]", result)
            if mm and int(mm.group(1)) > 10:
                result += "\n\n**Note**: The printed pandas DataFrame is truncated due to its size. Only **10 rows** are displayed, which may introduce observation bias due to the incomplete table. If you want to comprehensively understand the details without bias, please ask Executor using `df.head(X)` to display more rows."
            history.extend([{"role": "assistant", "content": code},
                            {"role": "user", "content": EXEC_SUMMARY.format(result=result)}])
            answer = llm(history)
            history.extend([{"role": "assistant", "content": answer}])
            return code, EXEC_CONCLUSION.format(answer=answer, result=result), True, history
        log["cell_errors"] += 1
        result = error
        prompt.append({"role": "assistant", "content": code})
        prompt.append({"role": "user", "content": f"Execution failed:\n{result}\nPlease revise your code and retry."})
        retry_flag = True
    err = "The Executor failed to complete the instruction, please re-write a new instruction for Executor."
    history.extend([{"role": "assistant", "content": err}])
    return err, err, True, history


def control_loop(obj, schema, cand, kernel, llm, log, max_steps=MAX_STEP):
    prompt = [{"role": "system", "content": CTRL_SYSTEM.format(objective=obj, format=CTRL_FORMAT, agent=AGENT_RULES,
                                                               background=schema)},
              {"role": "user", "content": "Let's begin."}]
    history, trajectory = [], []
    final = CTRL_SUMMARY.format(objective=obj, cand=cand)
    for step in range(max_steps):
        note = [{"role": "user", "content": f"Continue your reasoning process for the target issue:\n\n{obj}\n\nFollow the rules during issue solving:\n\n{AGENT_RULES}.\n\nResponse format:\n\n{CTRL_FORMAT}"}]
        response_raw = llm(prompt + note)
        try:
            if "```json" in response_raw:
                response_raw = re.search(r"```json\n(.*)\n```", response_raw, re.S).group(1).strip()
            if '"analysis":' not in response_raw or '"instruction":' not in response_raw or '"completed":' not in response_raw:
                log["format_errors"] += 1
                prompt.append({"role": "assistant", "content": response_raw})
                prompt.append({"role": "user", "content": "Please provide your analysis in requested JSON format."})
                continue
            response = json.loads(response_raw)
            instruction, completed = response["instruction"], response["completed"]
            log["steps"] = step + 1
            if str(completed) == "True":
                prompt.append({"role": "assistant", "content": response_raw})
                prompt.append({"role": "user", "content": final})
                answer = llm(prompt)
                prompt.append({"role": "assistant", "content": answer})
                return answer, trajectory, "completed", prompt
            code, result, _, history = execute_act(instruction, schema, history, kernel, llm, log)
            trajectory.append({"step": step + 1, "analysis": response.get("analysis"), "instruction": instruction,
                               "code": code, "result": result[:4000]})
            prompt.append({"role": "assistant", "content": response_raw})
            prompt.append({"role": "user", "content": f"{result}"})
        except (APIError, Overflow):
            raise
        except Exception as e:   # noqa: BLE001  原代码：任何解析异常都回传给 Controller 继续
            log["format_errors"] += 1
            prompt.append({"role": "assistant", "content": response_raw})
            prompt.append({"role": "user", "content": f"{str(e)}\nPlease provide your analysis in requested JSON format."})
    final = final.replace("Now, you have decided to finish your reasoning process. ",
                          "Now, the maximum steps of your reasoning have been reached. ")
    if prompt[-1]["role"] == "user":
        prompt[-1]["content"] = final
    else:
        prompt.append({"role": "user", "content": final})
    answer = llm(prompt)
    prompt.append({"role": "assistant", "content": answer})
    return answer, trajectory, "max_step", prompt


def to_predictions(answer):
    """按原格式解析最终 JSON（"root cause component" / "root cause reason"）。"""
    m = re.search(r"```json\n(.*)\n```", answer, re.S)
    text = m.group(1) if m else answer
    pairs = []
    try:
        obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
        for v in obj.values():
            if isinstance(v, dict):
                pairs.append((str(v.get("root cause component", "")).strip(), str(v.get("root cause reason", "")).strip()))
    except ValueError:   # 原格式里常带尾逗号 / 注释：退回逐条正则
        comps = re.findall(r'"root cause component"\s*:\s*"([^"]*)"', text)
        reasons = re.findall(r'"root cause reason"\s*:\s*"([^"]*)"', text)
        pairs = list(zip(comps + [""] * (len(reasons) - len(comps)), reasons))
    out = []
    for comp, reason in pairs[:4]:
        if not reason:
            continue
        r = reason_of(reason)
        cat = dict(_reason_pairs()).get(r) or next((c for kw, c in _KEYWORDS if kw in reason.lower()), "unknown")
        out.append({"type": cat, "headline": reason, "description": f"{reason}（类别 {cat}，位置 {comp}）。",
                    "location": comp, "evidence_templates": []} | ({"mechanism_id": r or reason} if _reason_desc() else {}))
    return out


def from_text(answer, comps):
    """兜底：最终回复没给出要求的 JSON（沿用控制器格式、或重复生成到 max_tokens）时，按出现顺序取文中逐字出现的候选原因。
    回复是控制器 JSON 时只看 instruction 字段（那里是它的结论，analysis 里常有被排除的假设）。"""
    text = answer
    try:
        text = str(json.loads(answer[answer.index("{"): answer.rindex("}") + 1])["instruction"])
    except (ValueError, KeyError, TypeError):
        pass
    low = text.lower()
    hits = sorted((low.find(r.lower()), r, c) for r, c in _reason_pairs() if r.lower() in low)
    loc = next((c for c in sorted(comps, key=lambda c: low.find(c.lower()) if c.lower() in low else 1 << 30)
                if c.lower() in low), "")
    out = []
    for pos, reason, cat in hits[:4]:
        ctx = text[max(0, pos - 200): pos + len(reason) + 100]
        out.append({"type": cat, "headline": reason, "description": f"{reason}（类别 {cat}，位置 {loc}）。{ctx}",
                    "location": loc, "evidence_templates": []} | ({"mechanism_id": reason} if _reason_desc() else {}))
    return out



def diagnose(log=None, metrics=None, *, label_space=None, max_steps=MAX_STEP,
             kernel_factory=None, kernel_image="profrca-kernel:1", cell_timeout=CELL_TIMEOUT):
    """Run the original controller/executor against only the supplied telemetry.
    The default executor is a network-disabled Docker container. A host may
    inject its own *isolated* kernel factory implementing run(code) and close().
    """
    import tempfile
    from contextlib import ExitStack
    from datetime import datetime, timezone

    if not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_STEP:
        raise ConfigurationError("max_steps must be an integer from 1 to 25")
    if label_space == "mechanisms":
        from ...pipeline import closed_label_space
        label_space = closed_label_space()
    if label_space is not None and (not isinstance(label_space, dict) or "category" not in label_space):
        raise ConfigurationError("label_space must be None, mechanisms, or a custom category mapping")
    token = _SPACE.set(label_space)
    llm, kernel = LLM(), None
    stats = {"steps": 0, "cells": 0, "cell_errors": 0, "oversize": 0, "format_errors": 0}
    try:
        with tempfile.TemporaryDirectory(prefix="profrca-agent-") as temporary, ExitStack() as cleanup:
            wd = Path(temporary)
            (wd / "telemetry").mkdir()
            files = {}
            if log is not None:
                files["log"] = wd / "telemetry" / "log.txt"
                shutil.copyfile(log, files["log"])
            if metrics is not None:
                files["metric"] = wd / "telemetry" / "metric.csv"
                metrics.to_csv(files["metric"], index=False)
            schema, cand = basic_prompt({}, files)
            if metrics is not None:
                start, end = (datetime.fromtimestamp(float(x), timezone.utc).isoformat()
                              for x in (metrics["time"].min(), metrics["time"].max()))
                obj = (f"Between {start} and {end} (UTC), the system experienced failures. "
                       "The exact number of failures is unknown. Identify the root cause components and reasons.")
            else:
                obj = ("Identify the root cause components and reasons of failures in the provided logs. "
                       "The fault window and the number of failures are unknown; use only the input observations.")
            kernel = (kernel_factory(wd) if kernel_factory is not None
                      else DockerKernel(wd, image=kernel_image, timeout=cell_timeout))
            cleanup.callback(kernel.close)
            ok, _, error = kernel.run(INIT_CODE)
            if not ok:
                raise RuntimeError("RCA-Agent kernel initialization failed")
            answer, trajectory, end, prompt = control_loop(obj, schema, cand, kernel, llm, stats, max_steps)
            preds, parse = to_predictions(answer), "json"
            if not preds:
                prompt.append({"role": "user", "content": "Please provide your final answer in the requested JSON format."})
                answer2 = llm(prompt)
                preds, parse = to_predictions(answer2), "json_reask"
                comps = entities(list(metrics.columns)[1:]) if metrics is not None else []
                if not preds:
                    preds, parse = from_text(answer2, comps) or from_text(answer, comps), "text_fallback"
                answer += "\n\n=== re-ask ===\n\n" + answer2
            return {"root_causes": preds, "usage": llm.usage, "route": "rca_agent_controller_executor",
                    "error": None if preds else "no root cause in final answer", "end": end, "parse": parse,
                    "trajectory": trajectory, "final_answer": answer, **stats}
    finally:
        _SPACE.reset(token)
