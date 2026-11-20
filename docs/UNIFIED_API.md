# 四种诊断方法统一接口（交接说明）

本阶段只交付可独立运行的方法库。系统前后端、benchmark 批量评测、评分器、LightGBM 新数据集训练及 LaTeX 报告不在本次修改范围。

## 1. 安装与快速调用

在任意目录名下 clone 后均可安装，不要求文件夹必须命名为 profrca：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e '.[tools,baselines,test]'
python -m profrca --list-methods
```

Python >= 3.9；`tools` 安装随包指标工具所需 LightGBM 4.6.0 / scikit-learn 1.4.2；
`baselines` 安装 D-Bot 的 scipy 和 RCA-Agent 的 tiktoken。随包 pickle 只加载可信仓库中的文件。
原 requirements.txt 保留模型训练时的依赖锁定，Python 3.12 可使用上面的安装方式。

```python
from profrca import diagnose, DiagnosisConfig, LLMConfig, load_tools

config = DiagnosisConfig(
    llm=LLMConfig(timeout=300),  # 默认读取环境变量，不在配置文件/源码中写密钥
    include_details=True,
)
result = diagnose(log='examples/log.txt', metrics='examples/metric.csv',
                  method='profrca', config=config, tools=load_tools(), line_refs=3)
# method 可选 profrca / llm / dbot / rca_agent
print(result['status'], result['root_causes'], result['evidence'])
```

LLM 服务由 `DEEPSEEK_API_BASE`（含 /v1）、`DEEPSEEK_MODEL`、`DEEPSEEK_API_KEY` 配置。
`LLMConfig(base_url=..., model=..., api_key=...)` 可覆盖环境变量，配置按请求隔离，不修改进程环境；
同一服务中不同请求可使用不同模型。密钥字段不会出现在配置 repr 或结果中。
可通过 `LLMConfig(chat=callable)` 接入主系统已有客户端，callable 接收 messages、max_tokens、temperature、timeout，
返回 `(text, usage_dict)`。这也是测试替身的注入点；默认执行真实 HTTP 请求，没有假诊断回退。

```bash
python -m profrca --method profrca --log examples/log.txt --metrics examples/metric.csv --lines 3
python -m profrca --method llm --log examples/log.txt --metrics examples/metric.csv
python -m profrca --method dbot --log examples/log.txt --metrics examples/metric.csv
python -m profrca --method rca_agent --log examples/log.txt --metrics examples/metric.csv --open-world --details
```

各方法均接收仅日志、仅指标、日志+指标；至少提供其中之一。CLI 支持 `--output result.json`。

## 2. 真实输入格式与转换边界

### 日志

UTF-8 / UTF-8 BOM 文本文件；物理行号从 1 开始。ProfRCA 继续使用既有 Drain3 解析及续行合并；
LLM 基线阅读全文；D-Bot 保留上游的告警/专家工具视图；RCA-Agent 的 Executor 读取副本。
空文件、非法 UTF-8、含 NUL 的二进制文件在诊断前报 InputError。

### 指标宽表

```csv
time,host|cpu_usage,host|memory_used
0,10,100
1,12,101
2,90,102
3,92,103
```

时间列默认 `time`，可通过 DiagnosisConfig(time_col='timestamp') 或 CLI `--time-col timestamp` 指定。
时间支持数值秒/序号或 ISO-8601，后者转换成 UTC epoch 秒；不会自动猜测毫秒/纳秒单位。
其他列必须为数值指标，推荐 `实体|指标`，旧的下划线列名原样保留。
至少 4 个不同时刻；允许局部 NaN，不允许全空指标、Infinity、重复宽表时间、重复列名、标签/文字列。
输入排序但不修改调用方 DataFrame；不做插值，不截取真实故障窗口。

### D-Bot 539 长表

实际文件为 `timestamp,phase,metric_name,metric_value,unit,source_scope`。
自动识别 timestamp / metric_name / metric_value 三列；仅消费这三列，将列名改成 `PostgreSQL|指标名`；
实体名可由 `entity` / `--entity` 指定。phase、unit、source_scope 及其他 metadata 不进入模型。
同一时间同一指标的重复观测取中位数，不按 phase 选择故障窗口。

### 通用长表

支持 `ts,entity,metric,value`，转换成 `time,实体|指标...`，重复观测取中位数。
`metric_format='auto'` 只识别上述两种明确结构；也可显式传 wide / long / dbot_long。
其他长表应先由调用方转换，不能只靠 csv 扩展名猜测。

**金标准隔离由评测方负责**：接口不读取相邻的 gold.json、ground_truth.json、metadata.json；
不接收 gold 参数。调用者应只传可观测数据，不能把标签伪装成数值指标列。
本方法库不负责决定训练/测试划分或裁剪 baseline/fault/recovery 窗口。

## 3. 统一输出

`schema_version='profrca.diagnosis.v1'`，返回值为可序列化 dict：

| 字段 | 含义 |
|---|---|
| method | profrca / llm / dbot / rca_agent |
| status | ok / no_prediction / error / cancelled |
| root_causes | 按方法原始排序的候选；每项含 rank、type、mechanism_id、headline、description、location、证据字段 |
| evidence | 统一证据列表：log_lines / reference / reported_text |
| evidence_status / evidence_note | 当前方法是否提供证据及缺失原因 |
| route | 实际执行路线，如 metric_tool / llm / llm_full_text / dbot / rca_agent_controller_executor |
| llm_calls / prompt_tokens / output_tokens / seconds | 方法成本和耗时；不是准确率 |
| error / error_code | 执行错误；不含密钥及服务响应正文 |
| input_summary | 识别的指标格式、转换后行数和指标列 |

证据 `verified=True` 仅用于 ProfRCA 已解析模板映射回原始日志的行号。
模型陈述及未解析的 T#/M# 仅作报告引用，`verified=False`；不能当作已核验的原文。
D-Bot / RCA-Agent 可能没有原文引用；指标工具不会凭空生成日志引用，此时 evidence=[] 并给出说明。
候选保留原生的 evidence_templates / evidence_refs / evidence_lines，方便太行继续扩展证据提取。

机制答案空间保持算法原设定：ProfRCA 默认为 30 机制；LLM / D-Bot 是上游开放类别提示词，
mechanism_id 未提供时为 null；RCA-Agent 可选 label_space='mechanisms' 或 None。
**统一结构不代表统一打分口径**；后续 evaluator 需要显式定义跨方法的标签映射，不能伪造 mechanism_id。

输入错误/配置错误抛出 InputError / ConfigurationError（均继承 ValueError）。
执行错误返回 status=error 和 error_code；调用方必须检查 status，不能把错误结果当成功。
ProfRCA 原先的 route、coverage、tool、profile、root_causes 等字段继续保留；
`profrca.pipeline.diagnose` 原始入口不变；顶层 `profrca.diagnose` 新增统一字段与输入验证。

## 4. RCA-Agent 的独立执行器

RCA-Agent 保留 Controller → Executor → 代码执行 → 总结 → Controller 的真实循环。
原 macOS sandbox-exec 绑定替换为 Docker，不在主机上直接执行模型生成代码。

先启动 Docker 并显式构建镜像（运行时不会自动下载镜像）：

```bash
docker build -t profrca-kernel:1 -f docker/Dockerfile.kernel .
python -m profrca --method rca_agent --log examples/log.txt --metrics examples/metric.csv --details
```

容器禁网、非 root、只读根文件系统、去除 capabilities、限制 CPU/内存/PID；
仅挂载当前案例遥测与 worker，输入只读，不挂载仓库、标签、模型密钥或 Docker socket。
每次调用独立临时目录、独立容器和 IPython 状态；结束清理该实例，超时终止，不能继续读取迟到响应。
未启动 Docker 或镜像未构建时，返回 ExecutionUnavailable，绝不伪装为成功。
宿主已有安全执行服务时可传 `kernel_factory(workdir)`，提供 run(code) -> (success, result, error) 和 close()。
这个扩展点只用于受控执行器；不能替换成没有隔离的生产主机 exec。

RCA-Agent 的 tiktoken 第一次使用可能下载 cl100k_base 编码表；离线部署先预热其缓存：
`python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"`。
若运行环境会切换临时目录，预热和运行时均将 TIKTOKEN_CACHE_DIR 指向同一持久可写目录。

## 5. 验证和协作范围

```bash
python -m pytest tests -q
# 显式启用真实 Docker 隔离测试：
# Linux/macOS: PROFRCA_RUN_DOCKER=1 python -m pytest tests/test_docker_kernel.py -q
# PowerShell: $env:PROFRCA_RUN_DOCKER='1'; python -m pytest tests/test_docker_kernel.py -q
```

普通测试使用可预测的 chat/kernel 协议替身，运行真实算法控制流程，不测诊断准确率。
Docker 测试验证真实 IPython 执行、变量保留、文件隔离、禁网与清理。
完整付费模型评测、benchmark 批处理与训练放在后续独立 evaluation 工程。

太行接入时：保留顶层 API，在 methods/adapters.py 或 pipeline.py 改进 ProfRCA 内部证据流程；
补充 root_causes 中的证据即可由公共 API 转换；若需要新的证据类型，先扩展统一契约并补对应测试。
本次没有修改系统主仓库、其 submodule 指针或技术报告；合并后由维护者更新主项目依赖版本。
