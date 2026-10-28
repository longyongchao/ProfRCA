# ProfRCA：基于异常画像的可信高效根因诊断

ProfRCA 把一次故障的**日志 TXT** 和/或**指标宽表 CSV** 变成按可能性排序的根因候选（最多 4 个），每个候选带一级类别、机制描述、根因位置和可回查的证据编号。

它有两条互补的链路：

| 链路 | 输入 | 怎么诊断 | LLM 调用 | 适用场景 |
|---|---|---|---|---|
| **有监督指标工具** | 指标宽表 | LightGBM 分类器（随包发布 3 个） | 0 次 | 指标 schema 与某个已训练工具重合 ≥ 90% |
| **画像 + LLM 推理** | 日志 和/或 指标 | 统计压缩成「异常画像」，再做 2~3 轮 LLM 推理 | 1~3 次 | 新系统、没有训练数据、或只有日志 |

`diagnose()` 自动选择：先按指标列名路由到有监督工具，没有工具能覆盖时再走 LLM 链路。

本目录是一个**自洽的 Python 包**：只依赖 `requirements.txt` 中的第三方库，不 import 宿主项目的任何代码，可以单独 clone、测试和使用。

---

## 1. 目录结构

```
profrca/
├── __init__.py            对外接口：diagnose, load_tools
├── __main__.py            命令行：python -m profrca
├── pipeline.py            一次诊断的组合逻辑（工具路由 → 回落 LLM）
├── fast_rca.py            日志解析（Drain3）→ 异常画像 → LLM 多轮推理
├── metric_events.py       指标宽表 → 给 LLM 的紧凑文本（原始表或变化事件）
├── metric_classifier.py   有监督指标工具：特征、分类器、路由器
├── mechanisms.py          封闭世界 30 个根因机制，以及各 benchmark 原生标签到机制的映射
├── entries.py / preprocessor.py   日志条目数据结构与预处理
├── models/                已训练的 LightGBM 模型 + MANIFEST.json（文件哈希、训练口径、评测结果）
├── train/                 训练脚本：公共配方、通用 CLI、三个已发布工具的复现脚本
│   ├── common.py
│   ├── train_benchmark.py     ← 新 benchmark 用这个
│   ├── train_dbpa.py / train_rcaeval.py / train_wangqing.py
│   ├── label_info/            标签的中文说明（可选）
│   └── splits/                case 列表与留出集（只有 case ID）
├── tests/                 不依赖任何 benchmark 数据的单元测试
└── requirements.txt
```

---

## 2. 安装

需要 Python ≥ 3.9。

```sh
python -m venv venv && . venv/bin/activate
python -m pip install -r profrca/requirements.txt
```

**macOS 注意**：如果 `import lightgbm` 报 `libomp.dylib` 找不到，有两种解决办法：

- `brew install libomp`；
- 或者复用 scikit-learn 自带的 libomp：

```sh
export DYLD_LIBRARY_PATH=$PWD/venv/lib/python3.9/site-packages/sklearn/.dylibs
```

模型文件是 pickle，用 `requirements.txt` 中的库版本训练。跨大版本加载（比如 LightGBM 3.x 或 5.x）不保证兼容；遇到问题时，按第 7 节重新训练即可。

运行测试：

```sh
python -m pytest profrca/tests -q
```

在宿主项目里运行，会额外检查旧导入路径的兼容性；单独运行时，这几项会自动 skip。

---

## 3. 快速使用

### 3.1 命令行

```sh
# 只有指标：先尝试有监督工具（0 次 LLM 调用）
python -m profrca --metrics case/metrics.csv --tools

# 日志 + 指标，走 LLM 链路
python -m profrca --log case/log.txt --metrics case/metrics.csv

# 只有日志
python -m profrca --log case/log.txt --details      # --details 保留完整异常画像
```

各选项说明：

- `--tools`：启用 `models/` 下的工具；
- `--models-dir`：换成别的模型目录；
- `--open-world`：答案空间不限于 30 个机制，改用一级类别白名单。

输出 JSON 打印到标准输出。只要有根因候选，退出码就是 0。

### 3.2 Python API

```python
from profrca import diagnose, load_tools

tools = load_tools()                       # 读取 models/MANIFEST.json 中 role=production 的工具
out = diagnose(log="log.txt", metrics="metrics.csv", tools=tools)

out["route"]          # "metric_tool"（有监督工具命中）或 "llm"
out["root_causes"]    # 按可能性排序的候选
out["llm_calls"], out["output_tokens"]   # 成本：调用次数、LLM 输出 token 总数
```

参数说明：

- `metrics` 可以是 CSV 路径，也可以是 `pandas.DataFrame`。
- `tools=None` 表示不用有监督工具。
- `label_space`：
  - `"mechanisms"`（默认）：封闭的 30 个机制；
  - `None`：开放的一级类别白名单；
  - 也可以传自定义 dict：`{"prompt": 给 LLM 的答案空间说明, "category": {机制ID: 一级类别}}`。
- 其余关键字参数会原样传给 `fast_rca.diagnose_profile`，例如 `request_timeout`、`world_mode`、`ordering_mode`、`metrics_only_mode`、`candidate_diversity`。

每个候选的字段：

| 字段 | 含义 |
|---|---|
| `type` | 一级类别（cpu / memory / disk_io / network / db_internal / code …） |
| `mechanism_id` | 30 个机制之一（仅封闭世界） |
| `headline` / `description` | 中文的机制名与机制说明；工具路径还会附上置信度和主要依据的指标 |
| `location` | 根因位置（服务、实体）；未知时为 `null` |
| `evidence_templates` / `evidence_refs` | LLM 链路引用的日志模板编号 `T#` 和指标证据编号 `M#`，可以回查画像 |
| `label` / `probability` | 仅工具路径：原生标签与概率 |

### 3.3 LLM 服务配置（只有 LLM 链路需要）

| 环境变量 | 说明 |
|---|---|
| `DEEPSEEK_API_BASE` | OpenAI 兼容的 `/chat/completions` 服务地址 |
| `DEEPSEEK_MODEL` 或 `FAST_RCA_MODEL` | 模型名 |
| `DEEPSEEK_API_KEY` | 密钥：只从环境变量读取，不要写进代码或提交到仓库 |
| `FAST_RCA_MAX_TOKENS` | 画像里保留的模板数上限，默认 40 |
| `FAST_RCA_THINKING` | 设为 `1` 时开启模型思考模式（输出 token 更多） |
| `FAST_RCA_PARSER` | 日志解析器。默认 `v2`（Drain3）；`v1` 是宿主项目的旧 LILAC 链路，只用于复现历史结果 |

有监督工具命中时完全不调用模型，不需要配置这些变量。

---

## 4. 输入格式

**日志**：任意文本日志，一行一条记录。带时间戳的行会被切分到时间桶；没有时间戳的续行合并到上一条记录。

**指标宽表**：第一列是 `time`，其余每列是 `实体|指标`，例如：

```
time,carts|cpu,carts|mem,orders|cpu,orders|latency-90
0,0.12,512,0.30,85
1,0.13,515,0.31,86
```

- 实体通常是服务、主机或数据库实例。单实体系统也可以写成 `PostgreSQL|total_cpu_usage.usr`。
- 列名中没有 `|` 时，按最后一个下划线切分实体和指标。
- `time` 可以是时间戳，也可以是序号。

---

## 5. 方法细节

### 5.1 日志 → 异常画像（`fast_rca.py`，纯统计，不调用 LLM）

1. **模板化**：用 Drain3 把每行日志归为一个模板。模板中的变量会预先掩码，包括 UUID、IP、数字、时间，以及含变量的 JSON 值。多行日志（如堆栈）会并入上一条；解析结束后，每行重新匹配到最终的聚类，避免同一条语句因 Drain3 前后泛化程度不同而被拆成两个模板。
2. **时间桶**：把整段日志均分成 40 个时间桶，得到「模板 × 时间桶」计数矩阵。
3. **变点**：在总量序列上估计异常起点（onset）。对每个模板计算起点前后速率变化的泊松似然比（LLR），据此排序。
4. **画像**：只把以下内容交给 LLM，每项都有数量上限：
   - LLR 排名靠前的模板，以及 ERROR/FATAL 级模板和起点后新出现的模板；
   - 十类信号词（cpu / memory / disk_io / network …）的命中对照表；
   - 组件分布，以及数值参数的变化表（如耗时、队列长度的前后中位数）。

   每个模板分配一个编号 `T#`，带 20 格字符曲线和一条示例原文。

十几万行的日志通常被压成几千 token 的画像，原始日志不会整段发给模型。

### 5.2 指标 → 证据文本（`metric_events.py`）

- 序列较短、预算放得下时，保留降采样后的原始表（`mode=raw`）。
- 否则逐条序列做稳健归一化，用 PELT 检测变点，识别电平跳变、尖峰、计数器重置和状态切换；然后合并同步变化，输出 top-K 变化事件（`mode=events`）。
- 每条证据分配一个编号 `M#`。
- `relation_mode` 可选负载归一化、工作量回归残差等机制证据（opt-in），默认关闭。

### 5.3 LLM 推理（`diagnose_profile`）

- **第一轮**：输入画像和指标证据，输出最多 4 个候选。每个候选包含类别、具体机制、位置、引用的 `T#`/`M#`，以及不超过 20 字的理由。封闭世界下，机制必须取自 `mechanisms.py` 的 30 个 ID 之一。
- **放大（zoom，日志路径）**：把首轮候选引用的模板，和与之相关的未入选模板，补充更多上下文，再交给模型。
- **复核（review）**：检查以下几点，然后重新排序或改写：
  - 证据是否真实存在、是否支持该机制；
  - 是否把症状当成了原因；
  - 候选之间是否重复。

  指标-only 路径的复核通过 `metrics_only_mode` 显式开启。
- 返回的 `usage` 记录每次调用的 token 数。`pipeline.diagnose` 把其中的 `completion_tokens` 汇总为 `output_tokens`。评测时用它衡量成本；响应时间受服务负载影响，不作为指标。

### 5.4 有监督指标工具（`metric_classifier.py`）

| 类 | 用途 | 特征 |
|---|---|---|
| `MetricClassifier` | 单实体或实体固定的系统（数据库实例、单主机） | 每个指标的整段统计量，加 CUSUM 估计的变点前后稳健水平 |
| `MetricEnsemble` | 多个特征集的等权概率平均 | `v1`（原始量级）、`shape_v1`（尺度无关，跨硬件泛化更好）、`rel_v1`（两者合并） |
| `ServiceFaultClassifier` | 微服务（实体名随系统变化） | 两段式：先逐实体二分类给出根因实体，再在该实体上做故障类型多分类。特征只描述「该实体相对同伴有多异常」，不含实体名 |
| `MetricToolRouter` | 在多个工具间路由 | 只看指标列名：输入列与训练 schema 的重合度 ≥ 0.9 时启用，否则返回 `None`，由调用方回落到 LLM |

各类的共同点：

- 特征**不读注入窗口或标签**，只用整段序列。
- 每个工具最多给 4 个候选，概率 < 0.01 的丢弃。
- 原生标签经 `label_info[标签]["mechanism_id"]` 或 `mechanisms.NATIVE` 映射到 30 机制，同一机制只保留一次。

---

## 6. 随包发布的模型（`models/`）

`models/MANIFEST.json` 是模型清单：`load_tools()` 只根据它加载模型。每个条目记录以下信息：

- 文件名、sha256、字节数；
- 训练时间和库版本；
- `role`：`production` 会默认加载，`ablation` 只在消融实验中加载；
- 类名、标签、训练数据规模、训练脚本与选项；
- 评测结果。

| 工具 | 文件 | 类型 | 标签数 | 训练规模 |
|---|---|---|---|---|
| `dbpa` | `dbpa_metric_ens_v1.pkl` | MetricEnsemble（v1 + shape_v1） | 9 | 19,397 条记录 |
| `rcaeval` | `rcaeval_metric_clf_v1.pkl` | ServiceFaultClassifier | 11 | 697 个 case |
| `wangqing` | `wangqing_metric_ens_v1.pkl` | MetricEnsemble（v1 + shape_v1） | 6 | 101 个 case |
| `dbpa_v1` / `dbpa_rel_v1` / `dbpa_shape_v1` | `dbpa_metric_clf_*.pkl` | 单特征集（消融） | 9 | 同 dbpa |

训练配方完全相同，也没有调参：

- `lightgbm.LGBMClassifier(random_state=42, verbose=-1)`；
- 复合根因按标签各复制一行；
- 定位模型额外使用 `class_weight="balanced"`。

### 6.1 评测结果

Acc 统一采用 Top-K_GT 计分：K = 金标准根因个数，`max(0, (命中数 − 0.1 × 错误数) / K)`。

| 工具 | 口径 | 结果 |
|---|---|---|
| dbpa | 留一硬件配置（主口径） | Acc 0.9391，Top-1 0.9429（MySQL 0.9928，PostgreSQL 0.9204） |
| dbpa | 随机：全量减去 10% 子集训练，在子集上测 | Acc 0.9802（上界参考） |
| dbpa | **每标签 K=20 个训练记录，5 个种子** | **Acc 0.8734 ± 0.0043**，Top-1 0.8859 |
| rcaeval | 留一系统（主口径） | 根因实体 Top-1 0.8811，实体 + 类型联合 Top-1 0.5178，机制级 Acc 0.6339 |
| rcaeval | 分组 5 折（系统 × 服务 × 故障） | 实体 0.9194，联合 0.6325 |
| rcaeval | **每标签 K=20，5 个种子** | **机制级 Acc 0.5847 ± 0.0079** |
| wangqing | 留一组合（主口径） | Acc 0.6678，Top-1 0.9406（标签频次先验基线 0.1337） |
| wangqing | 留一轮（同分布上界） | Acc 0.9129 |
| wangqing | **每标签 K=20，5 个种子** | **Acc 0.7007 ± 0.0111**，Top-1 0.9465 |

**论文主口径**是表中加粗的行：封闭世界 30 机制，每个标签只给 20 个训练 case，用 5 个种子取均值。这模拟的是「新系统只积累了少量带标签故障」的情形。

### 6.2 泄漏说明

先说明一个分界：

- **评测数据泄漏**：随机划分时，同一场实验的多次重复可能分在训练和测试两侧。
- **模型能力**：模型在没见过的组、配置或系统上的表现。

两者必须分开看，所以主口径都按组留出。

- DBPA：只用注入前的行做留一硬件评测，Acc 是 0.720，全窗口是 0.937，标签先验是 0.271。可见部分可区分性来自实验配置本身，并非都来自故障信号。
- RCAEval：RE3 的 out-of-fold 结果能认出实验批次，所以以留一系统为主口径。
- 「留一轮」「随机」两种口径只作为上界参考。

---

## 7. 训练

### 7.1 新 benchmark：通用训练器 `train_benchmark`

先把数据整理成 **case 目录**，每个子目录对应一个故障 case：

```
mybench/
├── case_0001/
│   ├── metrics.csv      指标宽表（第 4 节格式）
│   └── gold.json        金标准
└── case_0002/ ...
```

`gold.json` 的最小格式（文件名也可以叫 `label.json`）：

```json
{
  "root_causes": [{"native_label": "cpu", "location": "orders"}],
  "benchmark_attrs": {"system": "ob", "hardware": "32_128"}
}
```

字段说明：

- `native_label`（或 `label`）是本 benchmark 的原生标签。复合故障在列表中写多个。
- `location` 只有 service 模式需要。
- 其余字段可以用作交叉验证的分组键，用点号路径引用，如 `benchmark_attrs.hardware`。

训练与评测：

```sh
python -m profrca.train.train_benchmark --data mybench --name mybench \
    --mode multiclass --cv logo:benchmark_attrs.hardware --k-per-label 20 --seeds 5
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--mode multiclass` | 整段指标统计 → 多分类。适合单实体、实体固定的系统。`--sets` 指定特征集，默认 `v1 shape_v1` 等权集成 |
| `--mode service` | 两段式，先定位实体再判故障类型。适合微服务。每个 case 只取第一个根因 |
| `--cv` | 交叉验证口径，可给多个：`logo:<键>` 留一组；`group<K>:<键>` 分组 K 折（多个键用逗号连接）；`random<K>` 随机 K 折（只作上界）。键可以是点号路径、`labels`（标签组合）或 `location` |
| `--k-per-label K --seeds S` | 每折训练集里每个标签只抽 K 个 case，重复 S 个种子 |
| `--exclude` | 留出集：这些 case 不进入任何训练集（交叉验证各折和最终产物都不用），但仍参与交叉验证测试 |
| `--cases` | 只读这些 case，并按给定顺序读取。LightGBM 的结果与样本顺序有关，复现时要固定顺序 |
| `--label-info` | 标签的中文说明：`{标签: {category, headline, description, mechanism_id}}`。不给时按 `mechanisms.NATIVE` 自动映射；映射不到的标签记为 `unknown` |
| `--role` | 写进 MANIFEST 的角色，默认 `production` |
| `--no-save` | 只评测，不保存模型 |

输出：

- `--out`（默认 `runs/<name>`）下的 `summary.json`：各口径的 acc / top1 / hit4 / acc_mech（以及 loc1 / joint1），多种子时附带标准差；
- 同目录下的 `rows_<口径>.jsonl`：逐 case 预测，不含指标数据；
- 用全部非留出 case 训练的最终模型，写入 `--models-dir`，并登记到 `MANIFEST.json`，其中只记录文件名，不记本机路径。

训练完成后，`load_tools()` 会自动加载新工具。只要指标列名能匹配，`diagnose()` 就会路由到它。

**建议**：

- 交叉验证一定要按「同一场实验 / 同一配置 / 同一系统」分组，否则同场重复会让结果虚高。
- 新标签如果不在 30 机制中，要么在 `--label-info` 里给出 `mechanism_id`，要么在开放世界口径（`label_space=None`）下使用。

### 7.2 复现三个已发布工具

数据位于宿主项目的 `benchmarks-data/closed_world_rca/<bench>/`，与上面的 case 目录格式一致。

```sh
# wangqing：留一组合 + 留一轮 + 先验基线
python -m profrca.train.train_wangqing --data benchmarks-data/closed_world_rca/wangqing

# RCAEval：service 模式，35 个留出 case 不进训练，留一系统
python -m profrca.train.train_rcaeval --data benchmarks-data/closed_world_rca/rcaeval

# DBPA：需要官方完整 DBPA_dataset（训练用全量减去 10% 评测子集）
python -m profrca.train.train_dbpa --full /path/to/DBPA_dataset \
    --dbpa benchmarks-data/DBPA --cases benchmarks-data/closed_world_rca/dbpa \
    --k-per-label 20 --seeds 5 --save-members
```

几点说明：

- 三个脚本都可以追加通用参数，如 `--no-save`，或用 `--models-dir /tmp/x` 输出到别处，避免覆盖随包模型。
- `splits/` 只存 case ID，固定了读取顺序和留出集。
- 复现核对结果：
  - 在本机用 `train_wangqing` / `train_rcaeval` 重训后，与随包模型逐 case 比对，预测完全一致（wangqing 101/101，RCAEval 抽查 120/120）。评测数字与 6.1 节一致。
  - DBPA 的官方全量数据不在本仓库。`train_dbpa` 只用评测子集做过代码路径冒烟测试，随包的 DBPA 模型由宿主项目原脚本基于全量数据训练。

---

## 8. 在宿主项目中使用

本包作为 git 子模块放在宿主项目根目录的 `profrca/` 下。宿主项目的旧导入路径通过**模块别名**指向本包：

```python
# server/diagnose/fast_rca.py（metric_events.py、metric_classifier.py 同理）
import sys
from profrca import fast_rca as _m
sys.modules[__name__] = _m
```

使用模块别名而不是 `from profrca.fast_rca import *`，有两个原因：

- `server.diagnose.fast_rca` 与 `profrca.fast_rca` 是**同一个模块对象**，对其中一个做 monkeypatch，或修改模块级常量，在另一个上同样生效；
- 引用 `server.diagnose.metric_classifier` 的旧 pickle 仍能加载。

`benchmark/unified/closed_world.py` 同样是 `profrca.mechanisms` 的别名。宿主代码可以直接这样调用：

```python
from profrca import diagnose, load_tools
```

首次 clone 宿主项目后，需要拉取子模块：

```sh
git submodule update --init profrca
```

---

## 9. 安全与数据

- 密钥只从环境变量读取，代码和 MANIFEST 中都不含密钥。
- LLM 链路只把**统计画像和指标证据**发给配置的模型端点：模板、计数曲线和少量示例行，不含整段原始日志。示例行仍可能包含业务信息，接入外部服务前要确认数据合规。
- `models/*.pkl` 由 DBPA、RCAEval、wangqing 数据训练，属于派生数据，发布前要确认数据使用协议。
- pickle 只能加载可信来源的文件，加载不受信任的 pickle 等同于执行任意代码。
- `runs/`、评测产物和原始数据不入库，`.gitignore` 已排除 `runs/`。

---

## 10. 测试

```sh
python -m pytest profrca/tests -q
```

| 测试文件 | 覆盖内容 |
|---|---|
| `test_train.py` | 用合成数据跑通 multiclass / service 两种训练模式：交叉验证分组、K 抽样、MANIFEST 登记（不记本机路径）、训练后路由到新工具；schema 不匹配时回落 LLM 链路（测试中替换为假模型，不发网络请求） |
| `test_models.py` | 随包模型的 sha256 与 MANIFEST 一致；按 role 加载；按 schema 路由，输出映射到 30 机制且不重复；宿主项目的旧导入路径是别名 |

测试不依赖任何 benchmark 数据，也不调用 LLM。
