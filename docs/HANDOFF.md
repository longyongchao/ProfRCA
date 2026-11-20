# 品璋：统一方法库阶段交接

日期：2026-11-20。分支：`feat/unified-method-api`。

## 已完成

1. 将 ProfRCA、直接 LLM、D-Bot、RCA-Agent 接入同一工厂和 `diagnose(..., method=...)` 入口。
2. 从真实输入结构出发，支持宽表及 D-Bot 539 / 通用长表转换；检查日志编码、指标时间/数值及缺失输入。
3. 返回统一的根因、证据、状态、成本字段，保留原 ProfRCA 字段和 pipeline 入口。
4. 将三种基线从宿主评测工程中独立出来，记录固定源码提交与离线适配边界。
5. RCA-Agent 使用独立 Docker IPython 执行器，去除 macOS-only 的 sandbox-exec 绑定。
6. 提供可安装包、CLI、合成样例、迁移说明及本交接文档。

## 验证记录

环境：Windows 主机 Python 3.12.14，numpy 1.26.4、pandas 2.3.3、LightGBM 4.6.0、scikit-learn 1.4.2。
Docker worker：Linux、Python 3.10.16、IPython 8.18.1、numpy 1.26.4、pandas 2.2.3、scipy 1.13.1。

- 全部测试（包括真实 Docker 测试）：**40 passed, 3 skipped**。
- 3 个 skip 是现有测试对宿主 `server.diagnose.*` shim 的检查；独立仓库没有宿主服务，按原测试约定跳过。
- 覆盖四种方法 × 仅日志/仅指标/两者都有；这部分使用脚本化 chat 与执行器协议替身，走真实算法控制流。
- 覆盖长表转换、非法输入、并发客户端隔离、错误脱敏、CLI、随包模型哈希/加载/路由和原训练脚本回归。
- 真实 Docker 测试：IPython 状态持续、CSV 读取、只读输入、只读根文件系统、非 root、禁网、无 API key、超时清理。
- `pip check` 通过；wheel 构建、独立目标目录安装、四方法导入及三套随包工具加载通过，第三方许可证已包含。

合成日志 + 16 行合成指标还各运行一次真实 DeepSeek-V4-Flash 联通验证：

| 方法 | 运行状态 | LLM 调用 | 输出 token | 耗时（秒） |
|---|---|---:|---:|---:|
| 直接 LLM | ok | 1 | 343 | 5.40 |
| ProfRCA（画像链路） | ok | 1 | 62 | 2.31 |
| D-Bot | ok | 30 | 6540 | 114.46 |
| RCA-Agent | ok | 11 | 2542 | 57.85 |

RCA-Agent 此次为控制费用设置 `max_steps=3`，实际执行了 3 个分析 cell，按 max_step 分支汇总返回。
默认仍为上游 25 步。这里验证可调用、执行和返回，不证明任务收敛、诊断准确率或性能提升；
例子是合成输入，没有做 benchmark 打分，也没有在重建的 539 个案例上跑实验。
服务地址和密钥只在本地读取，未提交；验证记录不包含真实数据、服务地址或凭据。

## 给太行的入口

```python
from profrca import diagnose, DiagnosisConfig
out = diagnose(log='log.txt', metrics='metric.csv', method='profrca',
               config=DiagnosisConfig(include_details=True), line_refs=3)
```

- 使用指南及全部字段：[UNIFIED_API.md](UNIFIED_API.md)。
- 方法选择：`factory.py`、`methods/adapters.py`。
- 根因与证据标准化：`api.py` 的 `_evidence()` 和统一输出部分。
- ProfRCA 算法仍在 `pipeline.py`、`fast_rca.py`；后续日志证据改进可在此继续。
- 目前指标工具命中仍走原来的零 LLM 分类路径，没有日志引用，会明确返回 evidence 不可用；
  “指标 + 日志画像并行，再融合诊断并补日志证据”属于太行接下来的改进，本阶段没有提前实现。

## 本阶段未做

未修改或上传 sundb-ai-ops 主系统、报告仓库或其他用户仓库；未创建 evaluation 仓库；
未更改主仓库的 submodule 指针；未训练 D-Bot 新工具；未跑完整评测；未填写 LaTeX 实验表。
推送的是专用分支，主分支不覆盖。合并后由维护者决定何时更新主系统引用的 ProfRCA 提交。
