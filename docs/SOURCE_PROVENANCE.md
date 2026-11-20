# 源码来源与迁移差异

ProfRCA 基础提交：`ae0d7e81ddf4ffb7b4a7e5e3f41cb8825ddd87e0`。
三种基线取自主仓库固定提交：
`longyongchao/sundb-ai-ops@6ee89780d0d9cba70787a582debeb975e1550a1c`。
原主仓库仅只读获取，未修改。

| 本地文件 | 上游文件 |
|---|---|
| methods/engines/plain_llm.py | benchmark/unified/plain_llm_full.py |
| methods/engines/dbot.py | benchmark/unified/dbot_full.py |
| methods/engines/rca_agent.py | benchmark/unified/rca_agent_full.py |
| methods/engines/kernel_worker.py | benchmark/unified/rca_agent_kernel.py |

## 保留的算法行为

- LLM：读取完整原始日志和全量指标，不做画像压缩；JSON 解析失败重试一次；不静默截断超长输入。
- D-Bot：Chief DBA 描述告警 → 招募专家 → UCT 树搜索与反思/投票 → 根因提取 → 诊断摘要 → 交叉复核 → 精炼报告 → 结构化输出。
  仍采用最多 2 专家、2 simulations、2 vote candidates、24 节点深度，以及上游温度设置。
- RCA-Agent：原 Controller / Executor 提示词与两级循环、最多 25 controller 步、执行错误重写、输出长度门槛、结果总结和最终解析/追问。
- ProfRCA：指标工具优先、未命中回落画像+LLM 的流程未改；仍保留原始 pipeline 入口和训练脚本。

## 必要的工程迁移

- 移除三种基线对 full_eval/opt_eval、benchmark gold、宿主 server 路径及评测输出目录的导入。
- Case 的数据来源改为调用者传入的日志与已验证指标，不读取同目录标签或 metadata。
- 统一模型传输层支持逐请求配置，D-Bot 专家线程显式继承该次调用的配置；不改全局环境。
- D-Bot 的随机工具选择改为每案例局部 Random，防止并发请求互相改种子。
- RCA-Agent macOS sandbox-exec / select 管道实现改为跨平台 Docker + JSON-lines IPython worker；
  label space 使用请求内 ContextVar，退出恢复，不污染下一次调用。
- 增加正常输入验证、两种明确长表转换、工厂分发、统一输出、CLI 与可安装打包。
- LLM 基线输入中的指标 CSV 是统一校验后的全量宽表重新序列化，格式/排序可能变化；不删减数值观测、不降采样。
- 没有复制评分器、评测数据、真实日志、API 密钥或模型服务地址。

这些基线继承上游已经存在的离线适配：D-Bot 不使用作者未公开的知识库或在线数据库优化工具；
RCA-Agent 没有 trace 数据，其 schema/候选组件来自本次日志和指标。
因此应描述为“主项目已有离线基线的独立迁移”，不能声称重新严格复现了作者原始环境或论文准确率。

宿主仓库 Apache-2.0 许可证副本见 `third_party/sundb-ai-ops-LICENSE`。
原 D-Bot / OpenRCA 归属和适配说明保留在迁移文件头部；新的工程改动已在文件头与本说明中标注。
