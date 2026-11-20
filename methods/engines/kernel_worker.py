# Adapted packaging: this worker runs only inside the dedicated container.
#!/usr/bin/env python3
"""rca_agent_full.py 的代码执行子进程：一个有状态的 IPython shell（对应 OpenRCA 的 InteractiveShellEmbed），
在 sandbox-exec 里运行（禁网、只能写 case 临时目录、读不到数据集里的金标准）。

协议：stdin 每行一个 JSON {"code": ...}；对每条在协议 fd 上回一行 {"success", "result", "error"}。
result 与 OpenRCA 相同，取 str(exec.result)（最后一个表达式的值），print 的输出不回传。
"""

import json
import os
import sys
import traceback

proto = os.fdopen(os.dup(1), "w")
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 1)   # 被执行代码的 print / displayhook 输出丢弃，避免污染协议
os.dup2(devnull, 2)

from IPython.core.interactiveshell import InteractiveShell  # noqa: E402
from traitlets.config import Config  # noqa: E402

cfg = Config()
cfg.HistoryManager.enabled = False
shell = InteractiveShell.instance(config=cfg)

for line in sys.stdin:
    code = json.loads(line)["code"]
    try:
        r = shell.run_cell(code)
        if r.success:
            # 截到 1M 字符只为限制管道 / tiktoken 开销：远超 16384 token，控制器照样按「过长」丢弃
            out = {"success": True, "result": str(r.result)[:1_000_000], "error": None}
        else:
            e = r.error_in_exec or r.error_before_exec
            out = {"success": False, "result": "", "error": "".join(traceback.format_exception(type(e), e, e.__traceback__))}
    except BaseException as e:   # noqa: BLE001  被执行代码可能 raise SystemExit 等
        out = {"success": False, "result": "", "error": repr(e)}
    proto.write(json.dumps(out) + "\n")
    proto.flush()
