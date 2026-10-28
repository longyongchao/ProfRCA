"""命令行：python -m profrca --log app.log [--metrics m.csv] [--tools] [--open-world] [--details]

模型服务从环境变量读取：DEEPSEEK_API_BASE、DEEPSEEK_MODEL（或 FAST_RCA_MODEL）、DEEPSEEK_API_KEY。
有监督工具命中时不调用模型，不需要这些变量。
"""
import argparse
import json
import sys

from .pipeline import diagnose, load_tools


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m profrca", description="ProfRCA 根因诊断")
    ap.add_argument("--log", help="日志 TXT 文件")
    ap.add_argument("--metrics", help="指标宽表 CSV（首列 time，其余列 实体|指标）")
    ap.add_argument("--tools", action="store_true", help="启用 models/ 下的有监督指标工具")
    ap.add_argument("--models-dir", help="工具模型目录（默认包内 models/）")
    ap.add_argument("--open-world", action="store_true",
                    help="不用封闭 30 机制清单，改用一级类别白名单（宿主项目默认口径）")
    ap.add_argument("--details", action="store_true", help="输出中保留完整画像")
    a = ap.parse_args(argv)
    if not a.log and not a.metrics:
        ap.error("至少提供 --log 或 --metrics")
    tools = load_tools(a.models_dir) if a.tools else None
    out = diagnose(a.log, a.metrics, tools=tools, label_space=None if a.open_world else "mechanisms")
    if not a.details:
        out.pop("profile", None)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    return 0 if out.get("root_causes") else 1


if __name__ == "__main__":
    sys.exit(main())
