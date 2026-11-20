"""命令行：python -m profrca --log app.log [--metrics m.csv] [--tools] [--open-world] [--details]

模型服务从环境变量读取：DEEPSEEK_API_BASE、DEEPSEEK_MODEL（或 FAST_RCA_MODEL）、DEEPSEEK_API_KEY。
有监督工具命中时不调用模型，不需要这些变量。
"""
import argparse
import json
import sys

from .api import diagnose, DiagnosisConfig
from .pipeline import load_tools
from .factory import list_methods


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m profrca", description="ProfRCA 根因诊断")
    ap.add_argument("--method", choices=list_methods(), default="profrca", help="选择诊断方法")
    ap.add_argument("--list-methods", action="store_true", help="列出方法并退出")
    ap.add_argument("--log", help="日志 TXT 文件")
    ap.add_argument("--metrics", help="指标宽表 CSV（首列 time，其余列 实体|指标）")
    ap.add_argument("--tools", action="store_true", help="启用 models/ 下的有监督指标工具")
    ap.add_argument("--models-dir", help="工具模型目录（默认包内 models/）")
    ap.add_argument("--open-world", action="store_true",
                    help="不用封闭 30 机制清单，改用一级类别白名单（宿主项目默认口径）")
    ap.add_argument("--details", action="store_true", help="输出中保留完整画像")
    ap.add_argument("--lines", type=int, default=0, metavar="K",
                    help="每个候选附上所引用模板 T# 的行号（每个模板最多 K 行，优先取变点后）")
    ap.add_argument("--metric-format", choices=["auto", "wide", "long", "dbot_long"], default="auto")
    ap.add_argument("--time-col", default="time", help="宽表时间列名")
    ap.add_argument("--entity", default="PostgreSQL", help="D-Bot 长表转换后的实体名")
    ap.add_argument("--output", help="另存诊断 JSON 文件")
    ap.add_argument("--max-steps", type=int, default=25, help="RCA-Agent controller 步数上限")
    ap.add_argument("--kernel-image", default="profrca-kernel:1", help="RCA-Agent Docker 内核镜像")
    a = ap.parse_args(argv)
    if a.list_methods:
        print("\n".join(list_methods()))
        return 0
    if not a.log and not a.metrics:
        ap.error("至少提供 --log 或 --metrics")
    if a.method != "profrca" and (a.tools or a.models_dir or a.lines):
        ap.error("--tools / --models-dir / --lines 只适用于 profrca")
    options = {}
    if a.method == "profrca":
        options = dict(tools=load_tools(a.models_dir) if a.tools else None,
                       label_space=None if a.open_world else "mechanisms", line_refs=a.lines)
    elif a.method == "rca_agent":
        options = dict(max_steps=a.max_steps, kernel_image=a.kernel_image,
                       label_space=None if a.open_world else "mechanisms")
    try:
        out = diagnose(a.log, a.metrics, method=a.method,
                       config=DiagnosisConfig(metric_format=a.metric_format, time_col=a.time_col,
                                              entity=a.entity, include_details=a.details), **options)
    except ValueError as exc:
        ap.error(str(exc))
    if not a.details:
        out.pop("profile", None)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1, default=str)
    print()
    if a.output:
        from pathlib import Path
        Path(a.output).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0 if out.get("root_causes") else 1


if __name__ == "__main__":
    sys.exit(main())
