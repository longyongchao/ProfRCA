"""随包发布的模型与 MANIFEST 一致、可加载、按 schema 路由；宿主项目的旧导入路径是同一模块对象。"""
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from profrca import diagnose, load_tools
from profrca.pipeline import MODELS_DIR
from profrca.train.common import sha256

pytest.importorskip("lightgbm")
MANIFEST = json.loads((MODELS_DIR / "MANIFEST.json").read_text(encoding="utf-8"))


def test_manifest_hashes_match_files():
    for name, ent in MANIFEST["tools"].items():
        assert sha256(MODELS_DIR / ent["file"]) == ent["sha256"], name
        assert ent["role"] in ("production", "ablation")


def test_load_tools_roles():
    assert sorted(load_tools().tools) == ["dbpa", "rcaeval", "wangqing"]
    assert len(load_tools(include_ablation=True).tools) == len(MANIFEST["tools"])
    assert list(load_tools(names=["dbpa"]).tools) == ["dbpa"]


def test_route_by_schema_only():
    """用模型自己的训练 schema 造一张随机宽表：应路由到该工具，且输出映射到 30 机制。"""
    tools = load_tools()
    rng = np.random.default_rng(0)
    m = tools.tools["dbpa"].members[0]
    wide = pd.DataFrame({"time": np.arange(25)} | {f"PostgreSQL|{c}": rng.normal(size=25) for c in sorted(m.metrics)})
    out = diagnose(metrics=wide, tools=tools)
    assert out["route"] == "metric_tool" and out["tool"] == "dbpa" and out["output_tokens"] == 0
    from profrca.mechanisms import CATEGORY
    assert all(rc["mechanism_id"] in CATEGORY for rc in out["root_causes"])
    assert len({rc["mechanism_id"] for rc in out["root_causes"]}) == len(out["root_causes"])


@pytest.mark.parametrize("old,new", [("server.diagnose.fast_rca", "profrca.fast_rca"),
                                     ("server.diagnose.metric_events", "profrca.metric_events"),
                                     ("server.diagnose.metric_classifier", "profrca.metric_classifier")])
def test_host_shims_are_aliases(old, new):
    try:
        mod = importlib.import_module(old)
    except ImportError:
        pytest.skip("不在宿主项目中")
    assert mod is importlib.import_module(new)
