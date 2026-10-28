"""训练公共部分与通用训练器：用合成 case 目录跑通 multiclass / service 两种模式，不依赖任何 benchmark 数据。"""
import json

import numpy as np
import pandas as pd
import pytest

from profrca import diagnose, load_tools
from profrca.train import common as T
from profrca.train import train_benchmark

pytest.importorskip("lightgbm")

LABELS = {"cpu": "host|cpu_usage", "memory": "host|mem_used", "disk_io": "host|disk_await"}


def _series(rng, jump):
    x = rng.normal(10, 1, 60)
    if jump:
        x[30:] += 25
    return x


def _write(d, cid, wide, roots, **attrs):
    (d / cid).mkdir(parents=True)
    wide.to_csv(d / cid / "metrics.csv", index=False)
    (d / cid / "gold.json").write_text(json.dumps({"root_causes": roots, "benchmark_attrs": attrs}))


@pytest.fixture
def multiclass_dir(tmp_path):
    rng = np.random.default_rng(0)
    for n in range(60):
        lab = sorted(LABELS)[n % 3]
        wide = pd.DataFrame({"time": np.arange(60)} | {c: _series(rng, l == lab) for l, c in LABELS.items()})
        _write(tmp_path / "data", f"c{n:03d}", wide, [{"native_label": lab}], group=f"g{n % 4}")
    return tmp_path


@pytest.fixture
def service_dir(tmp_path):
    rng = np.random.default_rng(1)
    for n in range(120):
        lab, root = ("cpu", "mem")[n % 2], "abcd"[n % 4]
        cols = {"time": np.arange(60)}
        for e in "abcd":
            for k in ("cpu", "mem", "latency"):
                cols[f"{e}|{k}"] = _series(rng, (e == root and k == lab) or k == "latency")
        _write(tmp_path / "data", f"s{n:03d}", pd.DataFrame(cols), [{"label": lab, "location": root}],
               system=f"sys{n % 3}")
    return tmp_path


def test_dbot_and_sampling():
    assert T.dbot(["a", "b"], ["a", "b"]) == 1.0
    assert T.dbot(["a", "x"], ["a", "b"]) == pytest.approx(0.45)
    assert T.dbot(["x"], ["a"]) == 0.0
    labs = [["a"], ["a"], ["b"], ["a", "b"], ["c"]]
    got = T.sample_k(range(5), lambda i: labs[i], 1, np.random.default_rng(0))
    assert {l for i in got for l in labs[i]} == {"a", "b", "c"} and len(got) <= 3
    assert T.sample_k([1, 2], lambda i: labs[i], None, None) == [1, 2]


def test_folds_keep_groups_together():
    items = [{"gold": {"root_causes": [{"label": "x"}], "benchmark_attrs": {"g": g}}} for g in "aabbbcd"]
    logo = T.folds(items, "logo:benchmark_attrs.g")
    assert [f for f, _ in logo] == ["a", "b", "c", "d"]
    for spec in ("group2:benchmark_attrs.g", "random3"):
        tests = [t for _, t in T.folds(items, spec)]
        assert sorted(i for t in tests for i in t) == list(range(7))
    for _, t in T.folds(items, "group2:benchmark_attrs.g"):
        assert not ({2, 3, 4} & t) or {2, 3, 4} <= t                    # 同组整体落在同一折


def test_multiclass_train_register_and_route(multiclass_dir, monkeypatch):
    from profrca import fast_rca
    monkeypatch.setattr(fast_rca, "diagnose_profile", lambda *a, **k: {"root_causes": [], "usage": []})  # 不调模型
    models = multiclass_dir / "models"
    res = train_benchmark.main(["--data", str(multiclass_dir / "data"), "--name", "toy", "--workers", "1",
                                "--cv", "logo:benchmark_attrs.group", "--out", str(multiclass_dir / "runs"),
                                "--models-dir", str(models)])
    assert res["protocols"]["logo:benchmark_attrs.group"]["acc"] > 0.9
    assert "acc_mech" in res["protocols"]["logo:benchmark_attrs.group"]   # cpu/memory/disk_io 可映射到 30 机制
    man = json.loads((models / "MANIFEST.json").read_text())
    assert man["tools"]["toy"]["file"] == "toy_metric_tool.pkl"
    assert str(multiclass_dir) not in json.dumps(man)                     # 不记本机路径

    tools = load_tools(models)
    wide = pd.read_csv(multiclass_dir / "data" / "c001" / "metrics.csv")   # c001 = disk_io
    out = diagnose(metrics=wide, tools=tools)
    assert out["route"] == "metric_tool" and out["tool"] == "toy" and out["llm_calls"] == 0
    assert out["root_causes"][0]["mechanism_id"] == "disk_io_pressure"
    miss = diagnose(metrics=wide.rename(columns={"host|cpu_usage": "host|other"}), tools=tools)
    assert miss["route"] == "llm" and miss["llm_calls"] == 0               # schema 不够重合 → 回落 LLM 链路


def test_service_mode_k_per_label(service_dir):
    res = train_benchmark.main(["--data", str(service_dir / "data"), "--name", "svc", "--mode", "service",
                                "--workers", "1", "--cv", "logo:benchmark_attrs.system", "--k-per-label", "30",
                                "--seeds", "2", "--out", str(service_dir / "runs"), "--no-save"])
    r = res["protocols"]["logo:benchmark_attrs.system"]
    assert r["seeds"] == 2 and r["loc1"] > 0.8 and r["joint1"] > 0.6
    assert not (service_dir / "models").exists()
