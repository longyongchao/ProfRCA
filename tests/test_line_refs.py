"""T# 行号（opt-in）：默认画像与 prompt 不变；行号指回原日志中属于该模板的行，优先取变点后。"""
from profrca import fast_rca


def _log(tmp_path):
    lines = []
    for s in range(400):
        lines.append(f"2024-01-01 00:{s // 60:02d}:{s % 60:02d} INFO request ok id={s}")
        if s >= 200:
            lines.append(f"2024-01-01 00:{s // 60:02d}:{s % 60:02d} ERROR disk write timeout after {s} ms")
    p = tmp_path / "log.txt"
    p.write_text("\n".join(lines) + "\n")
    return p, lines


def test_default_profile_and_prompt_unchanged(tmp_path):
    p, _ = _log(tmp_path)
    entries = fast_rca.parse_log(str(p))
    base, refs = fast_rca.build_profile(entries), fast_rca.build_profile(entries, line_refs=3)
    assert all("lines" not in t for t in base["templates_stats"])
    assert fast_rca.render_profile(base)[0] == fast_rca.render_profile(refs)[0]


def test_line_refs_point_to_template_lines(tmp_path):
    p, lines = _log(tmp_path)
    prof = fast_rca.build_profile(fast_rca.parse_log(str(p)), line_refs=3)
    err = next(t for t in prof["templates_stats"] if "timeout" in t["key"])
    assert len(err["lines"]) == 3 and err["lines_scope"] == "after_onset"
    assert all("disk write timeout" in lines[n - 1] for n in err["lines"])          # 行号从 1 起算
    assert err["first_line"] == lines.index(next(l for l in lines if "timeout" in l)) + 1
    assert err["last_line"] == len(lines)

    _, chosen = fast_rca.render_profile(prof)
    i = next(i for i, (_, t) in enumerate(chosen) if t is err)
    preds = fast_rca._to_predictions({"root_causes": [{"category": "disk_io", "mechanism": "磁盘写 IO 压力",
                                                      "location": "", "evidence": [f"T{i}", "T999"]}]}, chosen)
    assert preds[0]["evidence_lines"] == {f"T{i}": err["lines"]}                   # 越界编号被丢弃
    base_chosen = fast_rca.render_profile(fast_rca.build_profile(fast_rca.parse_log(str(p))))[1]
    base = fast_rca._to_predictions({"root_causes": [{"category": "disk_io", "mechanism": "x", "location": "",
                                                     "evidence": [f"T{i}"]}]}, base_chosen)
    assert "evidence_lines" not in base[0]
