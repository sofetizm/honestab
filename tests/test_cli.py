import json

from honestab.cli import main


def test_example_and_analyze(tmp_path, capsys):
    data = tmp_path / "exp.csv"
    assert main(["example", str(data), "--users", "20000", "--revenue-lift", "0.3"]) == 0
    capsys.readouterr()
    assert main(["analyze", str(data), "--config", str(data.with_suffix(".yaml")), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "КАТИТЬ" and out["srm"]["ok"]
    assert [m["role"] for m in out["metrics"]] == ["primary", "secondary", "secondary", "guardrail"]
    report = tmp_path / "r.md"
    main(["analyze", str(data), "--config", str(data.with_suffix(".yaml")), "--out", str(report)])
    assert report.read_text().startswith("## Решение")


def test_json_config_and_errors(tmp_path, capsys):
    data = tmp_path / "exp.csv"
    main(["example", str(data), "--users", "2000"])
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"metrics": [{"name": "r", "column": "revenue"}]}))
    assert main(["analyze", str(data), "--config", str(cfg)]) == 2
    assert "primary" in capsys.readouterr().err


def test_design_and_srm(capsys):
    assert main(["design", "--kind", "proportion", "--baseline", "0.05", "--mde-rel", "0.05",
                 "--daily", "20000"]) == 0
    out = capsys.readouterr().out
    assert "122,121" in out and "13 days" in out
    assert main(["design", "--std", "1", "--mde", "0.1"]) == 0
    assert main(["srm", "5000", "5000"]) == 0
    assert main(["srm", "50000", "48000"]) == 1
