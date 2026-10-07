"""CLI tests: demo + audit subcommands, exit codes, output formats."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from citecheck.cli import main


def run_module(*args):
    project_root = Path(__file__).resolve().parent.parent
    return subprocess.run(
        [sys.executable, "-m", "citecheck", *args],
        capture_output=True, text=True, cwd=project_root,
    )


def test_demo_exit_code_and_verdicts():
    proc = run_module("demo", "--format", "text")
    assert proc.returncode == 0, proc.stderr
    for verdict in ("SUPPORTED", "WEAK", "UNSUPPORTED", "CONTRADICTED"):
        assert verdict in proc.stdout


def test_demo_json_valid():
    proc = run_module("demo", "--format", "json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["summary"]["total"] == 4


def test_audit_writes_report_file(tmp_path):
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"items": [
        {"id": "x1", "claim": "Water boils at 100 degrees Celsius.",
         "citation_text": "Water boils at 100 degrees Celsius at sea level."}
    ]}))
    out = tmp_path / "report.md"
    assert main(["audit", str(src), "--out", str(out)]) == 0
    text = out.read_text()
    assert "SUPPORTED" in text
    assert "Water boils" in text


def test_audit_missing_file_exit_2():
    assert main(["audit", "/nonexistent/file.json"]) == 2


def test_audit_threshold_flag(tmp_path, capsys):
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"items": [
        {"id": "x1", "claim": "Water boils at 100 degrees Celsius.",
         "citation_text": "Water boils at 100 degrees Celsius at sea level."}
    ]}))
    assert main(["audit", str(src), "--format", "text", "--threshold", "1.5"]) == 0
    assert "supported 0/1" in capsys.readouterr().out
