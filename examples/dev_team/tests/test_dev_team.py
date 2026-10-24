import json
import os
from pathlib import Path
import signal
import time

import pytest

import dev_team
from dev_team import DeterministicDeveloper, develop, route_after_test


class FixedDeveloper:
    def __init__(self, test_code):
        self.test_code = test_code
        self.calls = 0

    def develop(self, design, requirement):
        self.calls += 1
        return "def add(a, b): return a + b\n", self.test_code


def assert_not_delivered(out, workspace):
    assert out["test_passed"] is not True
    assert out["deployment_config"] is None
    assert out["documentation"] is None
    assert not (workspace / "Dockerfile").exists()
    assert not (workspace / "README.md").exists()
    report = json.loads((workspace / "failure_report.json").read_text())
    assert report == out["failure_report"]
    assert report["status"] == out["status"]
    assert report["delivered"] is False


def test_offline_dev_team_passes_tests(tmp_path):
    out = develop("算术模块", dev=DeterministicDeveloper(), workspace_dir=str(tmp_path))
    assert out["test_passed"] is True, out["test_output"]
    assert out["status"] == "delivered"
    assert out["failure_report"] is None
    assert out["attempts"][0]["returncode"] == 0
    for filename in ("app.py", "app_test.py", "Dockerfile", "README.md"):
        assert (tmp_path / filename).exists()


def test_persistent_failure_exhausts_budget_without_delivery(tmp_path):
    dev = FixedDeveloper("def test_wrong(): assert False, 'still broken'\n")
    out = develop("算术模块", dev=dev, workspace_dir=str(tmp_path), max_iterations=3)
    assert out["status"] == "budget_exhausted"
    assert dev.calls == out["iteration"] == 3
    assert len(out["attempts"]) == 3
    assert all(attempt["test_status"] == "failed" for attempt in out["attempts"])
    assert "still broken" in out["failure_report"]["attempts"][0]["output"]
    assert_not_delivered(out, tmp_path)


@pytest.mark.parametrize("previous_run", ["delivered", "failed", "unrelated"])
def test_nonempty_workspace_is_rejected_without_modification(tmp_path, previous_run):
    if previous_run == "delivered":
        develop("算术模块", workspace_dir=str(tmp_path))
    elif previous_run == "failed":
        develop("算术模块", dev=FixedDeveloper("def test_wrong(): assert False\n"),
                workspace_dir=str(tmp_path), max_iterations=1)
    else:
        (tmp_path / "user_notes.txt").write_text("User-owned content", encoding="utf-8")
    contents_before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }
    dev = FixedDeveloper("def test_ok(): assert True\n")
    with pytest.raises(ValueError, match="workspace_dir"):
        develop("算术模块", dev=dev, workspace_dir=str(tmp_path))
    assert dev.calls == 0
    assert contents_before == {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in tmp_path.rglob("*") if path.is_file()
    }


def test_nonexistent_workspace_is_created(tmp_path):
    workspace = tmp_path / "new" / "workspace"
    out = develop("算术模块", workspace_dir=str(workspace))
    assert out["status"] == "delivered"
    assert (workspace / "Dockerfile").is_file()


def test_failed_attempt_can_be_fixed(tmp_path):
    class RepairingDeveloper:
        calls = 0

        def develop(self, design, requirement):
            self.calls += 1
            if self.calls > 1:
                assert "上一轮测试反馈" in design
                assert "AssertionError" in design
            return "", f"def test_answer(): assert {self.calls} == 2\n"

    out = develop("算术模块", dev=RepairingDeveloper(), workspace_dir=str(tmp_path))
    assert out["status"] == "delivered"
    assert [attempt["test_status"] for attempt in out["attempts"]] == ["failed", "passed"]


def test_generation_exception_is_failed_without_delivery(tmp_path):
    class BrokenDeveloper:
        def develop(self, design, requirement):
            raise RuntimeError("generation unavailable")

    out = develop("算术模块", dev=BrokenDeveloper(), workspace_dir=str(tmp_path))
    assert out["status"] == "failed"
    assert out["failure_report"]["reason"] == "generation_error"
    assert out["attempts"] == []
    assert_not_delivered(out, tmp_path)


def test_collection_error_does_not_consume_retry_budget(tmp_path):
    dev = FixedDeveloper("def invalid syntax\n")
    out = develop("算术模块", dev=dev, workspace_dir=str(tmp_path))
    assert out["status"] == "failed"
    assert out["test_status"] == "error"
    assert dev.calls == 1
    assert_not_delivered(out, tmp_path)


def test_delivery_exception_does_not_report_success(tmp_path, monkeypatch):
    def unavailable_devops(state):
        raise OSError("deployment artifact could not be written")

    monkeypatch.setattr(dev_team, "devops", unavailable_devops)
    out = develop("算术模块", workspace_dir=str(tmp_path))
    assert out["test_passed"] is True
    assert out["status"] == "failed"
    assert out["failure_report"]["reason"] == "delivery_error"
    assert out["documentation"] is None
    assert not (tmp_path / "Dockerfile").exists()
    assert not (tmp_path / "README.md").exists()


def test_route_fixes_on_failure():
    assert route_after_test({"test_passed": False, "iteration": 1, "max_iterations": 3}) == "fix"


def test_route_delivers_when_passed():
    assert route_after_test({"test_passed": True, "iteration": 3, "max_iterations": 3}) == "deliver"


def test_route_stops_without_delivery_at_cap():
    assert route_after_test({"test_passed": False, "iteration": 3, "max_iterations": 3}) == "fail"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_max_iterations_requires_positive_integer(value):
    with pytest.raises(ValueError, match="max_iterations"):
        develop("算术模块", max_iterations=value)


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), "1", 10**400])
def test_timeout_requires_finite_positive_number(value):
    with pytest.raises(ValueError, match="test_timeout_seconds"):
        develop("算术模块", test_timeout_seconds=value)


def test_infinite_test_times_out_without_delivery(tmp_path):
    dev = FixedDeveloper("def test_hang():\n    while True: pass\n")
    started = time.monotonic()
    out = develop("算术模块", dev=dev, workspace_dir=str(tmp_path), test_timeout_seconds=1)
    assert time.monotonic() - started < 4
    assert out["status"] == "timed_out"
    assert dev.calls == 1
    assert out["attempts"][0]["returncode"] is None
    assert_not_delivered(out, tmp_path)


def process_is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # Linux minimal containers may not promptly reap orphaned children. A zombie
    # no longer executes code; its init/subreaper owns the final reap.
    stat_file = Path(f"/proc/{pid}/stat")
    if stat_file.exists():
        try:
            return stat_file.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        except FileNotFoundError:
            return False
    return True


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup requires macOS/Linux")
@pytest.mark.parametrize("hang", [False, True], ids=["parent_exits", "parent_times_out"])
def test_cleanup_includes_term_resistant_child(tmp_path, hang):
    child_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    generated_test = (
        "import pathlib, subprocess, sys, time\n"
        "def test_child():\n"
        f"    child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "    pathlib.Path('child.pid').write_text(str(child.pid))\n"
        "    time.sleep(0.1)\n"  # Give the child time to install its SIGTERM handler.
        + ("    while True: time.sleep(0.1)\n" if hang else "    assert True\n")
    )
    pid_file = tmp_path / "child.pid"
    try:
        # Normal exit tests cleanup, not pytest startup speed under concurrent builds.
        # Keep the hanging branch short so it still exercises the timeout path.
        out = develop("算术模块", dev=FixedDeveloper(generated_test), workspace_dir=str(tmp_path),
                      test_timeout_seconds=2 if hang else 10)
        assert out["status"] == ("timed_out" if hang else "delivered")
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2
        while process_is_running(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not process_is_running(pid), "generated test child survived cleanup"
    finally:
        # Ensure a regression cannot leak the deliberately long-lived test process.
        if pid_file.exists():
            pid = int(pid_file.read_text())
            if process_is_running(pid):
                os.kill(pid, signal.SIGKILL)


def test_cli_reports_failed_delivery(monkeypatch, capsys):
    monkeypatch.setattr(dev_team, "develop", lambda *args, **kwargs: {
        "status": "budget_exhausted", "test_passed": False,
        "workspace_dir": "/example", "test_output": "failed",
        "failure_report": {"reason": "max_iterations"},
    })
    assert dev_team.main([]) == 1
    assert "未交付" in capsys.readouterr().out
