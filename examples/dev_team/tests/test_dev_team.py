import os

from dev_team import (
    DeterministicDeveloper,
    develop,
    route_after_test,
)


def test_offline_dev_team_passes_tests():
    out = develop("用户管理系统", dev=DeterministicDeveloper())
    assert out["test_passed"] is True, out["test_output"]
    ws = out["workspace_dir"]
    # 角色真的写出了文件，而不是只返回文本
    assert os.path.exists(os.path.join(ws, "app.py"))
    assert os.path.exists(os.path.join(ws, "app_test.py"))
    assert os.path.exists(os.path.join(ws, "Dockerfile"))
    assert os.path.exists(os.path.join(ws, "README.md"))


def test_route_fixes_on_failure():
    assert route_after_test(
        {"test_passed": False, "iteration": 1, "max_iterations": 3}
    ) == "fix"


def test_route_delivers_when_passed():
    assert route_after_test(
        {"test_passed": True, "iteration": 1, "max_iterations": 3}
    ) == "deliver"


def test_route_delivers_at_cap():
    # 即使测试没过，达到最大迭代次数也要停止循环（防止死循环）
    assert route_after_test(
        {"test_passed": False, "iteration": 3, "max_iterations": 3}
    ) == "deliver"
