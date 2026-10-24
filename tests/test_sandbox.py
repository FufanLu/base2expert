from b2e.rsi.sandbox import check, run


def test_arithmetic_program_runs():
    r = run("revenue_2019 = 5260.0\nrevenue_2018 = 4981.0\nprint(round((revenue_2019 - revenue_2018) / revenue_2018 * 100, 2))\n")
    assert r.ok and r.value == 5.6


def test_allowed_import_and_last_line_is_the_answer():
    r = run("import math\nprint('working')\nprint(math.sqrt(16))\n")
    assert r.ok and r.value == 4.0


def test_static_check_rejects_escape_hatches():
    assert check("import os\nprint(1)") == "import: os"
    assert check("from subprocess import run") == "import: subprocess"
    assert check("print(open('/etc/passwd').read())") == "call: open"
    assert check("print(().__class__.__bases__)").startswith("attribute: __")
    assert check("__import__('os')") == "name: __import__"
    assert run("import socket").error == "rejected: import: socket"


def test_runtime_failures_are_reported_not_raised():
    assert run("while True:\n    pass\n", timeout_s=1, cpu_s=1).ok is False
    assert run("print(1/0)").error.startswith("exit 1: ZeroDivisionError")
    assert run("print('n/a')").error.startswith("not a number")
    assert run("x = 1").error == "no output"
    assert run("print(float('inf'))").error == "not finite"
