"""Run a model-written answer program and read its numeric result.

In the self-improvement loop (P7.5) the model writes a new finance problem
together with a short Python program that computes the answer, the way
FinQA's gold answers come from a program. The program's output becomes the
gold answer, so the program must be run, and it was written by a model: it is
checked and then executed in a separate, limited process.

Static check (before anything runs): parse with ast; imports only from
ALLOWED_MODULES; no names or attributes with double underscores; no calls to
open / exec / eval / compile / input / globals / locals / vars / getattr /
setattr / delattr / breakpoint / exit / quit.
Execution: `python -I -S` (isolated: no user site, no environment variables,
no current directory on sys.path) in a fresh temporary directory, with CPU
time, address space, file size and process count limited by setrlimit, an
empty environment and a wall-clock timeout. The program must print one
number as its last line of output.

This is a guard against accidents and against programs that do more than
arithmetic, not a security boundary against a determined attacker; the jobs
run as an unprivileged user on nodes without network access.
"""

from __future__ import annotations

import ast
import contextlib
import math
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass

ALLOWED_MODULES = {"math", "statistics", "decimal", "fractions"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "input", "globals", "locals", "vars", "getattr", "setattr",
                   "delattr", "breakpoint", "exit", "quit", "help", "memoryview"}


@dataclass
class Result:
    ok: bool
    value: float | None = None
    error: str = ""


def check(source: str) -> str | None:
    """None if the program passes the static check, else the reason."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"syntax: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_MODULES]
            if bad:
                return f"import: {bad[0]}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES or node.level:
                return f"import: {node.module}"
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            return f"name: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"attribute: {node.attr}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return f"call: {node.func.id}"
    return None


def _limits(cpu_s: int, mem_bytes: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    with contextlib.suppress(ValueError, OSError):  # macOS does not enforce RLIMIT_AS; Linux (the cluster) does
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # no child processes


def run(source: str, timeout_s: float = 5.0, cpu_s: int = 5, mem_mb: int = 512) -> Result:
    reason = check(source)
    if reason:
        return Result(False, error=f"rejected: {reason}")
    with tempfile.TemporaryDirectory(prefix="b2e-sandbox-") as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(source)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", path], cwd=d, env={}, capture_output=True, text=True,
                               timeout=timeout_s, preexec_fn=lambda: _limits(cpu_s, mem_mb << 20))
        except subprocess.TimeoutExpired:
            return Result(False, error="timeout")
    if p.returncode != 0:
        last = (p.stderr.strip().splitlines() or ["?"])[-1]
        return Result(False, error=f"exit {p.returncode}: {last[:200]}")
    lines = [line for line in p.stdout.strip().splitlines() if line.strip()]
    if not lines:
        return Result(False, error="no output")
    try:
        value = float(lines[-1].strip().replace(",", ""))
    except ValueError:
        return Result(False, error=f"not a number: {lines[-1][:80]!r}")
    if not math.isfinite(value):
        return Result(False, error="not finite")
    return Result(True, value=value)
