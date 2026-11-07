"""Benchmark runner: the same tasks through loops and pi, scored by pytest.

For every task x harness x run: copy the pristine task into a work directory,
run the harness headlessly with one prompt, then run the task's tests and
record pass/fail plus what the harness reported (rounds, tool calls, tokens,
wall time). Results go to results/<stamp>.json and results/<stamp>.md.

Usage: python run.py [--runs N] [--tasks a,b] [--harness loops,pi] [--timeout S]
       [--model google/gemma-4-e4b]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"
WORK = HERE / "work"
RESULTS = HERE / "results"
VENV_BIN = HERE / ".venv" / "Scripts"
PYTHON = VENV_BIN / "python.exe"
LOOPS = HERE.parent / "target" / "release" / "loops.exe"
MODEL = "google/gemma-4-e4b"  # LM Studio id; pi gets the part after the slash
# npm installs a `pi.cmd` shim on Windows; CreateProcess does not resolve bare `pi`.
PI = shutil.which("pi.cmd") or shutil.which("pi") or "pi"

PROMPT = (
    "Make the tests in this directory pass. Implement the stub in {stub}; "
    "INSTRUCTIONS.md describes the exercise. Check your work by running "
    "`python -m pytest -q` and keep going until it passes. "
    "Do not modify the test file."
)


def task_files(task: Path) -> tuple[str, str]:
    """(stub file name, test file name) of a task directory."""
    tests = [p.name for p in task.glob("*_test.py")]
    assert len(tests) == 1, task
    return tests[0].replace("_test.py", ".py"), tests[0]


def env_with_venv() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = str(VENV_BIN) + os.pathsep + env["PATH"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_pytest(cwd: Path) -> dict:
    """Runs the task's tests; returns passed/failed counts and whether all passed."""
    proc = subprocess.run(
        [str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=cwd, capture_output=True, text=True, timeout=120, env=env_with_venv(),
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", tail)) else 0
    return {
        "passed": passed, "failed": failed, "errors": errors,
        "ok": proc.returncode == 0 and passed > 0, "summary": tail,
    }


def kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        proc.kill()


def run_process(cmd: list[str], cwd: Path, timeout: int, env: dict) -> dict:
    """Runs cmd, capturing stdout/stderr; kills the tree on timeout."""
    started = time.monotonic()
    # No stdin: pi -p waits for EOF on an inherited pipe and never starts.
    proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", env=env)
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        out, err = proc.communicate()
        timed_out = True
    return {
        "stdout": out, "stderr": err, "returncode": proc.returncode,
        "timed_out": timed_out, "elapsed_s": round(time.monotonic() - started, 1),
    }


def run_loops(cwd: Path, prompt: str, timeout: int) -> dict:
    stats_path = cwd / ".loops-stats.json"
    cmd = [str(LOOPS), "--print", prompt, "--stats", str(stats_path),
           "--timeout", str(timeout), "--max-rounds", "40", "--model", MODEL]
    res = run_process(cmd, cwd, timeout + 30, env_with_venv())
    stats = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats_path.unlink()
    calls = stats.get("tool_calls", [])
    return {
        "outcome": "timeout" if res["timed_out"] else stats.get("outcome", "failed"),
        "error": stats.get("error"),
        "rounds": stats.get("rounds", 0),
        "tool_calls": [c["name"] for c in calls],
        "tool_errors": sum(1 for c in calls if c.get("is_error")),
        "prompt_tokens_last": stats.get("prompt_tokens_last", 0),
        "prompt_tokens_total": stats.get("prompt_tokens_total", 0),
        "completion_tokens_total": stats.get("completion_tokens_total", 0),
        "elapsed_s": res["elapsed_s"],
        "answer": stats.get("answer", res["stdout"])[-2000:],
        "log": res["stderr"][-6000:],
    }


def run_pi(cwd: Path, prompt: str, timeout: int) -> dict:
    cmd = [PI, "-p", "--mode", "json", "--provider", "lmstudio",
           "--model", MODEL.split("/")[-1],
           "--no-session", "--no-extensions", "--no-skills", "--no-context-files",
           "--no-mcp", prompt]
    res = run_process(cmd, cwd, timeout, env_with_venv())
    rounds = 0
    calls: list[str] = []
    tool_errors = 0
    prompt_last = prompt_total = completion_total = 0
    answer = ""
    error = None
    ended = False
    for line in res["stdout"].splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("type")
        if t == "message_end" and e["message"].get("role") == "assistant":
            m = e["message"]
            rounds += 1
            u = m.get("usage") or {}
            prompt_last = u.get("input", prompt_last)
            prompt_total += u.get("input", 0)
            completion_total += u.get("output", 0)
            if m.get("stopReason") == "error":
                error = m.get("errorMessage")
            texts = [c.get("text", "") for c in m.get("content", [])
                     if isinstance(c, dict) and c.get("type") == "text"]
            if texts:
                answer = "".join(texts)
        elif t == "tool_execution_end":
            calls.append(e.get("toolName", "?"))
            tool_errors += 1 if e.get("isError") else 0
        elif t == "agent_end":
            ended = True
    if res["timed_out"]:
        outcome = "timeout"
    elif error or not ended:
        outcome = "failed"
    else:
        outcome = "done"
    return {
        "outcome": outcome, "error": error, "rounds": rounds, "tool_calls": calls,
        "tool_errors": tool_errors, "prompt_tokens_last": prompt_last,
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": completion_total, "elapsed_s": res["elapsed_s"],
        "answer": answer[-2000:], "log": res["stderr"][-6000:],
    }


HARNESSES = {"loops": run_loops, "pi": run_pi}


def prepare(task: Path, harness: str, run: int) -> Path:
    work = WORK / harness / task.name / f"run{run}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for p in task.iterdir():
        shutil.copy(p, work / p.name)
    return work


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--harness", default="loops,pi")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    MODEL = args.model

    tasks = sorted(p for p in TASKS.iterdir() if p.is_dir())
    if args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in tasks if t.name in wanted]
    harnesses = args.harness.split(",")
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results: list[dict] = []

    for task in tasks:
        stub, _ = task_files(task)
        prompt = PROMPT.format(stub=stub)
        for run in range(1, args.runs + 1):
            for harness in harnesses:
                work = prepare(task, harness, run)
                print(f"{task.name:22} {harness:6} run {run} ...", end=" ", flush=True)
                r = HARNESSES[harness](work, prompt, args.timeout)
                tests = run_pytest(work)
                r.update(task=task.name, harness=harness, run=run, tests=tests,
                         model=MODEL)
                results.append(r)
                print(f"{'PASS' if tests['ok'] else 'fail':4} {r['outcome']:7} "
                      f"{r['elapsed_s']:6.1f}s rounds={r['rounds']:2} "
                      f"tools={len(r['tool_calls']):2} {tests['summary']}")
                (RESULTS / f"{stamp}.json").write_text(
                    json.dumps(results, indent=1), encoding="utf-8")

    (RESULTS / f"{stamp}.md").write_text(table(results, harnesses), encoding="utf-8")
    print(table(results, harnesses))
    return 0


def table(results: list[dict], harnesses: list[str]) -> str:
    """Per-task pass counts and per-harness totals, as markdown."""
    tasks = sorted({r["task"] for r in results})
    lines = ["| task | " + " | ".join(harnesses) + " |",
             "|---|" + "---|" * len(harnesses)]
    for t in tasks:
        cells = []
        for h in harnesses:
            rs = [r for r in results if r["task"] == t and r["harness"] == h]
            cells.append(f"{sum(r['tests']['ok'] for r in rs)}/{len(rs)}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("| harness | solved | pass rate | median time | mean rounds | mean tool calls | tool errors | timeouts | mean final context |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for h in harnesses:
        rs = [r for r in results if r["harness"] == h]
        if not rs:
            continue
        solved = sum(r["tests"]["ok"] for r in rs)
        times = sorted(r["elapsed_s"] for r in rs)
        median = times[len(times) // 2]
        mean = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
        lines.append(
            f"| {h} | {solved}/{len(rs)} | {100 * solved / len(rs):.0f}% | {median:.0f}s | "
            f"{mean('rounds'):.1f} | {sum(len(r['tool_calls']) for r in rs) / len(rs):.1f} | "
            f"{sum(r['tool_errors'] for r in rs)} | "
            f"{sum(r['outcome'] == 'timeout' for r in rs)} | {mean('prompt_tokens_last'):.0f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
