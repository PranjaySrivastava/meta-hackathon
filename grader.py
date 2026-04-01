"""
grader.py  –  Benchmark runner for the TextAgent.

Usage:
    python grader.py              # full benchmark, all 15 tasks
    python grader.py --rules      # rules-only mode (no API calls, instant)

Quota-safe:
    - 5s delay between tasks  →  max 12 req/min, free tier never exceeded
    - Cache in inference.py   →  duplicate states cost zero API calls
    - Model cascade           →  auto-falls to gemini-1.5-flash if 2.0-flash exhausted
"""

from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field


REQUEST_DELAY_SEC = 5.0   # seconds between API calls (safe for 15 req/min free tier)


# ── Dataclasses ───────────────────────────────────────────────────────

@dataclass
class GradeResult:
    task_id:          str
    difficulty:       str
    text_snippet:     str
    expected_action:  str
    predicted_action: str
    correct:          bool
    latency_ms:       float
    score:            float   # 1.0 correct | 0.5 near-miss | 0.0 wrong
    source:           str     # "llm" | "rule" | "cache"
    notes:            str = ""

    def __str__(self):
        mark = "✔" if self.correct else "✘"
        return (
            f"{mark} [{self.difficulty.upper():6}] {self.task_id}  "
            f"expected={self.expected_action:<20} "
            f"got={self.predicted_action:<20} "
            f"[{self.source:<5}] ({self.latency_ms:.0f} ms)"
        )


@dataclass
class BenchmarkReport:
    results:        list
    total:          int   = 0
    correct:        int   = 0
    accuracy:       float = 0.0
    avg_latency_ms: float = 0.0
    by_difficulty:  dict  = field(default_factory=dict)
    api_calls_made: int   = 0

    def __post_init__(self):
        self._compute()

    def _compute(self):
        self.total          = len(self.results)
        self.correct        = sum(r.correct for r in self.results)
        self.accuracy       = self.correct / self.total if self.total else 0.0
        self.avg_latency_ms = (
            sum(r.latency_ms for r in self.results) / self.total
            if self.total else 0.0
        )
        self.api_calls_made = sum(1 for r in self.results if r.source == "llm")

        for diff in ("easy", "medium", "hard"):
            subset = [r for r in self.results if r.difficulty == diff]
            if subset:
                c = sum(r.correct for r in subset)
                self.by_difficulty[diff] = {
                    "total": len(subset), "correct": c,
                    "accuracy": c / len(subset),
                }

    def summary(self) -> str:
        lines = [
            "━" * 60,
            "  Benchmark Report",
            "━" * 60,
            f"  Overall accuracy : {self.accuracy:.1%}  ({self.correct}/{self.total})",
            f"  Avg latency      : {self.avg_latency_ms:.0f} ms",
            f"  API calls made   : {self.api_calls_made} "
            f"(cache/rules saved {self.total - self.api_calls_made} calls)",
            "",
            "  Per difficulty:",
        ]
        for diff, s in self.by_difficulty.items():
            bar = "█" * int(s["accuracy"] * 20)
            lines.append(
                f"    {diff:<8} {s['accuracy']:.0%}  {bar:<20} "
                f"({s['correct']}/{s['total']})"
            )
        lines.append("━" * 60)
        return "\n".join(lines)

    def failed_tasks(self) -> list:
        return [r for r in self.results if not r.correct]


# ── Near-miss pairs (partial credit 0.5) ─────────────────────────────

NEAR_MISS_PAIRS = {
    frozenset({"escalate",          "apologize_and_fix"}),
    frozenset({"clarify",           "respond"}),
    frozenset({"answer",            "respond"}),
    frozenset({"transact",          "answer"}),
    frozenset({"apologize_and_fix", "acknowledge"}),
}


# ── Core grading function ─────────────────────────────────────────────

def grade(state: dict, expected_action: str, agent) -> GradeResult:
    """Grade a single state against the expected action."""
    from inference import _cache, _cache_key

    # Check cache BEFORE calling act() so source label is correct
    was_cached = _cache_key(state) in _cache

    t0 = time.perf_counter()
    predicted = agent.act(state)
    latency_ms = (time.perf_counter() - t0) * 1000

    correct = predicted == expected_action
    score   = 1.0 if correct else 0.0
    notes   = ""

    if not correct and frozenset({predicted, expected_action}) in NEAR_MISS_PAIRS:
        score = 0.5
        notes = "near-miss (partial credit 0.5)"

    if was_cached:
        source = "cache"
    elif not agent.use_llm:
        source = "rule"
    else:
        source = "llm"

    return GradeResult(
        task_id=state.get("_id", "?"),
        difficulty=state.get("_difficulty", "unknown"),
        text_snippet=state.get("text", "")[:55],
        expected_action=expected_action,
        predicted_action=predicted,
        correct=correct,
        latency_ms=latency_ms,
        score=score,
        source=source,
        notes=notes,
    )


# ── Benchmark runner ──────────────────────────────────────────────────

def run_benchmark(agent=None, tasks: list = None, verbose: bool = True) -> BenchmarkReport:
    """
    Run the agent over all tasks and return a BenchmarkReport.
    Adds REQUEST_DELAY_SEC between calls to stay within free-tier limits.
    """
    from tasks import TASKS
    from agent import TextAgent

    if tasks is None:
        tasks = TASKS
    if agent is None:
        agent = TextAgent(use_llm=True, verbose=False)

    results = []
    total = len(tasks)

    for i, task in enumerate(tasks):
        state = dict(task["state"])
        state["_id"]         = task["id"]
        state["_difficulty"] = task["difficulty"]

        result = grade(state, task["expected_action"], agent)

        if verbose:
            print(result)
            if result.notes:
                print(f"         ↳ {result.notes}")

        results.append(result)

        # Throttle — skip after last task
        if REQUEST_DELAY_SEC > 0 and i < total - 1:
            time.sleep(REQUEST_DELAY_SEC)

    report = BenchmarkReport(results=results)

    if verbose:
        print()
        print(report.summary())

        failed = report.failed_tasks()
        if failed:
            print(f"\n  Failed tasks ({len(failed)}):")
            for r in failed:
                line = f"    • {r.task_id}: expected '{r.expected_action}', got '{r.predicted_action}'"
                if r.notes:
                    line += f"  [{r.notes}]"
                print(line)

        from inference import quota_status
        print()
        print(quota_status())

    return report


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    use_llm = "--rules" not in sys.argv
    if not use_llm:
        print("Running in rules-only mode (no API calls)...\n")

    from agent import TextAgent
    agent = TextAgent(use_llm=use_llm, verbose=False)
    run_benchmark(agent=agent, verbose=True)