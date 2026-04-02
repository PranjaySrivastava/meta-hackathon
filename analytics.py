from grader import run_benchmark
from agent import TextAgent
from collections import Counter

def analyze():
    agent = TextAgent(use_llm=True, verbose=False)
    report = run_benchmark(agent=agent, verbose=False)

    print("\n=== ANALYTICS REPORT ===")

    # Difficulty-wise accuracy
    for diff, stats in report.by_difficulty.items():
        print(f"{diff.upper()} → {stats['accuracy']:.2%}")

    # Failure patterns
    failures = report.failed_tasks()
    pairs = Counter((f.expected_action, f.predicted_action) for f in failures)

    print("\nCommon Mistakes:")
    for k, v in pairs.items():
        print(f"{k} → {v} times")