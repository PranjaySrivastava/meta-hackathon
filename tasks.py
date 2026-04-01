"""
tasks.py  –  Task bank for benchmarking the TextAgent.

Import this file; never run it directly during a benchmark.
Run  python tasks.py  only to preview the task list.
"""

TASKS = [

    # ── EASY ─────────────────────────────────────────────────────────
    {
        "id": "E01", "difficulty": "easy",
        "description": "Direct question, high confidence",
        "state": {
            "text": "What are your business hours?",
            "intent": "question", "sentiment": "neutral",
            "confidence": 0.95, "context": [],
        },
        "expected_action": "answer",
    },
    {
        "id": "E02", "difficulty": "easy",
        "description": "Clear positive feedback",
        "state": {
            "text": "Thanks! Everything works perfectly now.",
            "intent": "praise", "sentiment": "positive",
            "confidence": 0.92, "context": [],
        },
        "expected_action": "acknowledge",
    },
    {
        "id": "E03", "difficulty": "easy",
        "description": "Explicit purchase intent",
        "state": {
            "text": "I'd like to subscribe to the premium plan.",
            "intent": "purchase", "sentiment": "neutral",
            "confidence": 0.93, "context": [],
        },
        "expected_action": "transact",
    },
    {
        "id": "E04", "difficulty": "easy",
        "description": "User says goodbye",
        "state": {
            "text": "Goodbye, I'm done for today.",
            "intent": "bye", "sentiment": "neutral",
            "confidence": 0.96, "context": [],
        },
        "expected_action": "close",
    },
    {
        "id": "E05", "difficulty": "easy",
        "description": "Explicit urgent emergency",
        "state": {
            "text": "URGENT: production system is down, emergency fix needed!",
            "intent": "support", "sentiment": "negative",
            "confidence": 0.91, "context": [],
        },
        "expected_action": "escalate",
    },

    # ── MEDIUM ───────────────────────────────────────────────────────
    {
        "id": "M01", "difficulty": "medium",
        "description": "Complaint disguised as a question",
        "state": {
            "text": "Why does this keep failing every single time?",
            "intent": "question", "sentiment": "negative",
            "confidence": 0.65, "context": [],
        },
        "expected_action": "apologize_and_fix",
    },
    {
        "id": "M02", "difficulty": "medium",
        "description": "Vague message, low confidence — needs clarification",
        "state": {
            "text": "Not sure what to do here.",
            "intent": "", "sentiment": "neutral",
            "confidence": 0.35, "context": [],
        },
        "expected_action": "clarify",
    },
    {
        "id": "M03", "difficulty": "medium",
        "description": "Positive opener but asks a genuine question",
        "state": {
            "text": "Love the product! How do I integrate with Zapier?",
            "intent": "question", "sentiment": "positive",
            "confidence": 0.72, "context": [],
        },
        "expected_action": "answer",
    },
    {
        "id": "M04", "difficulty": "medium",
        "description": "Escalation buried in polite language",
        "state": {
            "text": "I've been waiting a week and this is still not resolved. This is pretty critical for us.",
            "intent": "complaint", "sentiment": "negative",
            "confidence": 0.60, "context": [],
        },
        "expected_action": "escalate",
    },
    {
        "id": "M05", "difficulty": "medium",
        "description": "Purchase intent phrased as a question",
        "state": {
            "text": "Can I upgrade my account right now?",
            "intent": "question", "sentiment": "neutral",
            "confidence": 0.58, "context": [],
        },
        "expected_action": "transact",
    },

    # ── HARD ─────────────────────────────────────────────────────────
    {
        "id": "H01", "difficulty": "hard",
        "description": "Sarcasm — positive words, clearly negative intent",
        "state": {
            "text": "Oh great, it broke AGAIN. Just perfect.",
            "intent": "complaint", "sentiment": "positive",
            "confidence": 0.40, "context": [],
        },
        "expected_action": "apologize_and_fix",
    },
    {
        "id": "H02", "difficulty": "hard",
        "description": "Garbled / noisy input, very low confidence",
        "state": {
            "text": "asdklj 123 ??? what even is thsi lol",
            "intent": "", "sentiment": "neutral",
            "confidence": 0.12, "context": [],
        },
        "expected_action": "clarify",
    },
    {
        "id": "H03", "difficulty": "hard",
        "description": "Mixed buy + complaint in one message",
        "state": {
            "text": "I want to buy your product but I've heard it has serious bugs.",
            "intent": "purchase", "sentiment": "negative",
            "confidence": 0.50, "context": [],
        },
        "expected_action": "apologize_and_fix",
    },
    {
        "id": "H04", "difficulty": "hard",
        "description": "Mild crisis language, low classifier confidence",
        "state": {
            "text": "things are getting a bit critical on our end",
            "intent": "", "sentiment": "neutral",
            "confidence": 0.28, "context": [],
        },
        "expected_action": "escalate",
    },
    {
        "id": "H05", "difficulty": "hard",
        "description": "Multi-turn reversal — positive context, then negative",
        "state": {
            "text": "Actually, scratch that. It stopped working again.",
            "intent": "", "sentiment": "negative",
            "confidence": 0.45,
            "context": [[{"text": "It finally works!"}, "acknowledge"]],
        },
        "expected_action": "apologize_and_fix",
    },
]


def get_tasks(difficulty: str = None) -> list:
    """Return all tasks, or filter by 'easy' | 'medium' | 'hard'."""
    if difficulty is None:
        return TASKS
    return [t for t in TASKS if t["difficulty"] == difficulty]


def task_summary() -> str:
    counts = {}
    for t in TASKS:
        counts[t["difficulty"]] = counts.get(t["difficulty"], 0) + 1
    lines = [f"  {d}: {n} tasks" for d, n in counts.items()]
    return "Task bank\n" + "\n".join(lines) + f"\n  total: {len(TASKS)}"


if __name__ == "__main__":
    # Preview only — this block never runs when imported by grader.py
    print(task_summary())
    print()
    for t in TASKS:
        print(f"[{t['difficulty'].upper():6}] {t['id']} – {t['description']}")
        print(f"         expected → {t['expected_action']}")