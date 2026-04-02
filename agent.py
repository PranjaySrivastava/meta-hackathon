"""
agent.py
--------
Text/NLP Decision Agent.
  - Primary  : LLM via inference.py (Gemini with quota management)
  - Fallback : Rule-based heuristics (instant, zero API calls)
"""

from inference import run_inference, ACTIONS


class TextAgent:
    """
    Reads a text/NLP state dict and returns the best action string.

    State schema:
        text        : str
        intent      : str
        sentiment   : str
        confidence  : float
        context     : list
    """

    def __init__(self, use_llm: bool = True, verbose: bool = False):
        self.use_llm = use_llm
        self.verbose = verbose

    # ── Public API ────────────────────────────────────────────────────

    def act(self, state: dict) -> str:
        """Return the best action for the given state."""
        self._validate(state)

        if self.use_llm:
            try:
                action = run_inference(state)
                if self.verbose:
                    print(f"[LLM]  '{state['text'][:55]}' → {action}")
                return action
            except Exception as exc:
                if self.verbose:
                    print(f"[LLM FAILED] {exc} — using rules")

        action = self._rule_based(state)
        if self.verbose:
            print(f"[RULE] '{state['text'][:55]}' → {action}")
        return action

    # ── Rule-based fallback ───────────────────────────────────────────

    def _rule_based(self, state: dict) -> str:
        text       = state.get("text", "").lower()
        intent     = state.get("intent", "").lower()
        sentiment  = state.get("sentiment", "").lower()
        confidence = state.get("confidence", 0.5)

        # 1. Farewell → close
        farewell_phrases = {"bye", "goodbye", "quit", "exit", "see you"}
        text_words = set(text.split())
        if (any(w in text_words for w in farewell_phrases)
                or "done for today" in text
                or "done for now" in text
                or intent in {"bye", "quit", "exit", "done"}):
            return "close"

        # 2. CRISIS FIRST (FIXED PRIORITY 🔥)
        if any(w in text for w in {"urgent", "emergency", "critical", "asap",
                                  "danger", "shutdown", "down", "outage"}):
            return "escalate"

        # 3. Very low confidence → clarify
        if confidence < 0.3:
            return "clarify"

        # 4. Sarcasm detection
        sarcasm_triggers = {"again", "broke", "broken", "failed", "fail",
                            "crash", "error", "issue", "problem"}
        positive_words   = {"great", "perfect", "awesome", "wonderful", "fantastic"}
        if (any(p in text for p in positive_words)
                and any(s in text for s in sarcasm_triggers)):
            return "apologize_and_fix"

        # 5. Complaint / negative → apologize_and_fix
        complaint_words = {"wrong", "broken", "bad", "hate", "issue", "problem",
                          "bug", "fail", "failing", "awful", "terrible", "horrible"}
        if any(w in text for w in complaint_words) or sentiment == "negative":
            return "apologize_and_fix"

        # 6. Purchase intent → transact
        buy_words = {"buy", "purchase", "order", "subscribe", "upgrade",
                     "pro plan", "premium", "sign up", "get started"}
        if (any(w in text for w in buy_words)
                or intent in {"buy", "purchase", "order", "subscribe"}):
            return "transact"

        # 7. Medium ambiguity → clarify
        if confidence < 0.4 and not intent:
            return "clarify"

        # 8. Question → answer
        if (any(w in text for w in {"what", "why", "how", "when", "where", "who", "?"})
                or intent == "question"):
            return "answer"

        # 9. Positive / praise → acknowledge
        if (any(w in text for w in {"great", "good", "thanks", "love", "awesome",
                                   "perfect", "thank", "appreciate"})
                or sentiment == "positive"):
            return "acknowledge"

        return "respond"

    # ── Validation ────────────────────────────────────────────────────

    @staticmethod
    def _validate(state: dict):
        if "text" not in state or not isinstance(state["text"], str):
            raise ValueError("state must have a 'text' string field")


# ── Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = TextAgent(use_llm=True, verbose=True)

    samples = [
        {"text": "How do I reset my password?",
         "intent": "question",  "sentiment": "neutral",  "confidence": 0.85, "context": []},

        {"text": "This is broken and awful!",
         "intent": "complaint", "sentiment": "negative", "confidence": 0.90, "context": []},

        {"text": "Thanks, everything works great!",
         "intent": "praise",    "sentiment": "positive", "confidence": 0.88, "context": []},

        {"text": "I want to buy the pro plan.",
         "intent": "purchase",  "sentiment": "neutral",  "confidence": 0.80, "context": []},

        {"text": "emergency shutdown needed NOW",
         "intent": "",          "sentiment": "negative", "confidence": 0.55, "context": []},

        {"text": "hmm ok",
         "intent": "",          "sentiment": "neutral",  "confidence": 0.20, "context": []},
    ]

    for s in samples:
        print(agent.act(s))
        print()