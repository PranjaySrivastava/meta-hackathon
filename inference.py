"""
inference.py
------------
Gemini-powered inference with:
  - Response caching     → identical states never call the API twice
  - Multi-model fallback → gemini-2.0-flash → gemini-1.5-flash → rules
  - Per-minute throttle  → stays under 15 req/min automatically
  - Daily quota guard    → switches model before hitting the wall
  - Zero retries on daily-limit (429 with limit:0) → instant fallback
"""

import os
import json
import re
import time
import hashlib
import urllib.request
import urllib.error
from collections import deque

# ── Action space ──────────────────────────────────────────────────────
ACTIONS = [
    "answer",
    "clarify",
    "escalate",
    "apologize_and_fix",
    "acknowledge",
    "transact",
    "close",
    "respond",
]

# ── Model cascade (tries in order, skips if daily-limited) ────────────
MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

# ── Rate limit: max requests per minute per model ─────────────────────
MAX_RPM = 12          # stay safely under free-tier 15/min
WINDOW  = 60.0        # seconds

# ── Per-model state ───────────────────────────────────────────────────
_model_exhausted: dict[str, bool] = {m: False for m in MODELS}
_request_times:   dict[str, deque] = {m: deque() for m in MODELS}

# ── Response cache: hash(state_json) → action ─────────────────────────
_cache: dict[str, str] = {}

SYSTEM_PROMPT = f"""You are a decision-making agent for a text/NLP pipeline.

Given a JSON state object choose the SINGLE best action from this list:
{json.dumps(ACTIONS, indent=2)}

DECISION RULES (apply strictly in order — stop at first match):
1. Text contains farewell (bye, goodbye, done, quit, exit) → "close"
2. confidence < 0.3 AND text is vague/unclear → "clarify"
3. Text contains urgent/crisis words OR describes ongoing critical issue → "escalate"
   - "a bit critical", "getting critical", "still not resolved" ALL count as escalate
4. sentiment=negative AND (complaint OR broken OR failing) → "apologize_and_fix"
5. Complaint disguised as a question (why does X fail/break?) → "apologize_and_fix"
6. SARCASM: positive words (great, perfect) + something broke/failed again → "apologize_and_fix"
   - "Oh great, it broke AGAIN" = sarcasm → "apologize_and_fix", NOT "acknowledge"
7. intent=purchase OR text asks about upgrading/buying/subscribing → "transact"
8. confidence < 0.4 AND intent is empty AND text is ambiguous → "clarify"
9. Text contains a genuine question → "answer"
10. sentiment=positive OR text expresses thanks/praise → "acknowledge"
11. Default → "respond"

Respond with ONLY this JSON (no markdown, no extra keys):
{{"action": "<action>", "reason": "<one short sentence>"}}"""


# ── Helpers ───────────────────────────────────────────────────────────

def _cache_key(state: dict) -> str:
    """Stable hash of the state (ignores internal _ keys)."""
    clean = {k: v for k, v in state.items() if not k.startswith("_")}
    return hashlib.md5(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def _throttle(model: str):
    """Block until we are allowed to send another request for this model."""
    q = _request_times[model]
    now = time.time()
    # Drop timestamps older than the window
    while q and now - q[0] > WINDOW:
        q.popleft()
    if len(q) >= MAX_RPM:
        wait = WINDOW - (now - q[0]) + 0.2
        print(f"[THROTTLE] {model}: waiting {wait:.1f}s to respect rate limit...")
        time.sleep(wait)
    # NOTE: timestamp is recorded in _call_model only on successful request


def _is_daily_exhausted(err_body: str) -> bool:
    """Return True if the error is a daily quota exhaustion (limit: 0)."""
    try:
        j = json.loads(err_body)
        for v in j.get("error", {}).get("details", []):
            for violation in v.get("violations", []):
                if "PerDay" in violation.get("quotaId", ""):
                    return True
        # Also check the message text
        msg = j.get("error", {}).get("message", "")
        if "limit: 0" in msg and "PerDay" in msg:
            return True
    except Exception:
        pass
    return False


def _call_model(model: str, state: dict) -> str:
    """
    Call a single Gemini model. Returns action string.
    Raises RuntimeError on failure.
    Marks model as daily-exhausted when appropriate.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    _throttle(model)

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents":           [{"parts": [{"text": json.dumps(state)}]}],
        "generationConfig":   {"temperature": 0.1, "maxOutputTokens": 256},
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
        _request_times[model].append(time.time())  # record only on success
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        if e.code == 429:
            if _is_daily_exhausted(err_body):
                _model_exhausted[model] = True
                raise RuntimeError(f"[{model}] Daily quota exhausted — marking as unavailable")
            else:
                raise RuntimeError(f"[{model}] Rate limited (per-minute) — throttle will handle this")
        raise RuntimeError(f"[{model}] HTTP {e.code}: {err_body}") from e

    raw = body["candidates"][0]["content"]["parts"][0]["text"].strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$",        "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"[{model}] Bad JSON response: {raw!r}") from exc

    action = parsed.get("action", "").strip()
    if action not in ACTIONS:
        raise RuntimeError(f"[{model}] Unknown action: {action!r}")

    return action


# ── Public API ────────────────────────────────────────────────────────

def run_inference(state: dict) -> str:
    """
    Return the best action for the given state.

    Strategy:
      1. Return cached result if state was seen before.
      2. Try each model in MODELS order, skipping daily-exhausted ones.
      3. Raise RuntimeError if all models fail (agent.py will use rules).
    """
    key = _cache_key(state)
    if key in _cache:
        print(f"[CACHE HIT] Returning cached action: {_cache[key]}")
        return _cache[key]

    last_error = None
    for model in MODELS:
        if _model_exhausted[model]:
            print(f"[SKIP] {model} is daily-exhausted, trying next model...")
            continue
        try:
            action = _call_model(model, state)
            _cache[key] = action          # cache the result
            return action
        except RuntimeError as e:
            print(f"[WARN] {e}")
            last_error = e
            continue

    raise RuntimeError(
        f"All models failed. Last error: {last_error}. "
        "Falling back to rule-based agent."
    )


def quota_status() -> str:
    """Print current quota / cache status."""
    lines = ["── Inference Status ──────────────────"]
    for m in MODELS:
        status = "❌ daily-exhausted" if _model_exhausted[m] else "✔  available"
        used   = len(_request_times[m])
        lines.append(f"  {m:<30} {status}  ({used}/{MAX_RPM} req in last 60s)")
    lines.append(f"  Cache entries: {len(_cache)}")
    lines.append("──────────────────────────────────────")
    return "\n".join(lines)


# ── Standalone test ───────────────────────────────────────────────────
if __name__ == "__main__":
    test_state = {
        "text": "I need help urgently, the system is down!",
        "intent": "support",
        "sentiment": "negative",
        "confidence": 0.82,
        "context": [],
    }
    try:
        print(run_inference(test_state))
        # Call again — should be instant from cache
        print(run_inference(test_state))
        print(quota_status())
    except RuntimeError as e:
        print(f"Failed: {e}")