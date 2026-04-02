ACTION_MAP = {
    "answer": "Provide solution to user",
    "clarify": "Ask follow-up question",
    "escalate": "Send to human support",
    "apologize_and_fix": "Apologize and resolve issue",
    "acknowledge": "Respond positively",
    "transact": "Process user request",
    "close": "End conversation",
    "respond": "General reply"
}

def map_action(action):
    return ACTION_MAP.get(action, "Unknown")