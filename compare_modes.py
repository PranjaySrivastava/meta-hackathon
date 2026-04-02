from agent import TextAgent

state = {
    "text": "This is critical and not working!",
    "intent": "",
    "sentiment": "negative",
    "confidence": 0.7,
    "context": []
}

for mode in ["fast", "hybrid", "smart"]:
    agent = TextAgent(use_llm=(mode != "fast"))
    result = agent.act(state)
    print(f"{mode.upper()} → {result}")