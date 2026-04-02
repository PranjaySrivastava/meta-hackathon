from agent import TextAgent

agent = TextAgent(use_llm=False)

tests = [
    "",  # empty
    "????",  # noise
    "asdasdasd",  # random
    "OMG EVERYTHING IS BROKEN AGAIN GREAT",  # sarcasm
    "buy buy buy now",  # spam
]

for t in tests:
    state = {
        "text": t,
        "intent": "",
        "sentiment": "",
        "confidence": 0.2,
        "context": []
    }

    print(f"\nInput: {t}")
    print("Output:", agent.act(state))