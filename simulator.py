from agent import TextAgent
from memory import Memory

agent = TextAgent(use_llm=True)
memory = Memory()

while True:
    text = input("User: ")

    state = {
        "text": text,
        "intent": "",
        "sentiment": "",
        "confidence": 0.5,
        "context": memory.get_context()
    }

    action = agent.act(state)

    memory.add(state, action)

    print("Agent:", action)