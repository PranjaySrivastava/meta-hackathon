from agent import TextAgent
from memory import Memory

# Person 1
from environment import StudentEnv   # (their file)

# INIT
env = StudentEnv()
agent = TextAgent(use_llm=False)
memory = Memory()

state = env.reset()

for step in range(20):

    # 1. Add memory context
    state["context"] = memory.get_context()

    # 2. Agent decides
    action = agent.act(state)

    # 3. Save memory
    memory.add(state, action)

    # 4. Environment updates
    next_state, reward, done = env.step(action)

    print(f"\nSTEP {step}")
    print("State:", state)
    print("Action:", action)
    print("Reward:", reward)

    state = next_state

    if done:
        break