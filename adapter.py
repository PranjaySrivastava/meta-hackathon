def convert_env_to_agent_state(env_state, memory):

    text = f"""
    Student condition:
    stress={env_state.get('stress', 0)},
    performance={env_state.get('performance', 0)},
    attendance={env_state.get('attendance', 0)}
    """

    return {
        "text": text,
        "intent": "",
        "sentiment": "",
        "confidence": 0.5,
        "context": memory.get_context()
    }