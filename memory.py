class Memory:

    def __init__(self):
        self.history = []

    def add(self, state, action):
        self.history.append((state["text"], action))

    def get_context(self):
        return self.history[-5:]