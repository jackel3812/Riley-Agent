class RileyCore:
    def __init__(self):
        self.memory = []
        self.mode = "default"
        self.personality = "neutral"

    def set_mode(self, mode):
        self.mode = mode
        return f"Mode set to {mode}."

    def set_personality(self, personality):
        self.personality = personality
        return f"Personality profile set to {personality}."

    def think(self, prompt):
        # You can expand this logic later
        memory_snippet = " ".join(self.memory[-3:]) if self.memory else ""
        context = f"Mode: {self.mode}\nPersonality: {self.personality}\nMemory: {memory_snippet}\nUser: {prompt}\nRiley:"
        return context

    def remember(self, thought):
        self.memory.append(thought)
        if len(self.memory) > 50:
            self.memory.pop(0)  # Keep memory short-term for now
