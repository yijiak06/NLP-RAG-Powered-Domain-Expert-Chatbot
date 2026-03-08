class ConversationMemory:
    def __init__(self, max_turns=5):
        """
        Initialize the conversation memory.
        max_turns: Maximum number of conversation turns to keep (1 turn = 1 user msg + 1 assistant msg).
        """
        self.max_turns = max_turns
        self.buffer = []  # Stores tuples in the format: [("User", "question"), ("Assistant", "answer")]

    def add_user(self, message):
        """Record the user's message."""
        self.buffer.append(("User", message))
        self._trim_memory()

    def add_assistant(self, message):
        """Record the assistant's message."""
        self.buffer.append(("Assistant", message))
        self._trim_memory()

    def _trim_memory(self):
        """Remove the oldest records if the memory exceeds the maximum number of turns."""
        max_messages = self.max_turns * 2
        if len(self.buffer) > max_messages:
            self.buffer = self.buffer[-max_messages:]

    def get_context(self):
        """Format the conversation history into a string to feed into the LLM."""
        if not self.buffer:
            return "No previous conversation."
        
        context = ""
        for role, msg in self.buffer:
            context += f"{role}: {msg}\n"
        return context.strip()