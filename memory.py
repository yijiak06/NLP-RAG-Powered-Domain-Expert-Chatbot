from collections import deque


class ConversationMemory:
    def __init__(self, max_turns=10):
        """
        Stores the last `max_turns` user + assistant exchanges.
        Each turn consists of:
        - one user message
        - one assistant response
        """
        self.buffer = deque(maxlen=max_turns * 2)

    def add_user(self, text):
        """Add user message to memory."""
        self.buffer.append(("User", text))

    def add_assistant(self, text):
        """Add assistant response to memory."""
        self.buffer.append(("Assistant", text))

    def get_context(self):
        """
        Returns conversation history as formatted text
        to prepend to new prompts.
        """
        return "\n".join([f"{role}: {message}" for role, message in self.buffer])

    def clear(self):
        """Clears conversation memory."""
        self.buffer.clear()