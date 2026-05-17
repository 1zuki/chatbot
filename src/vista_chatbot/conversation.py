from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .llm import ChatTurn


@dataclass(frozen=True)
class SeenMessage:
    text: str
    timestamp: float


class ConversationMemory:
    def __init__(self, max_turns: int = 12):
        self.turns: deque[ChatTurn] = deque(maxlen=max_turns)

    def add_user(self, text: str) -> None:
        self.turns.append(ChatTurn("user", text))

    def add_assistant(self, text: str) -> None:
        self.turns.append(ChatTurn("assistant", text))

    def recent(self, n: int) -> list[ChatTurn]:
        if n <= 0:
            return []
        return list(self.turns)[-n:]

    def clear(self) -> None:
        self.turns.clear()
