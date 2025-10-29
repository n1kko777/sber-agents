from collections import deque
from typing import Deque, Dict, List, Tuple

Message = Tuple[str, str]


class InMemoryDialogStore:
    def __init__(self, limit: int = 6) -> None:
        self._dialogs: Dict[int, Deque[Message]] = {}
        self._limit = limit

    def add(self, user_id: int, role: str, text: str) -> List[Message]:
        history = self._dialogs.setdefault(user_id, deque(maxlen=self._limit))
        history.append((role, text))
        return list(history)

    def get(self, user_id: int) -> List[Message]:
        history = self._dialogs.get(user_id)
        if not history:
            return []
        return list(history)

    def reset(self, user_id: int) -> None:
        self._dialogs.pop(user_id, None)
