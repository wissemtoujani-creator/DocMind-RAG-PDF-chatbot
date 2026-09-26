"""Conversation history persistence.

History is stored per conversation so that two users asking about two documents
never share a memory. The previous implementation kept a single module-level
``chat_history`` list, which leaked every user's turns to every other user.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Protocol

from app.core.logging import get_logger
from app.services.rag.schemas import ChatMessage

logger = get_logger(__name__)


class ConversationRepository(Protocol):
    async def history(self, conversation_id: str, *, limit: int) -> list[ChatMessage]: ...

    async def append(self, conversation_id: str, messages: list[ChatMessage]) -> None: ...

    async def clear(self, conversation_id: str) -> None: ...


class InMemoryConversationRepository:
    def __init__(self, *, max_turns: int = 200) -> None:
        self._max_messages = max_turns * 2
        self._history: dict[str, deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self._max_messages)
        )
        self._lock = asyncio.Lock()

    async def history(self, conversation_id: str, *, limit: int) -> list[ChatMessage]:
        messages = list(self._history.get(conversation_id, ()))
        return messages[-limit:]

    async def append(self, conversation_id: str, messages: list[ChatMessage]) -> None:
        async with self._lock:
            self._history[conversation_id].extend(messages)
        logger.info(
            "repository.history_appended",
            conversation_id=conversation_id,
            messages=len(messages),
        )

    async def clear(self, conversation_id: str) -> None:
        async with self._lock:
            self._history.pop(conversation_id, None)
