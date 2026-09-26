"""Prompt construction for the grounded answer step.

Prompts live apart from the pipeline so they can be unit-tested, versioned, and
eventually swapped per-tenant without touching orchestration code.
"""

from __future__ import annotations

from app.services.rag.schemas import RetrievedChunk

SYSTEM_PROMPT = """\
You are DocMind, a precise document analyst.

Rules:
1. Answer using ONLY the numbered context passages below.
2. If the answer is absent from the passages, reply exactly: \
"I don't have this information in the document."
3. Cite the passage number for every factual claim, like [1] or [2][3].
4. Never invent page numbers, figures, or names. If a passage does not state it, \
it is not in the document.
5. Be concise and direct. No preamble, no restating the question.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered block the model can cite."""
    blocks: list[str] = []
    for position, retrieved in enumerate(chunks, start=1):
        page = retrieved.chunk.page
        blocks.append(f"[{position}] (page {page})\n{retrieved.chunk.text}")
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_block = build_context_block(chunks)
    return f"{SYSTEM_PROMPT}\nContext passages:\n{context_block}\n\nQuestion: {question}\n"


def build_history_block(messages: list[tuple[str, str]]) -> str:
    """Render prior turns as plain text.

    The conversation is flattened into the prompt rather than threaded through a
    message-native chain: it keeps the provider interface swappable and the
    prompt inspectable in tests and traces.
    """
    if not messages:
        return ""
    lines = [
        f"{'User' if role == 'user' else 'Assistant'}: {content}"
        for role, content in messages
    ]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"
