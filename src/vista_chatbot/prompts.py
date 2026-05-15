from __future__ import annotations

from typing import Iterable, List, Mapping

from .retriever import RetrievedDoc

SYSTEM_PROMPT = """You are Vista, a concise in-game Minecraft server assistant.
Rules:
- Answer in plain chat-friendly English.
- Prefer facts from the provided server/wiki context.
- If the context does not contain the answer and you are not sure, say you do not know.
- Keep replies short, useful, and non-toxic.
- Do not claim to be an API or external service.
""".strip()


def build_context_block(docs: Iterable[RetrievedDoc], max_chars: int) -> str:
    pieces: List[str] = []
    total = 0
    for i, doc in enumerate(docs, start=1):
        piece = f"[{i}] {doc.compact(max_chars=700)}"
        if total + len(piece) > max_chars:
            break
        pieces.append(piece)
        total += len(piece)
    return "\n".join(pieces)


def build_chat_messages(
    user_message: str,
    docs: Iterable[RetrievedDoc],
    history: List[Mapping[str, str]],
    max_context_chars: int = 1800,
) -> List[dict]:
    context = build_context_block(docs, max_context_chars)
    system = SYSTEM_PROMPT
    if context:
        system += "\n\nRelevant server/Minecraft context:\n" + context

    messages: List[dict] = [{"role": "system", "content": system}]
    messages.extend({"role": h["role"], "content": h["content"]} for h in history)
    messages.append({"role": "user", "content": user_message})
    return messages


def fallback_prompt_from_messages(messages: List[dict]) -> str:
    # Fallback for tokenizers without chat templates.
    out = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        if role == "system":
            out.append(f"System: {content}")
        elif role == "user":
            out.append(f"User: {content}")
        elif role == "assistant":
            out.append(f"Assistant: {content}")
    out.append("Assistant:")
    return "\n\n".join(out)
