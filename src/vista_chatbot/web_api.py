from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import BotConfig
from .conversation import ConversationMemory
from .retriever import UNKNOWN_WIKI_REPLY, SearchResult
from .runtime import BotEngine

URL_RE = re.compile(r"https?://[^\s)>\"]+", re.IGNORECASE)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=800)
    session_id: str | None = Field(default=None, max_length=80)


class SourceLink(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    links: list[str]
    sources: list[SourceLink]


@dataclass
class SessionStore:
    max_sessions: int
    max_turns: int

    def __post_init__(self) -> None:
        self._sessions: OrderedDict[str, ConversationMemory] = OrderedDict()

    def get(self, session_id: str) -> ConversationMemory:
        memory = self._sessions.get(session_id)
        if memory is None:
            memory = ConversationMemory(max_turns=self.max_turns)
            self._sessions[session_id] = memory
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return memory


class WebChatService:
    def __init__(
        self,
        *,
        engine: BotEngine,
        max_sessions: int,
        max_session_turns: int,
        public_wiki_base_url: str,
    ):
        self.engine = engine
        self._lock = threading.Lock()
        self.store = SessionStore(max_sessions=max_sessions, max_turns=max_session_turns)
        self.public_wiki_base_url = public_wiki_base_url.strip().rstrip("/")

    def chat(self, *, message: str, session_id: str) -> tuple[str, list[SearchResult]]:
        clean = message.strip()
        if not clean:
            return UNKNOWN_WIKI_REPLY, []
        with self._lock:
            self.engine.memory = self.store.get(session_id)
            reply, results = self.engine.answer_query(
                clean,
                speaker=f"web:{session_id}",
                rank="web",
                apply_cooldown=False,
            )
        return reply or UNKNOWN_WIKI_REPLY, results

    def source_links(self, results: list[SearchResult], *, limit: int = 3) -> list[SourceLink]:
        out: list[SourceLink] = []
        if not self.public_wiki_base_url:
            return out
        seen: set[str] = set()
        for r in results:
            url = source_path_to_url(r.chunk.source_path, base_url=self.public_wiki_base_url)
            if not url or url in seen:
                continue
            seen.add(url)
            title = " > ".join(r.chunk.heading_path) if r.chunk.heading_path else r.chunk.title
            out.append(SourceLink(title=title or "Wiki page", url=url))
            if len(out) >= limit:
                break
        return out


def create_app(
    *,
    config_path: str | Path,
    host_for_sources: str | None = None,
    cors_origins: list[str] | None = None,
    max_sessions: int = 500,
    max_session_turns: int = 20,
) -> FastAPI:
    cfg = BotConfig.load(config_path)
    engine = BotEngine(cfg)
    engine.warmup()
    public_base = host_for_sources or getattr(cfg.chat, "public_wiki_base_url", "")
    service = WebChatService(
        engine=engine,
        max_sessions=max_sessions,
        max_session_turns=max_session_turns,
        public_wiki_base_url=public_base,
    )

    app = FastAPI(title="Vista Chatbot API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        session_id = sanitize_session_id(payload.session_id) or new_session_id()
        reply, results = service.chat(message=payload.message, session_id=session_id)
        source_links = service.source_links(results)
        links = unique_urls_from_text(reply)
        for item in source_links:
            if item.url not in links:
                links.append(item.url)
        return ChatResponse(
            session_id=session_id,
            reply=reply,
            links=links,
            sources=source_links,
        )

    return app


def sanitize_session_id(value: str | None) -> str | None:
    if not value:
        return None
    out = value.strip()
    if not out:
        return None
    out = re.sub(r"[^a-zA-Z0-9_-]", "", out)
    return out[:80] or None


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def unique_urls_from_text(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_RE.findall(text):
        url = match.rstrip(".,;:")
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def source_path_to_url(source_path: str, *, base_url: str) -> str | None:
    if not base_url:
        return None
    p = source_path.replace("\\", "/").strip("/")
    for prefix in ("src/content/docs/", "content/docs/", "docs/"):
        if p.startswith(prefix):
            p = p[len(prefix) :]
            break
    p = re.sub(r"\.(md|mdx)$", "", p, flags=re.IGNORECASE)
    if p.endswith("/index"):
        p = p[: -len("/index")]
    p = p.strip("/")
    if not p:
        return base_url + "/"
    return f"{base_url}/{p}/"
