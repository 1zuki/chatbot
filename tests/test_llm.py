from pathlib import Path

from vista_chatbot.chunking import Chunk
from vista_chatbot.config import BotConfig, ChatConfig, LoggingConfig, ModelConfig, PromptConfig, RetrievalConfig, RulesConfig
from vista_chatbot.llm import LocalGenerator
from vista_chatbot.retriever import SearchResult, UNKNOWN_WIKI_REPLY


def _cfg(tmp_path: Path) -> BotConfig:
    return BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(triggers=["!vista"]),
        model=ModelConfig(enabled=True, fallback_to_extractive=True),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )


def _result(text: str, score: float = 0.6) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id="c1",
            source_path="wiki/test.mdx",
            title="Test",
            heading_path=["Test"],
            text=text,
            start_char=0,
            end_char=len(text),
        ),
        score=score,
    )


def test_clean_generation_removes_context_preamble_and_shortens():
    raw = (
        "Based on the context, To create a nation, type /n new [Nation Name]. "
        "You need at least 10 residents first. Let me know if you need anything else."
    )
    cleaned = LocalGenerator._clean_generation(raw)
    assert "based on the context" not in cleaned.lower()
    assert "let me know" not in cleaned.lower()
    assert cleaned.count(".") <= 2


def test_generate_or_fallback_skips_llm_when_retrieval_is_low_confidence(tmp_path):
    cfg = _cfg(tmp_path)
    gen = LocalGenerator(cfg)
    # Mark as "loaded" so code path would normally try model generation.
    gen.model = object()
    gen.tokenizer = object()

    out = gen.generate_or_fallback(
        prompt="dummy",
        query="banana telescope",
        results=[_result("Use /warp spawn for hub.")],
        max_chat_chars=180,
    )
    assert out == UNKNOWN_WIKI_REPLY
