import csv
from pathlib import Path

from vista_chatbot.config import BotConfig, ChatConfig, LoggingConfig, ModelConfig, PromptConfig, RetrievalConfig, RulesConfig
from vista_chatbot.runtime import BotEngine


def make_engine(tmp_path: Path) -> BotEngine:
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(triggers=["!vista"], ignore_after_send_seconds=8.0),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )
    return BotEngine(cfg)


def test_prefix_can_be_after_decorated_prefix(tmp_path):
    engine = make_engine(tmp_path)
    assert engine._strip_query_prefix("🏕 ➟ TOPAZ ➡ !vista what is fluff") == "what is fluff"


def test_prefix_does_not_match_similar_command(tmp_path):
    engine = make_engine(tmp_path)
    assert engine._strip_query_prefix("!vistaa what is fluff") is None


def test_admin_command_denied_for_non_admin(tmp_path):
    engine = make_engine(tmp_path)
    out = engine.handle_text("<Steve> !vista status")
    assert out == "No permission."


def test_admin_command_allowed_with_parsed_decorated_name(tmp_path):
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(
            triggers=["!vista"],
            ignore_after_send_seconds=8.0,
            admin_names=["Izu"],
            admin_only_commands=False,
        ),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )
    engine = BotEngine(cfg)
    out = engine.handle_text("🏕 ➟ TOPAZ Izu [❄ '24] ➡ !vista status")
    assert out is not None
    assert "Vista online." in out


def test_blacklisted_user_is_silently_ignored(tmp_path):
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(
            triggers=["!vista"],
            ignore_after_send_seconds=8.0,
            blacklisted_users=["Steve"],
            admin_only_commands=False,
        ),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )
    engine = BotEngine(cfg)
    out = engine.handle_text("<Steve> !vista status")
    assert out is None


def test_admin_command_allowed_with_rank_only(tmp_path):
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(
            triggers=["!vista"],
            ignore_after_send_seconds=8.0,
            admin_names=[],
            admin_ranks=["TOPAZ"],
            admin_only_commands=False,
        ),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )
    engine = BotEngine(cfg)
    out = engine.handle_text("🏕 ➟ TOPAZ Notch [❄ '24] ➡ !vista status")
    assert out is not None
    assert "rank=TOPAZ" in out


def test_critical_command_requires_rank_when_configured(tmp_path):
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(
            triggers=["!vista"],
            ignore_after_send_seconds=8.0,
            admin_names=["Izu"],
            admin_ranks=["TOPAZ"],
            admin_only_commands=False,
            require_rank_for_critical_admin_commands=True,
        ),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )
    engine = BotEngine(cfg)
    out = engine.handle_text("<Izu> !vista stop")
    assert out == "No permission."


def test_whoami_uses_parsed_speaker(tmp_path):
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(
            triggers=["!vista"],
            ignore_after_send_seconds=8.0,
            admin_names=["Izu"],
        ),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )
    engine = BotEngine(cfg)
    out = engine.handle_text("🏕 ➟ TOPAZ Izu [❄ '24] ➡ !vista whoami")
    assert out is not None
    assert "speaker=Izu" in out
    assert "rank=TOPAZ" in out


def test_history_toggle_and_status_command(tmp_path):
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(
            triggers=["!vista"],
            ignore_after_send_seconds=8.0,
            admin_names=["Izu"],
            admin_only_commands=False,
            history_enabled=True,
        ),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(log_file=str(tmp_path / "bot.log")),
    )
    engine = BotEngine(cfg)

    off = engine.handle_text("<Izu> !vista history off")
    assert off == "Conversation history disabled and context cleared."

    status = engine.handle_text("<Izu> !vista history status")
    assert status == "Conversation history is off."

    on = engine.handle_text("<Izu> !vista context on")
    assert on == "Conversation history enabled."


def test_user_queries_logged_to_separate_csv(tmp_path):
    query_log_path = tmp_path / "user_queries.csv"
    cfg = BotConfig(
        path=tmp_path / "bot.json",
        project_root=tmp_path,
        chat=ChatConfig(
            triggers=["!vista"],
            ignore_after_send_seconds=8.0,
            history_enabled=False,
        ),
        model=ModelConfig(enabled=False),
        retrieval=RetrievalConfig(enabled=False),
        rules=RulesConfig(command_prefix="!vista"),
        prompt=PromptConfig(),
        logging=LoggingConfig(
            log_file=str(tmp_path / "bot.log"),
            query_log_file=str(query_log_path),
            query_log_enabled=True,
        ),
    )
    engine = BotEngine(cfg)
    reply, _ = engine.answer_query("how to claim land", speaker="Izu", rank="TOPAZ")
    assert reply is not None

    with query_log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["rank", "user", "queries", "reply"]
    assert rows[1][0] == "TOPAZ"
    assert rows[1][1] == "Izu"
    assert rows[1][2] == "how to claim land"
    assert rows[1][3]


def test_long_reply_is_split_not_hard_truncated(tmp_path):
    engine = make_engine(tmp_path)
    long_reply = "word " * 500
    finalized = engine._finalize_reply(long_reply)
    assert finalized is not None
    assert len(finalized) > engine.config.chat.max_chat_chars * 3
    parts = engine.split_reply(finalized)
    assert len(parts) > 3
    assert all(len(p) <= engine.config.chat.max_chat_chars for p in parts)
