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
