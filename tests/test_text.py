from vista_chatbot.text import contains_trigger, parse_minecraft_chat, split_for_chat, strip_trigger


def test_parse_angle_chat():
    parsed = parse_minecraft_chat("<Steve> izu what is fluff?")
    assert parsed.speaker == "Steve"
    assert parsed.rank is None
    assert parsed.content == "izu what is fluff?"


def test_trigger_strip():
    assert contains_trigger("hello izukia what is marriage", ["izu", "izukia"])
    assert strip_trigger("izukia what is marriage", ["izu", "izukia"]) == "what is marriage"


def test_split_for_chat():
    parts = split_for_chat("a " * 300, 80)
    assert all(len(p) <= 80 for p in parts)
    assert len(parts) > 1


def test_parse_decorated_arrow_chat():
    parsed = parse_minecraft_chat("🏕 ➟ TOPAZ Izu [❄ '24] ➡ !vista whats bread")
    assert parsed.speaker == "Izu"
    assert parsed.rank == "TOPAZ"
    assert parsed.content == "!vista whats bread"


def test_parse_color_code_decorated_chat():
    parsed = parse_minecraft_chat("§a🏙 ➟ TOPAZ ➡ §f!vista status")
    assert parsed.speaker == "TOPAZ"
    assert parsed.rank is None
    assert parsed.content == "!vista status"


def test_parse_decorated_rank_name_without_flair():
    parsed = parse_minecraft_chat("🏕 ➟ TOPAZ Notch ➡ !vista status")
    assert parsed.speaker == "Notch"
    assert parsed.rank == "TOPAZ"
    assert parsed.content == "!vista status"
