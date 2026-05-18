from vista_chatbot.web_api import sanitize_session_id, source_path_to_url, unique_urls_from_text


def test_source_path_to_url_docs_path() -> None:
    out = source_path_to_url(
        "src/content/docs/earth/towny/nations/creating-a-nation.mdx",
        base_url="https://wiki.vistavalley.xyz",
    )
    assert out == "https://wiki.vistavalley.xyz/earth/towny/nations/creating-a-nation/"


def test_source_path_to_url_index_page() -> None:
    out = source_path_to_url("docs/home/index.md", base_url="https://wiki.example.com")
    assert out == "https://wiki.example.com/home/"


def test_sanitize_session_id_strips_invalid_chars() -> None:
    out = sanitize_session_id(" a@b#c? ")
    assert out == "abc"


def test_unique_urls_from_text_keeps_order_and_dedupes() -> None:
    text = "See https://a.test/docs and https://a.test/docs and http://b.test/x."
    out = unique_urls_from_text(text)
    assert out == ["https://a.test/docs", "http://b.test/x"]
