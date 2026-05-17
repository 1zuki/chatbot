from vista_chatbot.chunking import clean_mdx, load_wiki_chunks


def test_clean_mdx_removes_imports_and_keeps_links():
    raw = "---\ntitle: Test\n---\nimport X from 'y'\n# Hello\nUse [warp](/warp)."
    out = clean_mdx(raw)
    assert "import X" not in out
    assert "Hello" in out
    assert "warp" in out


def test_load_chunks(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "a.mdx").write_text("# Fluff\n" + "Fluff is a cosmetic item. " * 40, encoding="utf-8")
    chunks = load_wiki_chunks(wiki, globs=["**/*.mdx"], chunk_chars=250, chunk_overlap=30)
    assert chunks
    assert chunks[0].source_path == "a.mdx"
