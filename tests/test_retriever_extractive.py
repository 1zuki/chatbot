from vista_chatbot.chunking import Chunk
from vista_chatbot.retriever import SearchResult, extractive_answer, extractive_candidates


def _chunk(*, text: str, source: str = "wiki/page.mdx") -> Chunk:
    return Chunk(
        chunk_id="c1",
        source_path=source,
        title="Test",
        heading_path=["Test"],
        text=text,
        start_char=0,
        end_char=len(text),
    )


def test_extractive_candidates_rank_by_query_overlap():
    results = [
        SearchResult(
            chunk=_chunk(
                text="Use /claim to claim land. Then use /trust to add friends."
            ),
            score=0.60,
        ),
        SearchResult(
            chunk=_chunk(
                text="Fluff cosmetics are visual only and do not change gameplay.",
                source="wiki/fluff.mdx",
            ),
            score=0.95,
        ),
    ]
    candidates = extractive_candidates("how to claim land", results, max_candidates=4)
    assert candidates
    assert "claim land" in candidates[0].lower()


def test_extractive_answer_uses_top_chunk_when_no_sentence_overlap():
    results = [
        SearchResult(
            chunk=_chunk(text="Spawn warp: use /warp spawn for the main city hub."),
            score=0.81,
        )
    ]
    out = extractive_answer("banana telescope", results, max_chars=200)
    assert out.startswith("Wiki says:")
    assert "spawn warp" in out.lower()


def test_extractive_answer_prefers_creation_step_over_warning_sentence():
    results = [
        SearchResult(
            chunk=_chunk(
                text=(
                    "To create your nation, your town is required to have at least 10 residents. "
                    "Failure to afford the costs will lead to your nation being bankrupt."
                ),
                source="wiki/nations/create.mdx",
            ),
            score=0.666,
        )
    ]
    out = extractive_answer("how do i create a nation", results, max_chars=220).lower()
    assert "to create your nation" in out
    assert "failure to afford" not in out
