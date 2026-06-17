from __future__ import annotations

from pathlib import Path

import numpy as np

from vista_chatbot.chunking import Chunk
from vista_chatbot.retriever import EmbeddingIndex


def _chunk(text: str, source: str, heading: list[str]) -> Chunk:
    return Chunk(
        chunk_id=source,
        source_path=source,
        title=heading[0] if heading else "Test",
        heading_path=heading,
        text=text,
        start_char=0,
        end_char=len(text),
    )


def _index_with(chunks: list[Chunk], embeddings: np.ndarray) -> EmbeddingIndex:
    index = EmbeddingIndex(index_dir=Path("/tmp/vista-test-index"), embedding_model="stub")
    index.chunks = chunks
    index.embeddings = embeddings.astype("float32")

    class _StubModel:
        def __init__(self, vec: np.ndarray):
            self._vec = vec

        def encode(self, *_args, **_kwargs):
            return self._vec[None, :].astype("float32")

    # Query vector points "between" two chunks so cosine alone leaves the exact
    # command chunk ranked second; BM25 must pull it up on exact term match.
    index._model = _StubModel(np.array([0.6, 0.8, 0.0], dtype="float32"))
    return index


def _index_with_query(
    chunks: list[Chunk], embeddings: np.ndarray, query_vec: np.ndarray
) -> EmbeddingIndex:
    """Like ``_index_with`` but the stub model returns a caller-chosen query vector."""
    index = EmbeddingIndex(index_dir=Path("/tmp/vista-test-index"), embedding_model="stub")
    index.chunks = chunks
    index.embeddings = embeddings.astype("float32")

    class _StubModel:
        def __init__(self, vec: np.ndarray):
            self._vec = vec

        def encode(self, *_args, **_kwargs):
            return self._vec[None, :].astype("float32")

    index._model = _StubModel(query_vec.astype("float32"))
    return index


def _table_fragment(rows: list[str], source: str, heading: list[str], start_char: int) -> Chunk:
    """A chunk shaped like ``_chunk_table`` output: context word then ``|`` rows."""
    text = (heading[-1] if heading else "Table") + "\n" + "\n".join(rows)
    return Chunk(
        chunk_id=f"{source}:{start_char}",
        source_path=source,
        title=heading[0] if heading else "Test",
        heading_path=heading,
        text=text,
        start_char=start_char,
        end_char=start_char + len(text),
    )


def test_bm25_reorders_within_dense_gate():
    # The dense winner is semantically about taxation but shares NO query terms
    # ("set"/"town"/"taxes") — exactly the case where lexical match should pull
    # the exact-command chunk above a purely-semantic neighbour. (Note the path
    # tokens leak into the embed text, so the prose chunk's path avoids the
    # query terms too.)
    chunks = [
        _chunk("Mayors fund their settlement by charging residents a daily levy from the bank.",
               "earth/economy/levy.mdx", ["Resident Levy"]),
        _chunk("/town set taxes {$} Sets daily taxes.",
               "earth/towny/commands.mdx", ["/town"]),
    ]
    # Chunk 0 is the closer cosine match; chunk 1 clears the gate but ranks second.
    embeddings = np.array([[0.6, 0.8, 0.0], [0.5, 0.5, 0.7]], dtype="float32")
    index = _index_with(chunks, embeddings)

    results = index.search("set town taxes", top_k=2, min_score=0.18)
    assert results
    # Exact-term query should surface the command chunk first after fusion.
    assert results[0].chunk.source_path == "earth/towny/commands.mdx"
    # Score must remain the dense cosine value (not a fused/BM25 score).
    expected = float(index.embeddings[1] @ np.array([0.6, 0.8, 0.0], dtype="float32"))
    assert abs(results[0].score - expected) < 1e-5


def test_dense_gate_still_excludes_unrelated_chunks():
    chunks = [
        _chunk("Warps let you teleport across the map to shops and bases.",
               "earth/gameplay/warps.mdx", ["Warps"]),
        _chunk("Cactus farms are an unrelated topic about passive income strategies.",
               "earth/economy/farms.mdx", ["Farms"]),
    ]
    # Second chunk is orthogonal to the query vector -> cosine below the gate,
    # so even a strong BM25 term hit must not resurrect it.
    embeddings = np.array([[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    index = _index_with(chunks, embeddings)

    results = index.search("cactus farms unrelated topic", top_k=5, min_score=0.18)
    sources = {r.chunk.source_path for r in results}
    assert "earth/economy/farms.mdx" not in sources


def test_no_term_overlap_falls_back_to_dense_order():
    chunks = [
        _chunk("Alpha content about spawning and warps near the hub.",
               "a.mdx", ["A"]),
        _chunk("Beta content describing town claims and plots.",
               "b.mdx", ["B"]),
    ]
    embeddings = np.array([[0.6, 0.8, 0.0], [0.55, 0.78, 0.1]], dtype="float32")
    index = _index_with(chunks, embeddings)

    # Query terms appear in neither chunk -> BM25 contributes nothing, order
    # should match dense cosine ranking exactly.
    results = index.search("xyzzy plugh frobnicate", top_k=2, min_score=0.0)
    assert [r.chunk.source_path for r in results] == ["a.mdx", "b.mdx"]


def _table_fragment(rows: list[str], source: str, heading: list[str], start: int) -> Chunk:
    text = heading[-1] + "\n" + "\n".join(rows)
    return Chunk(
        chunk_id=f"{source}:{start}",
        source_path=source,
        title=heading[0] if heading else "Test",
        heading_path=heading,
        text=text,
        start_char=start,
        end_char=start + len(text),
    )


def test_split_table_pulls_in_low_cosine_sibling():
    # A 4-row Stars table split into two fragments sharing (source, heading).
    # Fragment A (ranks 1-2) is the cosine winner; fragment B (ranks 3-4) scores
    # below the gate and would never be retrieved on its own. Once A surfaces, B
    # must be spliced in so the row-aware extractor sees rank 3.
    frag_a = _table_fragment(
        ["| 1 | 5 votes | Silk Touch spawners |", "| 2 | $5,000 | Elevator |"],
        "earth/gameplay/stars.mdx", ["Stars"], start=0,
    )
    frag_b = _table_fragment(
        ["| 3 | $10,000 | Chest shop |", "| 4 | $15,000 | More chest shops |"],
        "earth/gameplay/stars.mdx", ["Stars"], start=200,
    )
    prose = _chunk("Unrelated page about cactus farms and passive income.",
                   "earth/economy/farms.mdx", ["Farms"])
    chunks = [frag_a, frag_b, prose]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32"
    )
    # Query points straight at fragment A; B's cosine is 0.0 (below gate).
    index = _index_with_query(chunks, embeddings, np.array([1.0, 0.0, 0.0]))

    results = index.search("star rank 3", top_k=5, min_score=0.18)
    sources = [r.chunk.source_path for r in results]
    # Both Stars fragments present; the sibling came in despite failing the gate.
    assert sources.count("earth/gameplay/stars.mdx") == 2
    assert "earth/economy/farms.mdx" not in sources
    by_start = {r.chunk.start_char: r for r in results}
    assert 200 in by_start
    # Sibling keeps its own (sub-gate) cosine; the trigger keeps its high score.
    assert abs(by_start[200].score - 0.0) < 1e-6
    assert abs(by_start[0].score - 1.0) < 1e-6
    # Sibling is spliced directly after its triggering fragment.
    assert results[0].chunk.start_char == 0
    assert results[1].chunk.start_char == 200


def test_single_chunk_table_has_no_siblings():
    # A table that fits in one chunk has no siblings to complete -> no-op.
    solo = _table_fragment(
        ["| Detonate | Excavate a 3x3 area |", "| Smelt | Auto-smelt drops |"],
        "earth/economy/enchants.mdx", ["Enchants"], start=0,
    )
    other = _chunk("Warps let you teleport across the map to shops and bases.",
                   "earth/gameplay/warps.mdx", ["Warps"])
    chunks = [solo, other]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32")
    index = _index_with_query(chunks, embeddings, np.array([1.0, 0.0, 0.0]))

    results = index.search("detonate enchant", top_k=5, min_score=0.18)
    # Only the one gated chunk; nothing spurious spliced in.
    assert [r.chunk.source_path for r in results] == ["earth/economy/enchants.mdx"]


def test_prose_chunks_are_not_grouped_as_siblings():
    # Two prose chunks sharing a heading must NOT be treated as table siblings:
    # a sub-gate prose neighbour should stay excluded.
    a = _chunk("Spawn is the central hub where new players arrive first.",
               "earth/gameplay/worlds.mdx", ["Worlds"])
    b = _chunk("The Nether is a separate dangerous dimension unlocked later.",
               "earth/gameplay/worlds.mdx", ["Worlds"])
    chunks = [a, b]
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float32")
    index = _index_with_query(chunks, embeddings, np.array([1.0, 0.0, 0.0]))

    results = index.search("spawn hub players", top_k=5, min_score=0.18)
    # Only the gated prose chunk; the sub-gate sibling is not pulled in.
    assert [r.chunk.start_char for r in results] == [0]


def test_prose_intro_pulls_in_sub_gate_table_under_same_heading():
    # The pricing-guide failure: a tier's prose intro ("Baseline Orb Price...")
    # matches a "price" query strongly, while the longer price-table chunk under
    # the same heading gets length-penalised below the gate and never surfaces.
    # Retrieving the intro must complete the table so the row-aware extractor
    # sees the queried row.
    intro = _chunk(
        "Heroic Tier\nBaseline Orb Price: 450,000$\nBaseline 100% Max Book Price: 200,000$",
        "earth/economy/pricing.mdx", ["Pricing", "Heroic Tier"],
    )
    intro = Chunk(**{**intro.__dict__, "start_char": 0, "end_char": len(intro.text)})
    table = _table_fragment(
        ["| Destruction | 500,000$ |", "| Detonate | 1,000,000$ |", "| Devour | 500,000$ |"],
        "earth/economy/pricing.mdx", ["Pricing", "Heroic Tier"], start=200,
    )
    other = _chunk("Unrelated page about cactus farms and passive income.",
                   "earth/economy/farms.mdx", ["Farms"])
    chunks = [intro, table, other]
    embeddings = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype="float32"
    )
    # Query points at the intro; the table's cosine is 0.0 (below the gate).
    index = _index_with_query(chunks, embeddings, np.array([1.0, 0.0, 0.0]))

    results = index.search("price for detonate", top_k=5, min_score=0.18)
    starts = [r.chunk.start_char for r in results]
    # The sub-gate table is spliced in right after its triggering intro.
    assert starts == [0, 200]
    assert "earth/economy/farms.mdx" not in [r.chunk.source_path for r in results]
    # Table keeps its own sub-gate cosine; the intro keeps its high score.
    by_start = {r.chunk.start_char: r.score for r in results}
    assert abs(by_start[200] - 0.0) < 1e-6
    assert abs(by_start[0] - 1.0) < 1e-6


def test_rare_anchor_rescued_when_crowded_out_of_top_k():
    # Faithful to the live "price for slingshot" case: several generic "price"
    # tier intros dominate the dense+fusion top_k, while the rare anchor
    # "slingshot" lives in one chunk that clears the gate but ranks below it.
    # RRF can't lift it (rank fusion discards the rare term's high IDF), so the
    # rare-anchor rescue must splice it back in.
    intros = [
        _chunk(
            f"{name} Tier Baseline Orb Price is {amt} dollars per orb here.",
            "earth/economy/pricing.mdx", ["Pricing", f"{name} Tier"],
        )
        for name, amt in [
            ("Common", "150,000"), ("Uncommon", "150,000"), ("Rare", "300,000"),
            ("Heroic", "450,000"), ("Legendary", "600,000"),
        ]
    ]
    answer = _chunk(
        "Mythical Tier price list includes Rush Slingshot at 250,000 here.",
        "earth/economy/pricing.mdx", ["Pricing", "Mythical Tier"],
    )
    chunks = intros + [answer]
    # Intros score high on the query vector; the slingshot chunk clears the 0.18
    # gate but sits last on dense, so RRF leaves it out of the top_k=2.
    embeddings = np.array(
        [[0.9, 0, 0], [0.88, 0, 0], [0.86, 0, 0], [0.84, 0, 0], [0.82, 0, 0], [0.25, 0, 0]],
        dtype="float32",
    )
    index = _index_with_query(chunks, embeddings, np.array([1.0, 0.0, 0.0]))

    results = index.search("price for slingshot", top_k=2, min_score=0.18)
    # Without the rescue the two highest "price" intros fill top_k and the
    # slingshot chunk is dropped; the rescue brings it back.
    assert any("Slingshot" in r.chunk.text for r in results)


def test_rare_anchor_rescued_below_dense_gate():
    # The "Atomic Detonate" price row sits below the 0.18 gate (dense 0.15) but
    # "atomic" (a rare anchor) names it exactly. The rescue reaches below the
    # gate down to its relaxed floor (0.10) — fusion never sees a sub-gate chunk.
    intro = _chunk(
        "Rare Tier Baseline Orb Price is 300,000 dollars per orb here.",
        "earth/economy/pricing.mdx", ["Pricing", "Rare Tier"],
    )
    answer = _chunk(
        "Legendary Tier price list includes Atomic Detonate at 2,000,000 here.",
        "earth/economy/pricing.mdx", ["Pricing", "Legendary Tier"],
    )
    chunks = [intro, answer]
    embeddings = np.array([[0.5, 0, 0], [0.15, 0, 0]], dtype="float32")
    index = _index_with_query(chunks, embeddings, np.array([1.0, 0.0, 0.0]))

    results = index.search("price for atomic detonate", top_k=5, min_score=0.18)
    assert any("Atomic Detonate" in r.chunk.text for r in results)


def test_rare_anchor_not_rescued_below_floor():
    # A chunk the embedder scored as truly unrelated (dense 0.05, below the 0.10
    # rescue floor) must NOT be resurrected even on an exact rare-term hit — the
    # floor is what keeps the relaxed gate from admitting noise.
    intro = _chunk(
        "Rare Tier Baseline Orb Price is 300,000 dollars per orb here.",
        "earth/economy/pricing.mdx", ["Pricing", "Rare Tier"],
    )
    answer = _chunk(
        "Legendary Tier price list includes Atomic Detonate at 2,000,000 here.",
        "earth/economy/pricing.mdx", ["Pricing", "Legendary Tier"],
    )
    chunks = [intro, answer]
    embeddings = np.array([[0.5, 0, 0], [0.05, 0, 0]], dtype="float32")
    index = _index_with_query(chunks, embeddings, np.array([1.0, 0.0, 0.0]))

    results = index.search("price for atomic detonate", top_k=5, min_score=0.18)
    assert not any("Atomic Detonate" in r.chunk.text for r in results)
