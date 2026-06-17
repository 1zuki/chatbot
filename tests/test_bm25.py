from vista_chatbot.bm25 import BM25Index, tokenize


def test_tokenize_normalizes_commands_and_hyphens():
    assert tokenize("`/town set taxes {$}`") == ["town", "set", "taxes"]
    assert tokenize("/warp <warp-name>") == ["warp", "warp", "name"]


def test_bm25_ranks_exact_term_match_first():
    docs = [
        tokenize("Town upkeep costs play a big part in the economy."),
        tokenize("/town set taxes {$} Sets daily taxes."),
        tokenize("Fluff cosmetics are visual only and do not change gameplay."),
    ]
    index = BM25Index.build(docs)
    scores = index.scores(tokenize("how do i set town taxes"))
    assert scores[1] == max(scores)
    assert scores[1] > 0.0


def test_bm25_unknown_terms_score_zero():
    docs = [tokenize("Warps allow players to travel."), tokenize("Nations have upkeep.")]
    index = BM25Index.build(docs)
    scores = index.scores(tokenize("banana telescope xyzzy"))
    assert scores == [0.0, 0.0]


def test_bm25_idf_nonnegative_for_common_terms():
    # A term in every document must not produce a negative score.
    docs = [tokenize("town town"), tokenize("town"), tokenize("town stuff")]
    index = BM25Index.build(docs)
    assert all(s >= 0.0 for s in index.scores(tokenize("town")))


def test_bm25_empty_query_returns_zeros():
    index = BM25Index.build([tokenize("a b c"), tokenize("d e f")])
    assert index.scores([]) == [0.0, 0.0]
