"""End-to-end "a kid asks the wiki bot" evaluation.

These tests drive the real retrieval + extractive-answer path that players hit
in chat. The default config sets ``model.enabled=False``, so the live bot's
``generate_or_fallback`` resolves to ``extractive_answer`` over the dense+BM25
retrieval results -- which is exactly what this eval calls directly.

Questions are phrased the way the bot's actual audience phrases them: lowercase,
misspelled, no punctuation, vague. This *is* the input distribution -- the bot
exists for the players who won't read the wiki themselves.

The suite needs the built embedding index (``artifacts/retriever``) and
``sentence-transformers`` installed. It skips cleanly when either is missing, so
it behaves as a local/integration eval rather than a CI unit test.

Cases are grouped:

* ``answered_well``   -- the bot returns the right fact today (hard assertions).
* ``out_of_scope``    -- not in the wiki; the bot must admit it doesn't know.
* ``known_weakness_*`` -- documented current failures, marked ``xfail`` so the
  suite stays green while tracking the gap. Two flavours:
    - false negative: the answer exists but the bot says "I don't know".
    - false positive: the bot confidently returns an unrelated/wrong snippet.
  If one starts passing (XPASS), the bot improved -- promote the case to
  ``answered_well`` and drop the marker.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# Keep model loading offline so a cached embedding model never blocks the eval
# on a network round-trip; the model is already downloaded if the index exists.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers not installed; kid-question eval is integration-only",
)

from vista_chatbot.config import BotConfig
from vista_chatbot.retriever import (
    UNKNOWN_WIKI_REPLY,
    EmbeddingIndex,
    _parse_source_date,
    extractive_answer,
)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "bot.json"


@pytest.fixture(scope="module")
def ask():
    """Return an ``ask(question) -> reply`` bound to the live index.

    Loads the embedding index once per module. Skips the whole module if the
    config or built index is absent, so the eval only runs where it can.
    """
    if not _CONFIG_PATH.exists():
        pytest.skip(f"config not found at {_CONFIG_PATH}")
    cfg = BotConfig.load(_CONFIG_PATH)
    index_dir = cfg.index_dir
    if not (index_dir / "embeddings.npy").exists():
        pytest.skip(f"no built index at {index_dir}; run scripts/build_wiki_index.py")

    index = EmbeddingIndex.load(index_dir, cfg.retrieval.embedding_model)

    def ask(question: str) -> str:
        results = index.search(
            question,
            top_k=cfg.retrieval.top_k,
            min_score=cfg.retrieval.min_score,
        )
        return extractive_answer(question, results, max_chars=cfg.chat.max_chat_chars)

    return ask


@pytest.fixture(scope="module")
def search():
    """Like ``ask`` but returns the raw ranked ``SearchResult`` list.

    Recency tests assert on which *source file* ranks first, which the composed
    text answer hides, so they need the results rather than the final string.
    """
    if not _CONFIG_PATH.exists():
        pytest.skip(f"config not found at {_CONFIG_PATH}")
    cfg = BotConfig.load(_CONFIG_PATH)
    index_dir = cfg.index_dir
    if not (index_dir / "embeddings.npy").exists():
        pytest.skip(f"no built index at {index_dir}; run scripts/build_wiki_index.py")

    index = EmbeddingIndex.load(index_dir, cfg.retrieval.embedding_model)

    def search(question: str):
        return index.search(
            question,
            top_k=cfg.retrieval.top_k,
            min_score=cfg.retrieval.min_score,
        )

    return search


def _is_unknown(reply: str) -> bool:
    return reply.strip() == UNKNOWN_WIKI_REPLY


# ---------------------------------------------------------------------------
# answered_well: kid-phrased questions the bot gets right today.
# Each case: (question, [expected substrings], mode) where mode is "any"/"all".
# Substrings are matched case-insensitively against the reply.
# ---------------------------------------------------------------------------
ANSWERED_WELL = [
    ("how do i make a town", ["/t new"], "any"),
    ("how do i claim land", ["/t claim"], "any"),
    ("how to join a nation", ["/nation join"], "any"),
    ("how do i vote", ["/vote"], "any"),
    ("what is the family system", ["family"], "any"),
    ("how do i sell stuff", ["auction"], "any"),
    ("how do i warp somewhere", ["/warp"], "any"),
    ("wat is custom enchant", ["enchant"], "any"),
    ("what does silk touch do", ["silk touch", "spawner"], "any"),
    ("how do i transfer my pet", ["leash", "pet ownership"], "any"),
    # Quantity question ("how much") about a plain fact, not a procedure: the
    # how-to confidence gate used to reject the $5,000 answer outright.
    ("how much money do i start with", ["5,000"], "any"),
    # Pricing guide, tier-specific rows (regression coverage for the live bot).
    ("whats the price for slingshot", ["250,000"], "any"),
    ("whats the price for inquisitive", ["6,000,000"], "any"),
    ("whats the price for detonate", ["1,000,000"], "any"),
    ("whats the price for atomic detonate", ["2,000,000"], "any"),
]


@pytest.mark.parametrize(
    "question, expected, mode",
    ANSWERED_WELL,
    ids=[q for q, _, _ in ANSWERED_WELL],
)
def test_kid_question_answered_well(ask, question, expected, mode):
    reply = ask(question).lower()
    assert not _is_unknown(reply), f"bot unexpectedly gave up on: {question!r}"
    needles = [e.lower() for e in expected]
    if mode == "all":
        missing = [n for n in needles if n not in reply]
        assert not missing, f"{question!r} -> {reply!r} missing {missing}"
    else:
        assert any(n in reply for n in needles), (
            f"{question!r} -> {reply!r} matched none of {needles}"
        )


# ---------------------------------------------------------------------------
# out_of_scope: nothing in the wiki answers these, so the bot must say so
# rather than grabbing a vaguely-related snippet.
# ---------------------------------------------------------------------------
OUT_OF_SCOPE = [
    "whats your favorite color",
    "do you like me",
]


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_says_dont_know(ask, question):
    reply = ask(question)
    assert _is_unknown(reply), f"{question!r} should be unknown, got: {reply!r}"


# ---------------------------------------------------------------------------
# known_weakness (false negative): the answer IS in the wiki, but the bot says
# "I don't know". Marked xfail; asserts the fact that SHOULD appear.
# ---------------------------------------------------------------------------
KNOWN_FALSE_NEGATIVES = [
    # Nether access unlocks at Star 5 (earth/gameplay/stars.mdx); the kid
    # phrasing "get to the nether" misses it.
    ("how do i get to the nether", "5"),
    # Flight in your own town unlocks at Star 25.
    ("what star do i need to fly", "25"),
    # The End unlocks at Star 10.
    ("what do i need for the end", "10"),
    # "how much is X" doesn't trip the cost path the way "price for X" does, so
    # the Detonate price (1,000,000$) is missed.
    ("how much is detonate", "1,000,000"),
]


@pytest.mark.parametrize(
    "question, expected",
    KNOWN_FALSE_NEGATIVES,
    ids=[q for q, _ in KNOWN_FALSE_NEGATIVES],
)
@pytest.mark.xfail(
    reason="known retrieval gap: answer exists in the wiki but the bot says it doesn't know",
    strict=False,
)
def test_known_weakness_false_negative(ask, question, expected):
    reply = ask(question).lower()
    assert not _is_unknown(reply), f"still giving up on {question!r}"
    assert expected.lower() in reply, f"{question!r} -> {reply!r} missing {expected!r}"


# ---------------------------------------------------------------------------
# known_weakness (false positive): the bot confidently returns an unrelated or
# flat-out wrong snippet. Marked xfail; asserts the known-wrong phrase is
# ABSENT (the behaviour we want once it's fixed). Currently the phrase is
# present, so the test xfails.
# ---------------------------------------------------------------------------
KNOWN_FALSE_POSITIVES = [
    # "teleport to a friend" returns the command that REMOVES all friends.
    ("how do i teleport to a friend", "removes all friends"),
    # "can i fly" returns the anti-cheat ban list line "Speed and fly hacks".
    ("can i fly", "hacks"),
    # "how to set home" returns plot join-day requirement commands.
    ("how to set home", "minjoindays"),
    # "wats spawn" (the hub) returns mob-spawner stacking mechanics.
    ("wats spawn", "nearby stacks"),
    # Dragon taming isn't a feature; the bot returns the Star 28 Dragon Head row.
    ("how do i tame a dragon", "dragon head"),
    # "will you be my friend" grabs the command that REMOVES all friends.
    ("will you be my friend", "removes all friends"),
    # "whats the meaning of life" matches the word "life" in remnants lore text.
    ("whats the meaning of life", "aethelgard"),
]


@pytest.mark.parametrize(
    "question, wrong_phrase",
    KNOWN_FALSE_POSITIVES,
    ids=[q for q, _ in KNOWN_FALSE_POSITIVES],
)
@pytest.mark.xfail(
    reason="known precision gap: bot confidently returns an unrelated/wrong snippet",
    strict=False,
)
def test_known_weakness_false_positive(ask, question, wrong_phrase):
    reply = ask(question).lower()
    assert wrong_phrase.lower() not in reply, (
        f"{question!r} still returns the wrong snippet containing {wrong_phrase!r}: {reply!r}"
    )


# ---------------------------------------------------------------------------
# regression guards: questions the wiki has no clean answer for, where the bot
# used to return a confidently-wrong snippet that has since been suppressed.
# Unlike the xfail cases above, these are hard assertions -- the wrong snippet
# must stay gone. (There is still no right answer to return, so we only assert
# the bad one is absent, not that any particular fact is present.)
# ---------------------------------------------------------------------------
SUPPRESSED_FALSE_POSITIVES = [
    # "how do u get stars": the generic verb "get" used to match a jail-bail
    # command. "get" is now a stopword, so that spurious hit is gone.
    ("how do u get stars", "jail"),
    # "how do i get a pet": used to return the unrelated /imageframe map command
    # on the same "get" match. The wiki has no get-a-pet mechanic (only pet
    # ownership transfer), so we only require the imageframe garbage to be gone.
    ("how do i get a pet", "imageframe"),
]


@pytest.mark.parametrize(
    "question, wrong_phrase",
    SUPPRESSED_FALSE_POSITIVES,
    ids=[q for q, _ in SUPPRESSED_FALSE_POSITIVES],
)
def test_suppressed_false_positive_stays_gone(ask, question, wrong_phrase):
    reply = ask(question).lower()
    assert wrong_phrase.lower() not in reply, (
        f"{question!r} regressed -- wrong snippet {wrong_phrase!r} is back: {reply!r}"
    )


# ---------------------------------------------------------------------------
# recency: "whats new / latest / changed" questions must surface the most
# recent dated entry, not an older one that happens to be a closer topical
# match. Dated content lives in wiki/{changelog,announcement,roadmap}/ named
# DD-MM-YY.md. These tests discover the newest file at runtime rather than
# hard-coding a date, so they don't rot as new entries are added.
# ---------------------------------------------------------------------------
_WIKI_DIR = Path(__file__).resolve().parents[1] / "wiki"
_DATED_DIRS = ("changelog", "announcement", "roadmap")


def _newest_changelog_stem() -> str | None:
    """Newest changelog filename stem (e.g. "10-06-26"), by parsed DD-MM-YY."""
    cl_dir = _WIKI_DIR / "changelog"
    if not cl_dir.is_dir():
        return None
    best: tuple[tuple[int, int, int], str] | None = None
    for p in cl_dir.glob("*.md"):
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2})", p.stem)
        if not m:
            continue
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        key = (yy, mm, dd)
        if best is None or key > best[0]:
            best = (key, p.stem)
    return best[1] if best else None


RECENCY_QUERIES = [
    "whats new",
    "latest update",
    "what was changed recently",
    "any new announcements",
]


@pytest.mark.parametrize("question", RECENCY_QUERIES)
def test_recency_query_surfaces_newest_changelog_first(search, question):
    newest = _newest_changelog_stem()
    if newest is None:
        pytest.skip("no dated changelog files to test recency against")
    results = search(question)
    assert results, f"{question!r} returned nothing"
    # The top result should be a dated entry, and no *older* dated entry should
    # outrank the newest changelog. We assert the newest changelog appears and
    # that the first dated result is not older than it.
    dated = [
        (r, re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2})", Path(r.chunk.source_path).stem))
        for r in results
    ]
    dated = [(r, m) for r, m in dated if m]
    assert dated, f"{question!r} surfaced no dated content: {[r.chunk.source_path for r in results]}"
    first_r, first_m = dated[0]
    first_key = (int(first_m.group(3)), int(first_m.group(2)), int(first_m.group(1)))
    nm = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2})", newest)
    newest_key = (int(nm.group(3)), int(nm.group(2)), int(nm.group(1)))
    assert first_key >= newest_key, (
        f"{question!r} ranked older dated entry {first_r.chunk.source_path!r} "
        f"above newest changelog {newest!r}"
    )


# Topical questions that must stay on the evergreen wiki, NOT be answered by a
# dated changelog/announcement — even when a changelog chunk out-*cosines* the
# wiki page on its own topic. "how can i increase my star rank" is the real
# regression from the live log: a Star Path Adjustments changelog scored 0.521,
# above every stars.mdx chunk, and won at the extraction layer on term overlap.
TOPICAL_NOT_DATED = [
    "how do capsules work",
    "how can i increase my star rank",
    "how do i increase my star",
]


@pytest.mark.parametrize("question", TOPICAL_NOT_DATED)
def test_topical_query_not_hijacked_by_changelog(search, question):
    results = search(question)
    assert results, f"{question!r} returned nothing"
    # No dated entry may sit at the top of a topical query when evergreen wiki
    # content is available; the whole top result must come from the wiki.
    top = results[0].chunk.source_path
    assert _parse_source_date(top) is None, (
        f"{question!r} hijacked by dated content: {top!r}"
    )
    assert top.startswith("vista-src/"), (
        f"{question!r} top result is not an evergreen wiki page: {top!r}"
    )
