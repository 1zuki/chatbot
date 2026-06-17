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
    assert "/wiki" in out.lower()
    assert "ask staff" in out.lower()


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


def test_extractive_answer_howto_can_merge_requirement_and_command():
    results = [
        SearchResult(
            chunk=_chunk(
                text=(
                    "To create your nation, your town is required to have at least 10 residents. "
                    "Once that requirement is met, type /n new [Nation Name]."
                ),
                source="wiki/nations/create.mdx",
            ),
            score=0.71,
        )
    ]
    out = extractive_answer("how do i create a nation", results, max_chars=260).lower()
    assert "required to have at least 10 residents" in out
    assert "/n new" in out


def test_extractive_answer_cost_query_can_pick_warning_like_line():
    results = [
        SearchResult(
            chunk=_chunk(
                text=(
                    "You should have at least $125 000 in-game money for establishing the nation. "
                    "Failure to afford the costs will lead to your nation being bankrupt."
                ),
                source="wiki/nations/create.mdx",
            ),
            score=0.69,
        )
    ]
    out = extractive_answer("what is nation creation cost", results, max_chars=220).lower()
    assert "$125 000" in out or "125 000" in out


def test_extractive_answer_prefers_navigation_command_over_set_command():
    results = [
        SearchResult(
            chunk=_chunk(
                text=(
                    "| | /town set spawn | Sets the town spawn. "
                    "| | /town spawn | Teleports you to your town spawn."
                ),
                source="wiki/town/commands.mdx",
            ),
            score=0.73,
        )
    ]
    out = extractive_answer("how do i go to my town spawn", results, max_chars=220).lower()
    assert "/town spawn" in out
    assert "/town set spawn" not in out


def test_extractive_answer_unknown_has_wiki_staff_hint():
    out = extractive_answer("hello", [], max_chars=220).lower()
    assert "/wiki" in out
    assert "ask staff" in out


def test_extractive_answer_avoids_guess_for_generic_richest_question():
    results = [
        SearchResult(
            chunk=_chunk(
                text=(
                    "Upon joining the server for the first time, players start with $5 000, "
                    "and a few basic items to assist with the early game experience."
                ),
                source="wiki/economy/getting-started.mdx",
            ),
            score=0.66,
        )
    ]
    out = extractive_answer("how to be richest in server", results, max_chars=220).lower()
    assert "/wiki" in out
    assert "ask staff" in out


def test_howto_promotes_command_when_it_is_the_dense_winner():
    # "how do i claim a land": a marketing line has higher raw overlap (claim +
    # land) than the actionable command (claim only), so the overlap-first sort
    # places it first. But the embedder scored the command strictly more relevant
    # (0.495 > 0.415), so it should be promoted to the answer.
    results = [
        SearchResult(
            chunk=_chunk(
                text="Use the Towny plugin to claim and secure your land and valuables.",
                source="wiki/towny/index.mdx",
            ),
            score=0.415,
        ),
        SearchResult(
            chunk=_chunk(
                text=(
                    "To claim additional chunks, stand in an adjacent unclaimed chunk "
                    "and type /t claim."
                ),
                source="wiki/towny/creating-a-town.mdx",
            ),
            score=0.495,
        ),
    ]
    out = extractive_answer("how do i claim a land", results, max_chars=240).lower()
    assert "/t claim" in out
    assert "towny plugin" not in out


def test_howto_does_not_promote_command_the_embedder_ranked_lower():
    # "how can i increase my star rank": the only command is a TOWN-rank command
    # (/t rank add) that the embedder scored well below the on-topic Stars intro
    # (0.287 < 0.388). Promoting it would give a confidently wrong (off-topic)
    # answer, so the score gate must keep it suppressed.
    results = [
        SearchResult(
            chunk=_chunk(
                text="Each star rank unlocks new perks that enhance the gameplay experience.",
                source="wiki/gameplay/stars.mdx",
            ),
            score=0.388,
        ),
        SearchResult(
            chunk=_chunk(
                text="Ranks can be given to residents with the command /t rank add [Username] [Rank].",
                source="wiki/towny/town-ranks.mdx",
            ),
            score=0.287,
        ),
    ]
    out = extractive_answer("how can i increase my star rank", results, max_chars=240).lower()
    assert "/t rank add" not in out
    assert "star rank unlocks" in out


def test_extractive_answer_navigation_query_requires_actionable_line():
    results = [
        SearchResult(
            chunk=_chunk(
                text="The End. Player-made explosions have been disabled.",
                source="wiki/worlds/the-end.mdx",
            ),
            score=0.62,
        )
    ]
    out = extractive_answer("how can i go to end", results, max_chars=220).lower()
    assert "/wiki" in out
    assert "ask staff" in out
