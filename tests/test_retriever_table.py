from vista_chatbot.chunking import Chunk
from vista_chatbot.retriever import SearchResult, extractive_answer


def _chunk(*, text: str, title: str = "Stars", heading: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id="c1",
        source_path="earth/gameplay/stars.mdx",
        title=title,
        heading_path=heading if heading is not None else [],
        text=text,
        start_char=0,
        end_char=len(text),
    )


# A chunk shaped like the chunker actually emits for the Stars table: a topic
# prefix line, then one row per line, header/separator already dropped.
STARS_CHUNK = (
    "Stars\n"
    "| 1🌟 | Complete 5 votes 2,000 XP Coal Block | Collect spawners with Silk Touch |\n"
    "| 5🌟 | $30,000 10,000 XP Gold Block | Access to the Nether Increased /home limit by 1 |\n"
    "| 7🌟 | $50,000 25,000 XP Quartz Block | Assign advanced Hopper Filters |\n"
    "| 10🌟 | $200,000 100,000 XP Netherite Block | Access to The End |"
)


def test_table_query_returns_matching_row_not_dropped_column():
    # The "Perks" answer lives in the 3rd column; older code dropped everything
    # past column 2 and returned a generic line instead.
    results = [SearchResult(chunk=_chunk(text=STARS_CHUNK), score=0.55)]
    out = extractive_answer("what do i get in star rank 5", results, max_chars=240).lower()
    assert "access to the nether" in out


def test_table_query_number_matches_first_cell_not_cost_digits():
    # "rank 5" must hit the 5🌟 row, not another row whose cost contains a 5
    # (e.g. $50,000 in the rank-7 row).
    results = [SearchResult(chunk=_chunk(text=STARS_CHUNK), score=0.55)]
    out = extractive_answer("what do i get in star rank 5", results, max_chars=240).lower()
    assert out.startswith("wiki says: 5")
    assert "quartz" not in out  # rank-7 row must not win on its $50,000 cost


def test_table_query_low_rank_not_beaten_by_generic_intro():
    # Rank 1 row should beat the definitional intro sentence that repeats "star".
    chunk_text = (
        "Each star rank unlocks new perks that enhance the gameplay experience.\n" + STARS_CHUNK
    )
    results = [SearchResult(chunk=_chunk(text=chunk_text), score=0.55)]
    out = extractive_answer("what perk i get from star rank 1", results, max_chars=240).lower()
    assert "silk touch" in out


def test_table_key_match_passes_confidence_gate():
    # A first-cell key hit ("star 7") is confident even if the row shares few
    # other query words, so it must not fall through to the unknown reply.
    results = [SearchResult(chunk=_chunk(text=STARS_CHUNK), score=0.40)]
    out = extractive_answer("what do i need for star 7", results, max_chars=240).lower()
    assert "hopper filters" in out
    assert "/wiki" not in out


def test_definition_query_prefers_intro_over_table_row():
    # A non-numbered "what is" query should return the prose definition, not an
    # arbitrary table row.
    chunk_text = (
        "Stars are free progression paths that reward dedicated players with perks.\n" + STARS_CHUNK
    )
    results = [SearchResult(chunk=_chunk(text=chunk_text), score=0.55)]
    out = extractive_answer("what is stars", results, max_chars=240).lower()
    assert "free progression paths" in out


# Town-levels table: rows keyed by a multi-word name, where one word ("Town")
# is shared across several keys. The row whose full key the query names should
# win over a row that only partially shares a generic word.
TOWN_LEVELS_CHUNK = (
    "Town Levels\n"
    "| Charter Town | 12 | 2.5 | 180 | 75 | 3 |\n"
    "| Large Town | 15 | 3 | 225 | 90 | 3 |\n"
    "| Metropolis | 24 | 5 | 360 | 135 | 5 |\n"
    "| Citadel | 27 | 6 | 405 | 150 | 5 |"
)


def test_name_keyed_row_full_key_beats_partial_topic_word():
    # "metropolis" fully names the Metropolis row's key; "Large Town" only
    # shares the generic "town". The fully-named row must win.
    results = [SearchResult(chunk=_chunk(text=TOWN_LEVELS_CHUNK, title="Levels"), score=0.5)]
    out = extractive_answer("what is town level metropolis", results, max_chars=240).lower()
    assert out.startswith("wiki says: metropolis")
    assert "large town" not in out


# Same key name ("Detonate") appears as a price-guide row and as an enchant
# description row. "What does X do" should prefer the description.
DETONATE_CHUNKS = [
    SearchResult(
        chunk=_chunk(
            text="Price Guide\n| Detonate | 1,000,000$ |\n| Atomic Detonate | 2,000,000$ |",
            title="Community Price Guide",
        ),
        score=0.5,
    ),
    SearchResult(
        chunk=_chunk(
            text=(
                "Heroic Enchantments\n"
                "| Detonate | Chance to instantly excavate blocks in a 3-block area. | 10 | Pickaxe |"
            ),
            title="Custom Enchantments",
        ),
        score=0.5,
    ),
]


def test_what_does_x_do_prefers_description_over_price_row():
    out = extractive_answer("what do detonate do", DETONATE_CHUNKS, max_chars=240).lower()
    assert "excavate blocks" in out
    assert "1,000,000" not in out


# Fix #1: a "create a town" how-to query must not be hijacked by a generic
# /town command row. The command's first cell ("/town") strips to the single
# topic word "town", which the query fully covers — that used to grant a bogus
# key_cover=1.0 and let the bare command outrank the actual how-to sentence.
CREATE_TOWN_CHUNK = (
    "Creating your own town in Towny is a straightforward process.\n"
    "To create your town (which also sets the homeblock at the same spot), "
    "stand at the desired location and type /t new [Town Name].\n"
    "| /town | Shows a player their town's town screen. |\n"
    "| /town online | Shows online players in your town. |\n"
    "| Shows /town commands available. |"
)


def test_howto_query_not_hijacked_by_command_row_topic_word():
    results = [SearchResult(chunk=_chunk(text=CREATE_TOWN_CHUNK, title="Towns"), score=0.55)]
    out = extractive_answer("how do i create a town", results, max_chars=240).lower()
    assert "/t new" in out
    # The bare screen command must not be the answer.
    assert "shows a player their town" not in out


def test_stray_single_cell_pipe_row_gets_no_key_cover():
    # A one-cell "| Shows /town commands available. |" line is prose with a
    # stray pipe, not a key/value data row. It must not win a key_cover match
    # on the topic word "town" over a real how-to sentence.
    results = [SearchResult(chunk=_chunk(text=CREATE_TOWN_CHUNK, title="Towns"), score=0.55)]
    out = extractive_answer("how do i create a town", results, max_chars=240).lower()
    assert "commands available" not in out


# Fix #2: a "what is X" definitional query should return the definition
# sentence even when a how-to sentence repeats the subject noun one extra
# time (higher raw overlap). Mirrors the live "what is custom enchant" case
# where the definition was retrieved at a *higher* dense score but lost on
# overlap.
CUSTOM_ENCHANT_CHUNK = (
    "Custom enchantments allow you to upgrade your tools, weapons, and armor "
    "with specialized abilities that extend far beyond vanilla limits.\n"
    "To enchant an item, open your inventory, click to select the custom "
    "enchantment book, and click directly on top of the equipment piece."
)


def test_definition_query_prefers_definition_over_howto_with_more_overlap():
    results = [SearchResult(chunk=_chunk(text=CUSTOM_ENCHANT_CHUNK, title="Enchants"), score=0.5)]
    out = extractive_answer("what is custom enchant", results, max_chars=240).lower()
    assert "allow you to upgrade" in out
    assert not out.startswith("wiki says: to enchant an item")


def test_definition_tier_inert_for_howto_queries():
    # The same chunk under a how-to query must still surface the how-to step,
    # proving the definition tier only fires for "what is" phrasing.
    results = [SearchResult(chunk=_chunk(text=CUSTOM_ENCHANT_CHUNK, title="Enchants"), score=0.5)]
    out = extractive_answer("how do i enchant an item", results, max_chars=240).lower()
    assert "open your inventory" in out


# Per server-owner request: a price question gets a community-pricing disclaimer
# instead of the "Wiki says:" prefix, since item prices are community-sourced
# averages, not official rates.
def test_price_query_uses_community_pricing_prefix():
    out = extractive_answer("whats the price for detonate", DETONATE_CHUNKS, max_chars=240)
    assert out.startswith("The pricing is set by the community (approx): ")
    assert not out.startswith("Wiki says:")
    assert "1,000,000" in out


def test_non_price_query_keeps_wiki_prefix():
    # A non-price question on the same chunks must keep the normal prefix, so the
    # community-pricing disclaimer is scoped to actual price queries only.
    out = extractive_answer("what do detonate do", DETONATE_CHUNKS, max_chars=240)
    assert out.startswith("Wiki says:")


def test_server_set_cost_query_is_not_community_priced():
    # "upkeep"/"tax"/"fee" are server-set costs, not community pricing, so they
    # must NOT get the community disclaimer even though they are cost questions.
    results = [
        SearchResult(
            chunk=_chunk(
                text="Town upkeep is charged daily. Overclaimed plots cost an extra $200 each.",
                title="Town Upkeep",
            ),
            score=0.5,
        )
    ]
    out = extractive_answer("how much is town upkeep", results, max_chars=240)
    assert not out.startswith("The pricing is set by the community")
