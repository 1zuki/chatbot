# Vista Chatbot

A local, no-API Minecraft wiki chatbot for [Minescript](https://minescript.net/).
It answers player questions in-game by retrieving from the server wiki — no
OpenAI key, no vector-DB service, no exported environment variables. Runs as a
pure retrieval shortener out of the box, with optional local TinyLlama
generation.

## What it does

Players ask questions in chat (`!timber how do i create a town`) and the bot
replies with a short, grounded answer pulled from the wiki. It's tuned for a
Minecraft server wiki, which is dominated by two content shapes: large
**command-reference tables** (`/town`, `/plot`, `/nation`) and **prose Q&A**
sections. It tells players when the wiki doesn't have an answer instead of
inventing one.

It runs on two surfaces from the same engine:

- **In-game** via Minescript (`mc_integration.py`)
- **Website widget** via a FastAPI `POST /api/chat` endpoint

## Features

- **Hybrid retrieval** — semantic embeddings (`all-MiniLM-L6-v2`) fused with a
  pure-stdlib BM25 lexical ranker via Reciprocal Rank Fusion. Catches both
  paraphrased questions and exact command/term matches.
- **Structure-aware chunking** — tables are kept row-intact, code spans like
  `/warp <warp-name>` are protected from MDX stripping, prose is packed by
  paragraph, and heading paths track real nesting.
- **Row-aware extraction** — table rows are matched by their key column, so a
  question about "star rank 5" returns that row, not generic intro text.
- **Recency-aware dated content** — changelogs, announcements, and roadmaps
  (named `DD-MM-YY.md`) are ranked newest-first for "what's new" questions, and
  kept out of the way of normal topical questions.
- **Confidence gating** — when retrieval isn't confident, the bot says it
  doesn't know rather than guessing.
- **No external dependencies for retrieval** — embeddings are a NumPy matrix on
  disk; BM25 is in-memory stdlib. Trivial to deploy inside Minescript.
- **Optional local generation** — TinyLlama (+ optional LoRA adapter) for
  free-form answers, with automatic fallback to extractive mode.

## How it works

```txt
in-game chat
   │
   ▼
parse + gate        strip decorations, enforce !timber prefix, cooldowns, blocklist
   │
   ▼
special-case rules  matched? → canned reply
   │ no match
   ▼
hybrid retrieval    dense cosine gate + BM25 rerank (RRF) → top-k wiki chunks
   │
   ▼
answer composition  extractive shortener  ·  LLM selector  ·  TinyLlama generation
   │
   ▼
length-split reply → Minecraft chat / web widget
```

The **dense `min_score` threshold is the hard inclusion gate**: BM25 only
*reorders* chunks that already cleared it, so an exact term hit can pull the
right row up but can never resurrect a chunk the embedder rejected.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build the index after adding/changing wiki files
python scripts/build_wiki_index.py --config config/bot.json

# try it without Minecraft
python scripts/query_rag.py how do i create a nation --show-context
```

Out of the box `model.enabled` is `false`, so the bot runs as a pure RAG
shortener (fast, fully offline). See [Configuration](#configuration) to turn on
generation.

## Wiki content

Place server docs under `wiki/` (`.md` / `.mdx`, nested folders fine). The
chunker keeps the relative source path in the index.

Dated content lives in three special folders, with each file named `DD-MM-YY.md`
(e.g. `10-06-26.md`):

```txt
wiki/changelog/      # patch notes
wiki/announcement/   # announcements
wiki/roadmap/        # roadmaps
```

Recency ranking parses the date from the filename. Any file *not* in that date
format is treated as ordinary evergreen content.

## Project layout

```txt
config/bot.json              # triggers, cooldowns, model, retrieval, rules
wiki/                        # server wiki (.md / .mdx) + dated content folders
artifacts/retriever/         # generated index: chunks.jsonl, embeddings.npy, meta.json
artifacts/logs/              # autoreply.log + user_queries.csv
src/vista_chatbot/
  chunking.py                # structure-aware MD/MDX cleaner + chunker
  retriever.py               # hybrid retrieval: dense (NumPy) + BM25, RRF-fused
  bm25.py                    # pure-stdlib BM25 ranker
  llm.py                     # TinyLlama / LoRA generation + extractive fallback
  web_api.py                 # FastAPI chat endpoint
  rules.py                   # contains/exact/regex special-case replies
  text.py                    # chat parsing, trigger stripping, output splitting
  runtime.py                 # production bot engine used by Minescript
scripts/
  build_wiki_index.py        # chunk wiki + build embeddings
  query_rag.py               # terminal retrieval test
  run_chat_api.py            # run the website chat API
  install_minescript_entry.py
mc_integration.py            # entrypoint copied into the Minescript folder
```

## Configuration

All config is in `config/bot.json`.

**Answer mode** (`model`):

| Mode | Settings | Behavior |
| --- | --- | --- |
| Extractive | `enabled: false` | Shortens the best wiki sentence/row. Fastest, fully offline. |
| LLM selector | `enabled: false`, `llm_select_extractive: true` | Local model picks the best candidate snippet. Cleaner, slower startup. |
| Generative | `enabled: true` | TinyLlama (+ optional LoRA) writes a grounded answer. |

Keep `fallback_to_extractive: true` so a model load/runtime failure still
answers from wiki text.

**Special-case rules** (`rules.special_cases`) — canned replies matched before
retrieval:

```json
{
  "name": "greeting",
  "kind": "exact_normalized",
  "patterns": ["hi izu", "hello izu"],
  "reply": "Meow. Ask me about the wiki, e.g. '!timber what is fluff?'",
  "stop": true
}
```

`kind` is `contains`, `exact_normalized`, or `regex`. Set `reply` to `null` to
silently ignore a message.

**Admin & moderation** (`chat`) — `admin_names`, `admin_ranks`,
`blacklisted_users`, per-command gating, and rank requirements for critical
commands (`stop`/`quit`). Command events and user queries are logged under
`artifacts/logs/`.

## In-game commands

```txt
!timber status            !timber history <on|off|toggle|status>
!timber help              !timber clear_context
!timber whoami            !timber reload_retriever
!timber stop
```

## Deployment

**Minescript:**

```bash
python scripts/install_minescript_entry.py --minescript-dir /path/to/.minecraft/minescript
```

Copies `mc_integration.py` and a generated `vista_chatbot_config.json` (with an
absolute `project_root` so imports resolve from the Minescript folder). Then run
`\mc_integration` in-game.

**Website widget:**

```bash
python scripts/run_chat_api.py --config config/bot.json \
  --host 127.0.0.1 --port 8787 \
  --cors-origins "http://localhost:4321,https://wiki.vistavalley.xyz" \
  --public-wiki-base-url "https://wiki.vistavalley.xyz"
```

Endpoints: `POST /api/chat` (`{"message": "...", "session_id": "optional"}`) and
`GET /api/health`. The response includes the reply, session id, and wiki source
links. For the Astro/Starlight wiki, the widget is injected via
`astro.config.mjs` and served from `public/vista-chatbot-widget.js`.

## Updating the wiki

After docs change, rebuild the index and restart the bot:

```bash
python scripts/build_wiki_index.py --config config/bot.json
```

## Testing

```bash
pytest
```

The suite covers chunking, retrieval, extraction, and runtime logic with stubbed
embeddings. `tests/test_kid_questions_eval.py` is an end-to-end eval against a
built index using the messy phrasing players actually type; it skips cleanly
when no index or `sentence-transformers` is present.
