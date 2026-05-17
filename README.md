# Vista Chatbot RAG

A local, no-API Minecraft wiki chatbot for Minescript.

The runtime flow is:

1. `mc_integration.py` listens to in-game chat through Minescript.
2. `BotEngine` filters spam, cooldowns, self-echoes, prefix-gated queries, commands, and special-case rules.
3. The RAG retriever searches pre-chunked `.md` / `.mdx` wiki pages.
4. The bot answers with either:
   - local TinyLlama + optional LoRA adapter, grounded on retrieved wiki context, or
   - extractive fallback that shortens the most relevant wiki chunk.
5. Optional: in extractive mode, an LLM selector can choose the best snippet from top candidates to reduce noisy/gibberish replies.

No server, no OpenAI API, and no exported environment variables are required.

## Project structure

```txt
vista-chatbot/
├── config/
│   └── bot.json                    # main config: triggers, cooldowns, model, retrieval, rules
├── data/                           # optional training / raw QA data, not needed for RAG runtime
├── wiki/                           # put server wiki files here, supports .md and .mdx
├── artifacts/
│   ├── retriever/                  # generated chunks.jsonl, embeddings.npy, meta.json
│   └── logs/                       # autoreply.log
├── src/vista_chatbot/
│   ├── config.py                   # dataclass config loader
│   ├── chunking.py                 # MD/MDX cleaner + chunker
│   ├── retriever.py                # sentence-transformers + NumPy cosine index
│   ├── llm.py                      # TinyLlama / LoRA local generation + fallback
│   ├── rules.py                    # contains/exact/regex special-case replies
│   ├── text.py                     # chat parsing, trigger stripping, output splitting
│   ├── conversation.py             # small recent context buffer
│   ├── logging_utils.py            # file + stream logging
│   └── runtime.py                  # production bot engine used by Minescript
├── scripts/
│   ├── build_wiki_index.py         # chunk wiki and build embeddings before Minecraft
│   ├── query_rag.py                # terminal test for retrieval/fallback answer
│   └── install_minescript_entry.py # copies entrypoint + generated config into Minescript folder
├── mc_integration.py               # file to move/copy into Minescript folder
├── requirements.txt
└── pyproject.toml
```

## Setup

Create a virtual environment from the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want 4-bit model loading, install bitsandbytes too:

```bash
pip install bitsandbytes
```

For CPU-only testing, set this in `config/bot.json`:

```json
"model": {
  "enabled": false,
  "fallback_to_extractive": true
}
```

That makes the bot run as a pure RAG shortener, which is usually enough while debugging the wiki.

## Add wiki files

Place your server docs under `wiki/`, for example:

```txt
wiki/src/content/docs/earth/fluff/cosmetics.mdx
wiki/src/content/docs/earth/fluff/flairs.mdx
wiki/src/content/docs/earth/fluff/tsunami.mdx
```

The chunker supports nested folders and keeps the relative source path in the index.

## Build the retriever index

Run this once after changing the wiki:

```bash
python scripts/build_wiki_index.py --config config/bot.json
```

Generated files:

```txt
artifacts/retriever/chunks.jsonl
artifacts/retriever/embeddings.npy
artifacts/retriever/meta.json
```

## Test outside Minecraft

```bash
python scripts/query_rag.py what is fluff --show-context
python scripts/query_rag.py how to use wraps
python scripts/query_rag.py what is tsunami
```

This tests the retriever and extractive fallback without loading Minescript.

## Install into Minescript

Run:

```bash
python scripts/install_minescript_entry.py --minescript-dir /path/to/.minecraft/minescript
```

It copies:

```txt
mc_integration.py
vista_chatbot_config.json
```

The generated `vista_chatbot_config.json` contains an absolute `project_root`, so the script can import `src/vista_chatbot` even after being copied into the Minescript folder.

In Minecraft, run:

```txt
\mc_integration
```

## Config rules / special cases

`config/bot.json` contains `rules.special_cases`:

```json
{
  "name": "greeting",
  "kind": "exact_normalized",
  "patterns": ["hi izu", "hello izu"],
  "reply": "Meow. Ask me about the wiki, e.g. '!vista what is fluff?'",
  "stop": true
}
```

Supported `kind` values:

- `contains`: substring match after normalization.
- `exact_normalized`: exact match after lowercasing, punctuation cleanup, and whitespace cleanup.
- `regex`: Python regular expression.

Set `reply` to `null` to silently ignore a message.

## Runtime commands

In-game:

```txt
!vista status
!vista help
!vista whoami
!vista clear_context
!vista reload_retriever
!vista stop
```

Admin control is configured in `chat`:

```json
"chat": {
  "admin_names": ["Izu"],
  "admin_ranks": ["TOPAZ", "RUBY"],
  "admin_only_commands": false,
  "admin_command_names": [
    "status",
    "ping",
    "whoami",
    "admins",
    "clear_context",
    "reload_retriever",
    "stop",
    "quit"
  ],
  "critical_admin_commands": ["stop", "quit"],
  "require_rank_for_critical_admin_commands": true,
  "log_command_events": true
}
```

Behavior:

- If `admin_only_commands=true`, every command requires admin.
- If `admin_only_commands=false`, only commands listed in `admin_command_names` require admin.
- Admin check passes if speaker name is in `admin_names` or parsed rank is in `admin_ranks`.
- If `require_rank_for_critical_admin_commands=true` and `admin_ranks` is non-empty, critical commands (default: `stop`, `quit`) require a staff rank match to reduce nickname spoof abuse.
- Command events are logged with timestamps in `artifacts/logs/autoreply.log` (allow/deny, command, speaker, rank).

## Production notes

Recommended first production setting:

```json
"model": {
  "enabled": false,
  "fallback_to_extractive": true
}
```

After the RAG answer quality is good, turn on TinyLlama:

```json
"model": {
  "enabled": true,
  "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "adapter_path": "artifacts/lora_full",
  "load_in_4bit": true,
  "fallback_to_extractive": true
}
```

If the model or LoRA fails to load, `fallback_to_extractive=true` keeps the bot alive and answers using retrieved wiki text.

If you want better extractive quality (slower, but usually cleaner), enable candidate selection by LLM:

```json
"model": {
  "enabled": false,
  "fallback_to_extractive": true,
  "llm_select_extractive": true,
  "llm_select_max_candidates": 6,
  "llm_select_max_new_tokens": 8
}
```

This mode asks the local model to pick only an index from top extractive candidates, then returns that chosen snippet.
Even with `"enabled": false`, the model still loads for this selector step, so expect slower startup.

## Updating the wiki

Whenever docs change:

```bash
python scripts/build_wiki_index.py --config config/bot.json
```

Then restart the Minescript bot.

## Safety behavior

The bot:

- answers only when the `!vista` prefix is used,
- ignores configured blocked substrings,
- parses decorated server chat lines like `🏕 ➟ TOPAZ Name [❄ '24] ➡ !vista ...`,
- extracts player name and rank from decorated lines so admin command checks can work,
- remembers its own recent messages to avoid replying to itself,
- rate-limits globally and per user,
- truncates and splits replies to Minecraft chat length,
- tells users when the wiki does not contain the answer instead of inventing one.
