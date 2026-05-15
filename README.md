# Vista Chatbot

A local, production-oriented Minescript chatbot for a Minecraft server.

`autoreply.py` loads the model locally when you run `\autoreply` in Minecraft, retrieves relevant wiki/QA context from a local FAISS index, generates a short answer, and sends it with `minescript.chat()`.

## Folder layout

```text
vista-chatbot/
├── autoreply.py                  # Minescript entrypoint; copy to .minecraft/minescript
├── configs/bot.json              # Runtime config, no exported env vars needed
├── data/                         # Put Minecraft QA data here
├── wiki/                         # Put server wiki .md/.mdx files here
├── src/vista_chatbot/            # Runtime, chunking, retrieval, training code
├── scripts/                      # CLI helpers
└── artifacts/                    # Generated corpus, retriever index, LoRA adapter, logs
```

## 1. Install

```bash
cd vista-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For GPU training/inference, install the PyTorch build matching your CUDA version first, then run the requirements install.

## 2. Add data

Wiki docs:

```text
wiki/*.md
wiki/*.mdx
wiki/**/**/*.mdx
```

Minecraft QA files can be `.json` or `.jsonl`. Each record should look like:

```json
{"question":"How do I claim land?","answer":"Use /claim ...","source":"minecraft_qa"}
```

Large JSON arrays also work. For huge data, `.jsonl` is still the nicest format, but top-level `.json` arrays like `[{"question": ..., "answer": ..., "source": ...}]` are streamed so the fast sampler can stop early.

## 3. Build balanced corpus

By default this keeps QA data roughly the same size as the wiki. For example, if the wiki produces 600 chunks, it keeps about 600 QA examples instead of all 700k rows.

```bash
python scripts/build_corpus.py \
  --wiki wiki \
  --qa data/minecraft_qa.jsonl \
  --out artifacts/corpus
```

Useful balance controls:

```bash
# keep 2 QA examples per wiki chunk
python scripts/build_corpus.py \
  --wiki wiki \
  --qa data/minecraft_qa.jsonl \
  --qa-per-wiki-chunk 2.0

# keep exactly 5000 QA examples
python scripts/build_corpus.py \
  --wiki wiki \
  --qa data/minecraft_qa.jsonl \
  --qa-target-records 5000

# fast default: reads only enough QA rows, then stops
python scripts/build_corpus.py \
  --wiki wiki \
  --qa data/minecraft_qa.jsonl \
  --qa-sampling head

# more representative sample, but scans the full QA dataset
python scripts/build_corpus.py \
  --wiki wiki \
  --qa data/minecraft_qa.jsonl \
  --qa-sampling reservoir
```

Outputs:

```text
artifacts/corpus/wiki_chunks.jsonl
artifacts/corpus/qa_retriever.jsonl
artifacts/corpus/retriever_corpus.jsonl
artifacts/corpus/sft_train.jsonl
artifacts/corpus/corpus_stats.json
```

## 4. Build balanced retriever index

`build_retriever.py` also balances by default. It indexes all wiki chunks and only about the same amount of QA docs. If `wiki_chunks.jsonl` and `qa_retriever.jsonl` exist beside the corpus, it uses those split files directly so it does not need to read a huge combined corpus.

```bash
python scripts/build_retriever.py \
  --corpus artifacts/corpus/retriever_corpus.jsonl \
  --out artifacts/retriever
```

Useful controls:

```bash
# index 2 QA docs per wiki chunk
python scripts/build_retriever.py --qa-per-wiki-doc 2.0

# index exactly 5000 QA docs
python scripts/build_retriever.py --qa-target-docs 5000

# disable balancing and index the corpus exactly as-is
python scripts/build_retriever.py --no-balance-to-wiki
```

This creates:

```text
artifacts/retriever/faiss.index
artifacts/retriever/metadata.jsonl
artifacts/retriever/config.json
artifacts/retriever/balanced_retriever_corpus.jsonl
```

## 5. Train LoRA

Small test run:

```bash
python scripts/train_lora.py \
  --train artifacts/corpus/sft_train.jsonl \
  --output artifacts/lora_full \
  --max-train-samples 2000 \
  --epochs 1
```

Balanced full run:

```bash
python scripts/train_lora.py \
  --train artifacts/corpus/sft_train.jsonl \
  --output artifacts/lora_full \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 8
```

Default base model is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Change it in `configs/bot.json` and pass the same value to `scripts/train_lora.py --base-model` if you switch models.

## 6. Smoke test outside Minecraft

Parsing only:

```bash
python scripts/smoke_test.py --skip-model --message "<Steve> izu how do I claim land?"
```

Full local generation:

```bash
python scripts/smoke_test.py --message "<Steve> izu how do I claim land?"
```

## 7. Install into Minescript

Copy `autoreply.py` and a generated runtime config into your Minecraft `minescript` folder:

```bash
python scripts/install_minescript_entry.py --minescript-dir ~/.minecraft/minescript
```

On Windows, use something like:

```powershell
python scripts/install_minescript_entry.py --minescript-dir "$env:APPDATA\.minecraft\minescript"
```

Then run in Minecraft chat:

```text
\autoreply
```

Minescript runs scripts from the Minecraft `minescript` folder with a backslash command and without `.py`, so this entrypoint bootstraps the repo path from `vista_chatbot_config.json` beside `autoreply.py`.

## Runtime commands

In chat:

```text
izu bot status
izu bot help
izu clear context
izu bot reload retriever
izu shutdown
```

To restrict admin commands, edit `configs/bot.json`:

```json
"admin_names": ["YourMinecraftName"],
"admin_only_commands": true
```

Then rerun `scripts/install_minescript_entry.py` so the Minescript folder gets the updated runtime config.

## Important config notes

`configs/bot.json` intentionally uses file config rather than exported environment variables because Minescript is launched from in-game chat. The installer writes an absolute `project_root` into `vista_chatbot_config.json` so imports work when `autoreply.py` is executed from `.minecraft/minescript`.

Key paths:

```json
"adapter_path": "artifacts/lora_full",
"index_dir": "artifacts/retriever",
"log_file": "artifacts/logs/autoreply.log"
```

All relative paths are resolved from `project_root`.

## Production checklist

- Build retriever after every big wiki update.
- Retrain or continue LoRA when QA data changes substantially.
- Keep `max_new_tokens` low enough for server chat.
- Use `admin_only_commands=true` on real servers.
- Check `artifacts/logs/autoreply.log` after crashes.
- Start with `--qa-per-wiki-chunk 1.0` or `2.0`; avoid sending all 700k QA rows through your laptop unless you really need to.

## Why both training and retrieval?

Fine-tuning teaches style and common Minecraft Q/A behavior. Retrieval keeps server-specific facts fresh without retraining every time the wiki changes. For production server docs, the retriever context should usually win over memorized model knowledge.
