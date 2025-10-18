# automatic-learning-amplifier
automatic-learning-amplifier turns raw documents into synthetic QA datasets, fine-tunes LLaMA-3 locally with MLX, and serves the tuned models for evaluation or deployment without leaving your own hardware.

## Why automatic-learning-amplifier
- **Document-first ingestion:** Handles PDF, DOCX, PPTX, HTML, and TXT, optionally captioning embedded images via `mlx_vlm` so visual context feeds the model.
- **Agentic QA generation:** Uses local 4-bit LLaMA-3 to craft diverse question/answer pairs, enriches prompts with subject-aware summaries, and tags low-confidence items through an automated verifier.
- **Built-in finetuning loop:** Streams verified QA into MLX LoRA/QLoRA trainers, fuses adapters, and can emit GGUF-ready artifacts for lightweight runtimes.
- **Evaluation & deployment hooks:** Compares tuned vs. baseline (or Anthropic) responses and can spin up `mlx-ui` so teams can chat with both models side by side.

## Pipeline at a glance
1. **Document staging** – `data.prepare.file_processor` indexes uploaded files, extracts text, captures captions (when `--images` is set), and chunks content with configurable overlap.
2. **Optional summaries** – `qa.summarizer` batches chunks, generates subject-focused abstracts (`--add_summary`, `--summary_batch_size`), and stores them for downstream prompting.
3. **QA synthesis** – `qa.generator` prompts LLaMA-3 with tailored instructions, attaches summaries when present, and writes JSON payloads per chunk.
4. **Verification & splitting** – `qa.verifier` validates each QA pair, `utils.jsonl` merges records, filters to validated data, and creates train/validation JSONL splits.
5. **Fine-tuning** – `models.training` wraps `mlx_lm.lora` to train adapters (LoRA or QLoRA), fuses weights, and prepares artifacts under `models/<ft_type>`.
6. **Assessment & deploy** – `models.comparison` benchmarks tuned answers against baseline or Anthropic Haiku, while `deploy.runtime` optionally clones `mlx-ui` to host side-by-side chats.

## Quick start
**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), Apple silicon (recommended for MLX acceleration). Install Node.js 20+ only if you plan to extend the UI.

```bash
# Set up environment
uv venv .venv
uv sync

# Stage documents and launch the Streamlit orchestration UI
mkdir -p data/documents
cp /path/to/your/*.pdf data/documents/
uv run streamlit run src/app.py -- --local

# (Optional) adjust defaults from the CLI parser
uv run python -m config.setup --help
```

Set `CLAUDE_API_KEY` and/or `OPENROUTER_API_KEY` in your shell when enabling those providers via the UI toggles.

## Key runtime switches
- **Generation controls:** `--word_limit`, `--question_amount`, `--focus` (processes/knowledge/formulas), `--images` for captioning, `--add_summary` plus `--summary_batch_size` for subject-specific context.
- **Verification & quality:** Enable `--verify` to add `validate` flags before dataset splitting; adjust `--retries`, `--starting_index`, `--ending_index`, and `--overlap` for large corpora.
- **Finetuning knobs:** Choose LoRA vs. QLoRA (`--ft_type`), set `--lora_layers`, `--epochs`, `--learning_rate`, precision (`--q4`/`--fp16`), and export GGUF-ready outputs with `--gguf`.
- **Evaluation & deployment:** Toggle comparisons against the initial model or Anthropic, and enable `--deploy` to boot `mlx-ui` once adapters are fused.

## Project layout
```
src/
├── app.py            # Streamlit entry point orchestrating the full run
├── config/           # Argument parsing, folder management, base model registry
├── data/             # Document readers, chunking, and prepared-data loaders
├── qa/               # Summaries, QA generation, verification utilities
├── models/           # Finetuning wrappers and response comparison helpers
├── deploy/           # mlx-ui bootstrap for serving finetuned models
└── utils/            # JSON extraction helpers and train/val splitting logic
data/
├── documents/        # Drop source corpora here
├── data_prepared/    # Chunked text produced during processing
├── qa_json/          # Raw QA outputs per chunk
├── data_ft/          # train.jsonl / valid.jsonl for MLX LoRA
├── summaries/        # Cached summaries (when enabled)
└── images/           # Saved images + captions (when enabled)
models/
└── {qlora|lora}/     # Finetuned adapters and fused weights
```

## Testing
Run the automated checks (mirrors CI) with:

```bash
uv run pytest
```

The suite validates directory setup, document chunking, and argument guards; extend it as you add new pipeline behaviors.
