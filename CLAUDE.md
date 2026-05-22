# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the application

```bash
# Install dependencies (Python 3.8+)
pip install -r requirements.txt

# Optional GPU torch build
pip install torch==2.1.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Start the Streamlit app (creates required dirs, loads .env, disables Streamlit watchdog)
python run.py

# Override port/host
python run.py --port 8502 --host 127.0.0.1

# Debug logs
LOG_LEVEL=DEBUG python run.py
```

`run.py` is the only correct entrypoint — it sets `STREAMLIT_WATCHDOG_DISABLE=1` (required to prevent crashes when torch is imported), seeds `TRANSFORMERS_CACHE`/`TORCH_HOME` to `models/cache/`, and creates `evaluation/evaluations/{forms,submissions}` and `evaluation/benchmarks/results` before launching. Do not invoke `streamlit run app.py` directly.

`.env` (gitignored) supplies `HF_TOKEN` (required for gated models like LLaMA-2) and optional `APP_PORT`, `APP_HOST`, `TRANSFORMERS_CACHE`, `LOG_LEVEL`. Logs land in `run.log` and `app.log`.

## Tests

There is no `tests/` directory and no test runner is wired up. The README mentions `pytest` and `requirements-dev.txt` but neither exists in the repo. Treat test-related claims in the README as aspirational.

## Architecture

### app.py is monolithic — `ui/` is dead code

`app.py` (~1400 lines) contains the **actual** Streamlit application. It defines its own page classes inline (`ModelSelectionPage`, `PromptEngineeringPage`, `EvaluationPage`, `BenchmarksPage`, `ModelComparisonPage`, `SettingsPage`) and routes between them in `main()` via `st.session_state.current_tab`. It also embeds `TorchUtils`, `SessionState`, and `DataLoader` helpers.

The `ui/pages/` and `ui/components/` directories contain a parallel, **unused** set of page/component modules. `app.py` does not import from `ui/` at all. When modifying UI behavior, edit `app.py` — changes to `ui/` will not be reflected anywhere. Be wary of duplicating logic between the two; if a refactor moves code into `ui/`, wire it through `app.py` explicitly.

### Backend module layout

- `models/model_loader.py` — `ModelConfig` registry (`MODEL_REPOS`, `MODEL_DETAILS`, `CPU_FRIENDLY_MODELS`), `load_model()`, quantization handling via `bitsandbytes`. Adding a model means appending to both `MODEL_REPOS` and `MODEL_DETAILS`.
- `models/inference.py` — `ModelInference` plus a `ThreadingUtils.run_with_timeout()` wrapper used to bound CPU generations.
- `prompts/library.py` — `PromptTemplate` + `PromptLibrary`, loads JSON templates from `prompts/templates/*.json` at startup.
- `prompts/strategies.py` — `PromptStrategy` base with `ZeroShot`, `FewShot`, `ChainOfThought` subclasses; instantiated via `create_prompt_strategy(name)`.
- `evaluation/metrics.py` — ROUGE/BLEU/BERTScore wrappers behind `EvaluationMetric` + `MetricsManager`. Downloads `nltk` punkt/stopwords on first import.
- `evaluation/human_eval.py` — form/submission persistence under `evaluation/evaluations/`.
- `evaluation/benchmarks.py` — `Benchmark`, `BenchmarkManager`, and the `create_policy_summary_benchmark` / `create_claim_response_benchmark` factories that seed `evaluation/benchmarks/*.json`.
- `utils/dataframe_utils.py` — `prepare_dataframe_for_display()` for Streamlit table rendering.

### CPU vs GPU paths diverge

`ModelConfig.CPU_FRIENDLY_MODELS` (Phi-2, Phi-1.5, TinyLLaMA, and GGUF-quantized 7B variants) gets special treatment in `load_model` and `ModelInference`: reduced max tokens, single-threaded generation, and timeout wrapping. Larger HF models (LLaMA-2, Mistral, Falcon) assume GPU + quantization. When the framework is run on CPU without a CPU-friendly model selected, the UI surfaces a warning — don't silently change defaults.

### Sample data is gitignored except specific files

`.gitignore` excludes `data/policies/*`, `data/claims/*`, `data/communications/*` but explicitly un-ignores three sample files (`sample_auto_policy.txt`, `sample_auto_claim.txt`, `sample_customer_inquiry.txt`). `DataLoader.load_sample_data()` in `app.py` hardcodes these three filenames. New sample files won't be picked up without code changes and gitignore exceptions.

### Session state

All cross-page state (`model`, `tokenizer`, `inference_engine`, `current_tab`, `generated_outputs`, `evaluation_results`) lives in `st.session_state`, initialized once by `SessionState.initialize()` at the top of `main()`. Loading a new model replaces `st.session_state.model` and you should call `TorchUtils.clear_gpu_memory()` before swapping to avoid CUDA OOM.

## Known README/reality drift

The README references several things that aren't in the repo: `Dockerfile`, `config/` directory with YAML files, `tests/`, `docs/`, `requirements-dev.txt`, and the `insurance_llm` import path used in the README's API example (the real imports are `models.model_loader`, `prompts.library`, etc. — flat top-level packages). Trust the code over the README when they disagree.
