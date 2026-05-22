# Insurance AI Agents

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Claude](https://img.shields.io/badge/Claude-Anthropic-orange.svg)](https://anthropic.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-blue.svg)](https://openai.com)
[![Google](https://img.shields.io/badge/Google-Gemini-red.svg)](https://google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

Insurance AI Agents is a full-stack AI application that brings LLM intelligence to insurance claim processing. It reads a policy document and a claim, then decides: **APPROVE, DENY, or ESCALATE** — with structured reasoning explaining every decision.

The system is provider-agnostic. You plug in Claude, GPT-4, or Gemini, and the application routes to the right API automatically. Swap models without touching a single line of code downstream.

Built as a portfolio project at the intersection of insurance operations and modern AI engineering.

---

## Why this project exists

Insurance claims teams deal with high volumes of repetitive decisions. A claim comes in, an adjuster checks the policy, evaluates coverage, flags red flags, and makes a call. This project explores how LLMs can assist that workflow — not replace the adjuster, but give them a structured first-pass analysis to act on.

The technical problem is interesting too: different LLM providers have different APIs, different strengths, and different costs. This app solves that with a clean abstraction layer so you can benchmark them against each other on real insurance tasks.

---

## Key Features

### Multi-Provider LLM Support
Connect to **Claude (Anthropic)**, **GPT-4 (OpenAI)**, or **Gemini (Google)** from a single UI. Each provider is wrapped behind a common interface — the rest of the app doesn't know or care which one is active. Switching providers takes seconds.

### Structured Claim Triage
The core use case: given a policy document and claim description, the model returns:
- **DECISION** — APPROVE, DENY, or ESCALATE
- **REASONING** — coverage analysis, policy status check, documentation review, red flags
- **SUMMARY** — a concise explanation suitable for the claims file

### Prompt Engineering Studio
Test and refine prompts interactively. Choose between zero-shot, few-shot, and chain-of-thought strategies. See streaming output in real time. Save outputs and compare across runs.

### Evaluation & Benchmarking
- **Automated metrics**: ROUGE, BLEU, BERTScore for text quality
- **Triage Accuracy metric**: domain-specific scoring that compares APPROVE/DENY/ESCALATE decisions against expected verdicts
- **15-claim benchmark suite** with realistic auto and homeowner scenarios, each with expected decisions and required phrases

### Model Comparison
Run the same claim through multiple providers and compare outputs side by side. Useful for evaluating which model handles edge cases best or fits your cost constraints.

---

## Supported Models

### Claude (Anthropic)
| Model | Speed | Best For |
|-------|-------|----------|
| Claude Haiku 4.5 | Fast | High-volume triage, cost-sensitive |
| Claude Sonnet 4.6 | Balanced | General use, good reasoning |
| Claude Opus 4.7 | Thorough | Complex edge cases, best accuracy |

### OpenAI
| Model | Speed | Best For |
|-------|-------|----------|
| GPT-4o Mini | Fast | Quick decisions, low cost |
| GPT-4o | Balanced | Strong general performance |
| GPT-4 Turbo | Thorough | Complex policy analysis |

### Google Gemini
| Model | Speed | Best For |
|-------|-------|----------|
| Gemini 2.5 Flash | Fast | Low latency, high throughput |
| Gemini 2.5 Lite | Lightweight | Minimal cost, simple claims |

---

## Quick Start

### Prerequisites
- Python 3.8+
- API key from at least one provider (Claude, OpenAI, or Gemini)

### Setup

```bash
# Clone the repo
git clone https://github.com/lukewarmrebel/insuragents.git
cd insuragents

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Open .env and add your key(s)
```

### Start the App

```bash
python run.py
```

Opens at **http://localhost:8501**

---

## How to Use

### Step 1 — Connect a Model
Go to the **Model Selection** page. Pick a provider (Claude / OpenAI / Gemini), paste your API key, choose a model, and click **Connect**. The key stays in your session and is never stored.

### Step 2 — Triage a Claim
Go to **Prompt Engineering**. The claim triage template is pre-loaded. Paste in:
- A **policy document** (use the sample auto policy to get started)
- A **claim description** (what happened, what's being claimed)

Click **Generate**. The model streams back a structured APPROVE/DENY/ESCALATE decision with full reasoning.

### Step 3 — Evaluate the Output
Go to **Evaluation**. Score the output against automated metrics (ROUGE, BLEU, BERTScore) or submit a human evaluation with custom criteria.

### Step 4 — Run the Benchmark
Go to **Benchmarks**. Run the full 15-claim test suite against your connected model. See aggregate accuracy, per-claim decisions, and compare against expected outcomes.

### Step 5 — Compare Providers (Optional)
Reconnect with a different provider and re-run the same claim or benchmark. Go to **Model Comparison** to review outputs side by side.

---

## Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Add the key(s) for your chosen provider(s):

```bash
# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GOOGLE_API_KEY=...

# Optional
APP_PORT=8501
APP_HOST=0.0.0.0
LOG_LEVEL=INFO
```

You only need **one** key to use the app. Add all three if you want to benchmark across providers.

---

## Project Structure

```
insuragents/
├── app.py                          # Streamlit app (all pages live here)
├── run.py                          # Entrypoint — sets env vars, starts server
├── requirements.txt
├── .env.example
│
├── models/                         # LLM provider layer
│   ├── base_inference.py           # Abstract interface (generate, stream)
│   ├── claude_inference.py         # Anthropic SDK wrapper
│   ├── openai_inference.py         # OpenAI SDK wrapper
│   ├── gemini_inference.py         # Google Generative AI wrapper
│   └── factory.py                  # Provider registry + factory function
│
├── prompts/
│   ├── library.py                  # Template loader and manager
│   ├── strategies.py               # Zero-shot, few-shot, CoT
│   └── templates/
│       └── claim_triage.json       # Claim triage prompt template
│
├── evaluation/
│   ├── metrics.py                  # ROUGE, BLEU, BERTScore, TriageAccuracy
│   ├── human_eval.py               # Human evaluation form handling
│   ├── benchmarks.py               # Benchmark runner
│   └── benchmarks/
│       └── claim_triage_benchmark.json   # 15 test claims
│
└── data/
    ├── policies/sample_auto_policy.txt
    ├── claims/sample_auto_claim.txt
    └── communications/sample_customer_inquiry.txt
```

---

## Architecture

The provider abstraction is the core design decision. Every LLM provider is wrapped in a class that implements the same two methods: `generate()` and `generate_with_streaming()`. The factory instantiates the right one at runtime based on user selection. Nothing downstream needs to change when you swap providers.

```
Streamlit UI (app.py)
    │
    ├── Model Selection → create_inference_engine(provider, key, model)
    │
    ▼
BaseInference (abstract)
    ├── ClaudeInference   → anthropic.Anthropic client
    ├── OpenAIInference   → openai.OpenAI client
    └── GeminiInference   → google.generativeai client
```

All pages — Prompt Engineering, Evaluation, Benchmarks — call `inference_engine.generate(prompt)`. The provider is transparent.

---

## Extending

### Add a New Provider

1. Create `models/myprovider_inference.py` extending `BaseInference`
2. Implement `generate()` and `generate_with_streaming()`
3. Add entries to `PROVIDER_MODELS` in `models/factory.py`
4. Add a branch in `create_inference_engine()`

### Add a Custom Metric

1. Subclass `EvaluationMetric` in `evaluation/metrics.py`
2. Implement `evaluate(generated_text, reference_text, context)`
3. Register with `self.register_metric(MyMetric())` in `_register_default_metrics()`

---

## Troubleshooting

**Nothing loads at localhost:8501**
Make sure you ran `python run.py`, not `streamlit run app.py` directly. The entrypoint sets required environment variables before launching.

**"API Key not found"**
Check your `.env` file has the right variable name for your provider (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`).

**Generation is slow**
Switch to a faster model — Haiku, GPT-4o Mini, or Gemini 2.5 Flash. Also try reducing `max_tokens`.

**Model Comparison page is empty**
Generate outputs on the Prompt Engineering page first. Comparison requires at least one saved run.

**Debug logs**
```bash
LOG_LEVEL=DEBUG python run.py
```
Logs write to `run.log` and `app.log`.

---

## License

MIT — see LICENSE file.

---

Built by [@lukewarmrebel](https://github.com/lukewarmrebel)
