# Insurance AI Agents

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Claude](https://img.shields.io/badge/Claude-Anthropic-orange.svg)](https://anthropic.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-blue.svg)](https://openai.com)
[![Google](https://img.shields.io/badge/Google-Gemini-red.svg)](https://google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is this?

Insurance AI Agents is a full-stack AI application that automates insurance workflows using large language models. It covers 7 distinct insurance operations — from triaging claims and summarizing policies to detecting fraud and checking regulatory compliance — all driven by the same prompt-engineering and evaluation framework.

The system is provider-agnostic. You plug in Claude, GPT-4, or Gemini, and the application routes to the right API automatically. Swap models without touching a single line of code downstream.

Built as a portfolio project at the intersection of insurance operations and modern AI engineering.

---

## Why this project exists

Insurance operations involve high volumes of structured but judgment-intensive work: a claim comes in, an adjuster checks coverage, flags red flags, drafts a response letter, and files it. A customer writes in confused about their policy. An underwriter needs a risk summary before renewal. These workflows are repetitive, well-defined, and exactly where LLMs add value as a first-pass assistant.

The technical problem is interesting too: different LLM providers have different APIs, strengths, and costs. This app solves that with a clean abstraction layer so you can benchmark them against each other on the same tasks.

---

## Insurance Workflows

The app ships with 7 pre-built workflows, each with its own prompt template and sample inputs.

### 1. Claim Triage
**The core decision workflow.** Given a policy and a claim, the model recommends APPROVE, DENY, or ESCALATE with structured reasoning.

Output format:
```
DECISION: APPROVE | DENY | ESCALATE
REASONING:
  - Coverage: is the loss type covered?
  - Policy status: active / lapsed / excluded driver?
  - Documentation: sufficient / insufficient?
  - Red flags: fraud indicators or ambiguities?
SUMMARY: plain-language explanation
```

---

### 2. Fraud Detection
**Flag suspicious claims before they reach a human adjuster.** Analyzes timing patterns, inconsistencies in the claim narrative, missing documentation, and financial motive to assign a fraud risk score.

Output format:
```
FRAUD RISK: LOW | MEDIUM | HIGH
INDICATORS:
  - Timing, claimant history, inconsistencies, documentation, financial motive
RECOMMENDED ACTION: PROCEED_NORMALLY | FLAG_FOR_REVIEW | REFER_TO_SIU
SUMMARY: plain-language risk explanation
```

---

### 3. Policy Summary
**Make dense policy documents readable.** Given a full policy document, the model extracts and summarizes coverage limits, deductibles, exclusions, and key conditions in plain language.

Useful for customer-facing explainers, onboarding materials, or internal quick-reference sheets.

---

### 4. Customer Communication
**Draft professional responses to customer inquiries.** Handles questions about coverage, claims status, billing, and policy changes. Addresses all aspects of the inquiry with specific policy details and actionable next steps.

Maintains an empathetic, professional tone appropriate for policyholder-facing communication.

---

### 5. Risk Assessment
**Structured risk analysis for underwriting and renewal decisions.** Given property or applicant details, generates a full report covering identified risks by category, likelihood and impact scores, mitigation strategies, and recommendations.

---

### 6. Compliance Check
**Catch regulatory issues before they become violations.** Analyzes insurance documents — cancellation notices, denial letters, policy language — against specified jurisdiction requirements. Flags non-compliant language and suggests remediation.

---

### 7. Policy Comparison
**Side-by-side policy analysis.** Compares two policies across coverage areas, limits, exclusions, premiums, and conditions. Surfaces key differences and highlights advantages and disadvantages of each — useful for renewal negotiations or competitive analysis.

---

## Key Features

### Multi-Provider LLM Support
Connect to **Claude (Anthropic)**, **GPT-4 (OpenAI)**, or **Gemini (Google)** from a single UI. Each provider is wrapped behind a common interface — the rest of the app doesn't know or care which one is active. Switching providers takes seconds.

### Prompt Engineering Studio
Test and refine prompts interactively across all 7 workflows. Choose between zero-shot, few-shot, and chain-of-thought strategies. See streaming output in real time. Save outputs and compare across runs.

### Evaluation & Benchmarking
- **Automated metrics**: ROUGE, BLEU, BERTScore for text quality
- **Triage Accuracy metric**: domain-specific scoring that compares APPROVE/DENY/ESCALATE decisions against expected verdicts
- **15-claim benchmark suite** with realistic auto and homeowner scenarios, each with expected decisions and required phrases

### Model Comparison
Run the same workflow through multiple providers and compare outputs side by side. Useful for evaluating which model handles edge cases best or fits your cost constraints.

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
Go to **Model Selection**. Pick a provider (Claude / OpenAI / Gemini), paste your API key, choose a model, and click **Connect**. The key stays in your session only.

### Step 2 — Choose a Workflow
Go to **Prompt Engineering**. Use the **Task Type** dropdown to pick a workflow:

| Task Type | Template | What you paste in |
|-----------|----------|-------------------|
| `claim_triage` | Claim Triage | policy document + claim description |
| `fraud_detection` | Fraud Detection | policy document + claim description |
| `policy_summary` | Policy Summary Concise | policy document |
| `customer_communication` | Customer Inquiry Detailed | customer inquiry + policy details |
| `risk_assessment` | Risk Assessment Detailed | risk info + assessment scope |
| `compliance_check` | Compliance Check | document text + jurisdiction/regulations |
| `policy_comparison` | Policy Comparison | policy 1 text + policy 2 text |

Each workflow has sample inputs built in — click **Load Sample** to populate the fields and run immediately.

### Step 3 — Generate & Review
Click **Generate**. The model streams back structured output in real time. Each workflow returns a different format — triage gives APPROVE/DENY/ESCALATE, fraud detection gives a risk score and recommended action, risk assessment gives a full report, and so on.

### Step 4 — Evaluate
Go to **Evaluation**. Score the output against automated metrics (ROUGE, BLEU, BERTScore) or submit a human evaluation with custom criteria.

### Step 5 — Benchmark
Go to **Benchmarks**. Run the 15-claim test suite against your connected model to measure aggregate accuracy and compare expected vs. actual decisions.

### Step 6 — Compare Providers (Optional)
Reconnect with a different provider and re-run the same workflow. Go to **Model Comparison** to review outputs side by side — useful for picking the right model for a given task.

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
