# Insurance AI Agents

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Claude](https://img.shields.io/badge/Claude-Anthropic-orange.svg)](https://anthropic.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-blue.svg)](https://openai.com)
[![Google](https://img.shields.io/badge/Google-Gemini-red.svg)](https://google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A portfolio-grade LLM application for insurance claim triage. Plug in your preferred LLM provider (Claude, OpenAI, or Gemini) and let AI help classify and route claims intelligently.

## Overview

Insurance claim processing requires speed and accuracy. This application uses large language models to automate claim triage—analyzing policy documents and claim details to recommend approval, denial, or escalation decisions. By supporting multiple LLM providers, you can choose the model and service that best fits your needs and budget.

### What it does

- **Claim Triage**: Analyze incoming claims against policy documents and make APPROVE/DENY/ESCALATE recommendations
- **Multi-Provider Support**: Use Claude (Anthropic), GPT-4 (OpenAI), or Gemini (Google) interchangeably
- **Prompt Engineering**: Test different prompt strategies (zero-shot, few-shot, chain-of-thought)
- **Evaluation Metrics**: Measure model accuracy with domain-specific benchmarks
- **Interactive Dashboard**: Streamlit UI for real-time testing and evaluation

### Design Philosophy

This application prioritizes **simplicity and flexibility**:
- No local model downloads or GPU dependencies
- Plug-and-play LLM providers via environment variables
- Minimal infrastructure—just Python and an API key
- Extensible architecture for adding new providers or metrics

---

## Quick Start

### Prerequisites
- Python 3.8+
- At least one LLM API key (Claude, OpenAI, or Gemini)

### Installation

```bash
# Clone the repository
git clone https://github.com/lukewarmrebel/insuragents.git
cd insuragents

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key(s)
cp .env.example .env
# Edit .env and add your API credentials
```

### Running the Application

```bash
# Start the application
python run.py

# Optionally specify port and host
python run.py --port 8502 --host 127.0.0.1

# Enable debug logging
LOG_LEVEL=DEBUG python run.py
```

The app opens at `http://localhost:8501`. Start on the **Model Selection** page to choose your LLM provider and model.

---

## Configuration

### Environment Variables (`.env`)

Create a `.env` file in the root directory. At minimum, provide **one** API key:

```bash
# Choose one or more providers:

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
OPENAI_API_KEY=sk-...

# Google Gemini
GOOGLE_API_KEY=...

# Optional app settings
APP_PORT=8501
APP_HOST=0.0.0.0
LOG_LEVEL=INFO
```

Use `.env.example` as a template.

---

## Supported Models

### Claude (Anthropic)
| Model | Use Case |
|-------|----------|
| **Claude Haiku 4.5** | Fast, cost-effective |
| **Claude Sonnet 4.6** | Balanced quality/speed |
| **Claude Opus 4.7** | Best quality |

### OpenAI
| Model | Use Case |
|-------|----------|
| **GPT-4o Mini** | Fast, affordable |
| **GPT-4o** | Best general-purpose |
| **GPT-4 Turbo** | Advanced reasoning |

### Google Gemini
| Model | Use Case |
|-------|----------|
| **Gemini 2.0 Flash** | Fastest |
| **Gemini 1.5 Flash** | Fast & capable |
| **Gemini 1.5 Pro** | Most powerful |

---

## Application Structure

```
insuragents/
├── app.py                          # Main Streamlit application
├── run.py                          # Application entrypoint
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
│
├── models/                         # LLM provider integrations
│   ├── base_inference.py           # Abstract base class
│   ├── claude_inference.py         # Claude API wrapper
│   ├── openai_inference.py         # OpenAI API wrapper
│   ├── gemini_inference.py         # Google Gemini wrapper
│   └── factory.py                  # Provider factory & model registry
│
├── prompts/                        # Prompt templates
│   ├── library.py                  # Template manager
│   ├── strategies.py               # Prompting strategies
│   └── templates/                  # JSON prompt definitions
│       └── claim_triage.json       # Claim triage template
│
├── evaluation/                     # Metrics and benchmarks
│   ├── metrics.py                  # Evaluation metrics (ROUGE, BLEU, BERTScore, Triage Accuracy)
│   ├── human_eval.py               # Human evaluation forms
│   ├── benchmarks.py               # Benchmark dataset management
│   ├── evaluations/                # Saved evaluation results
│   └── benchmarks/                 # Benchmark datasets
│       └── claim_triage_benchmark.json
│
├── data/                           # Sample insurance documents
│   ├── policies/                   # Sample policy files
│   ├── claims/                     # Sample claim files
│   └── communications/             # Sample communications
│
└── utils/                          # Utilities
    └── dataframe_utils.py          # Data formatting for Streamlit
```

---

## How It Works

### Architecture

The application uses a **provider abstraction layer** to support multiple LLM backends:

```
┌──────────────────────┐
│  Streamlit UI        │
│  (app.py)            │
└──────┬───────────────┘
       │
       ├─ Model Selection (choose provider + API key + model)
       └─ Create inference engine via factory
              │
              ▼
       ┌────────────────────┐
       │ BaseInference      │  (abstract interface)
       │ - generate()       │
       │ - generate_with_  │
       │   streaming()      │
       └──────┬─────────────┘
              │
       ┌──────┴──────────┬─────────────┐
       ▼                 ▼             ▼
   ClaudeInference  OpenAIInference  GeminiInference
   (anthropic)      (openai)         (google.generativeai)
```

All downstream components (`PromptEngineeringPage`, `EvaluationPage`, `BenchmarksPage`) call the standard interface, making provider swaps transparent to the rest of the application.

### Key Components

**models/factory.py**
- `PROVIDER_MODELS`: Nested dict of providers → models → model IDs
- `create_inference_engine()`: Instantiates the correct inference class based on user selection

**models/{claude,openai,gemini}_inference.py**
- Each implements `BaseInference` interface
- `generate()`: Single generation request
- `generate_with_streaming()`: Streaming generation with callback

**prompts/library.py**
- `PromptLibrary`: Loads JSON templates from `prompts/templates/`
- `PromptTemplate`: Variable substitution and formatting
- `PromptStrategy`: Zero-shot, few-shot, chain-of-thought implementations

**evaluation/metrics.py**
- Automated metrics: ROUGE, BLEU, BERTScore
- Domain-specific: `TriageAccuracyMetric` (compares APPROVE/DENY/ESCALATE decisions)
- `MetricsManager`: Batch evaluation

**evaluation/benchmarks.py**
- `Benchmark`, `BenchmarkManager`: Load and run benchmark datasets
- Example: `claim_triage_benchmark.json` with 15 realistic claim scenarios

---

## Usage Workflows

### 1. Claim Triage (Default)

1. **Model Selection** → Choose provider, enter API key, select model → Connect
2. **Prompt Engineering** → Load claim triage template → Paste policy + claim → Generate decision
3. **Evaluation** → View decision (APPROVE/DENY/ESCALATE), reasoning, and confidence
4. **Benchmarks** → Run triage against 15 benchmark claims → Compare model performance

### 2. Testing Multiple Providers

1. Go to **Model Selection**
2. Switch between Claude, OpenAI, and Gemini
3. Re-run the same prompt on each provider
4. Compare outputs in **Model Comparison** page

### 3. Prompt Refinement

1. **Prompt Engineering** → Modify the prompt template (zero-shot, few-shot, CoT)
2. Test against a few claims
3. Run **Benchmarks** to measure average accuracy across all test cases
4. Iterate on template until satisfied

---

## Sample Data

The application includes three sample documents:

- `data/policies/sample_auto_policy.txt` — Auto insurance policy
- `data/claims/sample_auto_claim.txt` — Claim for collision damage
- `data/communications/sample_customer_inquiry.txt` — Customer service inquiry

Reference these when testing prompts, or upload your own documents.

---

## Extending the Framework

### Adding a New LLM Provider

1. Create `models/my_provider_inference.py`:
   ```python
   from models.base_inference import BaseInference, SYSTEM_MESSAGE
   
   class MyProviderInference(BaseInference):
       def __init__(self, api_key: str, model_id: str):
           self.client = MyProvider(api_key=api_key)
           self.model_id = model_id
       
       def generate(self, prompt, max_length=1024, temperature=0.7, **kwargs):
           # Implement generation
           response = self.client.complete(prompt=prompt, ...)
           return [response.text]
       
       def generate_with_streaming(self, prompt, callback, **kwargs):
           # Implement streaming
           full_text = ""
           for chunk in self.client.stream(prompt=prompt, ...):
               callback(chunk)
               full_text += chunk
           return full_text
   ```

2. Update `models/factory.py`:
   ```python
   PROVIDER_MODELS["MyProvider"] = {
       "Model A": "model-a-id",
       "Model B": "model-b-id",
   }
   
   def create_inference_engine(provider, api_key, model_id):
       # ... existing code ...
       elif provider == "MyProvider":
           from models.my_provider_inference import MyProviderInference
           return MyProviderInference(api_key, model_id)
   ```

3. Update `.env.example` with the new API key variable.

### Adding a Custom Evaluation Metric

1. Create a metric class in `evaluation/metrics.py`:
   ```python
   class MyMetric(EvaluationMetric):
       def __init__(self):
           super().__init__(
               name="my_metric",
               description="My custom metric",
               max_score=1.0
           )
       
       def evaluate(self, generated_text, reference_text, context=None):
           score = my_scoring_logic(generated_text, reference_text)
           return EvaluationResult(
               metric_name=self.name,
               score=score,
               max_score=self.max_score,
               details={"analysis": "..."}
           )
   ```

2. Register it in `EvaluationMetrics._register_default_metrics()`:
   ```python
   self.register_metric(MyMetric())
   ```

---

## Troubleshooting

### "API Key not found" error

Ensure your `.env` file exists and contains the correct key for your chosen provider:
- `ANTHROPIC_API_KEY` for Claude
- `OPENAI_API_KEY` for OpenAI
- `GOOGLE_API_KEY` for Gemini

### "Rate limit exceeded" error

You've hit the provider's rate limit. Wait a few moments and retry, or consider upgrading your API plan.

### Generation is slow or times out

Try a **faster** model:
- Claude Haiku instead of Opus
- GPT-4o Mini instead of GPT-4 Turbo
- Gemini Flash instead of Gemini Pro

Reduce `max_tokens` in the prompt engineering page.

### Model comparison page is empty

Ensure you've generated outputs on the **Prompt Engineering** page first. Model comparison requires saved outputs.

### Debug logs

Enable debug logging to see detailed request/response logs:
```bash
LOG_LEVEL=DEBUG python run.py
```

Logs are written to `run.log` and `app.log`.

---

## Project Philosophy

This portfolio project demonstrates:

- **Multi-provider abstraction**: Clean factory pattern supporting Claude, OpenAI, and Gemini without downstream coupling
- **Domain-specific application**: Real-world insurance use case (claim triage) with structured evaluation metrics
- **Minimal infrastructure**: API-based, no local models, no GPU requirements—runs on any machine with internet
- **Extensibility**: Easy to add new providers, prompt templates, or evaluation metrics
- **Production-readiness**: Logging, error handling, streaming support, session state management

---

## Testing

The application includes:
- `evaluation/benchmarks/claim_triage_benchmark.json` — 15 test claims with expected outcomes
- `TriageAccuracyMetric` — Evaluates whether model decisions match expected verdicts
- **Benchmarks page** — Run full benchmark suite and compare providers

No formal test suite is present; validation is done via the interactive dashboard and benchmark runs.

---

## License

MIT License - see LICENSE file for details.

---

## Author

Built as a portfolio project exploring LLM integration, prompt engineering, and multi-provider abstraction.

**Repository**: [github.com/lukewarmrebel/insuragents](https://github.com/lukewarmrebel/insuragents)
