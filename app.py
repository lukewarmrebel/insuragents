"""
Insurance LLM Framework

A web application for prompt engineering and evaluation of LLMs for insurance tasks.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.factory import PROVIDER_MODELS, create_inference_engine
from prompts.library import get_prompt_library, PromptTemplate
from prompts.strategies import create_prompt_strategy
from evaluation.metrics import get_metrics_manager
from evaluation.human_eval import get_human_evaluation_manager
from evaluation.benchmarks import (
    get_benchmark_manager,
    create_policy_summary_benchmark,
    create_claim_response_benchmark
)
from utils.dataframe_utils import prepare_dataframe_for_display

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SessionState:
    """Class to manage Streamlit session state initialization and access."""

    @staticmethod
    def initialize():
        """Initialize session state variables."""
        if "model" not in st.session_state:
            st.session_state.model = None

        if "tokenizer" not in st.session_state:
            st.session_state.tokenizer = None

        if "inference_engine" not in st.session_state:
            st.session_state.inference_engine = None

        if "current_tab" not in st.session_state:
            st.session_state.current_tab = "model_selection"

        if "generated_outputs" not in st.session_state:
            st.session_state.generated_outputs = []

        if "evaluation_results" not in st.session_state:
            st.session_state.evaluation_results = {}

    @staticmethod
    def set_tab(tab_name: str):
        """Set the current tab."""
        st.session_state.current_tab = tab_name

class DataLoader:
    """Class to handle loading sample data."""

    @staticmethod
    def load_sample_data(data_type: str) -> str:
        """
        Load sample data for a specified type.

        Args:
            data_type: Type of data to load (e.g., 'policy', 'claim')

        Returns:
            Sample data text
        """
        data_dir = Path("data")

        if data_type == "policy":
            file_path = data_dir / "policies" / "sample_auto_policy.txt"
        elif data_type == "claim":
            file_path = data_dir / "claims" / "sample_auto_claim.txt"
        elif data_type == "customer_inquiry":
            file_path = data_dir / "communications" / "sample_customer_inquiry.txt"
        else:
            return "Sample data not found."

        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading sample data {file_path}: {str(e)}")
            return f"Error loading sample data: {str(e)}"

class ModelSelectionPage:
    """Class to handle multi-provider LLM configuration."""

    API_KEY_ENV = {
        "Claude": "ANTHROPIC_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Gemini": "GOOGLE_API_KEY",
    }

    def _display_system_info(self):
        """Display system information."""
        st.subheader("API Status")

        import platform
        import multiprocessing

        system_info = {
            "Python Version": os.sys.version.split()[0],
            "Streamlit Version": st.__version__,
            "CPU Cores": multiprocessing.cpu_count(),
            "Operating System": platform.system() + " " + platform.release(),
            "API Connected": "Yes" if st.session_state.model is not None else "No",
        }

        if st.session_state.model is not None:
            system_info["Provider"] = st.session_state.get("provider", "Unknown")
            system_info["Selected Model"] = st.session_state.get("model_id", "Unknown")

        system_df = pd.DataFrame(
            list(system_info.items()),
            columns=["Property", "Value"]
        )
        system_df = prepare_dataframe_for_display(system_df)
        st.table(system_df)

    def _connect(self, provider: str, api_key: str, model_id: str):
        """Connect to selected LLM provider."""
        try:
            engine = create_inference_engine(provider, api_key, model_id)
            st.session_state.model = engine
            st.session_state.tokenizer = None
            st.session_state.inference_engine = engine
            st.session_state.model_id = model_id
            st.session_state.provider = provider
            st.session_state.is_cpu_optimized = False

            st.success(f"Connected to {provider} with {model_id}!")
            SessionState.set_tab("prompt_engineering")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to connect to {provider}: {str(e)}")
            logger.error(f"{provider} API error: {str(e)}")

    def render(self):
        """Render the multi-provider LLM configuration page."""
        st.title("🤖 Insurance AI - Multi-Provider LLM Selection")

        with st.expander("About Multi-Provider Support", expanded=False):
            st.markdown("""
            This application supports multiple LLM providers:

            **Claude (Anthropic)**
            - No local model loading — works on any machine
            - Excellent reasoning for complex insurance decisions
            - Best for detailed policy analysis

            **GPT (OpenAI)**
            - Strong general-purpose reasoning
            - Wide deployment experience in enterprises
            - Good balance of cost and capability

            **Gemini (Google)**
            - Large context windows for long documents
            - Strong multimodal capabilities
            - Competitive pricing
            """)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("LLM Provider Configuration")

            provider = st.selectbox(
                "Select Provider",
                list(PROVIDER_MODELS.keys()),
                index=0,
                help="Choose your preferred LLM provider"
            )

            model_options = PROVIDER_MODELS[provider]
            selected_model_label = st.selectbox(
                "Select Model",
                list(model_options.keys()),
                help="Choose based on your speed/quality/cost preferences"
            )
            selected_model = model_options[selected_model_label]

            env_var_name = self.API_KEY_ENV[provider]
            api_key = st.text_input(
                f"{provider} API Key",
                value=os.environ.get(env_var_name, ""),
                type="password",
                help=f"Get your API key from the {provider} console"
            )

            if st.button("Connect to LLM"):
                if not api_key:
                    st.error(f"Please enter your {provider} API key")
                else:
                    with st.spinner(f"Connecting to {provider}..."):
                        self._connect(provider, api_key, selected_model)

        with col2:
            self._display_system_info()

class PromptEngineeringPage:
    """Class to handle the prompt engineering page UI and logic."""

    def render(self):
        """Render the prompt engineering page."""
        st.header("Prompt Engineering")

        if st.session_state.inference_engine is None:
            st.warning("Please load a model first.")
            if st.button("Go to Model Selection"):
                SessionState.set_tab("model_selection")
                st.rerun()
            return

        prompt_library = get_prompt_library()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Prompt Template Selection")

            task_types = prompt_library.list_task_types()
            task_type = st.selectbox("Select Task Type", task_types)

            templates = prompt_library.get_templates_by_task(task_type)
            template_names = [t.name for t in templates]
            selected_template_name = st.selectbox(
                "Select Template", template_names)

            selected_template = prompt_library.get_template(
                selected_template_name)

            if selected_template:
                st.text_area("Template Description",
                             selected_template.description, height=100)
                st.text_area(
                    "Template", selected_template.template, height=300)

                st.subheader("Input Variables")

                variables = {}

                for var in selected_template.variables:
                    if var not in ["examples", "cot_examples", "reasoning_request"]:
                        if var == "policy_text":
                            st.write(f"**{var}**")

                            if f"sample_{var}" not in st.session_state:
                                st.session_state[f"sample_{var}"] = False

                            sample_data_button = st.button(
                                "Load Sample Policy", key=f"load_sample_{var}")

                            if sample_data_button:
                                st.session_state[f"sample_{var}"] = True
                                st.session_state[f"{var}_content"] = DataLoader.load_sample_data(
                                    "policy")
                                st.rerun()

                            if st.session_state.get(f"sample_{var}", False):
                                variables[var] = st.session_state.get(
                                    f"{var}_content", "")
                                text_area = st.text_area(
                                    f"Enter {var}", value=variables[var], height=200, key=f"{var}_textarea")
                                variables[var] = text_area
                            else:
                                variables[var] = st.text_area(
                                    f"Enter {var}", "", height=200, key=f"{var}_textarea")
                        elif var == "claim_text":
                            st.write(f"**{var}**")

                            if f"sample_{var}" not in st.session_state:
                                st.session_state[f"sample_{var}"] = False

                            sample_data_button = st.button(
                                "Load Sample Claim", key=f"load_sample_{var}")

                            if sample_data_button:
                                st.session_state[f"sample_{var}"] = True
                                st.session_state[f"{var}_content"] = DataLoader.load_sample_data(
                                    "claim")
                                st.rerun()

                            if st.session_state.get(f"sample_{var}", False):
                                variables[var] = st.session_state.get(
                                    f"{var}_content", "")
                                text_area = st.text_area(
                                    f"Enter {var}", value=variables[var], height=200, key=f"{var}_textarea")
                                variables[var] = text_area
                            else:
                                variables[var] = st.text_area(
                                    f"Enter {var}", "", height=200, key=f"{var}_textarea")
                        elif var == "inquiry_text":
                            st.write(f"**{var}**")

                            if f"sample_{var}" not in st.session_state:
                                st.session_state[f"sample_{var}"] = False

                            sample_data_button = st.button(
                                "Load Sample Inquiry", key=f"load_sample_{var}")

                            if sample_data_button:
                                st.session_state[f"sample_{var}"] = True
                                st.session_state[f"{var}_content"] = DataLoader.load_sample_data(
                                    "customer_inquiry")
                                st.rerun()

                            if st.session_state.get(f"sample_{var}", False):
                                variables[var] = st.session_state.get(
                                    f"{var}_content", "")
                                text_area = st.text_area(
                                    f"Enter {var}", value=variables[var], height=200, key=f"{var}_textarea")
                                variables[var] = text_area
                            else:
                                variables[var] = st.text_area(
                                    f"Enter {var}", "", height=200, key=f"{var}_textarea")
                        else:
                            variables[var] = st.text_area(
                                f"Enter {var}", "", height=200, key=f"{var}_textarea")

                st.subheader("Prompt Strategy")
                strategy_type = st.selectbox(
                    "Select Prompt Strategy",
                    ["zero_shot", "few_shot", "chain_of_thought"],
                    index=0 if selected_template.strategy_type == "zero_shot" else (
                        1 if selected_template.strategy_type == "few_shot" else 2
                    )
                )

        with col2:
            st.subheader("Generation Settings")

            default_max_tokens = st.session_state.get(
                "default_max_tokens", 512)

            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                                    help="Higher values make output more random, lower values more deterministic")
            max_tokens = st.slider("Max Tokens", 64, 4096, default_max_tokens, 64,
                                   help="Maximum number of tokens to generate. Lower values are faster, especially on CPU.")

            is_cpu_optimized = st.session_state.get("is_cpu_optimized", False)
            if is_cpu_optimized:
                st.info(
                    "ℹ️ Running with CPU optimizations. Generation will be slower but more stable.")

                use_greedy = st.checkbox(
                    "Use greedy decoding",
                    value=False,
                    help="Faster but less creative text generation. Recommended for CPU."
                )

                st.warning(
                    "⚠️ CPU generation can take several minutes. For best results, keep your prompts short and max tokens small.")

            generate_button = st.button("Generate Output")

            if generate_button:

                if all(v.strip() for v in variables.values()):
                    try:

                        with st.spinner("Preparing prompt..."):

                            strategy_kwargs = {}
                            if strategy_type == "few_shot":

                                if task_type == "policy_summary":
                                    strategy_kwargs = {
                                        "examples": [
                                            {
                                                "input": "Auto insurance policy for driver with $100k/$300k liability, $500 deductible",
                                                "output": "This policy provides auto insurance with bodily injury liability limits of $100,000 per person and $300,000 per accident. The deductible for physical damage is $500."
                                            }
                                        ],
                                        "example_template": "Input: {input}\nOutput: {output}"
                                    }
                                elif task_type == "claim_response":
                                    strategy_kwargs = {
                                        "examples": [
                                            {
                                                "claim": "Rear-end collision with damage to bumper",
                                                "response": "We have approved your claim for the rear-end collision. Your policy covers the damage to your bumper, less your deductible."
                                            }
                                        ],
                                        "example_template": "Claim: {claim}\nResponse: {response}"
                                    }
                            elif strategy_type == "chain_of_thought":

                                if task_type == "policy_summary":
                                    strategy_kwargs = {
                                        "cot_examples": [
                                            {
                                                "input": "Auto insurance policy with $100k/$300k liability, $500 deductible",
                                                "reasoning": "This policy has liability limits of $100k per person and $300k per accident. It also has a $500 deductible for physical damage.",
                                                "output": "This policy provides auto insurance with bodily injury liability limits of $100,000 per person and $300,000 per accident. The deductible for physical damage is $500."
                                            }
                                        ],
                                        "cot_example_template": "Input: {input}\nReasoning: {reasoning}\nOutput: {output}"
                                    }

                            strategy = create_prompt_strategy(
                                strategy_type, **strategy_kwargs)

                            if strategy_type == "zero_shot":
                                formatted_prompt = selected_template.template.format(
                                    **variables)
                            else:
                                formatted_prompt = strategy.apply(
                                    variables, selected_template.template)

                            generation_params = {
                                "max_length": max_tokens,
                                "temperature": temperature,
                                "top_p": 0.9,
                                "top_k": 50,
                                "num_return_sequences": 1,

                                "do_sample": not (is_cpu_optimized and locals().get('use_greedy', False)),

                                "timeout_seconds": st.session_state.get("generation_timeout", 60)
                            }

                            if is_cpu_optimized:
                                if locals().get('use_greedy', False):

                                    generation_params["temperature"] = 1.0
                                    generation_params["top_p"] = 1.0
                                    generation_params["top_k"] = 1

                        with st.spinner("Generating text... Please wait."):

                            if is_cpu_optimized:
                                st.info(
                                    f"Generating with timeout of {generation_params['timeout_seconds']} seconds. CPU generation may take a while...")

                            progress_placeholder = st.empty()
                            progress_bar = progress_placeholder.progress(0.0)
                            status_text = st.empty()

                            status_text.text(
                                "Running model inference... (this may take a while on CPU)")

                            if is_cpu_optimized:

                                progress_bar.progress(0.5)
                                status_text.text(
                                    "Generating text... This may take several minutes on CPU. Please be patient.")

                            try:

                                generation_start_time = time.time()

                                output_container = st.empty()
                                generated_text = ""

                                def streaming_callback(token):
                                    nonlocal generated_text
                                    generated_text += token
                                    output_container.markdown(
                                        f"**Generated Output:**\n\n{generated_text}")

                                full_text = st.session_state.inference_engine.generate_with_streaming(
                                    formatted_prompt,
                                    callback=streaming_callback,
                                    max_length=generation_params["max_length"],
                                    temperature=generation_params["temperature"],
                                    top_p=generation_params["top_p"],
                                    top_k=generation_params["top_k"],
                                    do_sample=generation_params["do_sample"],
                                    timeout_seconds=generation_params["timeout_seconds"]
                                )

                                progress_bar.progress(1.0)

                                elapsed_time = time.time() - generation_start_time
                                status_text.text(
                                    f"Generation completed in {elapsed_time:.2f} seconds")

                                time.sleep(0.5)
                                progress_placeholder.empty()
                                status_text.empty()

                                if full_text.startswith("Generation timed out") or full_text.startswith("Generation error"):
                                    st.error(full_text)
                                    st.info(
                                        "Try reducing the max tokens, using a simpler prompt, or increasing the timeout setting in the model selection page.")

                                    if is_cpu_optimized:
                                        st.warning(
                                            "CPU-specific suggestions:")
                                        st.markdown("""
                                        - Try enabling 'greedy decoding' for faster generation
                                        - Use a much shorter prompt (under 100 words)
                                        - Reduce max tokens to 128 or less
                                        - Try a smaller model like Phi-2 or TinyLLaMA
                                        """)
                                else:

                                    if "generated_outputs" not in st.session_state:
                                        st.session_state.generated_outputs = []

                                    generation_time = time.time() - generation_start_time
                                    st.info(
                                        f"Generation completed in {generation_time:.2f} seconds")

                                    st.session_state.generated_outputs.append({
                                        "template_name": selected_template_name,
                                        "task_type": task_type,
                                        "strategy_type": strategy_type,
                                        "input_variables": variables.copy(),
                                        "generation_params": generation_params.copy(),
                                        "output": full_text,
                                        "timestamp": pd.Timestamp.now().isoformat(),
                                        "generation_time": generation_time
                                    })

                                    if st.button("Evaluate Output"):
                                        SessionState.set_tab("evaluation")
                                        st.rerun()

                            except Exception as e:
                                st.error(f"Error during generation: {str(e)}")
                                logger.error(
                                    f"Error during generation: {str(e)}")
                                st.info(
                                    "Troubleshooting tips: Try clearing your browser cache, restarting the app, or using a smaller model.")

                    except Exception as e:
                        st.error(f"Error preparing generation: {str(e)}")
                        logger.error(f"Error preparing generation: {str(e)}")
                else:
                    st.error("Please fill in all required variables.")

            if st.session_state.generated_outputs:
                st.subheader("Previous Outputs")

                for idx, output in enumerate(st.session_state.generated_outputs):
                    with st.expander(f"Output {idx+1}: {output['template_name']} ({output['timestamp']})"):
                        st.write(f"Task Type: {output['task_type']}")
                        st.write(f"Strategy: {output['strategy_type']}")
                        st.text_area(
                            f"Generated Text {idx+1}", output['output'], height=150)

                        if st.button(f"Evaluate Output {idx+1}"):

                            st.session_state.current_output_idx = idx
                            SessionState.set_tab("evaluation")
                            st.rerun()

class EvaluationPage:
    """Class to handle the evaluation page UI and logic."""

    def render(self):
        """Render the evaluation page."""
        st.header("Evaluation")

        if not st.session_state.generated_outputs:
            st.warning("No outputs to evaluate. Generate some outputs first.")
            if st.button("Go to Prompt Engineering"):
                SessionState.set_tab("prompt_engineering")
                st.rerun()
            return

        metrics_manager = get_metrics_manager()

        current_idx = getattr(st.session_state, "current_output_idx", len(
            st.session_state.generated_outputs) - 1)
        current_output = st.session_state.generated_outputs[current_idx]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Output to Evaluate")
            st.write(f"Task Type: {current_output['task_type']}")
            st.write(f"Template: {current_output['template_name']}")
            st.write(f"Strategy: {current_output['strategy_type']}")

            st.text_area("Input Variables", str(
                current_output['input_variables']), height=150)
            st.text_area("Generated Output",
                         current_output['output'], height=300)

            st.subheader("Reference Text (Optional)")
            reference_text = st.text_area("Enter reference text for comparison", "", height=200,
                                          help="Provide a reference text to compare the generated output against")

        with col2:
            st.subheader("Evaluation Metrics")

            available_metrics = metrics_manager.list_metrics()
            selected_metrics = st.multiselect(
                "Select metrics to use",
                [m["name"] for m in available_metrics],
                default=["relevance", "completeness", "complexity"]
            )

            st.subheader("Additional Context")

            context = {}

            if current_output['task_type'] == "policy_summary":
                required_sections = st.text_input(
                    "Required Sections (comma-separated)",
                    "coverages,limits,exclusions,premium",
                    help="Sections that should be included in the summary"
                )
                context["required_sections"] = [s.strip()
                                                for s in required_sections.split(",")]

                required_phrases = st.text_input(
                    "Required Phrases (comma-separated)",
                    "policy,coverage,deductible",
                    help="Phrases that should be included in the output"
                )
                context["required_phrases"] = [s.strip()
                                               for s in required_phrases.split(",")]

                prohibited_phrases = st.text_input(
                    "Prohibited Phrases (comma-separated)",
                    "uncertain,unclear,not mentioned",
                    help="Phrases that should not be included in the output"
                )
                context["prohibited_phrases"] = [s.strip()
                                                 for s in prohibited_phrases.split(",")]

            elif current_output['task_type'] == "claim_response":
                required_phrases = st.text_input(
                    "Required Phrases (comma-separated)",
                    "claim,policy,coverage",
                    help="Phrases that should be included in the output"
                )
                context["required_phrases"] = [s.strip()
                                               for s in required_phrases.split(",")]

                prohibited_phrases = st.text_input(
                    "Prohibited Phrases (comma-separated)",
                    "uncertain,unclear,not mentioned",
                    help="Phrases that should not be included in the output"
                )
                context["prohibited_phrases"] = [s.strip()
                                                 for s in prohibited_phrases.split(",")]

            if st.button("Run Evaluation"):
                with st.spinner("Evaluating..."):
                    try:

                        evaluation_results = metrics_manager.evaluate(
                            generated_text=current_output['output'],
                            metric_names=selected_metrics,
                            reference_text=reference_text if reference_text.strip() else None,
                            context=context
                        )

                        st.session_state.evaluation_results[current_idx] = {
                            "metrics": [result.as_dict() for result in evaluation_results.values()],
                            "reference_text": reference_text,
                            "context": context
                        }

                        st.success("Evaluation complete!")

                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        logger.error(f"Error during evaluation: {str(e)}")

            if current_idx in st.session_state.evaluation_results:
                st.subheader("Evaluation Results")

                results = st.session_state.evaluation_results[current_idx]

                results_df = pd.DataFrame([
                    {
                        "Metric": result["metric_name"],
                        "Score": result["score"],
                        "Max Score": result["max_score"],
                        "Normalized Score": result["normalized_score"]
                    }
                    for result in results["metrics"]
                ])

                results_df = prepare_dataframe_for_display(results_df)
                st.dataframe(results_df)

                fig = px.bar(
                    results_df,
                    x="Metric",
                    y="Normalized Score",
                    title="Evaluation Scores",
                    color="Normalized Score",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1]
                )
                st.plotly_chart(fig)

                for result in results["metrics"]:
                    with st.expander(f"Detailed Results for {result['metric_name']}"):
                        st.write(
                            f"Score: {result['score']:.3f} / {result['max_score']:.3f}")
                        st.write(
                            f"Normalized Score: {result['normalized_score']:.3f}")

                        if result["details"]:
                            st.write("Details:")
                            st.json(result["details"])

                if st.button("Export Results"):
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    export_path = f"evaluation_results_{timestamp}.json"

                    try:
                        with open(export_path, "w") as f:
                            json.dump({
                                "output": current_output,
                                "evaluation_results": results
                            }, f, indent=2)

                        st.success(f"Results exported to {export_path}")

                    except Exception as e:
                        st.error(f"Error exporting results: {str(e)}")
                        logger.error(f"Error exporting results: {str(e)}")

class BenchmarksPage:
    """Class to handle the benchmarks page UI and logic."""

    def render(self):
        """Render the benchmarks page."""
        st.header("Benchmarks")

        if st.session_state.inference_engine is None:
            st.warning("Please load a model first.")
            if st.button("Go to Model Selection"):
                SessionState.set_tab("model_selection")
                st.rerun()
            return

        benchmark_manager = get_benchmark_manager()

        if not benchmark_manager.list_benchmarks():

            logger.info(
                f"No benchmarks found in directory: {benchmark_manager.benchmarks_dir}")

            if st.button("Create Sample Benchmarks"):
                with st.spinner("Creating sample benchmarks..."):
                    try:
                        logger.info("Creating sample benchmarks...")

                        os.makedirs(
                            benchmark_manager.benchmarks_dir, exist_ok=True)
                        logger.info(
                            f"Ensured benchmark directory exists: {benchmark_manager.benchmarks_dir}")

                        policy_benchmark = create_policy_summary_benchmark()
                        if policy_benchmark is None:
                            raise ValueError(
                                "Failed to create policy summary benchmark")
                        logger.info(
                            f"Created policy benchmark: {policy_benchmark.name} with {len(policy_benchmark.examples)} examples")
                        benchmark_manager.benchmarks[policy_benchmark.name] = policy_benchmark

                        benchmark_path = os.path.join(
                            benchmark_manager.benchmarks_dir, f"{policy_benchmark.name}.json")
                        policy_benchmark.save(benchmark_path)
                        logger.info(
                            f"Saved policy benchmark to {benchmark_path}")

                        claim_benchmark = create_claim_response_benchmark()
                        if claim_benchmark is None:
                            raise ValueError(
                                "Failed to create claim response benchmark")
                        logger.info(
                            f"Created claim benchmark: {claim_benchmark.name} with {len(claim_benchmark.examples)} examples")
                        benchmark_manager.benchmarks[claim_benchmark.name] = claim_benchmark

                        benchmark_path = os.path.join(
                            benchmark_manager.benchmarks_dir, f"{claim_benchmark.name}.json")
                        claim_benchmark.save(benchmark_path)
                        logger.info(
                            f"Saved claim benchmark to {benchmark_path}")

                        st.success("Sample benchmarks created successfully!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error creating sample benchmarks: {str(e)}")
                        logger.error(
                            f"Error creating sample benchmarks: {str(e)}")
        else:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Available Benchmarks")

                benchmarks = benchmark_manager.list_benchmarks()
                benchmark_names = [b["name"] for b in benchmarks]
                selected_benchmark = st.selectbox(
                    "Select Benchmark", benchmark_names)

                if selected_benchmark:
                    benchmark = benchmark_manager.get_benchmark(
                        selected_benchmark)

                    if benchmark:
                        st.write(f"Task Type: {benchmark.task_type}")
                        st.write(f"Description: {benchmark.description}")
                        st.write(
                            f"Number of Examples: {len(benchmark.examples)}")
                        st.write(f"Metrics: {', '.join(benchmark.metrics)}")

                        with st.expander("View Examples"):
                            for idx, example in enumerate(benchmark.examples):
                                st.write(f"Example {idx+1}: {example.id}")
                                st.text_area(
                                    f"Input {idx+1}", example.input_text, height=150)
                                st.text_area(
                                    f"Reference Output {idx+1}", example.reference_output, height=150)

                        if st.button("Run Benchmark"):
                            with st.spinner(f"Running benchmark {selected_benchmark}..."):
                                try:

                                    def generate_output(input_text):
                                        generation_params = {
                                            "max_length": 512,
                                            "temperature": 0.7,
                                            "top_p": 0.9,
                                            "top_k": 50,
                                            "do_sample": True
                                        }

                                        if st.checkbox("Show generation in real-time", value=True):

                                            output_container = st.empty()
                                            generated_text = ""

                                            def streaming_callback(token):
                                                nonlocal generated_text
                                                generated_text += token
                                                output_container.markdown(
                                                    f"**Generating:**\n\n{generated_text}")

                                            full_text = st.session_state.inference_engine.generate_with_streaming(
                                                input_text,
                                                callback=streaming_callback,
                                                max_length=generation_params["max_length"],
                                                temperature=generation_params["temperature"],
                                                top_p=generation_params["top_p"],
                                                top_k=generation_params["top_k"],
                                                do_sample=generation_params["do_sample"]
                                            )

                                            return full_text
                                        else:

                                            generated_texts = st.session_state.inference_engine.generate(
                                                input_text,
                                                **generation_params
                                            )

                                            return generated_texts[0] if generated_texts else ""

                                    model_id = st.session_state.model.__class__.__name__
                                    result = benchmark_manager.run_benchmark(
                                        benchmark_name=selected_benchmark,
                                        model_id=model_id,
                                        generate_fn=generate_output
                                    )

                                    if "benchmark_results" not in st.session_state:
                                        st.session_state.benchmark_results = {}

                                    st.session_state.benchmark_results[selected_benchmark] = result

                                    st.success(
                                        f"Benchmark {selected_benchmark} completed successfully!")

                                except Exception as e:
                                    st.error(
                                        f"Error running benchmark: {str(e)}")
                                    logger.error(
                                        f"Error running benchmark: {str(e)}")

            with col2:
                st.subheader("Benchmark Results")

                if hasattr(st.session_state, "benchmark_results") and st.session_state.benchmark_results:

                    result_keys = list(
                        st.session_state.benchmark_results.keys())
                    selected_result = st.selectbox(
                        "Select Result", result_keys)

                    if selected_result:
                        result = st.session_state.benchmark_results[selected_result]

                        st.write("Aggregate Scores:")

                        scores_df = pd.DataFrame([
                            {"Metric": metric, "Score": score}
                            for metric, score in result.aggregate_scores.items()
                        ])

                        scores_df = prepare_dataframe_for_display(scores_df)
                        st.dataframe(scores_df)

                        metrics = [
                            m for m in result.aggregate_scores.keys() if m != "overall"]
                        scores = [result.aggregate_scores[m] for m in metrics]

                        fig = go.Figure()

                        fig.add_trace(go.Scatterpolar(
                            r=scores,
                            theta=metrics,
                            fill='toself',
                            name=result.model_id
                        ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )
                            ),
                            title=f"Benchmark Results: {selected_result}"
                        )

                        st.plotly_chart(fig)

                        if st.button("Export Benchmark Results"):
                            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                            export_path = f"benchmark_results_{selected_result}_{timestamp}.json"

                            try:
                                with open(export_path, "w") as f:
                                    json.dump(result.to_dict(), f, indent=2)

                                st.success(
                                    f"Results exported to {export_path}")

                            except Exception as e:
                                st.error(f"Error exporting results: {str(e)}")
                                logger.error(
                                    f"Error exporting results: {str(e)}")

                        with st.expander("View Detailed Results"):
                            for idx, example_result in enumerate(result.results):
                                st.write(
                                    f"Example {idx+1}: {example_result['example_id']}")

                                col3, col4 = st.columns(2)

                                with col3:
                                    st.text_area(
                                        f"Generated Output {idx+1}", example_result['generated_text'], height=150)

                                with col4:
                                    st.text_area(
                                        f"Reference Output {idx+1}", example_result['reference_output'], height=150)

                                st.write("Metrics:")
                                for metric, score in example_result['metrics'].items():
                                    st.write(f"{metric}: {score:.3f}")
                else:
                    st.info(
                        "No benchmark results available. Run a benchmark first.")

class ModelComparisonPage:
    """Class to handle the model comparison page UI and logic."""

    def render(self):
        """Render the model comparison page."""
        st.header("Model Comparison")

        if not hasattr(st.session_state, "benchmark_results") or not st.session_state.benchmark_results:
            st.warning("No benchmark results available. Run benchmarks first.")
            if st.button("Go to Benchmarks"):
                SessionState.set_tab("benchmarks")
                st.rerun()
            return

        benchmark_manager = get_benchmark_manager()

        result_keys = list(st.session_state.benchmark_results.keys())

        st.subheader("Compare Models")
        st.write("Load more models and run benchmarks to compare their performance.")

        if len(result_keys) > 1:

            selected_benchmark = st.selectbox("Select Benchmark", result_keys)

            results = {
                result.model_id: result
                for key, result in st.session_state.benchmark_results.items()
                if key == selected_benchmark
            }

            comparison_df = benchmark_manager.compare_models(
                selected_benchmark, results)

            st.subheader(f"Comparison for {selected_benchmark}")
            comparison_df = prepare_dataframe_for_display(comparison_df)
            st.dataframe(comparison_df)

            metrics = [
                col for col in comparison_df.columns if col.endswith("_score")]

            if metrics:

                fig1 = px.bar(
                    comparison_df,
                    x="model_id",
                    y="overall_score",
                    title="Overall Score Comparison",
                    color="overall_score",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1]
                )
                st.plotly_chart(fig1)

                fig2 = px.bar(
                    comparison_df.melt(
                        id_vars=["model_id"],
                        value_vars=metrics,
                        var_name="Metric",
                        value_name="Score"
                    ),
                    x="model_id",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Metric Score Comparison"
                )
                st.plotly_chart(fig2)

                metric_names = [m.replace("_score", "") for m in metrics]

                fig3 = go.Figure()

                for model_id in comparison_df["model_id"]:
                    model_data = comparison_df[comparison_df["model_id"] == model_id]
                    scores = [model_data[m].values[0] for m in metrics]

                    fig3.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=metric_names,
                        fill='toself',
                        name=model_id
                    ))

                fig3.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Model Comparison Radar Chart"
                )

                st.plotly_chart(fig3)
        else:
            st.info(
                "Run benchmarks with at least two different models to enable comparison.")

class SettingsPage:
    """Class to handle the settings page UI and logic."""

    def render(self):
        """Render the settings page."""
        st.header("Settings")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Application Settings")

            theme = st.selectbox(
                "Application Theme",
                ["Light", "Dark"],
                index=1
            )

            cache_models = st.checkbox("Cache Models", value=True,
                                       help="Keep models in memory between sessions")

            debug_mode = st.checkbox("Debug Mode", value=False,
                                     help="Show additional debugging information")

            if st.button("Save Settings"):
                st.success("Settings saved successfully!")

        with col2:
            st.subheader("API Information")

            system_info = {
                "Python Version": os.sys.version.split()[0],
                "Streamlit Version": st.__version__,
                "API Connected": "Yes" if st.session_state.model is not None else "No",
            }

            if st.session_state.model is not None:
                system_info["Selected Model"] = st.session_state.get("model_id", "Unknown")

            for key, value in system_info.items():
                st.text(f"{key}: {value}")

            if st.button("Clear Session"):

                for key in list(st.session_state.keys()):
                    del st.session_state[key]

                st.success("Session cleared successfully!")
                st.rerun()

def main():
    """Main function to run the Streamlit app."""

    st.set_page_config(
        page_title="Insurance LLM Framework",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    SessionState.initialize()

    st.sidebar.title("Insurance LLM Framework")
    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/3448/3448502.png", width=100)

    nav_options = {
        "Model Selection": "model_selection",
        "Prompt Engineering": "prompt_engineering",
        "Evaluation": "evaluation",
        "Benchmarks": "benchmarks",
        "Model Comparison": "model_comparison",
        "Settings": "settings"
    }

    for label, tab in nav_options.items():
        if st.sidebar.button(label, key=f"nav_{tab}"):
            SessionState.set_tab(tab)
            st.rerun()

    current_tab = st.session_state.current_tab

    if current_tab == "model_selection":
        model_selection_page = ModelSelectionPage()
        model_selection_page.render()
    elif current_tab == "prompt_engineering":
        prompt_engineering_page = PromptEngineeringPage()
        prompt_engineering_page.render()
    elif current_tab == "evaluation":
        evaluation_page = EvaluationPage()
        evaluation_page.render()
    elif current_tab == "benchmarks":
        benchmarks_page = BenchmarksPage()
        benchmarks_page.render()
    elif current_tab == "model_comparison":
        model_comparison_page = ModelComparisonPage()
        model_comparison_page.render()
    elif current_tab == "settings":
        settings_page = SettingsPage()
        settings_page.render()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "Claude-powered Insurance AI Framework. This application uses Claude API to provide "
        "expert-level insurance assistance for claim triage, policy analysis, risk assessment, "
        "and customer communication. "
        "[Get API key →](https://console.anthropic.com)"
    )

if __name__ == "__main__":
    main()
