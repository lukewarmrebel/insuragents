"""
Factory for creating inference engines based on provider selection.

This module handles instantiation of the correct inference engine (Claude, OpenAI, or Gemini)
based on user selection and provides a registry of available models per provider.
"""

import logging

logger = logging.getLogger(__name__)

PROVIDER_MODELS = {
    "Claude": {
        "Claude Haiku 4.5 (Fast)": "claude-haiku-4-5-20251001",
        "Claude Sonnet 4.6 (Balanced)": "claude-sonnet-4-6",
        "Claude Opus 4.7 (Best)": "claude-opus-4-7",
    },
    "OpenAI": {
        "GPT-4o Mini (Fast)": "gpt-4o-mini",
        "GPT-4o (Balanced)": "gpt-4o",
        "GPT-4 Turbo (Smart)": "gpt-4-turbo",
    },
    "Gemini": {
        "Gemini 2.5 Flash (Fast)": "gemini-2.5-flash",
        "Gemini 2.5 Lite (Balanced)": "gemini-2.5-lite",
    },
}


def create_inference_engine(provider: str, api_key: str, model_id: str):
    """
    Factory function to create the appropriate inference engine.

    Args:
        provider: Provider name ("Claude", "OpenAI", or "Gemini")
        api_key: API key for the provider
        model_id: Model identifier for the provider

    Returns:
        An instance of the appropriate inference engine

    Raises:
        ValueError: If provider is not recognized
    """
    if provider == "Claude":
        import anthropic
        from models.claude_inference import ClaudeInference
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Created Claude inference engine with model {model_id}")
        return ClaudeInference(client, model_id)

    elif provider == "OpenAI":
        from models.openai_inference import OpenAIInference
        logger.info(f"Created OpenAI inference engine with model {model_id}")
        return OpenAIInference(api_key, model_id)

    elif provider == "Gemini":
        from models.gemini_inference import GeminiInference
        logger.info(f"Created Gemini inference engine with model {model_id}")
        return GeminiInference(api_key, model_id)

    else:
        raise ValueError(f"Unknown provider: {provider}. Must be one of: Claude, OpenAI, Gemini")
