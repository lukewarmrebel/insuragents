"""
Google Gemini API inference module for the Insurance LLM Framework.

This module provides utilities for generating text with Google Gemini models via the Generative AI SDK.
"""

import logging
from typing import List, Optional, Callable

from models.base_inference import BaseInference, SYSTEM_MESSAGE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeminiInference(BaseInference):
    """Wrapper for Google Gemini API text generation."""

    def __init__(self, api_key: str, model_id: str):
        """
        Initialize Gemini inference engine.

        Args:
            api_key: Google API key
            model_id: Gemini model identifier (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
        """
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=SYSTEM_MESSAGE
        )
        self.model_id = model_id
        logger.info(f"Initialized GeminiInference with model: {model_id}")

    def generate(
        self,
        prompt: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
        timeout_seconds: int = 120,
        **kwargs
    ) -> List[str]:
        """
        Generate text using Gemini API.

        Args:
            prompt: Input prompt text
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (unused by Gemini)
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout (unused by Gemini)
            **kwargs: Additional parameters (unused)

        Returns:
            List containing the generated text
        """
        try:
            import google.generativeai as genai
            config = genai.types.GenerationConfig(
                max_output_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences or [],
            )
            response = self.model.generate_content(prompt, generation_config=config)
            text = response.text
            logger.info(f"Generated text with {len(text.split())} words")
            return [text]
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    def generate_with_streaming(
        self,
        prompt: str,
        callback: Callable[[str], None],
        max_length: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
        timeout_seconds: int = 120,
        **kwargs
    ) -> str:
        """
        Generate text using Gemini API with streaming callback.

        Args:
            prompt: Input prompt text
            callback: Function called with each streamed text chunk
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (unused by Gemini)
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout (unused by Gemini)
            **kwargs: Additional parameters (unused)

        Returns:
            Complete generated text
        """
        try:
            import google.generativeai as genai
            config = genai.types.GenerationConfig(
                max_output_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_sequences=stop_sequences or [],
            )
            full_text = ""
            for chunk in self.model.generate_content(
                prompt,
                generation_config=config,
                stream=True,
            ):
                if chunk.text:
                    callback(chunk.text)
                    full_text += chunk.text

            logger.info(f"Generated streaming text with {len(full_text.split())} words")
            return full_text
        except Exception as e:
            logger.error(f"Gemini API streaming error: {str(e)}")
            raise
