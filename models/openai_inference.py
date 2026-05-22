"""
OpenAI API inference module for the Insurance LLM Framework.

This module provides utilities for generating text with OpenAI models via the OpenAI SDK.
"""

import logging
from typing import List, Optional, Callable

from models.base_inference import BaseInference, SYSTEM_MESSAGE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OpenAIInference(BaseInference):
    """Wrapper for OpenAI API text generation."""

    def __init__(self, api_key: str, model_id: str):
        """
        Initialize OpenAI inference engine.

        Args:
            api_key: OpenAI API key
            model_id: OpenAI model identifier (e.g., "gpt-4o-mini", "gpt-4o")
        """
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model_id = model_id
        logger.info(f"Initialized OpenAIInference with model: {model_id}")

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
        Generate text using OpenAI API.

        Args:
            prompt: Input prompt text
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (unused by OpenAI)
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (unused by OpenAI)
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout (unused by OpenAI)
            **kwargs: Additional parameters (unused)

        Returns:
            List containing the generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences or None,
            )
            text = response.choices[0].message.content
            logger.info(f"Generated text with {len(text.split())} words")
            return [text]
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
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
        Generate text using OpenAI API with streaming callback.

        Args:
            prompt: Input prompt text
            callback: Function called with each streamed text chunk
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (unused by OpenAI)
            do_sample: Whether to use sampling (unused by OpenAI)
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout (unused by OpenAI)
            **kwargs: Additional parameters (unused)

        Returns:
            Complete generated text
        """
        try:
            full_text = ""
            with self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences or None,
                stream=True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        callback(delta)
                        full_text += delta

            logger.info(f"Generated streaming text with {len(full_text.split())} words")
            return full_text
        except Exception as e:
            logger.error(f"OpenAI API streaming error: {str(e)}")
            raise
