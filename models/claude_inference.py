"""
Claude API inference module for the Insurance LLM Framework.

This module provides utilities for generating text with Claude models via the Anthropic API.
"""

import logging
from typing import List, Optional, Callable

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClaudeInference:
    """Wrapper for Claude API text generation."""

    SYSTEM_MESSAGE = "You are an experienced insurance professional assistant. Provide accurate, detailed, and professional responses to insurance-related tasks."

    def __init__(self, client: anthropic.Anthropic, model_id: str):
        """
        Initialize Claude inference engine.

        Args:
            client: Anthropic API client
            model_id: Claude model identifier (e.g., "claude-haiku-4-5-20251001")
        """
        self.client = client
        self.model_id = model_id
        logger.info(f"Initialized ClaudeInference with model: {model_id}")

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
        Generate text using Claude API.

        Args:
            prompt: Input prompt text
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (unused by Claude, kept for compatibility)
            num_return_sequences: Number of sequences to generate (Claude always returns 1)
            do_sample: Whether to use sampling (ignored by Claude)
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout (unused, kept for compatibility)
            **kwargs: Additional parameters (unused)

        Returns:
            List containing the generated text
        """
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_length,
                system=self.SYSTEM_MESSAGE,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stop_sequences=stop_sequences or [],
            )
            text = response.content[0].text
            logger.info(f"Generated text with {len(text.split())} words")
            return [text]
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {str(e)}")
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
        Generate text using Claude API with streaming callback.

        Args:
            prompt: Input prompt text
            callback: Function called with each streamed text chunk
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (unused by Claude)
            do_sample: Whether to use sampling (ignored by Claude)
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout (unused)
            **kwargs: Additional parameters (unused)

        Returns:
            Complete generated text
        """
        try:
            full_text = ""
            with self.client.messages.stream(
                model=self.model_id,
                max_tokens=max_length,
                system=self.SYSTEM_MESSAGE,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stop_sequences=stop_sequences or [],
            ) as stream:
                for text in stream.text_stream:
                    callback(text)
                    full_text += text

            logger.info(f"Generated streaming text with {len(full_text.split())} words")
            return full_text
        except anthropic.APIError as e:
            logger.error(f"Claude API streaming error: {str(e)}")
            raise
