"""
Base class for inference engines supporting multiple LLM providers.

This module defines the abstract interface that all LLM providers must implement,
ensuring compatibility across Claude, OpenAI, Gemini, and future providers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable

SYSTEM_MESSAGE = "You are an experienced insurance professional assistant. Provide accurate, detailed, and professional responses to insurance-related tasks."


class BaseInference(ABC):
    """Abstract base class for LLM inference engines."""

    @abstractmethod
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
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout
            **kwargs: Additional provider-specific parameters

        Returns:
            List of generated texts
        """
        pass

    @abstractmethod
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
        Generate text with streaming callback.

        Args:
            prompt: Input prompt text
            callback: Function called with each text chunk
            max_length: Maximum output length in tokens
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            stop_sequences: Sequences that stop generation
            timeout_seconds: Request timeout
            **kwargs: Additional provider-specific parameters

        Returns:
            Complete generated text
        """
        pass
