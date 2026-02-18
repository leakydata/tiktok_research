"""
Multi-backend LLM clients with a unified generate() interface.

Backends:
  - OllamaClient (in ollama_client.py): local Ollama inference
  - OpenAICompatibleClient: OpenAI, DeepSeek, MiniMax (all OpenAI-compatible)
  - AnthropicClient: Claude models via Anthropic Messages API
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Base class for all LLM backends."""

    @abstractmethod
    def generate(
        self,
        model_key: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 50,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> dict:
        """Generate a completion. Returns:
        {
            'response': str,
            'tokens_generated': int,
            'processing_time_ms': int,
            'model_key': str,
            'model_name': str,
            'model_family': str,
            'model_size_b': float,
            'quantization': str,
            'inference_params': dict,
            'ollama_version': str,  # backend version string
        }
        """
        ...

    def ensure_model_loaded(self, model_key: str) -> None:
        """Pre-load a model. No-op for cloud backends."""
        pass

    def check_models_available(self) -> dict[str, bool]:
        """Check which configured models are available."""
        return {}

    def list_available_models(self) -> list[str]:
        """List available models."""
        return []


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI-compatible APIs: OpenAI, DeepSeek, MiniMax."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 backend_name: str = "openai", timeout: int = 600):
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.backend_name = backend_name
        self._version = f"{backend_name}/{openai.__version__}"

    def generate(
        self,
        model_key: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 50,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> dict:
        from config import MODELS_TO_TEST
        if model_key not in MODELS_TO_TEST:
            raise ValueError(f"Unknown model key: {model_key}")

        model_cfg = MODELS_TO_TEST[model_key]
        api_model = model_cfg['api_model_name']

        if temperature == 0.0 and seed is None:
            seed = 42

        # OpenAI newer models (gpt-5-*) require max_completion_tokens;
        # DeepSeek and MiniMax still use max_tokens.
        token_param = 'max_completion_tokens' if self.backend_name == 'openai' else 'max_tokens'
        params = {
            'model': api_model,
            'messages': [{'role': 'user', 'content': prompt}],
            token_param: max_tokens,
        }
        # Some models (gpt-5-nano) reject custom temperature/top_p values
        if not model_cfg.get('fixed_temperature'):
            params['temperature'] = temperature
            params['top_p'] = top_p
        if seed is not None:
            params['seed'] = seed
        if stop_sequences:
            params['stop'] = stop_sequences

        start = time.time()
        response = self.client.chat.completions.create(**params)
        elapsed_ms = int((time.time() - start) * 1000)

        text = response.choices[0].message.content or ''
        tokens = response.usage.completion_tokens if response.usage else 0

        inference_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'seed': seed,
            'stop': stop_sequences,
        }

        return {
            'response': text.strip(),
            'tokens_generated': tokens,
            'processing_time_ms': elapsed_ms,
            'model_key': model_key,
            'model_name': api_model,
            'model_family': model_cfg['family'],
            'model_size_b': model_cfg['size_b'],
            'quantization': model_cfg.get('quantization', 'none'),
            'inference_params': inference_params,
            'ollama_version': self._version,
        }

    def check_models_available(self) -> dict[str, bool]:
        from config import MODELS_TO_TEST
        return {
            k: True for k, v in MODELS_TO_TEST.items()
            if v.get('backend') == self.backend_name
        }


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Messages API (Claude models)."""

    def __init__(self, api_key: str, timeout: int = 600):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        self._version = f"anthropic/{anthropic.__version__}"

    def generate(
        self,
        model_key: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 50,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> dict:
        from config import MODELS_TO_TEST
        if model_key not in MODELS_TO_TEST:
            raise ValueError(f"Unknown model key: {model_key}")

        model_cfg = MODELS_TO_TEST[model_key]
        api_model = model_cfg['api_model_name']

        params = {
            'model': api_model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': temperature,
            'max_tokens': max_tokens,
            'top_p': top_p,
        }
        if stop_sequences:
            params['stop_sequences'] = stop_sequences
        # Anthropic does not support seed parameter

        start = time.time()
        response = self.client.messages.create(**params)
        elapsed_ms = int((time.time() - start) * 1000)

        text = response.content[0].text if response.content else ''
        tokens = response.usage.output_tokens if response.usage else 0

        inference_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'seed': None,
            'stop': stop_sequences,
        }

        return {
            'response': text.strip(),
            'tokens_generated': tokens,
            'processing_time_ms': elapsed_ms,
            'model_key': model_key,
            'model_name': api_model,
            'model_family': model_cfg['family'],
            'model_size_b': model_cfg['size_b'],
            'quantization': model_cfg.get('quantization', 'none'),
            'inference_params': inference_params,
            'ollama_version': self._version,
        }

    def check_models_available(self) -> dict[str, bool]:
        from config import MODELS_TO_TEST
        return {
            k: True for k, v in MODELS_TO_TEST.items()
            if v.get('backend') == 'anthropic'
        }
