"""
Step 4: Ollama client with full inference parameter recording for reproducibility.
"""

import requests
import time
import logging
from typing import Optional

from config import OLLAMA_BASE_URL, MODELS_TO_TEST, MAX_NUM_CTX
from llm_client import BaseLLMClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLMClient):
    """Client for local Ollama inference with full parameter tracking."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.ollama_version = self._get_version()

    def _get_version(self) -> str:
        try:
            resp = requests.get(f"{self.base_url}/api/version", timeout=5)
            return resp.json().get('version', 'unknown')
        except Exception:
            return 'unknown'

    def generate(
        self,
        model_key: str,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int = 50,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
    ) -> dict:
        """Generate a completion and return result with full inference metadata."""
        if model_key not in MODELS_TO_TEST:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODELS_TO_TEST.keys())}")

        model_cfg = MODELS_TO_TEST[model_key]
        ollama_name = model_cfg['ollama_name']

        # Cap num_ctx to avoid wasting GPU memory on unused context window
        num_ctx = min(model_cfg.get('context_length', 4096), MAX_NUM_CTX)

        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'num_ctx': num_ctx,
            'top_p': top_p,
            'top_k': top_k,
            'repeat_penalty': repeat_penalty,
        }
        if stop_sequences:
            options['stop'] = stop_sequences
        if seed is not None:
            options['seed'] = seed
        # For deterministic baseline: temperature=0 with fixed seed
        if temperature == 0.0 and seed is None:
            options['seed'] = 42

        payload = {
            'model': ollama_name,
            'messages': [
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
            'options': options,
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            result = resp.json()
            elapsed_ms = int((time.time() - start) * 1000)

            # Extract response text from chat format
            response_text = ''
            msg = result.get('message', {})
            if isinstance(msg, dict):
                response_text = msg.get('content', '')
            elif isinstance(result.get('response'), str):
                response_text = result['response']

            # Build the full inference params dict for DB storage
            inference_params = {
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'repeat_penalty': repeat_penalty,
                'num_predict': max_tokens,
                'seed': options.get('seed'),
                'stop': stop_sequences,
                'num_ctx': num_ctx,
            }

            return {
                'response': response_text.strip(),
                'tokens_generated': result.get('eval_count', 0),
                'processing_time_ms': elapsed_ms,
                'model_key': model_key,
                'model_name': ollama_name,
                'model_family': model_cfg['family'],
                'model_size_b': model_cfg['size_b'],
                'quantization': model_cfg.get('quantization', 'default'),
                'inference_params': inference_params,
                'ollama_version': self.ollama_version,
            }

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Is 'ollama serve' running?"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama request timed out for model {model_key}. "
                "The model may be too large or not loaded."
            )
        except Exception as e:
            logger.error(f"Ollama error for {model_key}: {e}")
            raise

    def ensure_model_loaded(self, model_key: str):
        """Pre-load a model into GPU memory using keep_alive (no inference needed)."""
        if model_key not in MODELS_TO_TEST:
            raise ValueError(f"Unknown model key: {model_key}")
        ollama_name = MODELS_TO_TEST[model_key]['ollama_name']
        logger.info(f"Loading model: {model_key} ({ollama_name})")
        try:
            # Use /api/chat with empty messages and keep_alive to load without inference
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={'model': ollama_name, 'messages': [], 'keep_alive': '30m'},
                timeout=120,
            )
            resp.raise_for_status()
            logger.info(f"Model {model_key} loaded successfully")
        except Exception:
            # Fallback: run a minimal generation if keep_alive-only doesn't work
            logger.debug(f"keep_alive load failed, falling back to minimal generate")
            self.generate(model_key, "hi", temperature=0.0, max_tokens=1)
            logger.info(f"Model {model_key} loaded successfully (via fallback)")

    def list_available_models(self) -> list[str]:
        """List models currently available in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get('models', [])
            return [m['name'] for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def check_models_available(self) -> dict[str, bool]:
        """Check which configured models are actually pulled in Ollama."""
        available = self.list_available_models()
        results = {}
        for key, cfg in MODELS_TO_TEST.items():
            if cfg.get('backend', 'ollama') != 'ollama':
                continue
            ollama_name = cfg['ollama_name']
            found = any(
                ollama_name in m or m.startswith(ollama_name.split(':')[0])
                for m in available
            )
            results[key] = found
        return results
