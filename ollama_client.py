"""
Step 4: Ollama client with full inference parameter recording for reproducibility.
"""

import requests
import time
import logging
from typing import Optional

from config import OLLAMA_BASE_URL, MODELS_TO_TEST

logger = logging.getLogger(__name__)


class OllamaClient:
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

        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
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
            'prompt': prompt,
            'stream': False,
            'options': options,
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            result = resp.json()
            elapsed_ms = int((time.time() - start) * 1000)

            # Build the full inference params dict for DB storage
            inference_params = {
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'repeat_penalty': repeat_penalty,
                'num_predict': max_tokens,
                'seed': options.get('seed'),
                'stop': stop_sequences,
                'num_ctx': model_cfg.get('context_length'),
            }

            return {
                'response': result.get('response', '').strip(),
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
        """Pre-load a model into GPU memory by running a trivial prompt."""
        if model_key not in MODELS_TO_TEST:
            raise ValueError(f"Unknown model key: {model_key}")
        logger.info(f"Loading model: {model_key} ({MODELS_TO_TEST[model_key]['ollama_name']})")
        self.generate(model_key, "Hello", temperature=0.0, max_tokens=1)
        logger.info(f"Model {model_key} loaded successfully")

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
            # Check if the ollama name (or a prefix of it) is in available models
            ollama_name = cfg['ollama_name']
            found = any(
                ollama_name in m or m.startswith(ollama_name.split(':')[0])
                for m in available
            )
            results[key] = found
        return results
