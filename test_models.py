"""
Quick smoke test: verify every configured model produces non-empty output
before committing to a full pipeline run. Supports Ollama and cloud backends.

Usage: python test_models.py
"""

import sys
from config import MODELS_TO_TEST, DEFAULT_MAX_TOKENS, BACKEND_CONFIGS
from label_parsing import normalize_label

TEST_PROMPT = """[Task]
Rate the CERTAINTY/HEDGING level in this health-related statement.

[Scale]
0.0-0.3 = HIGH HEDGING (maybe, possibly, I think, might be, could be)
0.7-1.0 = HIGH CERTAINTY (definitely, I know, confirmed, diagnosed, absolutely)

[Guidelines]
- Otherwise respond ONLY with a number between 0.0 and 1.0

[Segment]
I was officially diagnosed with ADHD last month by my psychiatrist after years of struggling.

[Response]
"""


def _get_client(model_key: str):
    """Create the right client for a model's backend."""
    model_cfg = MODELS_TO_TEST[model_key]
    backend = model_cfg.get('backend', 'ollama')

    if backend == 'ollama':
        from ollama_client import OllamaClient
        cfg = BACKEND_CONFIGS['ollama']
        return OllamaClient(base_url=cfg['base_url'])
    elif backend in ('openai', 'deepseek', 'minimax'):
        from llm_client import OpenAICompatibleClient
        cfg = BACKEND_CONFIGS[backend]
        return OpenAICompatibleClient(
            api_key=cfg['api_key'],
            base_url=cfg['base_url'],
            backend_name=backend,
        )
    elif backend == 'anthropic':
        from llm_client import AnthropicClient
        cfg = BACKEND_CONFIGS[backend]
        return AnthropicClient(api_key=cfg['api_key'])
    else:
        raise ValueError(f"Unknown backend: {backend}")


def main():
    print(f"Testing {len(MODELS_TO_TEST)} models\n")
    print(f"{'Model':<30s} {'Backend':>10} {'Tokens':>7} {'Time':>7} {'Kind':<10} {'Parsed':<15} {'Raw (first 80 chars)'}")
    print('-' * 130)

    passed = 0
    failed = 0
    skipped = 0

    for key, cfg in MODELS_TO_TEST.items():
        backend = cfg.get('backend', 'ollama')

        # Skip cloud models without API keys
        if backend != 'ollama':
            bcfg = BACKEND_CONFIGS.get(backend, {})
            if not bcfg.get('api_key'):
                print(f"{key:<30s} {backend:>10}   -- no API key set, skipping")
                skipped += 1
                continue

        try:
            client = _get_client(key)

            # For Ollama, check if model is pulled
            if backend == 'ollama':
                available = client.check_models_available()
                if not available.get(key, False):
                    print(f"{key:<30s} {backend:>10}   -- model not pulled, skipping")
                    failed += 1
                    continue

            result = client.generate(
                model_key=key,
                prompt=TEST_PROMPT,
                temperature=0.0,
                max_tokens=DEFAULT_MAX_TOKENS,
            )

            raw = result['response']
            tokens = result['tokens_generated']
            time_ms = result['processing_time_ms']
            parsed = normalize_label(raw, 'certainty_hedging')

            raw_preview = raw.replace('\n', ' ')[:80]
            is_empty = len(raw.strip()) == 0
            kind = parsed['label_kind']

            status = 'PASS' if not is_empty and kind in ('float', 'category', 'none') else 'FAIL'
            if status == 'PASS':
                passed += 1
            else:
                failed += 1

            label_str = ''
            if parsed['label_value_float'] is not None:
                label_str = f"{parsed['label_value_float']} ({parsed['label_bin']})"
            elif parsed['label_value_text']:
                label_str = parsed['label_value_text']
            else:
                label_str = kind

            print(f"{key:<30s} {backend:>10} {tokens:>7} {time_ms:>6}ms {kind:<10} {label_str:<15} {raw_preview}")

            if is_empty:
                print(f"  ^ WARNING: empty response!")

        except Exception as e:
            print(f"{key:<30s} {backend:>10}   {str(e)[:80]}")
            failed += 1

    print()
    total = len(MODELS_TO_TEST)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped (no API key) out of {total} models")

    if failed > 0:
        print("\nFix failing models before running the full pipeline.")
        sys.exit(1)
    else:
        print("\nAll tested models producing valid output. Safe to run the pipeline.")


if __name__ == '__main__':
    main()
