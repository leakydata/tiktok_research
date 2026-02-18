"""
Configuration for Multi-Run LLM Annotation Pipeline.
All secrets read from environment variables or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration - read from environment
DB_CONFIG = {
    'dbname': os.getenv('ANNOTATION_DB_NAME', 'tiktok_disorders'),
    'user': os.getenv('ANNOTATION_DB_USER', 'postgres'),
    'password': os.getenv('ANNOTATION_DB_PASSWORD', ''),
    'host': os.getenv('ANNOTATION_DB_HOST', 'localhost'),
    'port': int(os.getenv('ANNOTATION_DB_PORT', '5433'))
}

# Ollama configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')

# Cloud API keys (set in .env when needed)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
MINIMAX_API_KEY = os.getenv('MINIMAX_API_KEY', '')

# Backend connection configs
BACKEND_CONFIGS = {
    'ollama': {
        'base_url': OLLAMA_BASE_URL,
    },
    'openai': {
        'api_key': OPENAI_API_KEY,
        'base_url': 'https://api.openai.com/v1',
    },
    'anthropic': {
        'api_key': ANTHROPIC_API_KEY,
    },
    'deepseek': {
        'api_key': DEEPSEEK_API_KEY,
        'base_url': 'https://api.deepseek.com/v1',
    },
    'minimax': {
        'api_key': MINIMAX_API_KEY,
        'base_url': 'https://api.minimax.io/v1',
    },
}

# ── Local Ollama models ──────────────────────────────────────────────────
# Ordered smallest to largest for faster initial results.
# All models run locally so token limits are generous (models stop at EOS naturally).
MODELS_TO_TEST = {
    'glm-4.7-flash': {
        'backend': 'ollama',
        'ollama_name': 'glm-4.7-flash',
        'family': 'glm',
        'size_b': 4.7,
        'context_length': 8192,
    },
    'phi4:latest': {
        'backend': 'ollama',
        'ollama_name': 'phi4:latest',
        'family': 'phi',
        'size_b': 14.0,
        'context_length': 16384,
    },
    'gpt-oss:20b': {
        'backend': 'ollama',
        'ollama_name': 'gpt-oss:20b',
        'family': 'gpt',
        'size_b': 20.0,
        'context_length': 8192,
    },
    'alibayram/medgemma:27b': {
        'backend': 'ollama',
        'ollama_name': 'alibayram/medgemma:27b',
        'family': 'gemma',
        'size_b': 27.0,
        'context_length': 8192,
    },
    'gemma3:27b': {
        'backend': 'ollama',
        'ollama_name': 'gemma3:27b',
        'family': 'gemma',
        'size_b': 27.0,
        'context_length': 8192,
    },

    # ── Cloud foundational models ────────────────────────────────────────────
    # Set API keys in .env to enable. These are only used when explicitly
    # passed via --models; they won't run in Ollama-only experiments.
    'deepseek-chat': {
        'backend': 'deepseek',
        'api_model_name': 'deepseek-chat',       # DeepSeek-V3.2
        'family': 'deepseek',
        'size_b': 671.0,                          # 671B total, 37B active (MoE)
        'context_length': 128000,
    },
    'minimax-m2.5': {
        'backend': 'minimax',
        'api_model_name': 'MiniMax-M2.5',
        'family': 'minimax',
        'size_b': 0.0,                            # undisclosed
        'context_length': 204800,
    },
    'gpt-5-nano': {
        'backend': 'openai',
        'api_model_name': 'gpt-5-nano',
        'family': 'gpt',
        'size_b': 0.0,                            # undisclosed
        'context_length': 400000,
        'fixed_temperature': True,                 # API rejects custom temperature values
    },
}

# Maximum context window to allocate in Ollama.
# Set high enough for the full model context; Ollama handles memory internally.
MAX_NUM_CTX = 32768

# Default max tokens for generation.
# Models like deepseek-r1 and qwen3 output <think>...</think> reasoning blocks
# before the answer, which can use hundreds of tokens. Set generous to avoid
# truncating model output — models stop at EOS naturally.
DEFAULT_MAX_TOKENS = 4096

# Experimental parameters
NUM_RUNS = 5
TEMPERATURES = [0.0, 0.5]  # 0.0 = deterministic baseline, 0.5 = stochastic
DEFAULT_STABILITY_THRESHOLD = 0.8  # 4/5 agreement for categorical

# Construct-aware stability thresholds
STABILITY_THRESHOLDS = {
    # Categorical constructs: modal agreement ratio
    'temporal_orientation': {'type': 'categorical', 'threshold': 0.8},
    'agency_control': {'type': 'categorical', 'threshold': 0.8},
    'social_proof': {'type': 'categorical', 'threshold': 0.8},
    'medical_authority': {'type': 'categorical', 'threshold': 0.8},
    # Continuous constructs: max range across runs
    'certainty_hedging': {'type': 'continuous', 'max_range': 0.2, 'max_stdev': 0.10},
    'symptom_concreteness': {'type': 'continuous', 'max_range': 0.2, 'max_stdev': 0.10},
}

CONSTRUCTS = list(STABILITY_THRESHOLDS.keys())

# Canonical label vocabularies per construct
LABEL_VOCABULARIES = {
    'certainty_hedging': {
        'type': 'continuous',
        'range': (0.0, 1.0),
        'bins': {'low': (0.0, 0.29), 'moderate': (0.3, 0.69), 'high': (0.7, 1.0)},
    },
    'temporal_orientation': {
        'type': 'categorical',
        'allowed': ['past', 'present', 'future', 'mixed'],
    },
    'symptom_concreteness': {
        'type': 'continuous',
        'range': (0.0, 1.0),
        'bins': {'abstract': (0.0, 0.29), 'moderate': (0.3, 0.69), 'concrete': (0.7, 1.0)},
    },
    'agency_control': {
        'type': 'categorical',
        'allowed': ['active', 'passive', 'helpless', 'mixed'],
    },
    'social_proof': {
        'type': 'categorical',
        'allowed': ['present', 'absent'],
    },
    'medical_authority': {
        'type': 'categorical',
        'allowed': ['professional', 'self_research', 'mixed', 'none_observed'],
    },
}

# Chunking parameters
CHUNKING_CONFIGS = {
    'multi_sentence': {
        'min_chars': 150,
        'target_chars': 300,
        'max_chars': 500,
        'context_carry_words': 15,
    },
    'whole_transcript': {
        'max_chars': 2000,
    },
}
DEFAULT_CHUNKING_METHOD = 'multi_sentence'

# Batch sizes
CHUNKING_BATCH_SIZE = 500
ANNOTATION_BATCH_SIZE = 100  # Larger batches reduce DB overhead and progress-check frequency

# Task queue
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

# Cohort splits (methods paper framing)
COHORT_SPLITS = {
    'development': 0.20,   # Tune prompts and parameters
    'reliability': 0.60,   # Compute stability metrics (main analysis)
    'holdout': 0.20,        # Confirm findings generalize
}
