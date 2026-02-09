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

# Models to test - maps short name to Ollama model identifier
MODELS_TO_TEST = {
    'glm4-flash': {
        'ollama_name': 'glm-4-flash',
        'family': 'glm',
        'size_b': 4.7,
        'context_length': 8192,
    },
    'medgemma-27b': {
        'ollama_name': 'alibayram/medgemma:27b',
        'family': 'gemma',
        'size_b': 27.0,
        'context_length': 8192,
    },
    'gpt-oss-20b': {
        'ollama_name': 'gpt-oss:20b',
        'family': 'gpt',
        'size_b': 20.0,
        'context_length': 8192,
    },
    'qwen3-32b': {
        'ollama_name': 'qwen3:32b',
        'family': 'qwen',
        'size_b': 32.0,
        'context_length': 32768,
    },
}

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
ANNOTATION_BATCH_SIZE = 20

# Task queue
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5

# Cohort splits (methods paper framing)
COHORT_SPLITS = {
    'development': 0.20,   # Tune prompts and parameters
    'reliability': 0.60,   # Compute stability metrics (main analysis)
    'holdout': 0.20,        # Confirm findings generalize
}
