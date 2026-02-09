"""
Label normalization and parsing for LLM annotation responses.
Separated from annotate.py so it can be tested without database dependencies.
"""

import re
from config import LABEL_VOCABULARIES

# Synonym maps for categorical constructs (handles common LLM output variations)
LABEL_SYNONYMS = {
    'temporal_orientation': {
        'past': ['past', 'past-focused', 'past focused', 'historical'],
        'present': ['present', 'present-focused', 'present focused', 'current', 'ongoing'],
        'future': ['future', 'future-focused', 'future focused', 'prospective'],
        'mixed': ['mixed', 'combination', 'multiple', 'both'],
    },
    'agency_control': {
        'active': ['active', 'agentic', 'in control', 'self-directed'],
        'passive': ['passive', 'receiving', 'receptive'],
        'helpless': ['helpless', 'powerless', 'hopeless', 'defeated', 'loss of control'],
        'mixed': ['mixed', 'combination', 'both', 'unclear agency'],
    },
    'social_proof': {
        'present': ['present', 'yes', 'true', 'found', 'detected'],
        'absent': ['absent', 'no', 'false', 'not found', 'not detected', 'not present'],
    },
    'medical_authority': {
        'professional': ['professional', 'doctor', 'medical', 'clinical', 'professional-verified'],
        'self_research': ['self_research', 'self research', 'self-research', 'personal research',
                          'online research', 'self-directed'],
        'mixed': ['mixed', 'both', 'combination'],
        'none_observed': ['none_observed', 'none observed', 'none', 'no authority', 'no references'],
    },
}


def normalize_label(raw_response: str, construct_name: str) -> dict:
    """Parse and normalize a model response into a structured label.

    Returns dict with:
        label_kind: 'float' | 'category' | 'none' | 'unclear' | 'error'
        label_value_text: canonical text label (or None)
        label_value_float: float value (or None)
        label_bin: binned category for continuous (or None)
    """
    cleaned = raw_response.strip().lower()
    # Strip surrounding quotes and punctuation
    cleaned = cleaned.strip('"\'`.,;:!? ')

    # Check for "none" (no health content) — but NOT for medical_authority
    # where "none" means "none_observed" (handled by synonym map)
    if construct_name != 'medical_authority' and cleaned in (
        'none', 'n/a', 'no health content', 'not applicable'
    ):
        return {
            'label_kind': 'none',
            'label_value_text': 'none',
            'label_value_float': None,
            'label_bin': None,
        }

    # Check for "unclear"
    if cleaned in ('unclear', 'ambiguous', 'uncertain', 'cannot determine', "can't determine"):
        return {
            'label_kind': 'unclear',
            'label_value_text': 'unclear',
            'label_value_float': None,
            'label_bin': None,
        }

    # Empty / whitespace
    if not cleaned:
        return {
            'label_kind': 'unclear',
            'label_value_text': 'unclear',
            'label_value_float': None,
            'label_bin': None,
        }

    vocab = LABEL_VOCABULARIES.get(construct_name)
    if not vocab:
        return _make_error(f"Unknown construct: {construct_name}")

    # ── Continuous constructs ──
    if vocab['type'] == 'continuous':
        # Extract float value (including negative sign)
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if len(numbers) == 0:
            return _make_unclear("No number found in response")
        if len(numbers) > 1:
            # Multiple numbers = ambiguous, but take the first if they're close
            vals = [float(n) for n in numbers]
            if max(vals) - min(vals) <= 0.1:
                value = vals[0]
            else:
                return _make_unclear(f"Multiple ambiguous numbers: {numbers}")
        else:
            value = float(numbers[0])

        lo, hi = vocab['range']
        if not (lo <= value <= hi):
            return _make_unclear(f"Value {value} outside range [{lo}, {hi}]")

        # Determine bin
        label_bin = None
        for bin_name, (bin_lo, bin_hi) in vocab['bins'].items():
            if bin_lo <= value <= bin_hi:
                label_bin = bin_name
                break
        # Edge case: value exactly at boundary
        if label_bin is None and value == hi:
            label_bin = list(vocab['bins'].keys())[-1]

        return {
            'label_kind': 'float',
            'label_value_text': None,
            'label_value_float': round(value, 2),
            'label_bin': label_bin,
        }

    # ── Categorical constructs ──
    allowed = vocab['allowed']
    synonyms = LABEL_SYNONYMS.get(construct_name, {})

    # Direct match
    if cleaned in allowed:
        return {
            'label_kind': 'category',
            'label_value_text': cleaned,
            'label_value_float': None,
            'label_bin': None,
        }

    # Synonym match
    for canonical, syns in synonyms.items():
        if cleaned in syns:
            return {
                'label_kind': 'category',
                'label_value_text': canonical,
                'label_value_float': None,
                'label_bin': None,
            }

    # Substring match (e.g., response is "The temporal orientation is past.")
    for canonical in allowed:
        if canonical in cleaned:
            return {
                'label_kind': 'category',
                'label_value_text': canonical,
                'label_value_float': None,
                'label_bin': None,
            }

    # Synonym substring match
    for canonical, syns in synonyms.items():
        for syn in syns:
            if syn in cleaned:
                return {
                    'label_kind': 'category',
                    'label_value_text': canonical,
                    'label_value_float': None,
                    'label_bin': None,
                }

    return _make_unclear(f"Could not map '{raw_response.strip()[:80]}' to allowed labels")


def _make_unclear(reason: str) -> dict:
    return {
        'label_kind': 'unclear',
        'label_value_text': 'unclear',
        'label_value_float': None,
        'label_bin': None,
    }


def _make_error(reason: str) -> dict:
    return {
        'label_kind': 'error',
        'label_value_text': None,
        'label_value_float': None,
        'label_bin': None,
    }
