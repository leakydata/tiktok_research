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


def _strip_reasoning(text: str) -> str:
    """Remove reasoning/thinking blocks from model output.

    Handles all known reasoning tag formats:
      - <think>...</think>           (qwen3, deepseek-r1)
      - <|think|>...<|/think|>       (phi4-reasoning)
      - <reasoning>...</reasoning>   (generic)
      - <reflection>...</reflection> (generic)
    Returns only the final answer portion.
    """
    # All known reasoning tag patterns (closed)
    tag_patterns = [
        r'<think>.*?</think>',
        r'<\|think\|>.*?<\|/think\|>',
        r'<reasoning>.*?</reasoning>',
        r'<reflection>.*?</reflection>',
    ]
    stripped = text
    for pattern in tag_patterns:
        stripped = re.sub(pattern, '', stripped, flags=re.DOTALL | re.IGNORECASE)

    # Handle unclosed thinking blocks (model hit token limit mid-thought)
    unclosed_patterns = [r'<think>', r'<\|think\|>', r'<reasoning>', r'<reflection>']
    for tag in unclosed_patterns:
        tag_lower = tag.replace('\\', '').lower()
        if tag_lower in stripped.lower():
            parts = re.split(tag, stripped, flags=re.IGNORECASE)
            before = parts[0].strip()
            # Everything after the last unclosed tag is likely still reasoning, discard it
            # Use what came before if it has content
            stripped = before if before else ''

    # Some models wrap the final answer in **bold** or [brackets]
    stripped = re.sub(r'\*\*(.*?)\*\*', r'\1', stripped)

    # If still multi-line after stripping, take the last non-empty line
    # (many models explain first, then put the label on the final line)
    stripped = stripped.strip()
    if '\n' in stripped:
        lines = [ln.strip() for ln in stripped.split('\n') if ln.strip()]
        if lines:
            # Check if last line looks like a label (short, no sentences)
            last = lines[-1]
            if len(last) < 80:
                stripped = last

    return stripped.strip()


def normalize_label(raw_response: str, construct_name: str) -> dict:
    """Parse and normalize a model response into a structured label.

    Returns dict with:
        label_kind: 'float' | 'category' | 'none' | 'unclear' | 'error'
        label_value_text: canonical text label (or None)
        label_value_float: float value (or None)
        label_bin: binned category for continuous (or None)
    """
    # Strip reasoning blocks first, then clean
    cleaned = _strip_reasoning(raw_response).lower()
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

    # Direct match (entire cleaned response is exactly an allowed label)
    if cleaned in allowed:
        return _make_category(cleaned)

    # Synonym exact match
    for canonical, syns in synonyms.items():
        if cleaned in syns:
            return _make_category(canonical)

    # First-token extraction: models often respond "label  Explanation..."
    # Check the first 1-3 words against allowed labels and synonyms
    words = cleaned.split()
    first_tokens = [
        words[0] if len(words) >= 1 else '',
        ' '.join(words[:2]) if len(words) >= 2 else '',
        ' '.join(words[:3]) if len(words) >= 3 else '',
    ]
    for token in first_tokens:
        if token in allowed:
            return _make_category(token)
        for canonical, syns in synonyms.items():
            if token in syns:
                return _make_category(canonical)

    # Substring match — prefer the label that appears EARLIEST in the text
    # (avoids matching labels mentioned in explanation text over the actual answer)
    best_match = None
    best_pos = len(cleaned) + 1
    for canonical in allowed:
        pos = cleaned.find(canonical)
        if pos != -1 and pos < best_pos:
            best_match = canonical
            best_pos = pos
    if best_match is not None:
        return _make_category(best_match)

    # Synonym substring match — same earliest-occurrence logic
    best_match = None
    best_pos = len(cleaned) + 1
    for canonical, syns in synonyms.items():
        for syn in syns:
            pos = cleaned.find(syn)
            if pos != -1 and pos < best_pos:
                best_match = canonical
                best_pos = pos
    if best_match is not None:
        return _make_category(best_match)

    return _make_unclear(f"Could not map '{raw_response.strip()[:80]}' to allowed labels")


def _make_category(label: str) -> dict:
    return {
        'label_kind': 'category',
        'label_value_text': label,
        'label_value_float': None,
        'label_bin': None,
    }


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
