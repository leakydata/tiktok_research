"""
Unit tests for label normalization and parsing.
Tests that LLM output variations are correctly mapped to canonical labels.

Run with: python -m pytest tests/test_parsing.py -v
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import label_parsing
sys.path.insert(0, str(Path(__file__).parent.parent))

from label_parsing import normalize_label


class TestContinuousParsing:
    """Tests for certainty_hedging and symptom_concreteness (continuous 0-1)."""

    def test_simple_float(self):
        result = normalize_label("0.7", "certainty_hedging")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 0.7
        assert result['label_bin'] == 'high'

    def test_float_with_text(self):
        result = normalize_label("The certainty level is 0.5", "certainty_hedging")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 0.5
        assert result['label_bin'] == 'moderate'

    def test_low_value(self):
        result = normalize_label("0.2", "symptom_concreteness")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 0.2
        assert result['label_bin'] == 'abstract'

    def test_boundary_zero(self):
        result = normalize_label("0.0", "certainty_hedging")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 0.0
        assert result['label_bin'] == 'low'

    def test_boundary_one(self):
        result = normalize_label("1.0", "certainty_hedging")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 1.0
        assert result['label_bin'] == 'high'

    def test_out_of_range(self):
        result = normalize_label("2.5", "certainty_hedging")
        assert result['label_kind'] == 'unclear'

    def test_negative_value(self):
        result = normalize_label("-0.3", "certainty_hedging")
        assert result['label_kind'] == 'unclear'

    def test_multiple_ambiguous_numbers(self):
        result = normalize_label("0.3-0.8", "certainty_hedging")
        assert result['label_kind'] == 'unclear'

    def test_multiple_close_numbers(self):
        # Numbers within 0.1 of each other — takes first
        result = normalize_label("0.70 to 0.75", "certainty_hedging")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 0.7

    def test_no_number(self):
        result = normalize_label("moderate certainty", "certainty_hedging")
        assert result['label_kind'] == 'unclear'

    def test_integer_in_range(self):
        result = normalize_label("1", "certainty_hedging")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 1.0


class TestCategoricalParsing:
    """Tests for temporal_orientation, agency_control, social_proof, medical_authority."""

    def test_exact_match(self):
        result = normalize_label("past", "temporal_orientation")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'past'

    def test_case_insensitive(self):
        result = normalize_label("PRESENT", "temporal_orientation")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'present'

    def test_with_punctuation(self):
        result = normalize_label("present.", "temporal_orientation")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'present'

    def test_with_quotes(self):
        result = normalize_label('"future"', "temporal_orientation")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'future'

    def test_synonym_match(self):
        result = normalize_label("past-focused", "temporal_orientation")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'past'

    def test_substring_in_sentence(self):
        result = normalize_label("The temporal orientation is past.", "temporal_orientation")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'past'

    def test_agency_active(self):
        result = normalize_label("active", "agency_control")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'active'

    def test_agency_helpless(self):
        result = normalize_label("helpless", "agency_control")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'helpless'

    def test_social_proof_present(self):
        result = normalize_label("present", "social_proof")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'present'

    def test_social_proof_absent(self):
        result = normalize_label("absent", "social_proof")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'absent'

    def test_social_proof_yes_synonym(self):
        result = normalize_label("yes", "social_proof")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'present'

    def test_social_proof_no_synonym(self):
        result = normalize_label("no", "social_proof")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'absent'

    def test_medical_authority_professional(self):
        result = normalize_label("professional", "medical_authority")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'professional'

    def test_medical_authority_self_research(self):
        result = normalize_label("self_research", "medical_authority")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'self_research'

    def test_medical_authority_self_research_hyphen(self):
        result = normalize_label("self-research", "medical_authority")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'self_research'

    def test_medical_authority_none_observed(self):
        result = normalize_label("none_observed", "medical_authority")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'none_observed'

    def test_medical_authority_bare_none(self):
        """'none' for medical_authority means none_observed, not 'no health content'."""
        # This is tricky: "none" in medical_authority context means none_observed
        # but globally "none" means no health content.
        # The synonym map handles this: 'none' maps to 'none_observed'
        result = normalize_label("none", "medical_authority")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'none_observed'

    def test_invalid_label(self):
        result = normalize_label("something random and nonsensical", "temporal_orientation")
        assert result['label_kind'] == 'unclear'


class TestSpecialResponses:
    """Tests for none and unclear handling."""

    def test_none_response(self):
        result = normalize_label("none", "temporal_orientation")
        assert result['label_kind'] == 'none'

    def test_none_with_caps(self):
        result = normalize_label("NONE", "certainty_hedging")
        assert result['label_kind'] == 'none'

    def test_na_response(self):
        result = normalize_label("N/A", "social_proof")
        assert result['label_kind'] == 'none'

    def test_unclear_response(self):
        result = normalize_label("unclear", "agency_control")
        assert result['label_kind'] == 'unclear'

    def test_unclear_with_caps(self):
        result = normalize_label("Unclear", "certainty_hedging")
        assert result['label_kind'] == 'unclear'

    def test_cannot_determine(self):
        result = normalize_label("cannot determine", "temporal_orientation")
        assert result['label_kind'] == 'unclear'

    def test_empty_string(self):
        result = normalize_label("", "certainty_hedging")
        # Empty after strip -> should be unclear or none
        assert result['label_kind'] in ('unclear', 'none')

    def test_whitespace_only(self):
        result = normalize_label("   ", "temporal_orientation")
        assert result['label_kind'] in ('unclear', 'none')


class TestEdgeCases:
    """Edge cases from real LLM outputs."""

    def test_response_with_newline(self):
        result = normalize_label("present\n", "social_proof")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'present'

    def test_response_with_explanation(self):
        result = normalize_label("active. The speaker demonstrates agency by...", "agency_control")
        assert result['label_kind'] == 'category'
        assert result['label_value_text'] == 'active'

    def test_float_with_trailing_period(self):
        result = normalize_label("0.6.", "certainty_hedging")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 0.6

    def test_float_in_markdown(self):
        result = normalize_label("**0.8**", "symptom_concreteness")
        assert result['label_kind'] == 'float'
        assert result['label_value_float'] == 0.8


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-v'])
