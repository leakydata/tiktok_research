# Implementation Roadmap: Multi-Run LLM Annotation Pipeline

## Paper Thesis

**Core argument:** Multi-run stability filtering across diverse open-weight LLMs can
replace human annotators for subjective health language constructs. When multiple
models at multiple temperatures independently converge on the same label, that
convergence IS the validity evidence — the same way inter-rater reliability works
with human coders, except cheaper, faster, and fully reproducible.

**This is NOT a "match the human" paper.** This is a "humans aren't needed" paper.
The contribution is showing that LLM-as-rater reliability meets or exceeds published
human inter-rater reliability benchmarks for comparable constructs.

---

## Current State

### Implemented (working pipeline)
- [x] Cohort selection with dev/reliability/holdout splits (`cohort_selection.py`)
- [x] Multi-sentence chunking with context carry (`chunking.py`)
- [x] 6 construct-specific prompts with none/unclear distinction (`prompts.py`)
- [x] Multi-run annotation across 7 models x 2 temperatures x 5 runs (`annotate.py`)
- [x] Construct-aware stability: categorical agreement + continuous range/stdev (`stability.py`)
- [x] Krippendorff's alpha (nominal + interval) implementation (`stability.py`)
- [x] Final label derivation via modal/median (`final_labels.py`)
- [x] Validation against existing `narrative_elements` (`validation.py`)
- [x] CSV reporting with 8 table exports (`reporting.py`)
- [x] Pipeline orchestrator with skip/resume (`run_pipeline.py`)
- [x] Label parsing with 41 unit tests (`label_parsing.py`, `tests/test_parsing.py`)
- [x] Resume support for interrupted runs (`annotate.py --resume`)

### Database Schema
- `experiment_runs` - experiment tracking with config snapshots
- `study_cohort` - creator selection with splits
- `annotation_chunks` - chunked transcripts
- `annotation_tasks` - task queue with retry logic
- `llm_annotation_runs` - individual annotation results
- `annotation_stability_metrics` - per-chunk stability
- `group_reliability_metrics` - group-level Krippendorff alpha
- `final_annotations` - canonical labels
- `validation_comparisons` - comparison with narrative_elements
- `paraphrase_tests` - structure exists, not yet implemented

---

## Implementation Tasks

### Priority 1: Cross-Model Convergent Validity (CRITICAL)

**Why:** This is the centerpiece of the "humans aren't needed" argument. If 7 different
architectures (GLM, Phi, GPT-OSS, MedGemma, Gemma, Qwen, DeepSeek) independently
agree on labels, that convergence is powerful evidence — analogous to getting agreement
from 7 human raters with different training backgrounds.

**File:** Create `cross_model_validity.py`

**Implementation:**
1. For each chunk and construct, collect the stable final labels from ALL models
2. Compute cross-model agreement rate: what fraction of chunks get the same label
   from all models? from a majority?
3. Compute Krippendorff's alpha treating each model as a separate "rater" and each
   chunk as an "item" — this is the KEY metric for the paper
4. Build a cross-model agreement matrix (model x model pairwise agreement rates)
5. Report separately for T=0.0 and T=0.5

**Output tables/CSVs:**
- `cross_model_agreement_matrix.csv` — pairwise model agreement rates per construct
- `cross_model_krippendorff.csv` — alpha treating models as raters (THE headline number)
- `cross_model_unanimous.csv` — fraction of chunks where ALL models agree
- `cross_model_majority.csv` — fraction where >=5/7 models agree

**Key SQL pattern:**
```sql
-- Get final labels per chunk per model for cross-model comparison
SELECT chunk_id, construct_name, model_name, final_label_text, final_label_bin
FROM final_annotations
WHERE experiment_id = %s AND is_stable = TRUE
ORDER BY chunk_id, construct_name, model_name
```

**Context for published benchmarks:** Human inter-rater reliability for subjective
health constructs typically ranges from kappa 0.40-0.70. If cross-model alpha exceeds
0.60, that's a strong result. If it exceeds 0.70, that's the headline finding.

---

### Priority 2: Bootstrap Confidence Intervals (CRITICAL)

**Why:** Every metric currently reported is a point estimate. Reviewers will ask whether
differences between models/temperatures are statistically meaningful. CIs also
strengthen the comparison to published human reliability benchmarks.

**File:** Add to `stability.py` (new functions) and update `reporting.py`

**Implementation:**
1. Add `bootstrap_krippendorff_alpha(reliability_data, n_bootstrap=1000, ci=0.95)`
   - Resample items (chunks) WITH replacement
   - Compute alpha on each resample
   - Return (point_estimate, ci_lower, ci_upper)
2. Add `bootstrap_stability_rate(stability_booleans, n_bootstrap=1000, ci=0.95)`
   - Resample chunks with replacement
   - Compute fraction stable on each
   - Return (point_estimate, ci_lower, ci_upper)
3. Update `compute_group_reliability()` to store CI bounds
4. Update all reporting tables to include CI columns

**Schema addition** (add columns to `group_reliability_metrics`):
```sql
ALTER TABLE group_reliability_metrics
    ADD COLUMN IF NOT EXISTS alpha_ci_lower FLOAT,
    ADD COLUMN IF NOT EXISTS alpha_ci_upper FLOAT,
    ADD COLUMN IF NOT EXISTS stability_ci_lower FLOAT,
    ADD COLUMN IF NOT EXISTS stability_ci_upper FLOAT;
```

**Reporting format:** "alpha = 0.72 [0.65, 0.78]" in all tables.

---

### Priority 3: Single-Run vs. Multi-Run Comparison (CRITICAL)

**Why:** Must demonstrate the VALUE of doing 5 runs. If single-run labels are just as
consistent across models, the multi-run approach is unnecessary overhead.

**File:** Create `run_convergence.py`

**Implementation:**
1. For each chunk/construct/model/temperature, subsample runs:
   - R=1 (just run 1)
   - R=2 (runs 1-2)
   - R=3 (runs 1-3)
   - R=4 (runs 1-4)
   - R=5 (all runs)
2. At each R level, compute:
   - Stability rate (fraction of chunks that pass threshold)
   - Whether the "final label" matches the R=5 final label (convergence)
   - Cross-model agreement rate
3. Plot stability_rate vs R — expect diminishing returns, find the "elbow"
4. Report: "At R=1, X% of labels match R=5 consensus. At R=3, Y%. This suggests
   R=5 provides [meaningful/diminishing] improvement."

**Output:**
- `run_convergence.csv` — stability metrics at each R level
- Data for a convergence plot (Figure in paper)

**This directly answers:** "Why not just run once?" If single-run gives 60% cross-model
agreement but multi-run stable gives 85%, that's the justification.

---

### Priority 4: Statistical Significance Testing (STRONG)

**Why:** The temperature analysis (T=0.0 vs T=0.5) and model size comparisons need
p-values and effect sizes, not just averages.

**File:** Create `significance_tests.py`

**Implementation:**
1. **Temperature effect (paired test):**
   - For each chunk/construct/model, you have stability at T=0.0 and T=0.5
   - Paired Wilcoxon signed-rank test (non-normal data) or paired t-test
   - Cohen's d for effect size
   - Report per construct and overall
2. **Model size effect:**
   - Spearman correlation between model_size_b and stability_rate
   - Compare small (<=14B) vs medium (20-27B) vs large (>=32B) groups
   - Kruskal-Wallis test across groups
3. **Construct difficulty comparison:**
   - Which constructs are inherently harder for LLMs? (lower stability)
   - Friedman test across constructs (repeated measures)

**Output:**
- `significance_temperature.csv` — paired test results per construct
- `significance_model_size.csv` — correlation and group comparison
- `significance_constructs.csv` — construct difficulty ranking with tests

---

### Priority 5: Confusion Matrices & Error Analysis (STRONG)

**Why:** Knowing THAT models disagree isn't enough — knowing HOW they disagree reveals
systematic biases. "Models confuse passive with helpless" is actionable.

**File:** Create `error_analysis.py`

**Implementation:**
1. **Within-model confusion (instability analysis):**
   - For UNSTABLE chunks, build a co-occurrence matrix: when the model gives
     different labels across runs, which label pairs appear together?
   - E.g., for agency_control: how often does {active, mixed} co-occur vs
     {passive, helpless}?
2. **Cross-model confusion:**
   - When models disagree on a chunk, build a confusion matrix of model A label
     vs model B label
   - Identify systematic patterns (e.g., small models over-predict "mixed")
3. **Error by chunk characteristics:**
   - Do shorter chunks have more errors/instability?
   - Do chunks with context_carry have higher stability?
   - Do certain content_types cause systematic issues?

**Output:**
- `confusion_within_model.csv` — label co-occurrence for unstable chunks
- `confusion_cross_model.csv` — model x model disagreement patterns
- `error_by_chunk_length.csv` — stability vs char_length bins
- `error_by_content_type.csv` — stability by content_type

---

### Priority 6: Prompt Paraphrase Robustness (STRONG)

**Why:** If results change dramatically with minor prompt rewording, the method is
fragile. Showing stability across prompt variants is a strong methodological claim
the reference paper does NOT have.

**File:** Create `prompt_variants.py`, update `prompts.py`

**Implementation:**
1. Create 2 paraphrased variants per construct prompt (total: 3 versions each)
   - Variant A: Original (current prompts)
   - Variant B: Reworded instructions, same structure
   - Variant C: Different structure (e.g., examples-first instead of scale-first)
2. Keep the same scale/categories — only change the framing language
3. Run on development split only (saves compute)
4. Compare labels across prompt versions using Krippendorff alpha
5. Report: "Prompt paraphrase alpha = X, indicating [high/moderate] robustness"

**Use the existing `paraphrase_tests` table** for storing variants.

**Update `prompts.py`:**
```python
class HealthLanguagePrompts:
    VERSION = 'v1'  # Track prompt version

    @staticmethod
    def certainty_hedging(chunk_text, context_carry=None, variant='A'):
        # variant A = original, B = reworded, C = restructured
        ...
```

**Run as a separate experiment** with prompt_format='paraphrase_B' etc.,
then compare across prompt_format values.

---

### Priority 7: Stratified Analysis (DIFFERENTIATOR)

**Why:** Your TikTok dataset has rich metadata that Reddit datasets lack. Showing how
stability varies by content type, creator longitudinal position, and chunk properties
is a unique contribution.

**File:** Add methods to `reporting.py` or create `stratified_analysis.py`

**Implementation:**
1. **By content type:**
   ```sql
   SELECT ac.content_type, asm.construct_name,
          AVG(CASE WHEN asm.is_stable THEN 1.0 ELSE 0.0 END) AS stability_rate,
          COUNT(*) AS n
   FROM annotation_stability_metrics asm
   JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
   WHERE asm.experiment_id = %s
   GROUP BY ac.content_type, asm.construct_name
   ```
2. **By chunk length (binned):**
   - Bin char_length into quartiles
   - Report stability rate per bin per construct
   - Test for trend (Jonckheere-Terpstra or Spearman)
3. **By longitudinal position:**
   - Does stability differ for early vs late videos from the same creator?
   - Bin days_since_first_video into early/middle/late
4. **By claimed diagnosis:**
   - Do creators claiming specific diagnoses show different construct profiles?
   - This connects your linguistic constructs to clinical categories

**Output:**
- `stratified_by_content_type.csv`
- `stratified_by_chunk_length.csv`
- `stratified_by_temporal_position.csv`
- `stratified_by_diagnosis.csv`

---

### Priority 8: Discriminant Validity (STRONG)

**Why:** Constructs that SHOULD be independent should produce independent labels.
Constructs that should correlate should correlate. This is standard psychometric
validation and strengthens the "measurement quality" argument.

**File:** Create `discriminant_validity.py`

**Implementation:**
1. Build a construct x construct correlation matrix:
   - For categorical x categorical: Cramér's V
   - For continuous x continuous: Pearson/Spearman
   - For categorical x continuous: point-biserial or eta-squared
2. Expected patterns (state these as hypotheses):
   - `certainty_hedging` and `medical_authority=professional` should correlate
     positively (doctors cite things with certainty)
   - `agency_control=helpless` and `certainty_hedging` low (uncertain) might
     correlate (helplessness often expressed with hedging)
   - `social_proof` should be relatively independent of `temporal_orientation`
3. Report the correlation matrix as a heatmap-ready CSV

**Output:**
- `construct_correlation_matrix.csv`
- `discriminant_validity_tests.csv` (with expected vs observed direction)

---

### Priority 9: Qualitative Sanity Check (NECESSARY)

**Why:** Even without formal gold-standard annotation, reviewers expect to see that
labels make face-valid sense. This is NOT human annotation — it's a quick audit.

**File:** Create `qualitative_sample.py`

**Implementation:**
1. Sample 50 chunks stratified by:
   - 10 chunks where ALL models unanimously agree (high confidence)
   - 10 chunks where models maximally disagree (interesting cases)
   - 10 chunks labeled "none" (verify no health content)
   - 10 chunks labeled "unclear" (verify genuine ambiguity)
   - 10 random chunks from each construct
2. Export as a readable table: chunk_text | construct | all model labels | final label
3. In the paper, include a Table with 3-5 illustrative examples per construct
   showing chunk text and the assigned label — readers can judge face validity

**Output:**
- `qualitative_samples.csv` — for researcher review
- Use selected examples in paper's appendix/supplementary materials

---

### Priority 10: Proper Fleiss Kappa (POLISH)

**Why:** Currently `stability.py:431` sets `fleiss_kappa = alpha` which is an
approximation. Either implement properly or remove the column.

**File:** Update `stability.py`

**Implementation** (recommend implementing properly):
```python
def fleiss_kappa(reliability_data: list[list[Optional[str]]], categories: list[str]) -> float:
    """Proper Fleiss' kappa for multiple raters, multiple categories."""
    n_items = len(reliability_data)
    n_categories = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build rating matrix: items x categories (count of raters per category)
    matrix = np.zeros((n_items, n_categories))
    for i, item_labels in enumerate(reliability_data):
        for label in item_labels:
            if label is not None and label in cat_to_idx:
                matrix[i, cat_to_idx[label]] += 1

    n_raters = matrix.sum(axis=1)
    # ... standard Fleiss computation
```

Alternatively, remove `fleiss_kappa` column from schema and reporting if not needed.

---

### Priority 11: Unit Tests for Stability Computations (POLISH)

**Why:** 41 tests for label parsing but 0 for stability — the core statistical claim.

**File:** Create `tests/test_stability.py`

**Tests to add:**
1. `test_categorical_stability_unanimous` — 5 identical labels -> agreement=1.0, stable
2. `test_categorical_stability_split` — 3/5 agree -> agreement=0.6, unstable at 0.8
3. `test_continuous_stability_tight` — values within 0.05 range -> stable
4. `test_continuous_stability_spread` — values spanning 0.4 -> unstable
5. `test_krippendorff_alpha_perfect` — all raters agree -> alpha=1.0
6. `test_krippendorff_alpha_random` — random labels -> alpha near 0.0
7. `test_krippendorff_alpha_known_value` — use a published example with known alpha
8. `test_continuous_binning` — verify bin boundaries (0.29, 0.69 exclusive upper)

---

### Priority 12: Cost/Efficiency Analysis (POLISH)

**Why:** Part of the "humans aren't needed" argument is cost. Show total inference
time and compare to estimated human annotation cost.

**File:** Add to `reporting.py`

**Implementation:**
```sql
SELECT
    model_name,
    COUNT(*) AS total_tasks,
    SUM(processing_time_ms) / 1000.0 / 3600.0 AS total_hours,
    AVG(processing_time_ms) AS avg_ms_per_task,
    SUM(tokens_generated) AS total_tokens,
    AVG(tokens_generated) AS avg_tokens_per_task
FROM llm_annotation_runs
WHERE experiment_id = %s
GROUP BY model_name
ORDER BY total_hours
```

Compare to published human annotation rates (~50-100 items/hour for subjective
coding tasks). If your pipeline annotates 10,000 chunks x 6 constructs in X hours,
and a human team would take Y person-hours, that's a compelling efficiency argument.

**Output:** `cost_efficiency.csv`

---

## Paper Framing Notes

### Positioning vs. Hassan et al. (Dec 2024)

| Dimension | Hassan et al. | Your Paper |
|-----------|--------------|------------|
| Data source | Reddit text posts | TikTok video transcripts (spoken language) |
| Annotation target | Disorder labels (binary) | Linguistic constructs (categorical + continuous) |
| LLM approach | Single-pass cloud APIs | Multi-run local open-weight models |
| Validation | Match human labels | Cross-model convergence + stability |
| Models | 5 models (cloud) | 7 models (local, 4.7B-32B) |
| Key metric | F1, balanced accuracy | Krippendorff alpha, stability rate |
| Contribution | Synthetic multi-label dataset | Reliability-first annotation methodology |

### Key citations to gather
- Krippendorff (2004) — alpha as THE reliability coefficient
- Human inter-rater reliability benchmarks for health language coding (look for
  studies reporting kappa 0.40-0.70 on subjective constructs)
- Measurement theory: reliability as prerequisite for validity (Nunnally & Bernstein)
- LLM-as-rater literature (Gilardi et al. 2023, Ziems et al. 2024)
- Multi-rater agreement vs single-rater accuracy paradigm

### Abstract framing (draft direction)
> We propose multi-run stability filtering as a method for automating annotation
> of subjective health language constructs from social media transcripts. Rather
> than validating LLM annotations against human ground truth, we demonstrate that
> diverse open-weight language models (4.7B-32B parameters) achieve high internal
> consistency (within-model stability) and cross-model convergent validity
> (between-model agreement) that meets or exceeds published human inter-rater
> reliability benchmarks for comparable constructs. Across [N] TikTok transcript
> chunks and 6 constructs, [best model] achieved Krippendorff's alpha = [X]
> [CI_low, CI_high] with [Y]% of chunks producing stable labels. Cross-model
> agreement alpha = [Z], comparable to typical human coder agreement (0.40-0.70).
> These results suggest that multi-run LLM annotation can replace costly human
> coding for health language research while providing full reproducibility.

---

## Execution Order

Run these in order. Each builds on the previous.

1. **Bootstrap CIs** (Priority 2) — foundational, needed by everything else
2. **Cross-model validity** (Priority 1) — the headline result
3. **Single-run vs multi-run** (Priority 3) — justifies the methodology
4. **Significance tests** (Priority 4) — strengthens temperature/model claims
5. **Confusion matrices** (Priority 5) — explains disagreement patterns
6. **Discriminant validity** (Priority 8) — construct-level validation
7. **Prompt paraphrase** (Priority 6) — robustness claim
8. **Stratified analysis** (Priority 7) — exploits unique dataset properties
9. **Qualitative sample** (Priority 9) — face validity for appendix
10. **Fleiss kappa / tests / cost** (Priorities 10-12) — polish

---

## Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `cross_model_validity.py` | Cross-model agreement analysis | 1 |
| `run_convergence.py` | R=1 vs R=3 vs R=5 comparison | 3 |
| `significance_tests.py` | Paired tests and effect sizes | 4 |
| `error_analysis.py` | Confusion matrices, error patterns | 5 |
| `prompt_variants.py` | Paraphrase robustness testing | 6 |
| `stratified_analysis.py` | Breakdowns by content/length/etc | 7 |
| `discriminant_validity.py` | Construct correlation matrix | 8 |
| `qualitative_sample.py` | Face validity sample export | 9 |
| `tests/test_stability.py` | Unit tests for stability math | 11 |

## Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `stability.py` | Add bootstrap CI functions, proper Fleiss kappa | 2, 10 |
| `reporting.py` | Add CI columns to all tables, add cost table | 2, 12 |
| `schema.sql` | Add CI columns to group_reliability_metrics | 2 |
| `prompts.py` | Add variant='B'/'C' support per construct | 6 |
| `run_pipeline.py` | Add new steps to orchestrator | All |
