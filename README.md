# Multi-Run LLM Annotation Pipeline for Health Language Analysis

A reproducible pipeline demonstrating that **multi-run stability filtering across diverse open-weight LLMs** produces consistent, reliable annotations — eliminating the need for costly human coding in health communication research.

Extends [Hassan et al. (2024)](https://doi.org/10.48550/arXiv.2412.xxxxx) *"Automated Multi-Label Annotation for Mental Health Illnesses Using Large Language Models"* from Reddit to TikTok video transcripts, using local open-weight models via Ollama.

## Thesis

Traditional content analysis requires expensive, time-consuming human annotation with inter-rater reliability checks. This pipeline proves an alternative: by running **7 diverse LLM architectures** (4B–32B parameters) **5 times each** at **2 temperature settings**, then filtering for stability, researchers can automate annotation entirely.

**Cross-model convergence** — where architecturally independent LLMs agree on the same label — serves as the validity signal analogous to inter-rater reliability. If a 4.7B ChatGLM model, a 14B Phi model, and a 32B Qwen model all independently converge on the same annotation, this provides stronger evidence than any single human coder.

## Prerequisites

### Software
- **Python** >= 3.10
- **PostgreSQL** running on `localhost:5433` (configurable via `.env`)
- **Ollama** running on `localhost:11434` (configurable via `.env`)

### Existing Database
The pipeline extends an existing `tiktok_disorders` database that must already contain:
- `videos` — video metadata
- `transcripts` — full transcript text
- `claimed_diagnoses` — creator-level diagnosis claims
- `narrative_elements` — existing single-pass extraction (used for validation only)

### Models
All 7 models run locally via [Ollama](https://ollama.com). Pull them before running:

```bash
ollama pull glm-4.7-flash          # 4.7B  - ChatGLM family
ollama pull phi4:latest             # 14B   - Microsoft Phi
ollama pull gpt-oss:20b            # 20B   - GPT open-source
ollama pull alibayram/medgemma:27b # 27B   - Medical Gemma (domain-specific)
ollama pull gemma3:27b             # 27B   - Google Gemma
ollama pull qwen3:32b              # 32B   - Alibaba Qwen
ollama pull deepseek-r1:32b        # 32B   - DeepSeek reasoning
```

### Environment Setup

```bash
# Option 1: conda (recommended)
conda env create -f environment.yml
conda activate tiktok-research

# Option 2: pip
pip install psycopg2-binary numpy scipy requests python-dotenv nltk pytest
```

### Configuration

Create a `.env` file in the project root:

```
ANNOTATION_DB_NAME=tiktok_disorders
ANNOTATION_DB_USER=postgres
ANNOTATION_DB_PASSWORD=your_password
ANNOTATION_DB_HOST=localhost
ANNOTATION_DB_PORT=5433
OLLAMA_URL=http://localhost:11434
```

## Quick Start

```bash
# 1. Verify models respond
python test_models.py

# 2. Small pilot (20 chunks, 1 model, 1 temperature)
python run_pipeline.py --name "pilot_v1" --chunk-limit 20 \
    --models qwen3:32b --temperatures 0.0

# 3. Full study (all 15 steps, all models)
python run_pipeline.py --name "main_study_v1" --chunk-limit 500

# 4. Check publication readiness
python progress_report.py --experiment-id 1
```

## Pipeline Steps

The orchestrator (`run_pipeline.py`) runs 15 steps. Control execution with `--skip-to` and `--stop-after`.

### Core Pipeline (Steps 1–8)

| Step | Name | What It Does |
|------|------|-------------|
| 1 | `schema` | Creates/updates all database tables |
| 2 | `cohort` | Selects creators into development (20%), reliability (60%), holdout (20%) splits |
| 3 | `chunking` | Splits transcripts into multi-sentence chunks (150–500 chars) with context carry |
| 4 | `annotate` | Runs each chunk x 6 constructs x 7 models x 2 temperatures x 5 runs |
| 5 | `stability` | Per-chunk stability metrics + group-level reliability (alpha, kappa, bootstrap CIs) |
| 6 | `final_labels` | Derives canonical labels: modal for categorical, median for continuous |
| 7 | `validation` | Compares annotations against existing `narrative_elements` data |
| 8 | `reporting` | Generates publication CSV tables (Tables 1–6, threshold curves, per-chunk export) |

### Extended Publication Analyses (Steps 9–15)

| Step | Name | What It Does |
|------|------|-------------|
| 9 | `cross_model` | Cross-model convergent validity — **the headline analysis** |
| 10 | `convergence` | Demonstrates R=5 outperforms R=1 (justifies the multi-run approach) |
| 11 | `significance` | Wilcoxon (temperature effect), Spearman (model size), Friedman (construct difficulty) |
| 12 | `errors` | Confusion matrices: within-model instability + cross-model disagreement |
| 13 | `discriminant` | Construct correlation matrix (proves constructs measure distinct things) |
| 14 | `stratified` | Breakdowns by content type, chunk length, temporal position, diagnosis |
| 15 | `qualitative` | Stratified sample export for face validity review |

## Constructs

Six health language constructs are annotated:

| Construct | Type | Scale / Labels | Stability Rule |
|-----------|------|---------------|----------------|
| `certainty_hedging` | Continuous | 0.0–1.0 (low / moderate / high) | range <= 0.2 AND stdev <= 0.10 |
| `symptom_concreteness` | Continuous | 0.0–1.0 (abstract / moderate / concrete) | range <= 0.2 AND stdev <= 0.10 |
| `temporal_orientation` | Categorical | past / present / future / mixed | agreement >= 80% (4/5 runs) |
| `agency_control` | Categorical | active / passive / helpless / mixed | agreement >= 80% |
| `social_proof` | Categorical | present / absent | agreement >= 80% |
| `medical_authority` | Categorical | professional / self_research / mixed / none_observed | agreement >= 80% |

Every prompt also supports two special labels:
- **`none`** — no health-related content in the chunk (tracked as coverage rate)
- **`unclear`** — health content present but the construct is genuinely ambiguous (tracked as clarity rate)

## Usage

### Full Pipeline

```bash
# All models, both temperatures, development + reliability splits
python run_pipeline.py --name "main_study_v1" --chunk-limit 500

# Specific models or temperatures
python run_pipeline.py --name "qwen_study" --models qwen3:32b --temperatures 0.0 0.5

# Run on holdout set for confirmation
python run_pipeline.py --name "holdout_confirm" --splits holdout --chunk-limit 200
```

### Resume After a Crash

The annotation step supports full resume. If the pipeline crashes mid-annotation:

```bash
# Resume from annotation step — resets stale tasks, picks up where it left off
python run_pipeline.py --experiment-id 1 --skip-to annotate

# Or resume annotation directly
python annotate.py --resume 1
```

### Skip to Specific Steps

```bash
# Re-run only stability + reporting for an existing experiment
python run_pipeline.py --experiment-id 1 --skip-to stability --stop-after reporting

# Run only the extended analyses (steps 9–15)
python run_pipeline.py --experiment-id 1 --skip-to cross_model

# Generate reports only
python run_pipeline.py --experiment-id 1 --skip-to reporting --stop-after reporting
```

### Run Individual Modules

```bash
python stability.py --experiment-id 1
python reporting.py --experiment-id 1
python cross_model_validity.py --experiment-id 1
python run_convergence.py --experiment-id 1
python significance_tests.py --experiment-id 1
python error_analysis.py --experiment-id 1
python discriminant_validity.py --experiment-id 1
python stratified_analysis.py --experiment-id 1
python qualitative_sample.py --experiment-id 1
python progress_report.py --experiment-id 1
```

### CLI Reference

```
python run_pipeline.py [OPTIONS]

Options:
  --name TEXT               Experiment name (required for new runs)
  --description TEXT        Experiment description
  --experiment-id INT       Resume from existing experiment ID
  --chunk-limit INT         Max chunks to annotate (default: 100)
  --models MODEL [...]      Model keys (default: all 7)
  --temperatures FLOAT [...] Temperatures (default: 0.0 0.5)
  --num-runs INT            Runs per condition (default: 5)
  --splits SPLIT [...]      Cohort splits (default: development reliability)
  --chunking-method TEXT    Chunking method (default: multi_sentence)
  --skip-to STEP            Skip to a specific step
  --stop-after STEP         Stop after a specific step
```

## Output Files

All outputs are written to `outputs/experiment_<id>/`.

### Core Results Tables (Step 8)

| File | Paper Section | Contents |
|------|--------------|----------|
| `table1_group_reliability.csv` | Table 1 (Main Results) | Krippendorff alpha with 95% bootstrap CI, Fleiss kappa, stability rate with CI, coverage, clarity — per construct x model x temperature |
| `table2_stability_by_model.csv` | Table 2 | Mean stability rate, alpha, coverage by model |
| `table3_stability_by_temperature.csv` | Table 3 | T=0.0 vs T=0.5 effect per construct |
| `table4_coverage_clarity.csv` | Table 4 | Coverage (1 - none_rate) and clarity (1 - unclear_rate) |
| `table5_label_distributions.csv` | Table 5 | Label frequency distributions for stable annotations |
| `table6_cost_efficiency.csv` | Table 6 | Processing time, tokens, speedup vs estimated human coding |

### Validity and Robustness (Steps 9–15)

| File | Paper Section | Contents |
|------|--------------|----------|
| `cross_model_agreement_matrix.csv` | Convergent Validity | Pairwise agreement rate between every model pair |
| `cross_model_krippendorff.csv` | Convergent Validity | Alpha across models (models as raters) with bootstrap CIs |
| `cross_model_consensus.csv` | Convergent Validity | Unanimous + majority agreement rates |
| `run_convergence.csv` | Justifying R=5 | Stability rate and label match at R=1,2,3,4,5 |
| `significance_temperature.csv` | Statistical Tests | Wilcoxon paired test, Cohen's d for temperature effect |
| `significance_model_size.csv` | Statistical Tests | Spearman correlation: model size vs reliability |
| `significance_constructs.csv` | Statistical Tests | Friedman test for construct difficulty |
| `construct_correlation_matrix.csv` | Discriminant Validity | Pairwise construct correlations (should be low/moderate) |
| `confusion_within_model.csv` | Error Analysis | Label pair co-occurrences for unstable chunks |
| `confusion_cross_model.csv` | Error Analysis | Model-vs-model disagreement patterns |
| `error_by_chunk_length.csv` | Error Analysis | Stability by text length quartiles |
| `error_by_content_type.csv` | Error Analysis | Stability by content type |
| `stratified_by_content_type.csv` | Stratified Analysis | Stability x content type |
| `stratified_by_chunk_length.csv` | Stratified Analysis | Stability x chunk length quartiles |
| `stratified_by_temporal_position.csv` | Stratified Analysis | Stability x early/middle/late in timeline |
| `stratified_by_diagnosis.csv` | Stratified Analysis | Stability x creator's claimed diagnosis |
| `stratified_by_context_carry.csv` | Stratified Analysis | Stability with vs without prior context |
| `qualitative_samples.csv` | Appendix / Face Validity | Stratified sample: unanimous, disagreement, none, unclear |

### Raw Data Exports

| File | Purpose |
|------|---------|
| `full_per_chunk_stability.csv` | Complete per-chunk metrics joined with metadata |
| `figure_stability_threshold_curve.csv` | Data for plotting coverage vs threshold trade-off curves |
| `validation_results.csv` | Comparison with existing `narrative_elements` data |

## Monitoring a Run

```sql
-- Task progress
SELECT status, COUNT(*) FROM annotation_tasks
WHERE experiment_id = 1 GROUP BY status;

-- Progress by construct
SELECT construct_name, status, COUNT(*)
FROM annotation_tasks WHERE experiment_id = 1
GROUP BY construct_name, status ORDER BY construct_name;

-- Estimated time remaining
SELECT
  COUNT(*) FILTER (WHERE status='completed') AS done,
  COUNT(*) FILTER (WHERE status='pending') AS remaining,
  ROUND(
    EXTRACT(EPOCH FROM NOW() - MIN(started_at))
    / NULLIF(COUNT(*) FILTER (WHERE status='completed'), 0)
    * COUNT(*) FILTER (WHERE status='pending') / 60.0, 1
  ) AS est_minutes_left
FROM annotation_tasks WHERE experiment_id = 1;
```

Or use the progress report:
```bash
python progress_report.py --experiment-id 1
```

## Publication Readiness Thresholds

The `progress_report.py` script evaluates results against these criteria:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Krippendorff alpha | >= 0.667 | Accepted threshold for content analysis |
| Stability rate | >= 70% | Majority of chunks produce stable labels |
| Coverage rate | >= 80% | Most chunks contain relevant health content |
| Clarity rate | >= 90% | LLMs rarely resort to "unclear" |
| Cross-model agreement | >= 60% | Majority consensus across architectures |

## Recommended Workflow for Publication

### Phase 1: Development Tuning (20% split)
```bash
python run_pipeline.py --name "dev_v1" --chunk-limit 100 \
    --splits development --stop-after reporting
```
Review outputs. Adjust prompts in `prompts.py` or thresholds in `config.py` if needed.

### Phase 2: Main Study (60% reliability split, all models)
```bash
python run_pipeline.py --name "main_study" --chunk-limit 500 --splits reliability
```
This runs all 15 steps. Check readiness with `progress_report.py`.

### Phase 3: Holdout Confirmation (20% holdout split)
```bash
python run_pipeline.py --name "holdout_confirm" --splits holdout --chunk-limit 200
```
Confirm that Phase 2 reliability metrics generalize.

### Phase 4: Assemble Paper

Map outputs to paper sections:

1. **Table 1** — `table1_group_reliability.csv` (alpha, kappa, stability rate with CIs)
2. **Table 2** — `table2_stability_by_model.csv` (model comparison)
3. **Table 3** — `table3_stability_by_temperature.csv` (temperature effect)
4. **Table 4** — `table4_coverage_clarity.csv` (coverage and clarity)
5. **Table 5** — `table5_label_distributions.csv` (what the data looks like)
6. **Table 6** — `table6_cost_efficiency.csv` (LLM hours vs human hours)
7. **Figure 1** — `figure_stability_threshold_curve.csv` (threshold vs coverage trade-off)
8. **Headline finding** — `cross_model_*.csv` (independent architectures converge)
9. **Method justification** — `run_convergence.csv` (R=5 outperforms R=1)
10. **Significance** — `significance_*.csv` (p-values for all effects)
11. **Discriminant validity** — `construct_correlation_matrix.csv`
12. **Robustness** — `stratified_by_*.csv` (stable across content types, lengths, diagnoses)
13. **Appendix** — `qualitative_samples.csv` (example chunks for face validity)

## Tests

```bash
# Run all tests (71 total)
pytest tests/ -v

# Label normalization tests (41 tests)
pytest tests/test_parsing.py -v

# Stability metric tests (30 tests)
pytest tests/test_stability.py -v
```

Tests mock `psycopg2` to avoid requiring a database connection.

## Database Schema Migration

If upgrading from an earlier version, add the bootstrap CI columns:

```sql
ALTER TABLE group_reliability_metrics
    ADD COLUMN IF NOT EXISTS alpha_ci_lower FLOAT,
    ADD COLUMN IF NOT EXISTS alpha_ci_upper FLOAT,
    ADD COLUMN IF NOT EXISTS stability_ci_lower FLOAT,
    ADD COLUMN IF NOT EXISTS stability_ci_upper FLOAT;
```

Or re-run step 1 (`schema`) which uses `CREATE TABLE IF NOT EXISTS`.

## Project Structure

```
tiktok_research/
├── .env                         # Database and Ollama credentials (not committed)
├── config.py                    # All configuration: models, constructs, thresholds, bins
├── schema.sql                   # PostgreSQL table definitions
├── run_pipeline.py              # Main orchestrator (15 steps)
│
│   ── Core Pipeline ──
├── cohort_selection.py          # Step 2: Study cohort selection
├── chunking.py                  # Step 3: Transcript chunking (multi-sentence)
├── prompts.py                   # Construct prompt templates
├── label_parsing.py             # Label normalization (no DB dependency)
├── ollama_client.py             # Ollama API wrapper with parameter recording
├── annotate.py                  # Step 4: Multi-run annotation engine with resume
├── stability.py                 # Step 5: Stability metrics + reliability (alpha, kappa, CIs)
├── final_labels.py              # Step 6: Canonical label derivation
├── validation.py                # Step 7: Validation against narrative_elements
├── reporting.py                 # Step 8: Publication tables and CSV exports
│
│   ── Extended Analyses ──
├── cross_model_validity.py      # Step 9:  Cross-model convergent validity
├── run_convergence.py           # Step 10: R=1..5 convergence analysis
├── significance_tests.py        # Step 11: Statistical significance tests
├── error_analysis.py            # Step 12: Confusion matrices and error patterns
├── discriminant_validity.py     # Step 13: Construct independence
├── stratified_analysis.py       # Step 14: Stratified breakdowns by metadata
├── qualitative_sample.py        # Step 15: Face validity samples
│
│   ── Utilities ──
├── progress_report.py           # Pipeline status and publication readiness check
├── test_models.py               # Smoke test for Ollama models
│
├── tests/
│   ├── test_parsing.py          # 41 label normalization tests
│   └── test_stability.py        # 30 stability metric tests
│
├── outputs/                     # Generated CSV reports (per experiment)
│   └── experiment_<id>/
│       ├── table1_group_reliability.csv
│       ├── ...
│       └── qualitative_samples.csv
│
├── pyproject.toml               # Project metadata and dependencies
├── environment.yml              # Conda environment specification
└── IMPLEMENTATION_ROADMAP.md    # Development roadmap and design rationale
```

## Key Design Decisions

- **Stability over accuracy**: The core claim is that multi-run agreement is a stronger reliability signal than single-pass confidence scores. No human gold standard is needed.
- **Cross-model convergence as validity**: When architecturally diverse models independently agree, this is analogous to inter-rater reliability — but with LLMs as the raters.
- **Construct-aware thresholds**: Categorical constructs use agreement ratio (>= 0.8); continuous constructs use max range (<= 0.2) and stdev (<= 0.10).
- **Exclusive upper bounds on bins**: `low` = 0.0–0.29, `moderate` = 0.3–0.69, `high` = 0.7–1.0 — avoids ambiguity at boundaries.
- **None vs unclear**: `none` = no health content detected; `unclear` = health content present but construct is ambiguous. Tracked separately as coverage rate and clarity rate.
- **`none_observed` for medical authority**: Distinguishes "no medical authority cited" from "no health content in chunk."
- **R=5 runs**: Provides statistical power for stability while keeping compute feasible. The convergence analysis (step 10) validates this choice empirically.
- **T=0.0 and T=0.5**: Temperature 0.0 tests deterministic consistency; 0.5 tests whether the model's probability distribution survives sampling noise.
- **experiment_id FK**: Every table links back to experiment_id for full reproducibility.
- **Local models only**: All inference via Ollama for cost control and reproducibility.
- **Resume support**: Crashed runs can be resumed without losing completed work.

## Total Annotation Volume

For a full study with default settings:

```
Chunks (e.g., 500) x 6 constructs x 7 models x 2 temperatures x 5 runs
= 500 x 6 x 7 x 2 x 5 = 210,000 individual LLM inferences
```

This volume provides the statistical power needed for the reliability claims.
