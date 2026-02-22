# Multi-Run LLM Annotation Pipeline for Health Language Analysis

A reproducible pipeline demonstrating that **multi-run stability filtering across diverse LLMs** produces consistent, reliable annotations — eliminating the need for costly human coding in health communication research.

Supports both **local models via Ollama** and **cloud APIs** (OpenAI, DeepSeek, MiniMax, Anthropic) for cross-architecture comparison.

Extends [Hassan et al. (2024)](https://doi.org/10.48550/arXiv.2412.xxxxx) *"Automated Multi-Label Annotation for Mental Health Illnesses Using Large Language Models"* from Reddit to TikTok video transcripts.

## Thesis

Traditional content analysis requires expensive, time-consuming human annotation with inter-rater reliability checks. This pipeline proves an alternative: by running **diverse LLM architectures** (4B–671B parameters, local and cloud) **5 times each** at **2 temperature settings**, then filtering for stability, researchers can automate annotation entirely.

**Cross-model convergence** — where architecturally independent LLMs agree on the same label — serves as the validity signal analogous to inter-rater reliability. If a 4.7B ChatGLM model, a 14B Phi model, and a 671B DeepSeek model all independently converge on the same annotation, this provides stronger evidence than any single human coder.

## Prerequisites

### Software
- **Python** >= 3.10
- **PostgreSQL** running on `localhost:5433` (configurable via `.env`)
- **Ollama** running on `localhost:11434` (for local models, configurable via `.env`)

### Existing Database
The pipeline extends an existing `tiktok_disorders` database that must already contain:
- `videos` — video metadata
- `transcripts` — full transcript text
- `claimed_diagnoses` — creator-level diagnosis claims
- `narrative_elements` — existing single-pass extraction (used for validation only)

### Models

#### Local (Ollama)
Pull models before running:

```bash
ollama pull glm-4.7-flash          # 4.7B  - ChatGLM family
ollama pull phi4:latest             # 14B   - Microsoft Phi
ollama pull gpt-oss:20b            # 20B   - GPT open-source
ollama pull alibayram/medgemma:27b # 27B   - Medical Gemma (domain-specific)
ollama pull gemma3:27b             # 27B   - Google Gemma
```

#### Cloud APIs (optional)
Set API keys in `.env` to enable:

| Model | Backend | API Model Name | Notes |
|-------|---------|----------------|-------|
| `deepseek-chat` | DeepSeek | deepseek-chat | DeepSeek-V3.2, 671B MoE (37B active) |
| `minimax-m2.5` | MiniMax | MiniMax-M2.5 | Undisclosed size |
| `gpt-5-nano` | OpenAI | gpt-5-nano | Fixed temperature only (API restriction) |
| `claude-haiku-4.5` | Anthropic | claude-haiku-4-5-20251001 | Anthropic Haiku 4.5 |

### Environment Setup

```bash
# Option 1: conda (recommended)
conda env create -f environment.yml
conda activate tiktok-research

# Option 2: pip
pip install -e .

# Or manually:
pip install psycopg2-binary numpy scipy requests python-dotenv nltk matplotlib pandas openai anthropic pytest
```

### Configuration

Create a `.env` file in the project root:

```
# Database
ANNOTATION_DB_NAME=tiktok_disorders
ANNOTATION_DB_USER=postgres
ANNOTATION_DB_PASSWORD=your_password
ANNOTATION_DB_HOST=localhost
ANNOTATION_DB_PORT=5433

# Ollama
OLLAMA_URL=http://localhost:11434

# Cloud API keys (set to enable cloud models)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
MINIMAX_API_KEY=
```

## Quick Start

```bash
# 1. Verify all configured models respond
python test_models.py

# 2. Development run (100 chunks, local models)
python run_pipeline.py --name "dev_v1" --chunk-limit 100 \
    --splits development --stop-after reporting

# 3. Cloud model comparison (same chunks as above)
python run_pipeline.py --name "cloud_v1" --chunk-limit 100 \
    --splits development --models deepseek-chat minimax-m2.5 gpt-5-nano \
    --stop-after annotate

# 4. Check publication readiness
python progress_report.py
```

## Pipeline Steps

The orchestrator (`run_pipeline.py`) runs 15 steps. Control execution with `--skip-to` and `--stop-after`.

### Core Pipeline (Steps 1-8)

| Step | Name | What It Does |
|------|------|-------------|
| 1 | `schema` | Creates/updates all database tables |
| 2 | `cohort` | Selects creators into development (20%), reliability (60%), holdout (20%) splits |
| 3 | `chunking` | Splits transcripts into multi-sentence chunks (150-500 chars) with context carry |
| 4 | `annotate` | Runs each chunk x constructs x models x temperatures x runs |
| 5 | `stability` | Per-chunk stability metrics + group-level reliability (alpha, kappa, bootstrap CIs) |
| 6 | `final_labels` | Derives canonical labels: modal for categorical, median for continuous |
| 7 | `validation` | Compares annotations against existing `narrative_elements` data |
| 8 | `reporting` | Generates publication CSV tables (Tables 1-6, threshold curves, per-chunk export) |

### Extended Publication Analyses (Steps 9-15)

| Step | Name | What It Does |
|------|------|-------------|
| 9 | `cross_model` | Cross-model convergent validity -- **the headline analysis** |
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
| `certainty_hedging` | Continuous | 0.0-1.0 (low / moderate / high) | range <= 0.2 AND stdev <= 0.10 |
| `symptom_concreteness` | Continuous | 0.0-1.0 (abstract / moderate / concrete) | range <= 0.2 AND stdev <= 0.10 |
| `temporal_orientation` | Categorical | past / present / future / mixed | agreement >= 80% (4/5 runs) |
| `agency_control` | Categorical | active / passive / helpless / mixed | agreement >= 80% |
| `social_proof` | Categorical | present / absent | agreement >= 80% |
| `medical_authority` | Categorical | professional / self_research / mixed / none_observed | agreement >= 80% |

Every prompt also supports two special labels:
- **`none`** -- no health-related content in the chunk (tracked as coverage rate)
- **`unclear`** -- health content present but the construct is genuinely ambiguous (tracked as clarity rate)

## Usage

### Local Models Only
```bash
python run_pipeline.py --name "local_study" --chunk-limit 100 \
    --models glm-4.7-flash phi4:latest gpt-oss:20b gemma3:27b alibayram/medgemma:27b
```

### Cloud Models Only
```bash
python run_pipeline.py --name "cloud_study" --chunk-limit 100 \
    --models deepseek-chat minimax-m2.5 gpt-5-nano \
    --splits development --stop-after annotate
```

### Resume After a Crash
```bash
# Resume annotation -- resets stale tasks, picks up where it left off
python annotate.py --resume <EXPERIMENT_ID>

# Or via the orchestrator
python run_pipeline.py --experiment-id <ID> --skip-to annotate
```

### Add a New Model to an Existing Experiment
```bash
# Create tasks only (for batch submission)
python annotate.py --add-models <EXPERIMENT_ID> --models claude-haiku-4.5 --create-only

# Or create and run immediately (full-price)
python annotate.py --add-models <EXPERIMENT_ID> --models claude-haiku-4.5
```

### Batch Processing (50% Cost Savings)
Anthropic and OpenAI offer batch APIs with 50% discounts (24-hour turnaround).

```bash
# Create tasks without running them
python annotate.py --add-models 14 --models claude-haiku-4.5 --create-only

# Submit via batch API (50% off)
python batch_submit.py run --experiment-id 14 --model claude-haiku-4.5

# Or step-by-step:
python batch_submit.py submit --experiment-id 14 --model claude-haiku-4.5 --batch-size 1000
python batch_submit.py status --batch-id <BATCH_ID> --model claude-haiku-4.5
python batch_submit.py collect --experiment-id 14 --model claude-haiku-4.5 --batch-id <BATCH_ID>
```

Batch state is saved to `outputs/batch_state/` for recovery if disconnected.

### Run Stability + Reporting on Completed Experiment
```bash
python run_pipeline.py --experiment-id <ID> --skip-to stability
```

### Skip to Specific Steps
```bash
# Re-run only stability + reporting
python run_pipeline.py --experiment-id <ID> --skip-to stability --stop-after reporting

# Run only the extended analyses (steps 9-15)
python run_pipeline.py --experiment-id <ID> --skip-to cross_model

# Generate reports only
python run_pipeline.py --experiment-id <ID> --skip-to reporting --stop-after reporting
```

### CLI Reference

```
python run_pipeline.py [OPTIONS]

Options:
  --name TEXT               Experiment name (required for new runs)
  --description TEXT        Experiment description
  --experiment-id INT       Resume from existing experiment ID
  --chunk-limit INT         Max chunks to annotate (default: 100)
  --models MODEL [...]      Model keys (default: all configured)
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
| `table1_group_reliability.csv` | Table 1 (Main Results) | Krippendorff alpha with 95% bootstrap CI, Fleiss kappa, stability rate with CI, coverage, clarity -- per construct x model x temperature |
| `table2_stability_by_model.csv` | Table 2 | Mean stability rate, alpha, coverage by model |
| `table3_stability_by_temperature.csv` | Table 3 | T=0.0 vs T=0.5 effect per construct |
| `table4_coverage_clarity.csv` | Table 4 | Coverage (1 - none_rate) and clarity (1 - unclear_rate) |
| `table5_label_distributions.csv` | Table 5 | Label frequency distributions for stable annotations |
| `table6_cost_efficiency.csv` | Table 6 | Processing time, tokens, speedup vs estimated human coding |

### Validity and Robustness (Steps 9-15)

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
| `stratified_by_*.csv` | Stratified Analysis | Stability by content type, chunk length, temporal position, diagnosis |
| `qualitative_samples.csv` | Appendix / Face Validity | Stratified sample: unanimous, disagreement, none, unclear |

## Monitoring a Run

```bash
python progress_report.py
```

Or query directly:
```sql
SELECT status, COUNT(*) FROM annotation_tasks
WHERE experiment_id = 14 GROUP BY status;
```

## Publication Readiness Thresholds

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

### Phase 1b: Cloud Model Comparison
```bash
python run_pipeline.py --name "cloud_v1" --chunk-limit 100 \
    --splits development --models deepseek-chat minimax-m2.5 gpt-5-nano \
    --stop-after annotate
# Then run stability:
python run_pipeline.py --experiment-id <ID> --skip-to stability
```

### Phase 2: Main Study (60% reliability split)
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

1. **Table 1** -- `table1_group_reliability.csv` (alpha, kappa, stability rate with CIs)
2. **Table 2** -- `table2_stability_by_model.csv` (model comparison)
3. **Table 3** -- `table3_stability_by_temperature.csv` (temperature effect)
4. **Table 4** -- `table4_coverage_clarity.csv` (coverage and clarity)
5. **Table 5** -- `table5_label_distributions.csv` (what the data looks like)
6. **Table 6** -- `table6_cost_efficiency.csv` (LLM hours vs human hours)
7. **Figure 1** -- `figure_stability_threshold_curve.csv` (threshold vs coverage trade-off)
8. **Headline finding** -- `cross_model_*.csv` (independent architectures converge)
9. **Method justification** -- `run_convergence.csv` (R=5 outperforms R=1)
10. **Significance** -- `significance_*.csv` (p-values for all effects)
11. **Discriminant validity** -- `construct_correlation_matrix.csv`
12. **Robustness** -- `stratified_by_*.csv` (stable across content types, lengths, diagnoses)
13. **Appendix** -- `qualitative_samples.csv` (example chunks for face validity)

## Tests

```bash
# Label normalization + client contract tests (57 tests)
pytest tests/test_parsing.py tests/test_clients.py -v

# Stability metric tests (30 tests)
pytest tests/test_stability.py -v
```

Tests mock external dependencies to avoid requiring a database or API connection.

## Project Structure

```
tiktok_research/
+-- .env                         # Credentials (not committed)
+-- .gitignore
+-- config.py                    # All configuration: models, constructs, thresholds, bins
+-- schema.sql                   # PostgreSQL table definitions
+-- run_pipeline.py              # Main orchestrator (15 steps)
|
|   -- Core Pipeline --
+-- cohort_selection.py          # Step 2: Study cohort selection
+-- chunking.py                  # Step 3: Transcript chunking (multi-sentence)
+-- prompts.py                   # Construct prompt templates
+-- label_parsing.py             # Label normalization (no DB dependency)
+-- llm_client.py                # Multi-backend ABC + OpenAI/Anthropic clients
+-- ollama_client.py             # Ollama API wrapper (subclasses BaseLLMClient)
+-- annotate.py                  # Step 4: Multi-run annotation engine with resume
+-- stability.py                 # Step 5: Stability metrics + reliability
+-- final_labels.py              # Step 6: Canonical label derivation
+-- validation.py                # Step 7: Validation against narrative_elements
+-- reporting.py                 # Step 8: Publication tables and CSV exports
|
|   -- Extended Analyses --
+-- cross_model_validity.py      # Step 9:  Cross-model convergent validity
+-- run_convergence.py           # Step 10: R=1..5 convergence analysis
+-- significance_tests.py        # Step 11: Statistical significance tests
+-- error_analysis.py            # Step 12: Confusion matrices and error patterns
+-- discriminant_validity.py     # Step 13: Construct independence
+-- stratified_analysis.py       # Step 14: Stratified breakdowns by metadata
+-- qualitative_sample.py        # Step 15: Face validity samples
|
|   -- Utilities --
+-- progress_report.py           # Pipeline status and publication readiness check
+-- test_models.py               # Smoke test for all configured models
|
+-- tests/
|   +-- test_parsing.py          # 48 label normalization tests
|   +-- test_clients.py          # 9 multi-backend client contract tests
|   +-- test_stability.py        # 30 stability metric tests
|
+-- outputs/                     # Generated CSV reports (per experiment)
|   +-- experiment_<id>/
|
+-- pyproject.toml               # Project metadata and dependencies
+-- environment.yml              # Conda environment specification
+-- IMPLEMENTATION_ROADMAP.md    # Development roadmap and design rationale
```

## Architecture: Multi-Backend LLM Support

```
                    +------------------+
                    | BaseLLMClient    |  (ABC in llm_client.py)
                    | - generate()     |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                |                |
   +--------+-------+ +-----+------+ +-------+--------+
   | OllamaClient   | | OpenAI     | | AnthropicClient|
   | (ollama_client) | | Compatible | | (llm_client)   |
   +----------------+ +-----+------+ +----------------+
                             |
                    +--------+--------+
                    |        |        |
                 OpenAI  DeepSeek  MiniMax
```

The `AnnotationPipeline` in `annotate.py` uses a lazy client registry -- backends are instantiated on first use and cached. This means:
- Local-only experiments never import `openai` or `anthropic`
- Cloud API keys are only checked when cloud models are requested
- All backends return the same result dict for uniform DB storage

## Key Design Decisions

- **Stability over accuracy**: Multi-run agreement is a stronger reliability signal than single-pass confidence scores. No human gold standard is needed.
- **Cross-model convergence as validity**: Architecturally diverse models agreeing is analogous to inter-rater reliability.
- **Multi-backend support**: Local Ollama models and cloud APIs use the same pipeline, enabling direct comparison between open-weight and proprietary models.
- **Construct-aware thresholds**: Categorical constructs use agreement ratio (>= 0.8); continuous constructs use max range (<= 0.2) and stdev (<= 0.10).
- **Exclusive upper bounds on bins**: `low` = 0.0-0.29, `moderate` = 0.3-0.69, `high` = 0.7-1.0.
- **None vs unclear**: `none` = no health content; `unclear` = health content but ambiguous construct. Tracked separately.
- **R=5 runs**: Statistical power for stability; validated empirically by convergence analysis (step 10).
- **T=0.0 and T=0.5**: Tests deterministic consistency vs sampling robustness.
- **experiment_id FK**: Every table links back for full reproducibility.
- **Resume support**: Crashed runs resume without losing completed work or spending money twice.
