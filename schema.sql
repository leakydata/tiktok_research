-- ============================================================
-- MULTI-RUN LLM ANNOTATION PIPELINE - DATABASE SCHEMA
-- Run against: tiktok_disorders database (extends existing schema)
-- ============================================================

-- ============================================================
-- EXPERIMENT TRACKING (create first - others reference it)
-- ============================================================
CREATE TABLE IF NOT EXISTS experiment_runs (
    experiment_id SERIAL PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    description TEXT,

    -- Configuration snapshot
    num_chunks_tested INTEGER,
    models_tested TEXT[],
    temperatures_tested FLOAT[],
    num_runs INTEGER,
    chunking_method TEXT,
    chunking_version TEXT,
    prompt_version TEXT,

    -- Status
    status TEXT CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,

    -- Full config snapshot (JSON)
    config_snapshot JSONB,
    results_summary JSONB,
    notes TEXT
);

-- ============================================================
-- STUDY COHORT
-- ============================================================
CREATE TABLE IF NOT EXISTS study_cohort (
    cohort_id SERIAL PRIMARY KEY,
    creator_username TEXT UNIQUE NOT NULL,

    -- Selection criteria
    video_count INTEGER,
    transcript_count INTEGER,
    first_video_date DATE,
    last_video_date DATE,
    days_span INTEGER,

    -- Diagnoses claimed
    diagnoses_claimed TEXT[],
    primary_diagnosis TEXT,

    -- Study assignment (methods paper framing)
    cohort_split TEXT CHECK (cohort_split IN ('development', 'reliability', 'holdout')),
    included_reason TEXT,
    research_notes TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cohort_username ON study_cohort(creator_username);
CREATE INDEX IF NOT EXISTS idx_cohort_split ON study_cohort(cohort_split);

-- ============================================================
-- ANNOTATION CHUNKS
-- Multi-sentence chunks with char budget, not single sentences
-- ============================================================
CREATE TABLE IF NOT EXISTS annotation_chunks (
    chunk_id SERIAL PRIMARY KEY,
    transcript_id INTEGER REFERENCES transcripts(id),
    video_id INTEGER REFERENCES videos(id),
    chunk_sequence INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    sentence_count INTEGER,
    char_length INTEGER,

    -- Chunking method metadata
    chunking_method TEXT NOT NULL DEFAULT 'multi_sentence',
    chunking_version TEXT NOT NULL DEFAULT 'v1',
    chunking_params JSONB,

    -- Context carry (prior chunk tail for context-dependent constructs)
    context_carry_text TEXT,

    -- Temporal context
    video_date DATE,
    creator_username TEXT,
    days_since_first_video INTEGER,

    -- Content type from existing narrative_elements (for stratification)
    content_type TEXT,

    -- Prior diagnosis context
    prior_diagnoses_claimed TEXT[],

    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(transcript_id, chunk_sequence, chunking_method, chunking_version)
);

CREATE INDEX IF NOT EXISTS idx_chunks_transcript ON annotation_chunks(transcript_id);
CREATE INDEX IF NOT EXISTS idx_chunks_video ON annotation_chunks(video_id);
CREATE INDEX IF NOT EXISTS idx_chunks_creator ON annotation_chunks(creator_username);
CREATE INDEX IF NOT EXISTS idx_chunks_method ON annotation_chunks(chunking_method);

-- ============================================================
-- ANNOTATION TASK QUEUE
-- Tracks individual annotation jobs with retry logic
-- ============================================================
CREATE TABLE IF NOT EXISTS annotation_tasks (
    task_id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiment_runs(experiment_id),
    chunk_id INTEGER REFERENCES annotation_chunks(chunk_id),
    construct_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    temperature FLOAT NOT NULL,
    prompt_format TEXT NOT NULL,
    run_number INTEGER NOT NULL,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    retries INTEGER DEFAULT 0,
    last_error TEXT,

    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    UNIQUE(experiment_id, chunk_id, construct_name, model_name, temperature, prompt_format, run_number)
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON annotation_tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_experiment ON annotation_tasks(experiment_id);

-- ============================================================
-- LLM ANNOTATION RUNS
-- Individual annotation attempts with proper type separation
-- ============================================================
CREATE TABLE IF NOT EXISTS llm_annotation_runs (
    run_id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES annotation_tasks(task_id),
    experiment_id INTEGER REFERENCES experiment_runs(experiment_id),
    chunk_id INTEGER REFERENCES annotation_chunks(chunk_id),

    -- Construct being annotated
    construct_name TEXT NOT NULL,

    -- Experimental conditions
    run_number INTEGER NOT NULL,
    temperature FLOAT NOT NULL,
    prompt_format TEXT NOT NULL CHECK (prompt_format IN ('single_label', 'multi_label', 'unrestricted')),

    -- Model details
    model_name TEXT NOT NULL,
    model_family TEXT,
    model_size_b FLOAT,

    -- Label output: separate storage for float vs text
    label_kind TEXT NOT NULL CHECK (label_kind IN ('float', 'category', 'binary', 'none', 'unclear', 'error')),
    label_value_text TEXT,
    label_value_float DOUBLE PRECISION,
    label_bin TEXT,  -- binned version of continuous labels (low/moderate/high)

    -- Raw data
    raw_response TEXT,
    processing_time_ms INTEGER,
    tokens_generated INTEGER,

    -- Full inference parameters for reproducibility
    inference_params JSONB,  -- {top_p, top_k, repeat_penalty, num_ctx, seed, num_predict, stop}
    ollama_version TEXT,
    quantization TEXT,

    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(experiment_id, chunk_id, construct_name, run_number, temperature, prompt_format, model_name)
);

CREATE INDEX IF NOT EXISTS idx_runs_chunk ON llm_annotation_runs(chunk_id);
CREATE INDEX IF NOT EXISTS idx_runs_construct ON llm_annotation_runs(construct_name);
CREATE INDEX IF NOT EXISTS idx_runs_model ON llm_annotation_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON llm_annotation_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_label_kind ON llm_annotation_runs(label_kind);

-- ============================================================
-- ANNOTATION STABILITY METRICS
-- Construct-aware: different metrics for continuous vs categorical
-- ============================================================
CREATE TABLE IF NOT EXISTS annotation_stability_metrics (
    metric_id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiment_runs(experiment_id),
    chunk_id INTEGER REFERENCES annotation_chunks(chunk_id),
    construct_name TEXT NOT NULL,

    -- Experimental conditions
    num_runs INTEGER NOT NULL,
    temperature FLOAT NOT NULL,
    prompt_format TEXT NOT NULL,
    model_name TEXT NOT NULL,

    -- Common metrics
    none_rate FLOAT,       -- fraction labeled 'none' (no health content)
    unclear_rate FLOAT,    -- fraction labeled 'unclear'
    valid_run_count INTEGER,  -- runs that produced a usable label

    -- Categorical reliability metrics (NULL for continuous constructs)
    agreement_ratio FLOAT,
    label_entropy FLOAT,
    modal_label TEXT,
    modal_count INTEGER,

    -- Continuous reliability metrics (NULL for categorical constructs)
    mean_value FLOAT,
    median_value FLOAT,
    stdev_value FLOAT,
    iqr_value FLOAT,
    range_value FLOAT,
    modal_bin TEXT,         -- most common bin (low/moderate/high)
    bin_agreement_ratio FLOAT,  -- agreement when binned

    -- Stability determination (construct-aware)
    is_stable BOOLEAN,
    stability_rule TEXT,    -- description of rule applied
    construct_type TEXT CHECK (construct_type IN ('categorical', 'continuous')),

    -- All labels observed
    labels_observed JSONB,  -- array of all labels
    label_distribution JSONB,

    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(experiment_id, chunk_id, construct_name, temperature, prompt_format, model_name)
);

CREATE INDEX IF NOT EXISTS idx_stability_chunk ON annotation_stability_metrics(chunk_id);
CREATE INDEX IF NOT EXISTS idx_stability_construct ON annotation_stability_metrics(construct_name);
CREATE INDEX IF NOT EXISTS idx_stability_stable ON annotation_stability_metrics(is_stable);
CREATE INDEX IF NOT EXISTS idx_stability_experiment ON annotation_stability_metrics(experiment_id);

-- ============================================================
-- GROUP-LEVEL RELIABILITY
-- Krippendorff alpha etc. computed across many chunks
-- ============================================================
CREATE TABLE IF NOT EXISTS group_reliability_metrics (
    metric_id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiment_runs(experiment_id),
    construct_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    temperature FLOAT NOT NULL,
    prompt_format TEXT NOT NULL,

    -- Sample info
    num_chunks INTEGER,
    num_stable_chunks INTEGER,
    coverage_rate FLOAT,     -- 1 - none_rate across all chunks
    clarity_rate FLOAT,      -- 1 - unclear_rate among non-none chunks

    -- Group-level reliability (categorical)
    krippendorff_alpha FLOAT,
    fleiss_kappa FLOAT,      -- proper Fleiss kappa across items

    -- Group-level reliability (continuous)
    icc_value FLOAT,         -- intraclass correlation
    mean_stdev FLOAT,        -- average within-chunk stdev
    mean_range FLOAT,

    -- Stability summary
    stability_rate FLOAT,    -- fraction of chunks that are stable
    mean_agreement FLOAT,

    computed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(experiment_id, construct_name, model_name, temperature, prompt_format)
);

-- ============================================================
-- FINAL ANNOTATIONS
-- Canonical "winning" label per chunk/construct after stability filtering
-- ============================================================
CREATE TABLE IF NOT EXISTS final_annotations (
    annotation_id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiment_runs(experiment_id),
    chunk_id INTEGER REFERENCES annotation_chunks(chunk_id),
    construct_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    temperature FLOAT NOT NULL,
    prompt_format TEXT NOT NULL,

    -- Final label
    final_label_text TEXT,
    final_label_float DOUBLE PRECISION,
    final_label_bin TEXT,

    -- How it was derived
    decision_rule TEXT NOT NULL,  -- 'modal_label', 'median_value', 'excluded_unstable'
    decision_rule_version TEXT NOT NULL DEFAULT 'v1',

    -- Stability snapshot
    is_stable BOOLEAN NOT NULL,
    agreement_ratio FLOAT,
    stdev_value FLOAT,
    num_valid_runs INTEGER,

    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(experiment_id, chunk_id, construct_name, model_name, temperature, prompt_format)
);

CREATE INDEX IF NOT EXISTS idx_final_chunk ON final_annotations(chunk_id);
CREATE INDEX IF NOT EXISTS idx_final_construct ON final_annotations(construct_name);
CREATE INDEX IF NOT EXISTS idx_final_stable ON final_annotations(is_stable);

-- ============================================================
-- VALIDATION COMPARISON
-- Compare multi-run annotations with existing narrative_elements
-- ============================================================
CREATE TABLE IF NOT EXISTS validation_comparisons (
    comparison_id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiment_runs(experiment_id),
    chunk_id INTEGER REFERENCES annotation_chunks(chunk_id),
    video_id INTEGER REFERENCES videos(id),

    -- Multi-run annotation result
    construct_name TEXT NOT NULL,
    annotation_label TEXT,
    annotation_is_stable BOOLEAN,

    -- Existing narrative_elements value
    narrative_field TEXT NOT NULL,  -- column name from narrative_elements
    narrative_value TEXT,

    -- Agreement
    agrees BOOLEAN,
    comparison_notes TEXT,

    computed_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================
-- PARAPHRASE TESTS (structure for future Phase 2)
-- ============================================================
CREATE TABLE IF NOT EXISTS paraphrase_tests (
    paraphrase_id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiment_runs(experiment_id),
    original_chunk_id INTEGER REFERENCES annotation_chunks(chunk_id),
    paraphrase_number INTEGER,
    paraphrase_text TEXT NOT NULL,
    paraphrase_model TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
