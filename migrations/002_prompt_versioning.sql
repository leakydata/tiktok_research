-- ============================================================
-- Migration 002: Prompt versioning columns
-- Adds per-task prompt_version, prompt_template_name, and
-- per-run prompt_hash (sha256 of full rendered prompt).
--
-- Safe to re-run: uses ADD COLUMN IF NOT EXISTS.
-- V1 experiments are backfilled with defaults.
-- ============================================================

-- annotation_tasks: store which prompt version was used per task
ALTER TABLE annotation_tasks
    ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT 'v1',
    ADD COLUMN IF NOT EXISTS prompt_template_name TEXT;

-- llm_annotation_runs: store version, template name, and exact prompt hash
ALTER TABLE llm_annotation_runs
    ADD COLUMN IF NOT EXISTS prompt_version TEXT NOT NULL DEFAULT 'v1',
    ADD COLUMN IF NOT EXISTS prompt_template_name TEXT,
    ADD COLUMN IF NOT EXISTS prompt_hash TEXT;

-- Indexes for filtering by prompt version
CREATE INDEX IF NOT EXISTS idx_tasks_prompt_version
    ON annotation_tasks(prompt_version);
CREATE INDEX IF NOT EXISTS idx_runs_prompt_version
    ON llm_annotation_runs(prompt_version);

-- Backfill template names for existing rows
UPDATE annotation_tasks
SET prompt_template_name = construct_name || '_v1'
WHERE prompt_template_name IS NULL;

UPDATE llm_annotation_runs
SET prompt_template_name = construct_name || '_v1'
WHERE prompt_template_name IS NULL;
