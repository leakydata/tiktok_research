"""
Main orchestrator: runs the full multi-run annotation pipeline.

Usage:
    # Full pipeline (development + reliability splits, 100 chunks)
    python run_pipeline.py --name "pilot_run_v1" --chunk-limit 100

    # Specific models/temperatures
    python run_pipeline.py --name "qwen_only" --models qwen3-32b --temperatures 0.0 0.5

    # Just stability + reporting for an existing experiment
    python run_pipeline.py --experiment-id 1 --skip-to stability

    # Run on holdout set
    python run_pipeline.py --name "holdout_validation" --splits holdout --chunk-limit 50
"""

import argparse
import sys
import logging
import subprocess

from config import (
    DB_CONFIG, MODELS_TO_TEST, NUM_RUNS, TEMPERATURES,
    DEFAULT_CHUNKING_METHOD,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

STEPS = ['schema', 'cohort', 'chunking', 'annotate', 'stability', 'final_labels', 'validation', 'reporting']


def run_schema():
    """Apply database schema."""
    import psycopg2
    conn = psycopg2.connect(**DB_CONFIG)
    schema_path = 'schema.sql'

    logger.info("Applying database schema...")
    try:
        with open(schema_path, 'r') as f:
            sql = f.read()
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
        logger.info("Schema applied successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Schema application failed: {e}")
        raise
    finally:
        conn.close()


def run_cohort():
    """Select study cohort."""
    from cohort_selection import select_cohort
    select_cohort()


def run_chunking(method: str, splits: list[str]):
    """Chunk transcripts."""
    from chunking import TranscriptChunker
    chunker = TranscriptChunker(method=method)
    chunker.run(splits=splits)


def run_annotation(name: str, description: str, chunk_limit: int,
                   models: list[str], temperatures: list[float],
                   num_runs: int, splits: list[str]) -> int:
    """Run multi-run annotation and return experiment_id."""
    from annotate import AnnotationPipeline

    pipeline = AnnotationPipeline()
    try:
        # Check which models are available
        available = pipeline.ollama.check_models_available()
        missing = [k for k, v in available.items() if not v and k in models]
        if missing:
            logger.warning(f"Models not found in Ollama: {missing}")
            logger.warning("Run 'ollama pull <model>' to download them.")
            models = [m for m in models if m not in missing]
            if not models:
                raise RuntimeError("No models available. Pull at least one model first.")

        exp_id = pipeline.create_experiment(
            name=name,
            description=description,
            models=models,
            temperatures=temperatures,
            num_runs=num_runs,
            chunk_limit=chunk_limit,
        )
        pipeline.create_task_queue(
            experiment_id=exp_id,
            chunk_limit=chunk_limit,
            models=models,
            temperatures=temperatures,
            num_runs=num_runs,
            splits=splits,
        )
        pipeline.run_tasks(exp_id)
        return exp_id
    finally:
        pipeline.close()


def run_stability(experiment_id: int):
    """Compute stability metrics."""
    from stability import StabilityAnalyzer
    analyzer = StabilityAnalyzer()
    try:
        analyzer.run(experiment_id)
    finally:
        analyzer.close()


def run_final_labels(experiment_id: int):
    """Derive final annotations."""
    from final_labels import FinalLabelDeriver
    deriver = FinalLabelDeriver()
    try:
        deriver.run(experiment_id)
    finally:
        deriver.close()


def run_validation(experiment_id: int):
    """Run validation against narrative_elements."""
    from validation import ValidationComparator
    v = ValidationComparator()
    try:
        v.run(experiment_id)
        v.show_summary(experiment_id)
    finally:
        v.close()


def run_reporting(experiment_id: int):
    """Generate publication reports."""
    from reporting import ReportGenerator
    rg = ReportGenerator(experiment_id)
    try:
        rg.run_all()
    finally:
        rg.close()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Run LLM Annotation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--name', default='experiment',
                        help='Experiment name (for new runs)')
    parser.add_argument('--description', default='',
                        help='Experiment description')
    parser.add_argument('--experiment-id', type=int, default=None,
                        help='Resume from existing experiment ID')
    parser.add_argument('--chunk-limit', type=int, default=100,
                        help='Max chunks to annotate')
    parser.add_argument('--models', nargs='+', default=None,
                        help=f'Model keys to use (available: {list(MODELS_TO_TEST.keys())})')
    parser.add_argument('--temperatures', nargs='+', type=float, default=None,
                        help=f'Temperatures to test (default: {TEMPERATURES})')
    parser.add_argument('--num-runs', type=int, default=NUM_RUNS,
                        help=f'Runs per annotation (default: {NUM_RUNS})')
    parser.add_argument('--splits', nargs='+',
                        default=['development', 'reliability'],
                        help='Cohort splits to include')
    parser.add_argument('--chunking-method', default=DEFAULT_CHUNKING_METHOD,
                        help=f'Chunking method (default: {DEFAULT_CHUNKING_METHOD})')
    parser.add_argument('--skip-to', choices=STEPS, default=None,
                        help='Skip to a specific step')
    parser.add_argument('--stop-after', choices=STEPS, default=None,
                        help='Stop after a specific step')

    args = parser.parse_args()
    models = args.models or list(MODELS_TO_TEST.keys())
    temperatures = args.temperatures or TEMPERATURES

    logger.info("=" * 60)
    logger.info("MULTI-RUN LLM ANNOTATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Temperatures: {temperatures}")
    logger.info(f"Runs per annotation: {args.num_runs}")
    logger.info(f"Chunk limit: {args.chunk_limit}")
    logger.info(f"Splits: {args.splits}")

    # Determine which steps to run
    start_idx = STEPS.index(args.skip_to) if args.skip_to else 0
    stop_idx = STEPS.index(args.stop_after) + 1 if args.stop_after else len(STEPS)
    steps_to_run = STEPS[start_idx:stop_idx]

    experiment_id = args.experiment_id

    try:
        if 'schema' in steps_to_run:
            logger.info("\n[STEP 1/8] Applying database schema...")
            run_schema()

        if 'cohort' in steps_to_run:
            logger.info("\n[STEP 2/8] Selecting study cohort...")
            run_cohort()

        if 'chunking' in steps_to_run:
            logger.info("\n[STEP 3/8] Chunking transcripts...")
            run_chunking(args.chunking_method, args.splits)

        if 'annotate' in steps_to_run:
            logger.info("\n[STEP 4/8] Running multi-run annotation...")
            experiment_id = run_annotation(
                name=args.name,
                description=args.description,
                chunk_limit=args.chunk_limit,
                models=models,
                temperatures=temperatures,
                num_runs=args.num_runs,
                splits=args.splits,
            )
            logger.info(f"Experiment ID: {experiment_id}")

        if experiment_id is None:
            logger.error("No experiment_id available. Run annotation first or pass --experiment-id")
            sys.exit(1)

        if 'stability' in steps_to_run:
            logger.info("\n[STEP 5/8] Computing stability metrics...")
            run_stability(experiment_id)

        if 'final_labels' in steps_to_run:
            logger.info("\n[STEP 6/8] Deriving final annotations...")
            run_final_labels(experiment_id)

        if 'validation' in steps_to_run:
            logger.info("\n[STEP 7/8] Running validation against narrative_elements...")
            run_validation(experiment_id)

        if 'reporting' in steps_to_run:
            logger.info("\n[STEP 8/8] Generating reports...")
            run_reporting(experiment_id)

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE!")
        logger.info(f"Experiment ID: {experiment_id}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user. You can resume with:")
        if experiment_id:
            logger.warning(f"  python run_pipeline.py --experiment-id {experiment_id} --skip-to <step>")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        if experiment_id:
            logger.info(f"To resume: python run_pipeline.py --experiment-id {experiment_id} --skip-to <step>")
        sys.exit(1)


if __name__ == "__main__":
    main()
