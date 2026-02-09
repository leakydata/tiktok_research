"""
Step 5: Multi-run annotation pipeline.
- Creates experiment records for reproducibility
- Uses a task queue with retry logic
- Canonical label parsing with normalization per construct
- Stores float and text labels separately
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import json
import time
import logging
from typing import Optional

from config import (
    DB_CONFIG, MODELS_TO_TEST, NUM_RUNS, TEMPERATURES,
    CONSTRUCTS, LABEL_VOCABULARIES, ANNOTATION_BATCH_SIZE,
    MAX_RETRIES, RETRY_BACKOFF_SECONDS, CHUNKING_CONFIGS,
    DEFAULT_CHUNKING_METHOD, DEFAULT_STABILITY_THRESHOLD,
)
from ollama_client import OllamaClient
from prompts import HealthLanguagePrompts
from label_parsing import normalize_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── Annotation Pipeline ───────────────────────────────────────────────────

class AnnotationPipeline:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.ollama = OllamaClient()

    def create_experiment(
        self,
        name: str,
        description: str = "",
        models: list[str] = None,
        temperatures: list[float] = None,
        num_runs: int = NUM_RUNS,
        chunk_limit: int = None,
    ) -> int:
        """Create an experiment record and return its ID."""
        models = models or list(MODELS_TO_TEST.keys())
        temperatures = temperatures or TEMPERATURES

        config_snapshot = {
            'models': models,
            'temperatures': temperatures,
            'num_runs': num_runs,
            'constructs': CONSTRUCTS,
            'stability_thresholds': {k: str(v) for k, v in {}. items()},
            'chunking_method': DEFAULT_CHUNKING_METHOD,
            'chunking_config': CHUNKING_CONFIGS.get(DEFAULT_CHUNKING_METHOD, {}),
            'chunk_limit': chunk_limit,
            'label_vocabularies': {k: str(v) for k, v in LABEL_VOCABULARIES.items()},
        }

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO experiment_runs (
                    experiment_name, description, models_tested,
                    temperatures_tested, num_runs, chunking_method,
                    chunking_version, prompt_version, status, config_snapshot
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'running', %s)
                RETURNING experiment_id
            """, (
                name, description, models, temperatures, num_runs,
                DEFAULT_CHUNKING_METHOD, 'v1', 'v1', json.dumps(config_snapshot),
            ))
            experiment_id = cur.fetchone()[0]
            self.conn.commit()
            logger.info(f"Created experiment {experiment_id}: {name}")
            return experiment_id

    def create_task_queue(
        self,
        experiment_id: int,
        chunk_limit: int = None,
        models: list[str] = None,
        temperatures: list[float] = None,
        num_runs: int = NUM_RUNS,
        splits: list[str] = None,
    ) -> int:
        """Populate the annotation_tasks queue for an experiment."""
        models = models or list(MODELS_TO_TEST.keys())
        temperatures = temperatures or TEMPERATURES
        splits = splits or ['development', 'reliability']

        # Get chunks
        split_placeholders = ','.join(['%s'] * len(splits))
        chunk_query = f"""
            SELECT chunk_id FROM annotation_chunks ac
            INNER JOIN study_cohort sc ON ac.creator_username = sc.creator_username
            WHERE sc.cohort_split IN ({split_placeholders})
            ORDER BY ac.chunk_id
        """
        params = list(splits)
        if chunk_limit:
            chunk_query += " LIMIT %s"
            params.append(chunk_limit)

        with self.conn.cursor() as cur:
            cur.execute(chunk_query, params)
            chunk_ids = [row[0] for row in cur.fetchall()]

        if not chunk_ids:
            logger.warning("No chunks found for task queue")
            return 0

        logger.info(
            f"Creating task queue: {len(chunk_ids)} chunks x {len(models)} models x "
            f"{len(CONSTRUCTS)} constructs x {len(temperatures)} temps x {num_runs} runs"
        )

        total_tasks = len(chunk_ids) * len(models) * len(CONSTRUCTS) * len(temperatures) * num_runs
        logger.info(f"Total tasks: {total_tasks:,}")

        # Batch insert tasks
        task_rows = []
        for chunk_id in chunk_ids:
            for model in models:
                for construct in CONSTRUCTS:
                    for temp in temperatures:
                        for run_num in range(1, num_runs + 1):
                            task_rows.append((
                                experiment_id, chunk_id, construct, model,
                                temp, 'single_label', run_num,
                            ))

            if len(task_rows) >= 5000:
                self._insert_tasks(task_rows)
                task_rows = []

        if task_rows:
            self._insert_tasks(task_rows)

        # Update experiment with chunk count
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE experiment_runs SET num_chunks_tested = %s WHERE experiment_id = %s",
                (len(chunk_ids), experiment_id)
            )
            self.conn.commit()

        logger.info(f"Task queue created: {total_tasks:,} tasks")
        return total_tasks

    def _insert_tasks(self, rows):
        query = """
        INSERT INTO annotation_tasks (
            experiment_id, chunk_id, construct_name, model_name,
            temperature, prompt_format, run_number
        ) VALUES %s
        ON CONFLICT DO NOTHING
        """
        with self.conn.cursor() as cur:
            execute_values(cur, query, rows)
            self.conn.commit()

    def run_tasks(self, experiment_id: int, batch_size: int = ANNOTATION_BATCH_SIZE):
        """Process pending tasks from the queue."""
        while True:
            # Fetch a batch of pending tasks
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        t.task_id, t.chunk_id, t.construct_name, t.model_name,
                        t.temperature, t.prompt_format, t.run_number,
                        ac.chunk_text, ac.context_carry_text
                    FROM annotation_tasks t
                    INNER JOIN annotation_chunks ac ON t.chunk_id = ac.chunk_id
                    WHERE t.experiment_id = %s
                      AND t.status = 'pending'
                    ORDER BY t.model_name, t.chunk_id, t.construct_name, t.run_number
                    LIMIT %s
                """, (experiment_id, batch_size))
                tasks = cur.fetchall()

            if not tasks:
                break

            # Group by model to minimize model swaps
            current_model = None
            results = []

            for task in tasks:
                # Mark as running
                self._update_task_status(task['task_id'], 'running')

                # Load model if changed
                if task['model_name'] != current_model:
                    current_model = task['model_name']
                    try:
                        self.ollama.ensure_model_loaded(current_model)
                    except Exception as e:
                        logger.error(f"Failed to load model {current_model}: {e}")
                        self._fail_task(task['task_id'], str(e))
                        continue

                # Run annotation
                try:
                    result = self._annotate_single(task, experiment_id)
                    if result:
                        results.append(result)
                        self._update_task_status(task['task_id'], 'completed')
                    else:
                        self._fail_task(task['task_id'], "No result returned")
                except Exception as e:
                    logger.error(f"Task {task['task_id']} failed: {e}")
                    self._retry_or_fail(task['task_id'], str(e))

            # Batch insert results
            if results:
                self._insert_results(results)

            # Progress report
            self._log_progress(experiment_id)

        # Mark experiment complete
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE experiment_runs
                SET status = 'completed', completed_at = NOW()
                WHERE experiment_id = %s
            """, (experiment_id,))
            self.conn.commit()

        logger.info(f"Experiment {experiment_id} complete!")

    def _annotate_single(self, task: dict, experiment_id: int) -> Optional[dict]:
        """Run a single annotation and return the result dict."""
        prompt_func = HealthLanguagePrompts.get_prompt_func(task['construct_name'])
        prompt = prompt_func(
            task['chunk_text'],
            context_carry=task.get('context_carry_text'),
        )

        result = self.ollama.generate(
            model_key=task['model_name'],
            prompt=prompt,
            temperature=task['temperature'],
            max_tokens=50,
        )

        parsed = normalize_label(result['response'], task['construct_name'])

        return {
            'task_id': task['task_id'],
            'experiment_id': experiment_id,
            'chunk_id': task['chunk_id'],
            'construct_name': task['construct_name'],
            'run_number': task['run_number'],
            'temperature': task['temperature'],
            'prompt_format': task['prompt_format'],
            'model_name': task['model_name'],
            'model_family': result['model_family'],
            'model_size_b': result['model_size_b'],
            'label_kind': parsed['label_kind'],
            'label_value_text': parsed['label_value_text'],
            'label_value_float': parsed['label_value_float'],
            'label_bin': parsed['label_bin'],
            'raw_response': result['response'],
            'processing_time_ms': result['processing_time_ms'],
            'tokens_generated': result['tokens_generated'],
            'inference_params': json.dumps(result['inference_params']),
            'ollama_version': result['ollama_version'],
            'quantization': result['quantization'],
        }

    def _insert_results(self, results: list[dict]):
        query = """
        INSERT INTO llm_annotation_runs (
            task_id, experiment_id, chunk_id, construct_name,
            run_number, temperature, prompt_format,
            model_name, model_family, model_size_b,
            label_kind, label_value_text, label_value_float, label_bin,
            raw_response, processing_time_ms, tokens_generated,
            inference_params, ollama_version, quantization
        ) VALUES %s
        ON CONFLICT DO NOTHING
        """
        values = [
            (
                r['task_id'], r['experiment_id'], r['chunk_id'], r['construct_name'],
                r['run_number'], r['temperature'], r['prompt_format'],
                r['model_name'], r['model_family'], r['model_size_b'],
                r['label_kind'], r['label_value_text'], r['label_value_float'], r['label_bin'],
                r['raw_response'], r['processing_time_ms'], r['tokens_generated'],
                r['inference_params'], r['ollama_version'], r['quantization'],
            )
            for r in results
        ]
        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
            self.conn.commit()
        logger.debug(f"Inserted {len(values)} annotation results")

    def _update_task_status(self, task_id: int, status: str):
        ts_col = 'started_at' if status == 'running' else 'completed_at'
        with self.conn.cursor() as cur:
            cur.execute(
                f"UPDATE annotation_tasks SET status = %s, {ts_col} = NOW() WHERE task_id = %s",
                (status, task_id)
            )
            self.conn.commit()

    def _fail_task(self, task_id: int, error: str):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE annotation_tasks SET status = 'failed', last_error = %s WHERE task_id = %s",
                (error[:500], task_id)
            )
            self.conn.commit()

    def _retry_or_fail(self, task_id: int, error: str):
        with self.conn.cursor() as cur:
            cur.execute("SELECT retries FROM annotation_tasks WHERE task_id = %s", (task_id,))
            retries = cur.fetchone()[0]
            if retries < MAX_RETRIES:
                cur.execute(
                    "UPDATE annotation_tasks SET status = 'pending', retries = retries + 1, "
                    "last_error = %s WHERE task_id = %s",
                    (error[:500], task_id)
                )
                self.conn.commit()
                time.sleep(RETRY_BACKOFF_SECONDS * (retries + 1))
            else:
                self._fail_task(task_id, f"Max retries exceeded. Last error: {error[:400]}")

    def _log_progress(self, experiment_id: int):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT status, COUNT(*) AS cnt
                FROM annotation_tasks
                WHERE experiment_id = %s
                GROUP BY status
            """, (experiment_id,))
            counts = {row['status']: row['cnt'] for row in cur.fetchall()}

        total = sum(counts.values())
        done = counts.get('completed', 0)
        failed = counts.get('failed', 0)
        pending = counts.get('pending', 0)
        pct = (done / total * 100) if total else 0
        logger.info(
            f"Progress: {done:,}/{total:,} ({pct:.1f}%) | "
            f"pending={pending:,} failed={failed:,}"
        )

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-run LLM annotation")
    parser.add_argument('--name', required=True, help='Experiment name')
    parser.add_argument('--description', default='', help='Experiment description')
    parser.add_argument('--chunk-limit', type=int, default=100, help='Max chunks to annotate')
    parser.add_argument('--models', nargs='+', default=None, help='Model keys to use')
    parser.add_argument('--temperatures', nargs='+', type=float, default=None)
    parser.add_argument('--num-runs', type=int, default=NUM_RUNS)
    parser.add_argument('--splits', nargs='+', default=['development', 'reliability'])
    args = parser.parse_args()

    pipeline = AnnotationPipeline()
    try:
        exp_id = pipeline.create_experiment(
            name=args.name,
            description=args.description,
            models=args.models,
            temperatures=args.temperatures,
            num_runs=args.num_runs,
            chunk_limit=args.chunk_limit,
        )
        pipeline.create_task_queue(
            experiment_id=exp_id,
            chunk_limit=args.chunk_limit,
            models=args.models,
            temperatures=args.temperatures,
            num_runs=args.num_runs,
            splits=args.splits,
        )
        pipeline.run_tasks(exp_id)
    finally:
        pipeline.close()
