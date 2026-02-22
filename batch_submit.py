"""
Batch processing for cloud LLM annotation — 50% cost savings.

Supports Anthropic Message Batches API and OpenAI Batch API.
Both offer 50% off standard pricing with a 24-hour completion window.

Usage:
  # Submit a batch for Haiku on experiment 14
  python batch_submit.py submit --experiment-id 14 --model claude-haiku-4.5

  # Check status of a submitted batch
  python batch_submit.py status --batch-id msgbatch_01abc123

  # Collect results and write to DB
  python batch_submit.py collect --experiment-id 14 --batch-id msgbatch_01abc123

  # All-in-one: submit, wait, collect
  python batch_submit.py run --experiment-id 14 --model claude-haiku-4.5
"""

import argparse
import json
import time
import logging
import tempfile
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from config import (
    DB_CONFIG, MODELS_TO_TEST, BACKEND_CONFIGS,
    CONSTRUCTS, DEFAULT_MAX_TOKENS,
)
from prompts import HealthLanguagePrompts
from label_parsing import normalize_label

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STATE_DIR = Path(__file__).parent / 'outputs' / 'batch_state'
STATE_DIR.mkdir(parents=True, exist_ok=True)


class BatchProcessor:
    def __init__(self, experiment_id: int, model_key: str):
        self.experiment_id = experiment_id
        self.model_key = model_key
        self.model_cfg = MODELS_TO_TEST[model_key]
        self.backend = self.model_cfg['backend']
        self.conn = psycopg2.connect(**DB_CONFIG)

    def get_pending_tasks(self):
        """Fetch all pending tasks for this model+experiment."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT at.task_id, at.chunk_id, at.construct_name,
                       at.model_name, at.temperature, at.run_number,
                       at.prompt_format,
                       ac.chunk_text, ac.context_carry_text
                FROM annotation_tasks at
                INNER JOIN annotation_chunks ac ON at.chunk_id = ac.chunk_id
                WHERE at.experiment_id = %s
                  AND at.model_name = %s
                  AND at.status = 'pending'
                ORDER BY at.task_id
            """, (self.experiment_id, self.model_key))
            return cur.fetchall()

    def build_prompt(self, task):
        """Build the annotation prompt for a task."""
        prompt_func = HealthLanguagePrompts.get_prompt_func(task['construct_name'])
        return prompt_func(
            task['chunk_text'],
            context_carry=task.get('context_carry_text'),
        )

    # ── Anthropic Batch ──────────────────────────────────────────────────

    def submit_anthropic(self, tasks):
        """Submit batch via Anthropic Message Batches API. Returns batch_id."""
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        client = anthropic.Anthropic(api_key=BACKEND_CONFIGS['anthropic']['api_key'])
        api_model = self.model_cfg['api_model_name']

        requests = []
        for task in tasks:
            prompt = self.build_prompt(task)
            temp = task['temperature']

            params = MessageCreateParamsNonStreaming(
                model=api_model,
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=temp,
                messages=[{'role': 'user', 'content': prompt}],
            )
            requests.append(Request(
                custom_id=str(task['task_id']),
                params=params,
            ))

        logger.info(f"Submitting {len(requests)} requests to Anthropic Batch API...")
        batch = client.messages.batches.create(requests=requests)
        logger.info(f"Batch created: {batch.id} (status: {batch.processing_status})")

        # Save mapping for later collection
        task_map = {str(t['task_id']): dict(t) for t in tasks}
        state_file = STATE_DIR / f'{batch.id}.json'
        with open(state_file, 'w') as f:
            json.dump({
                'batch_id': batch.id,
                'backend': 'anthropic',
                'experiment_id': self.experiment_id,
                'model_key': self.model_key,
                'task_count': len(tasks),
                'task_ids': list(task_map.keys()),
            }, f, indent=2)

        return batch.id

    def poll_anthropic(self, batch_id):
        """Poll until batch completes. Returns final batch object."""
        import anthropic
        client = anthropic.Anthropic(api_key=BACKEND_CONFIGS['anthropic']['api_key'])

        while True:
            batch = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
            done = counts.succeeded + counts.errored + counts.canceled + counts.expired
            logger.info(
                f"  [{batch_id}] {done}/{total} done "
                f"(succeeded={counts.succeeded}, errored={counts.errored}, "
                f"expired={counts.expired}) — status: {batch.processing_status}"
            )
            if batch.processing_status == 'ended':
                return batch
            time.sleep(60)

    def collect_anthropic(self, batch_id):
        """Retrieve results and insert into DB."""
        import anthropic
        client = anthropic.Anthropic(api_key=BACKEND_CONFIGS['anthropic']['api_key'])

        # Reload task data from DB for parsing
        task_map = {}
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT at.task_id, at.chunk_id, at.construct_name,
                       at.model_name, at.temperature, at.run_number,
                       at.prompt_format
                FROM annotation_tasks at
                WHERE at.experiment_id = %s AND at.model_name = %s
            """, (self.experiment_id, self.model_key))
            for row in cur.fetchall():
                task_map[str(row['task_id'])] = dict(row)

        results = []
        counts = {'succeeded': 0, 'errored': 0, 'canceled': 0, 'expired': 0}

        for result in client.messages.batches.results(batch_id):
            task_id = result.custom_id
            task = task_map.get(task_id)
            if not task:
                logger.warning(f"Unknown custom_id: {task_id}")
                continue

            result_type = result.result.type
            counts[result_type] = counts.get(result_type, 0) + 1

            if result_type == 'succeeded':
                msg = result.result.message
                text = msg.content[0].text if msg.content else ''
                tokens = msg.usage.output_tokens if msg.usage else 0

                parsed = normalize_label(text.strip(), task['construct_name'])
                results.append({
                    'task_id': int(task_id),
                    'experiment_id': self.experiment_id,
                    'chunk_id': task['chunk_id'],
                    'construct_name': task['construct_name'],
                    'run_number': task['run_number'],
                    'temperature': task['temperature'],
                    'prompt_format': task['prompt_format'],
                    'model_name': task['model_name'],
                    'model_family': self.model_cfg['family'],
                    'model_size_b': self.model_cfg['size_b'],
                    'label_kind': parsed['label_kind'],
                    'label_value_text': parsed['label_value_text'],
                    'label_value_float': parsed['label_value_float'],
                    'label_bin': parsed['label_bin'],
                    'raw_response': text.strip(),
                    'processing_time_ms': 0,  # batch doesn't report per-request timing
                    'tokens_generated': tokens,
                    'inference_params': json.dumps({
                        'temperature': task['temperature'],
                        'max_tokens': DEFAULT_MAX_TOKENS,
                        'batch_id': batch_id,
                    }),
                    'ollama_version': 'anthropic/batch',
                    'quantization': 'none',
                })
            elif result_type == 'errored':
                # Docs: result.result.error.type is 'invalid_request' or server error
                error = getattr(result.result, 'error', None)
                if error:
                    err_obj = getattr(error, 'error', error)  # ErrorResponse wraps .error
                    error_msg = getattr(err_obj, 'message', str(err_obj))
                    error_type = getattr(err_obj, 'type', 'unknown')
                else:
                    error_msg = 'no detail'
                    error_type = 'unknown'
                if counts['errored'] <= 5:
                    logger.warning(f"Task {task_id} {error_type}: {error_msg}")
                elif counts['errored'] == 6:
                    logger.warning("(suppressing further error details...)")
            elif result_type == 'expired':
                if counts['expired'] <= 3:
                    logger.warning(f"Task {task_id} expired (batch hit 24h limit)")
            # 'canceled' — user-initiated, no action needed

        # Insert into DB
        if results:
            self._insert_results(results)
            self._mark_tasks_completed([r['task_id'] for r in results])

        logger.info(
            f"Collected: {counts['succeeded']} succeeded, {counts['errored']} errored, "
            f"{counts['expired']} expired, {counts['canceled']} canceled"
        )
        return counts['succeeded']

    # ── OpenAI Batch ─────────────────────────────────────────────────────

    def submit_openai(self, tasks):
        """Submit batch via OpenAI Batch API (JSONL upload). Returns batch_id."""
        import openai
        client = openai.OpenAI(
            api_key=BACKEND_CONFIGS['openai']['api_key'],
            base_url=BACKEND_CONFIGS['openai']['base_url'],
        )
        api_model = self.model_cfg['api_model_name']
        is_fixed_temp = self.model_cfg.get('fixed_temperature', False)

        # Build JSONL
        lines = []
        for task in tasks:
            prompt = self.build_prompt(task)
            body = {
                'model': api_model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_completion_tokens': DEFAULT_MAX_TOKENS,
            }
            if not is_fixed_temp:
                body['temperature'] = task['temperature']
                body['top_p'] = 0.9
            if task['temperature'] == 0.0:
                body['seed'] = 42

            lines.append(json.dumps({
                'custom_id': str(task['task_id']),
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': body,
            }))

        # Write to temp file and upload
        jsonl_content = '\n'.join(lines)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write(jsonl_content)
            jsonl_path = f.name

        logger.info(f"Uploading {len(lines)} requests as JSONL ({len(jsonl_content)} bytes)...")
        with open(jsonl_path, 'rb') as f:
            file_obj = client.files.create(file=f, purpose='batch')

        logger.info(f"File uploaded: {file_obj.id}")

        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint='/v1/chat/completions',
            completion_window='24h',
        )
        logger.info(f"Batch created: {batch.id} (status: {batch.status})")

        # Save state
        state_file = STATE_DIR / f'{batch.id}.json'
        with open(state_file, 'w') as f_out:
            json.dump({
                'batch_id': batch.id,
                'backend': 'openai',
                'experiment_id': self.experiment_id,
                'model_key': self.model_key,
                'task_count': len(tasks),
                'input_file_id': file_obj.id,
            }, f_out, indent=2)

        # Clean up temp file
        Path(jsonl_path).unlink(missing_ok=True)

        return batch.id

    def poll_openai(self, batch_id):
        """Poll until OpenAI batch completes."""
        import openai
        client = openai.OpenAI(
            api_key=BACKEND_CONFIGS['openai']['api_key'],
            base_url=BACKEND_CONFIGS['openai']['base_url'],
        )

        while True:
            batch = client.batches.retrieve(batch_id)
            counts = batch.request_counts
            logger.info(
                f"  [{batch_id}] completed={counts.completed}/{counts.total} "
                f"failed={counts.failed} — status: {batch.status}"
            )
            if batch.status in ('completed', 'failed', 'expired', 'cancelled'):
                return batch
            time.sleep(60)

    def collect_openai(self, batch_id):
        """Retrieve OpenAI batch results and insert into DB."""
        import openai
        client = openai.OpenAI(
            api_key=BACKEND_CONFIGS['openai']['api_key'],
            base_url=BACKEND_CONFIGS['openai']['base_url'],
        )

        batch = client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            logger.error(f"No output file for batch {batch_id} (status: {batch.status})")
            return 0

        # Download results
        content = client.files.content(batch.output_file_id)
        result_lines = content.text.strip().split('\n')

        # Load task map
        task_map = {}
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT task_id, chunk_id, construct_name, model_name,
                       temperature, run_number, prompt_format
                FROM annotation_tasks
                WHERE experiment_id = %s AND model_name = %s
            """, (self.experiment_id, self.model_key))
            for row in cur.fetchall():
                task_map[str(row['task_id'])] = dict(row)

        results = []
        succeeded = 0
        errored = 0

        for line in result_lines:
            entry = json.loads(line)
            task_id = entry['custom_id']
            task = task_map.get(task_id)
            if not task:
                continue

            resp = entry.get('response', {})
            if resp.get('status_code') == 200:
                body = resp.get('body', {})
                choices = body.get('choices', [])
                text = choices[0]['message']['content'] if choices else ''
                usage = body.get('usage', {})
                tokens = usage.get('completion_tokens', 0)

                parsed = normalize_label(text.strip(), task['construct_name'])
                results.append({
                    'task_id': int(task_id),
                    'experiment_id': self.experiment_id,
                    'chunk_id': task['chunk_id'],
                    'construct_name': task['construct_name'],
                    'run_number': task['run_number'],
                    'temperature': task['temperature'],
                    'prompt_format': task['prompt_format'],
                    'model_name': task['model_name'],
                    'model_family': self.model_cfg['family'],
                    'model_size_b': self.model_cfg['size_b'],
                    'label_kind': parsed['label_kind'],
                    'label_value_text': parsed['label_value_text'],
                    'label_value_float': parsed['label_value_float'],
                    'label_bin': parsed['label_bin'],
                    'raw_response': text.strip(),
                    'processing_time_ms': 0,
                    'tokens_generated': tokens,
                    'inference_params': json.dumps({
                        'temperature': task['temperature'],
                        'top_p': 0.9,
                        'max_tokens': DEFAULT_MAX_TOKENS,
                        'batch_id': batch_id,
                    }),
                    'ollama_version': 'openai/batch',
                    'quantization': 'none',
                })
                succeeded += 1
            else:
                errored += 1
                logger.warning(f"Task {task_id} failed: {resp.get('status_code')}")

        if results:
            self._insert_results(results)
            self._mark_tasks_completed([r['task_id'] for r in results])

        logger.info(f"Collected: {succeeded} succeeded, {errored} errored")
        return succeeded

    # ── Shared DB operations ─────────────────────────────────────────────

    def _insert_results(self, results):
        """Insert annotation results into llm_annotation_runs."""
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
        logger.info(f"  Inserted {len(values)} annotation results")

    def _mark_tasks_completed(self, task_ids):
        """Mark tasks as completed in annotation_tasks."""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE annotation_tasks
                SET status = 'completed', completed_at = NOW()
                WHERE task_id = ANY(%s)
            """, (task_ids,))
            self.conn.commit()
        logger.info(f"  Marked {len(task_ids)} tasks as completed")

    # ── Dispatch ─────────────────────────────────────────────────────────

    def submit(self, batch_size: int = 1000):
        """Submit pending tasks in batches. Returns list of batch_ids."""
        tasks = self.get_pending_tasks()
        if not tasks:
            logger.info("No pending tasks found.")
            return []

        logger.info(f"Found {len(tasks)} pending tasks for {self.model_key}")

        if self.backend not in ('anthropic', 'openai'):
            raise ValueError(
                f"Batch processing not supported for backend '{self.backend}'. "
                f"Only 'anthropic' and 'openai' offer batch APIs (50% discount). "
                f"For DeepSeek/MiniMax, use 'python annotate.py --resume' instead."
            )

        # Split into smaller batches for recoverability
        chunks = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        logger.info(f"Splitting into {len(chunks)} batch(es) of up to {batch_size} tasks")

        batch_ids = []
        submit_fn = self.submit_anthropic if self.backend == 'anthropic' else self.submit_openai
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Submitting batch {i}/{len(chunks)} ({len(chunk)} tasks)...")
            bid = submit_fn(chunk)
            batch_ids.append(bid)
            logger.info(f"  Batch {i} submitted: {bid}")

        return batch_ids

    def poll(self, batch_id):
        """Poll batch until done."""
        if self.backend == 'anthropic':
            return self.poll_anthropic(batch_id)
        elif self.backend == 'openai':
            return self.poll_openai(batch_id)

    def collect(self, batch_id):
        """Collect results and write to DB."""
        if self.backend == 'anthropic':
            return self.collect_anthropic(batch_id)
        elif self.backend == 'openai':
            return self.collect_openai(batch_id)

    def run(self, batch_size: int = 1000):
        """Submit, poll, collect — all in one."""
        batch_ids = self.submit(batch_size=batch_size)
        if not batch_ids:
            return

        total_count = 0
        for i, batch_id in enumerate(batch_ids, 1):
            logger.info(f"Waiting for batch {i}/{len(batch_ids)} ({batch_id}) to complete (up to 24h)...")
            self.poll(batch_id)

            logger.info(f"Collecting results for batch {i}/{len(batch_ids)}...")
            count = self.collect(batch_id)
            total_count += count

        logger.info(f"All done! {total_count} total results written to DB.")

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Batch processing for cloud LLM annotation (50% cost savings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported backends (50% batch discount):
  - anthropic: Claude Haiku 4.5, Sonnet, etc.
  - openai:    GPT-5-nano, GPT-4o, etc.

Not supported (use 'python annotate.py --resume' instead):
  - deepseek, minimax: No batch API available
        """,
    )

    sub = parser.add_subparsers(dest='command', required=True)

    # Submit
    p_submit = sub.add_parser('submit', help='Submit pending tasks as a batch')
    p_submit.add_argument('--experiment-id', type=int, required=True)
    p_submit.add_argument('--model', required=True, help='Model key (e.g., claude-haiku-4.5)')
    p_submit.add_argument('--batch-size', type=int, default=1000,
                          help='Max tasks per batch (default: 1000)')

    # Status
    p_status = sub.add_parser('status', help='Check batch status')
    p_status.add_argument('--batch-id', required=True)
    p_status.add_argument('--model', required=True)

    # Collect
    p_collect = sub.add_parser('collect', help='Collect results and write to DB')
    p_collect.add_argument('--experiment-id', type=int, required=True)
    p_collect.add_argument('--model', required=True)
    p_collect.add_argument('--batch-id', required=True)

    # Run (all-in-one)
    p_run = sub.add_parser('run', help='Submit, wait, and collect (all-in-one)')
    p_run.add_argument('--experiment-id', type=int, required=True)
    p_run.add_argument('--model', required=True)
    p_run.add_argument('--batch-size', type=int, default=1000,
                        help='Max tasks per batch (default: 1000)')

    args = parser.parse_args()

    if args.command == 'submit':
        proc = BatchProcessor(args.experiment_id, args.model)
        try:
            batch_ids = proc.submit(batch_size=args.batch_size)
            for bid in batch_ids:
                print(f"\nBatch submitted: {bid}")
                print(f"  Check status:  python batch_submit.py status --batch-id {bid} --model {args.model}")
                print(f"  Collect later: python batch_submit.py collect --experiment-id {args.experiment_id} --model {args.model} --batch-id {bid}")
        finally:
            proc.close()

    elif args.command == 'status':
        proc = BatchProcessor(1, args.model)  # experiment_id not needed for status
        try:
            proc.poll(args.batch_id)
        except KeyboardInterrupt:
            print("\nStopped polling (batch continues processing).")
        finally:
            proc.close()

    elif args.command == 'collect':
        proc = BatchProcessor(args.experiment_id, args.model)
        try:
            proc.collect(args.batch_id)
        finally:
            proc.close()

    elif args.command == 'run':
        proc = BatchProcessor(args.experiment_id, args.model)
        try:
            proc.run(batch_size=args.batch_size)
        except KeyboardInterrupt:
            print("\nInterrupted. Batch(es) continue server-side.")
            print(f"Check state files in: {STATE_DIR}")
            print(f"Resume with: python batch_submit.py status --batch-id <id> --model {args.model}")
        finally:
            proc.close()


if __name__ == '__main__':
    main()
