"""
Step 8: Derive final annotations from multi-run stability analysis.
- Categorical: modal label (if stable)
- Continuous: median value (if stable)
- Unstable chunks marked but still stored for analysis
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import logging

from config import DB_CONFIG, STABILITY_THRESHOLDS, LABEL_VOCABULARIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DECISION_RULE_VERSION = 'v1'


class FinalLabelDeriver:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)

    def run(self, experiment_id: int):
        """Derive final annotations from stability metrics."""
        logger.info(f"Deriving final annotations for experiment {experiment_id}...")

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    experiment_id, chunk_id, construct_name,
                    model_name, temperature, prompt_format,
                    construct_type, is_stable,
                    modal_label, modal_count, agreement_ratio,
                    mean_value, median_value, stdev_value, modal_bin,
                    valid_run_count
                FROM annotation_stability_metrics
                WHERE experiment_id = %s
            """, (experiment_id,))
            rows = cur.fetchall()

        if not rows:
            logger.warning("No stability metrics found")
            return

        logger.info(f"Processing {len(rows)} stability records")

        batch = []
        for row in rows:
            final = self._derive_one(row)
            batch.append(final)

            if len(batch) >= 1000:
                self._insert_batch(batch)
                batch = []

        if batch:
            self._insert_batch(batch)

        # Summary
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    construct_name,
                    model_name,
                    COUNT(*) AS total,
                    SUM(CASE WHEN is_stable THEN 1 ELSE 0 END) AS stable,
                    ROUND(
                        SUM(CASE WHEN is_stable THEN 1 ELSE 0 END)::numeric / COUNT(*), 3
                    ) AS stable_rate
                FROM final_annotations
                WHERE experiment_id = %s
                GROUP BY construct_name, model_name
                ORDER BY construct_name, stable_rate DESC
            """, (experiment_id,))
            summary = cur.fetchall()

        logger.info("=== FINAL ANNOTATION SUMMARY ===")
        for s in summary:
            logger.info(
                f"  {s['construct_name']:25s} | {s['model_name']:15s}: "
                f"{s['stable']}/{s['total']} stable ({s['stable_rate']})"
            )

        logger.info("Final label derivation complete!")

    def _derive_one(self, row: dict) -> tuple:
        """Derive the final label for one chunk/construct/condition."""
        construct_type = row['construct_type']

        if not row['is_stable']:
            # Unstable: record but mark as excluded
            return (
                row['experiment_id'], row['chunk_id'], row['construct_name'],
                row['model_name'], row['temperature'], row['prompt_format'],
                row.get('modal_label'),           # best guess text
                row.get('median_value'),           # best guess float
                row.get('modal_bin'),              # best guess bin
                'excluded_unstable',
                DECISION_RULE_VERSION,
                False,
                row.get('agreement_ratio'),
                row.get('stdev_value'),
                row.get('valid_run_count'),
            )

        if construct_type == 'continuous':
            return (
                row['experiment_id'], row['chunk_id'], row['construct_name'],
                row['model_name'], row['temperature'], row['prompt_format'],
                None,                              # no text label for continuous
                row['median_value'],               # median as final value
                row['modal_bin'],
                'median_value',
                DECISION_RULE_VERSION,
                True,
                row.get('agreement_ratio'),
                row.get('stdev_value'),
                row.get('valid_run_count'),
            )
        else:
            return (
                row['experiment_id'], row['chunk_id'], row['construct_name'],
                row['model_name'], row['temperature'], row['prompt_format'],
                row['modal_label'],                # modal label as final
                None,
                None,
                'modal_label',
                DECISION_RULE_VERSION,
                True,
                row.get('agreement_ratio'),
                None,
                row.get('valid_run_count'),
            )

    def _insert_batch(self, batch: list[tuple]):
        query = """
        INSERT INTO final_annotations (
            experiment_id, chunk_id, construct_name,
            model_name, temperature, prompt_format,
            final_label_text, final_label_float, final_label_bin,
            decision_rule, decision_rule_version,
            is_stable, agreement_ratio, stdev_value, num_valid_runs
        ) VALUES %s
        ON CONFLICT (experiment_id, chunk_id, construct_name, model_name, temperature, prompt_format)
        DO UPDATE SET
            final_label_text = EXCLUDED.final_label_text,
            final_label_float = EXCLUDED.final_label_float,
            is_stable = EXCLUDED.is_stable,
            created_at = NOW()
        """
        with self.conn.cursor() as cur:
            execute_values(cur, query, batch)
            self.conn.commit()

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    deriver = FinalLabelDeriver()
    try:
        deriver.run(args.experiment_id)
    finally:
        deriver.close()
