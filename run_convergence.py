"""
Run convergence analysis: demonstrates value of multi-run (R=5) vs single-run.

Subsamples runs at R=1,2,3,4,5 and measures:
  - Stability rate at each R level
  - Whether the label at R=k matches the R=5 consensus
  - Cross-model agreement at each R level

Answers: "Why not just run once?"

Outputs:
  - run_convergence.csv  (metrics at each R level)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import logging
from pathlib import Path
from collections import defaultdict, Counter

from config import DB_CONFIG, STABILITY_THRESHOLDS, LABEL_VOCABULARIES
from stability import (
    compute_categorical_stability,
    compute_continuous_stability,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'


class RunConvergenceAnalyzer:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_all_runs(self):
        """Fetch all annotation runs grouped by chunk/construct/model/temperature."""
        query = """
            SELECT
                chunk_id, construct_name, model_name, temperature,
                run_number, label_kind, label_value_text, label_value_float, label_bin
            FROM llm_annotation_runs
            WHERE experiment_id = %s
              AND label_kind NOT IN ('error')
            ORDER BY chunk_id, construct_name, model_name, temperature, run_number
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        # Group: {(chunk_id, construct, model, temp): [runs ordered by run_number]}
        groups = defaultdict(list)
        for row in rows:
            key = (row['chunk_id'], row['construct_name'], row['model_name'], row['temperature'])
            groups[key].append(row)

        return groups

    def derive_label_at_r(self, runs, construct_name, r):
        """Given first r runs, derive the label using the same logic as the pipeline.

        Returns (label, is_stable) or (None, False) if insufficient data.
        """
        subset = runs[:r]
        construct_cfg = STABILITY_THRESHOLDS.get(construct_name, {})
        construct_type = construct_cfg.get('type', 'categorical')
        vocab = LABEL_VOCABULARIES.get(construct_name)

        if construct_type == 'continuous':
            values = [
                row['label_value_float'] for row in subset
                if row['label_kind'] == 'float' and row['label_value_float'] is not None
            ]
            if len(values) < 2:
                return None, False
            metrics = compute_continuous_stability(
                values,
                bins=vocab.get('bins', {}),
                max_range=construct_cfg.get('max_range', 0.2),
                max_stdev=construct_cfg.get('max_stdev', 0.10),
            )
            return metrics['modal_bin'], metrics['is_stable']
        else:
            labels = [
                row['label_value_text'] for row in subset
                if row['label_kind'] == 'category' and row['label_value_text'] is not None
            ]
            if len(labels) < 2:
                return labels[0] if labels else None, False
            threshold = construct_cfg.get('threshold', 0.8)
            metrics = compute_categorical_stability(labels, threshold)
            return metrics['modal_label'], metrics['is_stable']

    def run(self):
        """Compute convergence metrics at R=1..5."""
        logger.info(f"Run convergence analysis for experiment {self.experiment_id}")
        groups = self.get_all_runs()
        logger.info(f"Found {len(groups)} annotation groups")

        max_r = max(len(runs) for runs in groups.values()) if groups else 5
        r_levels = list(range(1, max_r + 1))

        results = []

        # Per-construct, per-model, per-temperature analysis
        # First get the R=max reference labels
        reference_labels = {}
        for key, runs in groups.items():
            label, is_stable = self.derive_label_at_r(runs, key[1], len(runs))
            reference_labels[key] = label

        # Now compute at each R level
        construct_model_temps = defaultdict(list)
        for key in groups:
            construct_model_temps[(key[1], key[2], key[3])].append(key)

        for (construct, model, temp), keys in construct_model_temps.items():
            for r in r_levels:
                stable_count = 0
                match_count = 0
                total = 0
                labels_at_r = []

                for key in keys:
                    runs = groups[key]
                    if len(runs) < r:
                        continue

                    total += 1
                    label_r, is_stable_r = self.derive_label_at_r(runs, construct, r)
                    ref_label = reference_labels.get(key)

                    if is_stable_r:
                        stable_count += 1
                    if label_r is not None and label_r == ref_label:
                        match_count += 1
                    if label_r is not None:
                        labels_at_r.append(label_r)

                if total == 0:
                    continue

                results.append({
                    'construct_name': construct,
                    'model_name': model,
                    'temperature': temp,
                    'r_level': r,
                    'total_chunks': total,
                    'stable_count': stable_count,
                    'stability_rate': round(stable_count / total, 4),
                    'match_to_full_r': match_count,
                    'match_rate': round(match_count / total, 4),
                    'unique_labels': len(set(labels_at_r)),
                })

        # Also compute aggregated (across models) per construct
        construct_temps = defaultdict(list)
        for r in results:
            construct_temps[(r['construct_name'], r['temperature'], r['r_level'])].append(r)

        aggregated = []
        for (construct, temp, r), rows in construct_temps.items():
            total = sum(r_row['total_chunks'] for r_row in rows)
            stable = sum(r_row['stable_count'] for r_row in rows)
            matched = sum(r_row['match_to_full_r'] for r_row in rows)

            aggregated.append({
                'construct_name': construct,
                'model_name': 'ALL_MODELS',
                'temperature': temp,
                'r_level': r,
                'total_chunks': total,
                'stable_count': stable,
                'stability_rate': round(stable / total, 4) if total else 0,
                'match_to_full_r': matched,
                'match_rate': round(matched / total, 4) if total else 0,
                'unique_labels': None,
            })

        all_results = results + aggregated

        # Write CSV
        filepath = self.output_dir / 'run_convergence.csv'
        if all_results:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
                writer.writeheader()
                writer.writerows(all_results)
            logger.info(f"Wrote run_convergence.csv ({len(all_results)} rows)")
        else:
            logger.warning("No convergence data to write")

        # Log summary
        logger.info("\n=== CONVERGENCE SUMMARY (pooled across models) ===")
        for row in aggregated:
            if row['model_name'] == 'ALL_MODELS' and row['temperature'] is None or row['temperature'] == 0.0:
                logger.info(
                    f"  {row['construct_name']:25s} R={row['r_level']}: "
                    f"stable={row['stability_rate']}, match={row['match_rate']}"
                )

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run convergence analysis (R=1 vs R=5)")
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    analyzer = RunConvergenceAnalyzer(args.experiment_id)
    try:
        analyzer.run()
    finally:
        analyzer.close()
