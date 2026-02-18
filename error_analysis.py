"""
Error analysis: confusion matrices and systematic error patterns.

Analyzes:
  1. Within-model instability: when a model gives different labels across runs,
     which label pairs co-occur?
  2. Cross-model disagreement: when models disagree, which labels get confused?
  3. Error by chunk characteristics: stability vs chunk length, content type, etc.

Outputs:
  - confusion_within_model.csv
  - confusion_cross_model.csv
  - error_by_chunk_length.csv
  - error_by_content_type.csv
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'


class ErrorAnalyzer:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def within_model_confusion(self):
        """For unstable chunks, analyze which label pairs co-occur across runs."""
        query = """
            SELECT
                asm.construct_name, asm.model_name,
                asm.is_stable, asm.labels_observed, asm.label_distribution
            FROM annotation_stability_metrics asm
            WHERE asm.experiment_id = %s
              AND asm.is_stable = FALSE
              AND asm.labels_observed IS NOT NULL
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        # Count label pair co-occurrences for unstable chunks
        pair_counts = defaultdict(Counter)  # {(construct, model): Counter of (labelA, labelB)}

        for row in rows:
            labels = json.loads(row['labels_observed']) if isinstance(row['labels_observed'], str) else row['labels_observed']
            if not labels or len(labels) < 2:
                continue

            unique_labels = set(labels)
            if len(unique_labels) < 2:
                continue

            key = (row['construct_name'], row['model_name'])
            for a, b in combinations(sorted(unique_labels), 2):
                pair_counts[key][(a, b)] += 1

        results = []
        for (construct, model), counter in pair_counts.items():
            for (label_a, label_b), count in counter.most_common():
                results.append({
                    'construct_name': construct,
                    'model_name': model,
                    'label_a': label_a,
                    'label_b': label_b,
                    'co_occurrence_count': count,
                })

        self._write_csv('confusion_within_model.csv', results)
        logger.info(f"Within-model confusion: {len(results)} label pairs")
        return results

    def cross_model_confusion(self):
        """When models disagree on a chunk, which labels do they assign?"""
        query = """
            SELECT
                fa.chunk_id, fa.construct_name, fa.model_name,
                fa.final_label_text, fa.final_label_bin,
                asm.construct_type
            FROM final_annotations fa
            INNER JOIN annotation_stability_metrics asm
                ON fa.experiment_id = asm.experiment_id
                AND fa.chunk_id = asm.chunk_id
                AND fa.construct_name = asm.construct_name
                AND fa.model_name = asm.model_name
                AND fa.temperature = asm.temperature
            WHERE fa.experiment_id = %s
              AND fa.is_stable = TRUE
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        # Group by (chunk_id, construct_name)
        groups = defaultdict(dict)
        for row in rows:
            key = (row['chunk_id'], row['construct_name'])
            label = row['final_label_text'] if row['construct_type'] == 'categorical' else row['final_label_bin']
            if label:
                groups[key][row['model_name']] = label

        # For each pair of models, count agreement/disagreement
        pair_confusion = defaultdict(Counter)  # {(construct, modelA, modelB): Counter of (labelA, labelB)}

        for (chunk_id, construct), model_labels in groups.items():
            models = sorted(model_labels.keys())
            for m1, m2 in combinations(models, 2):
                l1 = model_labels[m1]
                l2 = model_labels[m2]
                if l1 != l2:
                    pair_key = (construct, m1, m2)
                    pair_confusion[pair_key][(l1, l2)] += 1

        results = []
        for (construct, m1, m2), counter in pair_confusion.items():
            total_disagreements = sum(counter.values())
            for (label_a, label_b), count in counter.most_common():
                results.append({
                    'construct_name': construct,
                    'model_a': m1,
                    'model_b': m2,
                    'label_model_a': label_a,
                    'label_model_b': label_b,
                    'count': count,
                    'proportion_of_disagreements': round(count / total_disagreements, 4),
                })

        self._write_csv('confusion_cross_model.csv', results)
        logger.info(f"Cross-model confusion: {len(results)} disagreement patterns")
        return results

    def error_by_chunk_length(self):
        """Stability rate by chunk length quartiles."""
        query = """
            SELECT
                ac.char_length,
                asm.construct_name, asm.model_name,
                asm.is_stable, asm.agreement_ratio
            FROM annotation_stability_metrics asm
            INNER JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
            WHERE asm.experiment_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        if not rows:
            return []

        # Compute quartile boundaries
        lengths = sorted(set(r['char_length'] for r in rows if r['char_length']))
        if not lengths:
            return []

        import numpy as np
        q25, q50, q75 = np.percentile(lengths, [25, 50, 75])

        def length_bin(l):
            if l <= q25:
                return f'Q1 (<=  {int(q25)})'
            elif l <= q50:
                return f'Q2 ({int(q25)+1}-{int(q50)})'
            elif l <= q75:
                return f'Q3 ({int(q50)+1}-{int(q75)})'
            else:
                return f'Q4 (>{int(q75)})'

        # Aggregate by length bin x construct
        bins = defaultdict(lambda: {'stable': 0, 'total': 0, 'agreement_sum': 0})
        for r in rows:
            if r['char_length'] is None:
                continue
            lb = length_bin(r['char_length'])
            key = (lb, r['construct_name'])
            bins[key]['total'] += 1
            if r['is_stable']:
                bins[key]['stable'] += 1
            if r['agreement_ratio'] is not None:
                bins[key]['agreement_sum'] += r['agreement_ratio']

        results = []
        for (lb, construct), vals in sorted(bins.items()):
            results.append({
                'length_bin': lb,
                'construct_name': construct,
                'total_chunks': vals['total'],
                'stable_chunks': vals['stable'],
                'stability_rate': round(vals['stable'] / vals['total'], 4) if vals['total'] else 0,
                'mean_agreement': round(vals['agreement_sum'] / vals['total'], 4) if vals['total'] else 0,
            })

        self._write_csv('error_by_chunk_length.csv', results)
        logger.info(f"Error by chunk length: {len(results)} rows")
        return results

    def error_by_content_type(self):
        """Stability rate by content type."""
        query = """
            SELECT
                ac.content_type,
                asm.construct_name, asm.model_name,
                asm.is_stable, asm.agreement_ratio
            FROM annotation_stability_metrics asm
            INNER JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
            WHERE asm.experiment_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        bins = defaultdict(lambda: {'stable': 0, 'total': 0, 'agreement_sum': 0})
        for r in rows:
            ct = r['content_type'] or 'unknown'
            key = (ct, r['construct_name'])
            bins[key]['total'] += 1
            if r['is_stable']:
                bins[key]['stable'] += 1
            if r['agreement_ratio'] is not None:
                bins[key]['agreement_sum'] += r['agreement_ratio']

        results = []
        for (ct, construct), vals in sorted(bins.items()):
            results.append({
                'content_type': ct,
                'construct_name': construct,
                'total_chunks': vals['total'],
                'stable_chunks': vals['stable'],
                'stability_rate': round(vals['stable'] / vals['total'], 4) if vals['total'] else 0,
                'mean_agreement': round(vals['agreement_sum'] / vals['total'], 4) if vals['total'] else 0,
            })

        self._write_csv('error_by_content_type.csv', results)
        logger.info(f"Error by content type: {len(results)} rows")
        return results

    def run(self):
        """Run full error analysis."""
        logger.info(f"Error analysis for experiment {self.experiment_id}")

        logger.info("\n--- Within-Model Confusion ---")
        self.within_model_confusion()

        logger.info("\n--- Cross-Model Confusion ---")
        self.cross_model_confusion()

        logger.info("\n--- Error by Chunk Length ---")
        self.error_by_chunk_length()

        logger.info("\n--- Error by Content Type ---")
        self.error_by_content_type()

        logger.info(f"\nError analysis reports written to {self.output_dir}")

    def _write_csv(self, filename, rows):
        if not rows:
            logger.warning(f"  No data for {filename}")
            return
        filepath = self.output_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"  Wrote {filename} ({len(rows)} rows)")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Error analysis and confusion matrices")
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    analyzer = ErrorAnalyzer(args.experiment_id)
    try:
        analyzer.run()
    finally:
        analyzer.close()
