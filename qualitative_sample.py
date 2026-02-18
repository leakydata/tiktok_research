"""
Qualitative sample export for face-validity review.

Selects a stratified sample of chunks for manual inspection:
  - Unanimous agreement chunks (high confidence)
  - Maximum disagreement chunks (interesting edge cases)
  - "None" labeled chunks (verify no health content)
  - "Unclear" labeled chunks (verify genuine ambiguity)
  - Random sample per construct

Outputs:
  - qualitative_samples.csv  (for researcher review / paper appendix)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import json
import logging
from pathlib import Path
from collections import defaultdict

from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'

SAMPLE_SIZE_PER_STRATUM = 10


class QualitativeSampler:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_high_agreement_chunks(self, construct: str, n: int = SAMPLE_SIZE_PER_STRATUM):
        """Chunks where ALL models agree on the same stable label."""
        query = """
            SELECT
                fa.chunk_id, ac.chunk_text,
                fa.construct_name, fa.model_name,
                fa.final_label_text, fa.final_label_bin, fa.final_label_float,
                fa.agreement_ratio
            FROM final_annotations fa
            INNER JOIN annotation_chunks ac ON fa.chunk_id = ac.chunk_id
            WHERE fa.experiment_id = %s
              AND fa.construct_name = %s
              AND fa.is_stable = TRUE
            ORDER BY fa.agreement_ratio DESC NULLS LAST
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id, construct))
            rows = cur.fetchall()

        # Group by chunk_id, find chunks where all models have the same label
        chunks = defaultdict(list)
        for row in rows:
            chunks[row['chunk_id']].append(row)

        unanimous = []
        for chunk_id, chunk_rows in chunks.items():
            labels = set()
            for r in chunk_rows:
                label = r['final_label_text'] or r['final_label_bin']
                if label:
                    labels.add(label)
            if len(labels) == 1 and len(chunk_rows) >= 2:
                unanimous.append(chunk_rows[0])

        return unanimous[:n]

    def get_high_disagreement_chunks(self, construct: str, n: int = SAMPLE_SIZE_PER_STRATUM):
        """Chunks with lowest agreement ratio (most unstable)."""
        query = """
            SELECT DISTINCT ON (asm.chunk_id)
                asm.chunk_id, ac.chunk_text,
                asm.construct_name, asm.model_name,
                asm.agreement_ratio, asm.is_stable,
                asm.labels_observed, asm.label_distribution
            FROM annotation_stability_metrics asm
            INNER JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
            WHERE asm.experiment_id = %s
              AND asm.construct_name = %s
              AND asm.agreement_ratio IS NOT NULL
            ORDER BY asm.chunk_id, asm.agreement_ratio ASC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id, construct))
            rows = cur.fetchall()

        # Sort by agreement (lowest first) and take top n
        rows.sort(key=lambda r: r['agreement_ratio'] or 999)
        return rows[:n]

    def get_none_chunks(self, construct: str, n: int = SAMPLE_SIZE_PER_STRATUM):
        """Chunks most frequently labeled 'none' (no health content)."""
        query = """
            SELECT DISTINCT ON (asm.chunk_id)
                asm.chunk_id, ac.chunk_text,
                asm.construct_name, asm.model_name,
                asm.none_rate
            FROM annotation_stability_metrics asm
            INNER JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
            WHERE asm.experiment_id = %s
              AND asm.construct_name = %s
              AND asm.none_rate > 0.5
            ORDER BY asm.chunk_id, asm.none_rate DESC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id, construct))
            rows = cur.fetchall()

        rows.sort(key=lambda r: r['none_rate'] or 0, reverse=True)
        return rows[:n]

    def get_unclear_chunks(self, construct: str, n: int = SAMPLE_SIZE_PER_STRATUM):
        """Chunks most frequently labeled 'unclear'."""
        query = """
            SELECT DISTINCT ON (asm.chunk_id)
                asm.chunk_id, ac.chunk_text,
                asm.construct_name, asm.model_name,
                asm.unclear_rate
            FROM annotation_stability_metrics asm
            INNER JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
            WHERE asm.experiment_id = %s
              AND asm.construct_name = %s
              AND asm.unclear_rate > 0.3
            ORDER BY asm.chunk_id, asm.unclear_rate DESC
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id, construct))
            rows = cur.fetchall()

        rows.sort(key=lambda r: r['unclear_rate'] or 0, reverse=True)
        return rows[:n]

    def get_all_model_labels_for_chunk(self, chunk_id: int, construct: str):
        """Get labels from all models for one chunk/construct."""
        query = """
            SELECT model_name, final_label_text, final_label_bin, final_label_float,
                   is_stable, agreement_ratio
            FROM final_annotations
            WHERE experiment_id = %s AND chunk_id = %s AND construct_name = %s
            ORDER BY model_name
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id, chunk_id, construct))
            return cur.fetchall()

    def run(self):
        """Generate stratified qualitative sample."""
        logger.info(f"Qualitative sampling for experiment {self.experiment_id}")

        # Get all constructs
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT construct_name FROM annotation_stability_metrics
                WHERE experiment_id = %s ORDER BY construct_name
            """, (self.experiment_id,))
            constructs = [row[0] for row in cur.fetchall()]

        all_samples = []

        for construct in constructs:
            logger.info(f"\n  --- {construct} ---")

            # High agreement
            for row in self.get_high_agreement_chunks(construct):
                model_labels = self.get_all_model_labels_for_chunk(row['chunk_id'], construct)
                label_summary = "; ".join(
                    f"{ml['model_name']}={ml['final_label_text'] or ml['final_label_bin']}"
                    for ml in model_labels
                )
                all_samples.append({
                    'stratum': 'unanimous_agreement',
                    'construct_name': construct,
                    'chunk_id': row['chunk_id'],
                    'chunk_text': row['chunk_text'],
                    'final_label': row.get('final_label_text') or row.get('final_label_bin', ''),
                    'agreement_ratio': row.get('agreement_ratio', ''),
                    'all_model_labels': label_summary,
                    'none_rate': '',
                    'unclear_rate': '',
                })

            # High disagreement
            for row in self.get_high_disagreement_chunks(construct):
                model_labels = self.get_all_model_labels_for_chunk(row['chunk_id'], construct)
                label_summary = "; ".join(
                    f"{ml['model_name']}={ml['final_label_text'] or ml['final_label_bin'] or 'unstable'}"
                    for ml in model_labels
                )
                dist = row.get('label_distribution', '')
                if isinstance(dist, dict):
                    dist = json.dumps(dist)
                all_samples.append({
                    'stratum': 'max_disagreement',
                    'construct_name': construct,
                    'chunk_id': row['chunk_id'],
                    'chunk_text': row['chunk_text'],
                    'final_label': dist,
                    'agreement_ratio': row.get('agreement_ratio', ''),
                    'all_model_labels': label_summary,
                    'none_rate': '',
                    'unclear_rate': '',
                })

            # None chunks
            for row in self.get_none_chunks(construct):
                all_samples.append({
                    'stratum': 'none_labeled',
                    'construct_name': construct,
                    'chunk_id': row['chunk_id'],
                    'chunk_text': row['chunk_text'],
                    'final_label': 'none',
                    'agreement_ratio': '',
                    'all_model_labels': '',
                    'none_rate': row.get('none_rate', ''),
                    'unclear_rate': '',
                })

            # Unclear chunks
            for row in self.get_unclear_chunks(construct):
                all_samples.append({
                    'stratum': 'unclear_labeled',
                    'construct_name': construct,
                    'chunk_id': row['chunk_id'],
                    'chunk_text': row['chunk_text'],
                    'final_label': 'unclear',
                    'agreement_ratio': '',
                    'all_model_labels': '',
                    'none_rate': '',
                    'unclear_rate': row.get('unclear_rate', ''),
                })

            logger.info(f"  Sampled for {construct}")

        # Write CSV
        if all_samples:
            filepath = self.output_dir / 'qualitative_samples.csv'
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_samples[0].keys())
                writer.writeheader()
                writer.writerows(all_samples)
            logger.info(f"\nWrote qualitative_samples.csv ({len(all_samples)} rows)")
        else:
            logger.warning("No samples to write")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qualitative sample for face validity")
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    sampler = QualitativeSampler(args.experiment_id)
    try:
        sampler.run()
    finally:
        sampler.close()
