"""
Stratified analysis: exploits the unique TikTok longitudinal dataset.

Breakdowns by:
  1. Content type (narrative, informational, etc.)
  2. Chunk length (quartiles)
  3. Temporal position (early/middle/late in creator's timeline)
  4. Claimed diagnosis
  5. Context carry presence (with vs without prior context)

Outputs:
  - stratified_by_content_type.csv
  - stratified_by_chunk_length.csv
  - stratified_by_temporal_position.csv
  - stratified_by_diagnosis.csv
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
import csv
import logging
from pathlib import Path
from collections import defaultdict

from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'


class StratifiedAnalyzer:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_stability_with_chunk_metadata(self):
        """Fetch stability metrics joined with chunk and cohort metadata."""
        query = """
            SELECT
                asm.chunk_id, asm.construct_name, asm.model_name,
                asm.temperature, asm.is_stable, asm.agreement_ratio,
                asm.none_rate, asm.unclear_rate, asm.construct_type,
                ac.char_length, ac.content_type, ac.creator_username,
                ac.days_since_first_video, ac.context_carry_text,
                sc.primary_diagnosis, sc.diagnoses_claimed
            FROM annotation_stability_metrics asm
            INNER JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
            LEFT JOIN study_cohort sc ON ac.creator_username = sc.creator_username
            WHERE asm.experiment_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            return cur.fetchall()

    def _aggregate_by_group(self, rows, group_key_func, group_label):
        """Generic aggregation: group rows and compute stability metrics per group."""
        groups = defaultdict(lambda: {
            'stable': 0, 'total': 0, 'agreement_sum': 0.0,
            'none_sum': 0.0, 'unclear_sum': 0.0,
        })

        for row in rows:
            key = group_key_func(row)
            if key is None:
                continue
            g = groups[(key, row['construct_name'])]
            g['total'] += 1
            if row['is_stable']:
                g['stable'] += 1
            if row['agreement_ratio'] is not None:
                g['agreement_sum'] += row['agreement_ratio']
            if row['none_rate'] is not None:
                g['none_sum'] += row['none_rate']
            if row['unclear_rate'] is not None:
                g['unclear_sum'] += row['unclear_rate']

        results = []
        for (group_val, construct), g in sorted(groups.items()):
            t = g['total']
            if t == 0:
                continue
            results.append({
                group_label: group_val,
                'construct_name': construct,
                'total_chunks': t,
                'stable_chunks': g['stable'],
                'stability_rate': round(g['stable'] / t, 4),
                'mean_agreement': round(g['agreement_sum'] / t, 4),
                'mean_none_rate': round(g['none_sum'] / t, 4),
                'mean_unclear_rate': round(g['unclear_sum'] / t, 4),
            })

        return results

    def by_content_type(self, rows):
        """Stability by content type."""
        results = self._aggregate_by_group(
            rows,
            lambda r: r['content_type'] or 'unknown',
            'content_type',
        )
        self._write_csv('stratified_by_content_type.csv', results)
        logger.info(f"  Content type: {len(results)} rows")
        return results

    def by_chunk_length(self, rows):
        """Stability by chunk length quartiles."""
        lengths = [r['char_length'] for r in rows if r['char_length']]
        if not lengths:
            return []
        q25, q50, q75 = np.percentile(lengths, [25, 50, 75])

        def length_bin(r):
            l = r['char_length']
            if l is None:
                return None
            if l <= q25:
                return f'Q1 (<={int(q25)})'
            elif l <= q50:
                return f'Q2 ({int(q25)+1}-{int(q50)})'
            elif l <= q75:
                return f'Q3 ({int(q50)+1}-{int(q75)})'
            else:
                return f'Q4 (>{int(q75)})'

        results = self._aggregate_by_group(rows, length_bin, 'length_bin')
        self._write_csv('stratified_by_chunk_length.csv', results)
        logger.info(f"  Chunk length: {len(results)} rows")
        return results

    def by_temporal_position(self, rows):
        """Stability by position in creator's timeline (early/middle/late)."""
        days = [r['days_since_first_video'] for r in rows if r['days_since_first_video'] is not None]
        if not days:
            return []
        t33, t66 = np.percentile(days, [33, 66])

        def temporal_bin(r):
            d = r['days_since_first_video']
            if d is None:
                return None
            if d <= t33:
                return 'early'
            elif d <= t66:
                return 'middle'
            else:
                return 'late'

        results = self._aggregate_by_group(rows, temporal_bin, 'temporal_position')
        self._write_csv('stratified_by_temporal_position.csv', results)
        logger.info(f"  Temporal position: {len(results)} rows")
        return results

    def by_diagnosis(self, rows):
        """Stability by creator's primary claimed diagnosis."""
        results = self._aggregate_by_group(
            rows,
            lambda r: r['primary_diagnosis'] or 'unknown',
            'primary_diagnosis',
        )
        self._write_csv('stratified_by_diagnosis.csv', results)
        logger.info(f"  Diagnosis: {len(results)} rows")
        return results

    def by_context_carry(self, rows):
        """Stability comparison: chunks with vs without context carry."""
        def has_context(r):
            return 'with_context' if r['context_carry_text'] else 'no_context'

        results = self._aggregate_by_group(rows, has_context, 'context_carry')
        self._write_csv('stratified_by_context_carry.csv', results)
        logger.info(f"  Context carry: {len(results)} rows")
        return results

    def run(self):
        """Run all stratified analyses."""
        logger.info(f"Stratified analysis for experiment {self.experiment_id}")

        rows = self._get_stability_with_chunk_metadata()
        logger.info(f"Loaded {len(rows)} stability records with metadata")

        logger.info("\nRunning stratifications:")
        self.by_content_type(rows)
        self.by_chunk_length(rows)
        self.by_temporal_position(rows)
        self.by_diagnosis(rows)
        self.by_context_carry(rows)

        logger.info(f"\nStratified analysis reports written to {self.output_dir}")

    def _write_csv(self, filename, rows):
        if not rows:
            logger.warning(f"  No data for {filename}")
            return
        filepath = self.output_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stratified analysis by metadata dimensions")
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    analyzer = StratifiedAnalyzer(args.experiment_id)
    try:
        analyzer.run()
    finally:
        analyzer.close()
