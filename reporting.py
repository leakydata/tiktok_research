"""
Step 9: Generate publication-quality tables and CSV exports.
Produces the standard outputs reviewers expect:
  - Coverage vs stability threshold curves
  - Stability vs temperature
  - Stability vs model size
  - Unclear rate comparisons by model/construct
  - Group-level reliability summary table
  - Per-construct label distributions
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import os
import logging
from pathlib import Path

from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'


class ReportGenerator:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self):
        """Generate all reports."""
        logger.info(f"Generating reports for experiment {self.experiment_id}")
        logger.info(f"Output directory: {self.output_dir}")

        self.table_group_reliability()
        self.table_stability_by_model()
        self.table_stability_by_temperature()
        self.table_coverage_clarity()
        self.table_label_distributions()
        self.csv_stability_threshold_curve()
        self.csv_per_chunk_stability()
        self.csv_validation_results()

        logger.info(f"All reports generated in {self.output_dir}")

    def table_group_reliability(self):
        """Table 1: Group-level reliability metrics (main results table)."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    construct_name,
                    model_name,
                    temperature,
                    num_chunks,
                    coverage_rate,
                    clarity_rate,
                    krippendorff_alpha,
                    icc_value,
                    stability_rate,
                    mean_agreement
                FROM group_reliability_metrics
                WHERE experiment_id = %s
                ORDER BY construct_name, model_name, temperature
            """, (self.experiment_id,))
            rows = cur.fetchall()

        self._write_csv('table1_group_reliability.csv', rows)
        logger.info(f"  Table 1: Group reliability ({len(rows)} rows)")

    def table_stability_by_model(self):
        """Table 2: Stability rate comparison across models."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    model_name,
                    ROUND(AVG(stability_rate), 3) AS avg_stability_rate,
                    ROUND(AVG(krippendorff_alpha), 3) AS avg_alpha,
                    ROUND(AVG(coverage_rate), 3) AS avg_coverage,
                    ROUND(AVG(clarity_rate), 3) AS avg_clarity,
                    SUM(num_chunks) AS total_chunks
                FROM group_reliability_metrics
                WHERE experiment_id = %s
                GROUP BY model_name
                ORDER BY avg_stability_rate DESC
            """, (self.experiment_id,))
            rows = cur.fetchall()

        self._write_csv('table2_stability_by_model.csv', rows)
        logger.info(f"  Table 2: Stability by model ({len(rows)} rows)")

    def table_stability_by_temperature(self):
        """Table 3: Temperature=0 (deterministic) vs temperature=0.5 (stochastic)."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    temperature,
                    construct_name,
                    ROUND(AVG(stability_rate), 3) AS avg_stability_rate,
                    ROUND(AVG(krippendorff_alpha), 3) AS avg_alpha,
                    ROUND(AVG(mean_agreement), 3) AS avg_agreement
                FROM group_reliability_metrics
                WHERE experiment_id = %s
                GROUP BY temperature, construct_name
                ORDER BY construct_name, temperature
            """, (self.experiment_id,))
            rows = cur.fetchall()

        self._write_csv('table3_stability_by_temperature.csv', rows)
        logger.info(f"  Table 3: Stability by temperature ({len(rows)} rows)")

    def table_coverage_clarity(self):
        """Table 4: Coverage (1-none_rate) and clarity (1-unclear_rate) by construct."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    construct_name,
                    model_name,
                    ROUND(AVG(none_rate), 3) AS avg_none_rate,
                    ROUND(AVG(unclear_rate), 3) AS avg_unclear_rate,
                    ROUND(1 - AVG(none_rate), 3) AS coverage,
                    ROUND(1 - AVG(unclear_rate), 3) AS clarity,
                    COUNT(*) AS num_chunks
                FROM annotation_stability_metrics
                WHERE experiment_id = %s
                GROUP BY construct_name, model_name
                ORDER BY construct_name, coverage DESC
            """, (self.experiment_id,))
            rows = cur.fetchall()

        self._write_csv('table4_coverage_clarity.csv', rows)
        logger.info(f"  Table 4: Coverage/clarity ({len(rows)} rows)")

    def table_label_distributions(self):
        """Table 5: Label distributions for stable annotations per construct."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    construct_name,
                    model_name,
                    final_label_text,
                    final_label_bin,
                    COUNT(*) AS count,
                    ROUND(
                        COUNT(*)::numeric / SUM(COUNT(*)) OVER (
                            PARTITION BY construct_name, model_name
                        ), 3
                    ) AS proportion
                FROM final_annotations
                WHERE experiment_id = %s
                  AND is_stable = TRUE
                GROUP BY construct_name, model_name,
                         final_label_text, final_label_bin
                ORDER BY construct_name, model_name, count DESC
            """, (self.experiment_id,))
            rows = cur.fetchall()

        self._write_csv('table5_label_distributions.csv', rows)
        logger.info(f"  Table 5: Label distributions ({len(rows)} rows)")

    def csv_stability_threshold_curve(self):
        """Data for Figure: Coverage vs stability threshold curves."""
        # Simulate different thresholds for categorical constructs
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rows = []

        for thresh in thresholds:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        construct_name,
                        model_name,
                        COUNT(*) AS total_chunks,
                        SUM(CASE WHEN agreement_ratio >= %s THEN 1 ELSE 0 END) AS stable_chunks,
                        ROUND(
                            SUM(CASE WHEN agreement_ratio >= %s THEN 1 ELSE 0 END)::numeric
                            / NULLIF(COUNT(*), 0), 3
                        ) AS stability_rate
                    FROM annotation_stability_metrics
                    WHERE experiment_id = %s
                      AND construct_type = 'categorical'
                    GROUP BY construct_name, model_name
                """, (thresh, thresh, self.experiment_id))
                for r in cur.fetchall():
                    rows.append({
                        'threshold': thresh,
                        'construct_name': r['construct_name'],
                        'model_name': r['model_name'],
                        'total_chunks': r['total_chunks'],
                        'stable_chunks': r['stable_chunks'],
                        'stability_rate': r['stability_rate'],
                    })

        # For continuous constructs, vary max_range
        range_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
        for max_range in range_thresholds:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        construct_name,
                        model_name,
                        COUNT(*) AS total_chunks,
                        SUM(CASE WHEN range_value <= %s THEN 1 ELSE 0 END) AS stable_chunks,
                        ROUND(
                            SUM(CASE WHEN range_value <= %s THEN 1 ELSE 0 END)::numeric
                            / NULLIF(COUNT(*), 0), 3
                        ) AS stability_rate
                    FROM annotation_stability_metrics
                    WHERE experiment_id = %s
                      AND construct_type = 'continuous'
                    GROUP BY construct_name, model_name
                """, (max_range, max_range, self.experiment_id))
                for r in cur.fetchall():
                    rows.append({
                        'threshold': max_range,
                        'construct_name': r['construct_name'],
                        'model_name': r['model_name'],
                        'total_chunks': r['total_chunks'],
                        'stable_chunks': r['stable_chunks'],
                        'stability_rate': r['stability_rate'],
                    })

        self._write_csv('figure_stability_threshold_curve.csv', rows)
        logger.info(f"  Threshold curve data ({len(rows)} rows)")

    def csv_per_chunk_stability(self):
        """Full per-chunk stability export for external analysis."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    asm.chunk_id,
                    ac.creator_username,
                    ac.video_date,
                    ac.content_type,
                    ac.char_length,
                    asm.construct_name,
                    asm.model_name,
                    asm.temperature,
                    asm.construct_type,
                    asm.is_stable,
                    asm.agreement_ratio,
                    asm.stdev_value,
                    asm.range_value,
                    asm.none_rate,
                    asm.unclear_rate,
                    asm.valid_run_count,
                    asm.modal_label,
                    asm.mean_value,
                    asm.median_value
                FROM annotation_stability_metrics asm
                INNER JOIN annotation_chunks ac ON asm.chunk_id = ac.chunk_id
                WHERE asm.experiment_id = %s
                ORDER BY asm.construct_name, asm.model_name, asm.chunk_id
            """, (self.experiment_id,))
            rows = cur.fetchall()

        self._write_csv('full_per_chunk_stability.csv', rows)
        logger.info(f"  Per-chunk export ({len(rows)} rows)")

    def csv_validation_results(self):
        """Validation comparison export."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    construct_name,
                    narrative_field,
                    annotation_label,
                    narrative_value,
                    agrees,
                    COUNT(*) AS count
                FROM validation_comparisons
                WHERE experiment_id = %s
                GROUP BY construct_name, narrative_field,
                         annotation_label, narrative_value, agrees
                ORDER BY construct_name, narrative_field
            """, (self.experiment_id,))
            rows = cur.fetchall()

        self._write_csv('validation_results.csv', rows)
        logger.info(f"  Validation results ({len(rows)} rows)")

    def _write_csv(self, filename: str, rows: list[dict]):
        """Write a list of dicts to CSV."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    rg = ReportGenerator(args.experiment_id)
    try:
        rg.run_all()
    finally:
        rg.close()
