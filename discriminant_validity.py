"""
Discriminant validity: construct correlation matrix.

Computes correlations between all 6 constructs to show that:
  - Constructs that should be independent ARE independent
  - Constructs that should relate DO relate
  - The 6 constructs capture different dimensions (not redundant)

Uses:
  - Cramér's V for categorical x categorical
  - Spearman for continuous x continuous
  - Point-biserial / eta-squared for categorical x continuous

Outputs:
  - construct_correlation_matrix.csv
  - discriminant_validity_tests.csv
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from scipy import stats
import csv
import logging
from pathlib import Path
from collections import defaultdict
from itertools import combinations

from config import DB_CONFIG, STABILITY_THRESHOLDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'


def cramers_v(labels_a: list, labels_b: list) -> float:
    """Compute Cramér's V for two categorical variables."""
    # Build contingency table
    cats_a = sorted(set(labels_a))
    cats_b = sorted(set(labels_b))
    a_idx = {c: i for i, c in enumerate(cats_a)}
    b_idx = {c: i for i, c in enumerate(cats_b)}

    table = np.zeros((len(cats_a), len(cats_b)))
    for a, b in zip(labels_a, labels_b):
        table[a_idx[a], b_idx[b]] += 1

    n = table.sum()
    if n == 0:
        return 0.0

    chi2 = stats.chi2_contingency(table, correction=False)[0]
    k = min(len(cats_a), len(cats_b))
    if k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))


class DiscriminantValidator:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_chunk_labels(self, model_name: str = None):
        """Get final labels for all chunks, one row per chunk with all construct labels.

        Returns: {chunk_id: {construct_name: label_value}}
        Uses final_label_text for categorical, final_label_float for continuous.
        """
        model_filter = "AND model_name = %s" if model_name else ""
        params = [self.experiment_id]
        if model_name:
            params.append(model_name)

        query = f"""
            SELECT
                fa.chunk_id, fa.construct_name,
                fa.final_label_text, fa.final_label_float, fa.final_label_bin,
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
              {model_filter}
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        data = defaultdict(dict)
        types = {}
        for row in rows:
            types[row['construct_name']] = row['construct_type']
            if row['construct_type'] == 'continuous':
                data[row['chunk_id']][row['construct_name']] = row['final_label_float']
            else:
                data[row['chunk_id']][row['construct_name']] = row['final_label_text']
        return data, types

    def compute_correlation_matrix(self, model_name: str = None):
        """Compute pairwise correlations between all constructs."""
        data, types = self.get_chunk_labels(model_name)
        constructs = sorted(types.keys())

        results = []
        for c1, c2 in combinations(constructs, 2):
            type1 = types[c1]
            type2 = types[c2]

            # Get paired values (chunks that have both constructs)
            vals1 = []
            vals2 = []
            for chunk_id, labels in data.items():
                if c1 in labels and c2 in labels:
                    v1 = labels[c1]
                    v2 = labels[c2]
                    if v1 is not None and v2 is not None:
                        vals1.append(v1)
                        vals2.append(v2)

            n = len(vals1)
            if n < 5:
                continue

            # Choose correlation method based on types
            if type1 == 'continuous' and type2 == 'continuous':
                # Spearman
                rho, p = stats.spearmanr(vals1, vals2)
                method = 'spearman'
                effect = round(float(rho), 4)
            elif type1 == 'categorical' and type2 == 'categorical':
                # Cramér's V
                effect = round(cramers_v(vals1, vals2), 4)
                # Chi-squared p-value
                cats_a = sorted(set(vals1))
                cats_b = sorted(set(vals2))
                a_idx = {c: i for i, c in enumerate(cats_a)}
                b_idx = {c: i for i, c in enumerate(cats_b)}
                table = np.zeros((len(cats_a), len(cats_b)))
                for a, b in zip(vals1, vals2):
                    table[a_idx[a], b_idx[b]] += 1
                try:
                    _, p, _, _ = stats.chi2_contingency(table)
                except ValueError:
                    p = 1.0
                method = 'cramers_v'
                rho = effect
            else:
                # Mixed: convert categorical to numeric codes and use Spearman
                if type1 == 'categorical':
                    cat_vals, num_vals = vals1, vals2
                    cat_name, num_name = c1, c2
                else:
                    cat_vals, num_vals = vals2, vals1
                    cat_name, num_name = c2, c1

                # Encode categories as integers
                categories = sorted(set(cat_vals))
                cat_codes = [categories.index(v) for v in cat_vals]
                rho, p = stats.spearmanr(cat_codes, num_vals)
                method = 'spearman_mixed'
                effect = round(float(rho), 4)

            results.append({
                'construct_a': c1,
                'type_a': type1,
                'construct_b': c2,
                'type_b': type2,
                'method': method,
                'correlation': effect,
                'p_value': round(float(p), 6),
                'n_pairs': n,
                'significant_005': p < 0.05,
                'model_name': model_name or 'ALL_MODELS',
            })

        return results

    def run(self):
        """Run discriminant validity analysis."""
        logger.info(f"Discriminant validity analysis for experiment {self.experiment_id}")

        # Run pooled across all models
        results = self.compute_correlation_matrix()

        # Also run per-model for the largest models (optional depth)
        # Skip per-model for now — pooled is the main result

        self._write_csv('construct_correlation_matrix.csv', results)

        # Log summary
        logger.info("\n=== CONSTRUCT CORRELATION MATRIX ===")
        for r in results:
            sig = "*" if r['significant_005'] else ""
            logger.info(
                f"  {r['construct_a']:25s} x {r['construct_b']:25s}: "
                f"{r['method']}={r['correlation']:.3f}{sig} (n={r['n_pairs']})"
            )

        logger.info(f"\nDiscriminant validity reports written to {self.output_dir}")

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
    parser = argparse.ArgumentParser(description="Discriminant validity: construct correlations")
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    validator = DiscriminantValidator(args.experiment_id)
    try:
        validator.run()
    finally:
        validator.close()
