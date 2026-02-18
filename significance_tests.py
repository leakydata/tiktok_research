"""
Statistical significance tests for key comparisons.

Tests:
  1. Temperature effect: T=0.0 vs T=0.5 (paired Wilcoxon signed-rank)
  2. Model size effect: Spearman correlation + group comparison
  3. Construct difficulty: Friedman test across constructs

Outputs:
  - significance_temperature.csv
  - significance_model_size.csv
  - significance_constructs.csv
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from scipy import stats
import csv
import logging
from pathlib import Path
from collections import defaultdict

from config import DB_CONFIG, MODELS_TO_TEST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'


class SignificanceTester:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def temperature_effect(self):
        """Paired test: T=0.0 vs T=0.5 stability per chunk/construct/model.

        For each chunk that has stability metrics at both temperatures,
        compare the agreement_ratio (or is_stable) between T=0.0 and T=0.5.
        """
        query = """
            SELECT
                chunk_id, construct_name, model_name, temperature,
                agreement_ratio, is_stable, stdev_value, range_value
            FROM annotation_stability_metrics
            WHERE experiment_id = %s
              AND temperature IN (0.0, 0.5)
              AND agreement_ratio IS NOT NULL
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        # Organize: {(chunk_id, construct, model): {temp: metrics}}
        paired = defaultdict(dict)
        for row in rows:
            key = (row['chunk_id'], row['construct_name'], row['model_name'])
            paired[key][row['temperature']] = row

        results = []
        # Per construct
        constructs = sorted(set(k[1] for k in paired.keys()))
        for construct in constructs:
            t0_vals = []
            t5_vals = []
            t0_stable = []
            t5_stable = []

            for key, temps in paired.items():
                if key[1] != construct:
                    continue
                if 0.0 in temps and 0.5 in temps:
                    t0_vals.append(temps[0.0]['agreement_ratio'])
                    t5_vals.append(temps[0.5]['agreement_ratio'])
                    t0_stable.append(1 if temps[0.0]['is_stable'] else 0)
                    t5_stable.append(1 if temps[0.5]['is_stable'] else 0)

            n_pairs = len(t0_vals)
            if n_pairs < 5:
                continue

            t0_arr = np.array(t0_vals)
            t5_arr = np.array(t5_vals)
            diff = t0_arr - t5_arr

            # Wilcoxon signed-rank test
            try:
                stat_w, p_wilcoxon = stats.wilcoxon(diff, alternative='two-sided')
            except ValueError:
                stat_w, p_wilcoxon = 0, 1.0

            # Cohen's d (paired)
            mean_diff = np.mean(diff)
            sd_diff = np.std(diff, ddof=1) if n_pairs > 1 else 1.0
            cohens_d = mean_diff / sd_diff if sd_diff > 0 else 0.0

            # Stability rate comparison
            stab_t0 = np.mean(t0_stable)
            stab_t5 = np.mean(t5_stable)

            results.append({
                'construct_name': construct,
                'n_pairs': n_pairs,
                'mean_agreement_t0': round(float(np.mean(t0_arr)), 4),
                'mean_agreement_t05': round(float(np.mean(t5_arr)), 4),
                'mean_diff': round(float(mean_diff), 4),
                'cohens_d': round(float(cohens_d), 4),
                'wilcoxon_stat': round(float(stat_w), 2),
                'p_value': round(float(p_wilcoxon), 6),
                'significant_005': p_wilcoxon < 0.05,
                'stability_rate_t0': round(float(stab_t0), 4),
                'stability_rate_t05': round(float(stab_t5), 4),
            })

            logger.info(
                f"  {construct}: T=0.0 mean={np.mean(t0_arr):.3f}, T=0.5 mean={np.mean(t5_arr):.3f}, "
                f"d={cohens_d:.3f}, p={p_wilcoxon:.4f}"
            )

        self._write_csv('significance_temperature.csv', results)
        return results

    def model_size_effect(self):
        """Correlation between model size and reliability metrics."""
        query = """
            SELECT
                model_name,
                krippendorff_alpha,
                stability_rate,
                coverage_rate,
                clarity_rate,
                construct_name
            FROM group_reliability_metrics
            WHERE experiment_id = %s
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        if not rows:
            logger.warning("No group reliability data found")
            return []

        # Map model names to sizes
        model_sizes = {}
        for name, info in MODELS_TO_TEST.items():
            model_sizes[name] = info['size_b']

        results = []
        constructs = sorted(set(r['construct_name'] for r in rows))

        for construct in constructs:
            c_rows = [r for r in rows if r['construct_name'] == construct]
            sizes = []
            alphas = []
            stab_rates = []

            for r in c_rows:
                size = model_sizes.get(r['model_name'])
                if size is not None and r['krippendorff_alpha'] is not None:
                    sizes.append(size)
                    alphas.append(r['krippendorff_alpha'])
                    stab_rates.append(r['stability_rate'] or 0)

            if len(sizes) < 3:
                continue

            # Spearman correlation: size vs alpha
            rho_alpha, p_alpha = stats.spearmanr(sizes, alphas)
            rho_stab, p_stab = stats.spearmanr(sizes, stab_rates)

            results.append({
                'construct_name': construct,
                'n_models': len(sizes),
                'spearman_rho_alpha': round(float(rho_alpha), 4),
                'p_value_alpha': round(float(p_alpha), 6),
                'spearman_rho_stability': round(float(rho_stab), 4),
                'p_value_stability': round(float(p_stab), 6),
            })

            logger.info(
                f"  {construct}: size-alpha rho={rho_alpha:.3f} (p={p_alpha:.4f}), "
                f"size-stability rho={rho_stab:.3f} (p={p_stab:.4f})"
            )

        # Overall (pooled across constructs)
        all_sizes = []
        all_alphas = []
        for r in rows:
            size = model_sizes.get(r['model_name'])
            if size is not None and r['krippendorff_alpha'] is not None:
                all_sizes.append(size)
                all_alphas.append(r['krippendorff_alpha'])

        if len(all_sizes) >= 3:
            rho, p = stats.spearmanr(all_sizes, all_alphas)
            results.append({
                'construct_name': 'ALL_CONSTRUCTS',
                'n_models': len(all_sizes),
                'spearman_rho_alpha': round(float(rho), 4),
                'p_value_alpha': round(float(p), 6),
                'spearman_rho_stability': None,
                'p_value_stability': None,
            })

        self._write_csv('significance_model_size.csv', results)
        return results

    def construct_difficulty(self):
        """Compare stability rates across constructs (Friedman test)."""
        query = """
            SELECT construct_name, model_name, temperature, stability_rate
            FROM group_reliability_metrics
            WHERE experiment_id = %s
              AND stability_rate IS NOT NULL
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (self.experiment_id,))
            rows = cur.fetchall()

        # Build matrix: conditions (model x temp) x constructs
        constructs = sorted(set(r['construct_name'] for r in rows))
        conditions = sorted(set((r['model_name'], r['temperature']) for r in rows))

        # For Friedman, need same number of observations per construct
        data_by_construct = defaultdict(dict)
        for r in rows:
            cond = (r['model_name'], r['temperature'])
            data_by_construct[r['construct_name']][cond] = r['stability_rate']

        # Find conditions present in ALL constructs
        common_conds = set(conditions)
        for construct in constructs:
            common_conds &= set(data_by_construct[construct].keys())
        common_conds = sorted(common_conds)

        if len(common_conds) < 3 or len(constructs) < 3:
            logger.warning("Not enough data for Friedman test")
            return []

        # Build array: rows = conditions, columns = constructs
        matrix = []
        for cond in common_conds:
            row = [data_by_construct[c].get(cond, 0) for c in constructs]
            matrix.append(row)
        matrix = np.array(matrix)

        # Friedman test
        try:
            stat_f, p_friedman = stats.friedmanchisquare(*[matrix[:, i] for i in range(len(constructs))])
        except ValueError:
            stat_f, p_friedman = 0, 1.0

        results = []
        for i, construct in enumerate(constructs):
            vals = matrix[:, i]
            results.append({
                'construct_name': construct,
                'mean_stability_rate': round(float(np.mean(vals)), 4),
                'median_stability_rate': round(float(np.median(vals)), 4),
                'std_stability_rate': round(float(np.std(vals, ddof=1)), 4),
                'rank': i + 1,
                'friedman_statistic': round(float(stat_f), 2) if i == 0 else None,
                'friedman_p_value': round(float(p_friedman), 6) if i == 0 else None,
                'n_conditions': len(common_conds),
            })

        # Sort by mean stability (hardest first)
        results.sort(key=lambda x: x['mean_stability_rate'])
        for i, r in enumerate(results):
            r['difficulty_rank'] = i + 1

        logger.info(f"Friedman test: chi2={stat_f:.2f}, p={p_friedman:.6f}")
        for r in results:
            logger.info(
                f"  {r['construct_name']:25s}: mean_stability={r['mean_stability_rate']:.3f} "
                f"(rank {r['difficulty_rank']})"
            )

        self._write_csv('significance_constructs.csv', results)
        return results

    def run(self):
        """Run all significance tests."""
        logger.info(f"Significance tests for experiment {self.experiment_id}")

        logger.info("\n--- Temperature Effect ---")
        self.temperature_effect()

        logger.info("\n--- Model Size Effect ---")
        self.model_size_effect()

        logger.info("\n--- Construct Difficulty ---")
        self.construct_difficulty()

        logger.info(f"\nSignificance test reports written to {self.output_dir}")

    def _write_csv(self, filename, rows):
        if not rows:
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
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    tester = SignificanceTester(args.experiment_id)
    try:
        tester.run()
    finally:
        tester.close()
