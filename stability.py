"""
Step 6: Construct-aware stability analysis.
- Categorical constructs: agreement_ratio, entropy, modal label
- Continuous constructs: stdev, IQR, range, mean, median
- Group-level: Krippendorff alpha (proper implementation across items)
- Separate coverage (1 - none_rate) and clarity (1 - unclear_rate)
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import numpy as np
from collections import Counter
import json
import logging
from typing import Optional

from config import DB_CONFIG, STABILITY_THRESHOLDS, LABEL_VOCABULARIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ── Per-Chunk Stability Metrics ───────────────────────────────────────────

def compute_categorical_stability(labels: list[str], threshold: float) -> dict:
    """Compute stability metrics for categorical labels."""
    counts = Counter(labels)
    n = len(labels)
    modal_label, modal_count = counts.most_common(1)[0]
    agreement_ratio = modal_count / n

    # Shannon entropy (base 2)
    probs = np.array([c / n for c in counts.values()])
    label_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    is_stable = agreement_ratio >= threshold

    return {
        'agreement_ratio': round(agreement_ratio, 4),
        'label_entropy': round(label_entropy, 4),
        'modal_label': modal_label,
        'modal_count': modal_count,
        'is_stable': is_stable,
        'stability_rule': f'agreement_ratio >= {threshold}',
        # Continuous fields are NULL
        'mean_value': None,
        'median_value': None,
        'stdev_value': None,
        'iqr_value': None,
        'range_value': None,
        'modal_bin': None,
        'bin_agreement_ratio': None,
    }


def compute_continuous_stability(
    values: list[float],
    bins: dict,
    max_range: float,
    max_stdev: float,
) -> dict:
    """Compute stability metrics for continuous labels."""
    arr = np.array(values)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    stdev_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr_val = q75 - q25
    range_val = float(np.max(arr) - np.min(arr))

    # Bin each value
    bin_labels = []
    for v in values:
        assigned = None
        for bin_name, (lo, hi) in bins.items():
            if lo <= v <= hi:
                assigned = bin_name
                break
        bin_labels.append(assigned or 'unknown')

    bin_counts = Counter(bin_labels)
    modal_bin, modal_bin_count = bin_counts.most_common(1)[0]
    bin_agreement = modal_bin_count / len(bin_labels)

    is_stable = range_val <= max_range and stdev_val <= max_stdev

    return {
        'mean_value': round(mean_val, 4),
        'median_value': round(median_val, 4),
        'stdev_value': round(stdev_val, 4),
        'iqr_value': round(iqr_val, 4),
        'range_value': round(range_val, 4),
        'modal_bin': modal_bin,
        'bin_agreement_ratio': round(bin_agreement, 4),
        'is_stable': is_stable,
        'stability_rule': f'range <= {max_range} AND stdev <= {max_stdev}',
        # Categorical fields get bin-based values
        'agreement_ratio': round(bin_agreement, 4),
        'label_entropy': None,
        'modal_label': modal_bin,
        'modal_count': modal_bin_count,
    }


# ── Group-Level Reliability (Krippendorff Alpha) ──────────────────────────

def krippendorff_alpha_nominal(reliability_data: list[list[Optional[str]]]) -> float:
    """Compute Krippendorff's alpha for nominal data.

    Args:
        reliability_data: List of items, where each item is a list of coder labels.
                         None indicates missing data.

    Returns:
        Alpha coefficient (-1 to 1, where 1 = perfect agreement).
    """
    # Collect all non-None values
    all_values = set()
    for item_labels in reliability_data:
        for label in item_labels:
            if label is not None:
                all_values.add(label)

    if len(all_values) <= 1:
        return 1.0  # Perfect agreement (or trivial case)

    # Build coincidence matrix
    value_list = sorted(all_values)
    val_to_idx = {v: i for i, v in enumerate(value_list)}
    k = len(value_list)
    coincidence = np.zeros((k, k), dtype=float)

    n_total = 0
    for item_labels in reliability_data:
        valid = [l for l in item_labels if l is not None]
        m = len(valid)
        if m < 2:
            continue
        n_total += m
        for i in range(m):
            for j in range(m):
                if i != j:
                    ci = val_to_idx[valid[i]]
                    cj = val_to_idx[valid[j]]
                    coincidence[ci][cj] += 1.0 / (m - 1)

    if n_total < 2:
        return 0.0

    # Marginals
    marginals = coincidence.sum(axis=1)
    n_pairs = marginals.sum()

    if n_pairs == 0:
        return 0.0

    # Observed disagreement
    D_o = 1.0 - np.trace(coincidence) / n_pairs

    # Expected disagreement
    D_e = 1.0 - np.sum(marginals * (marginals - 1)) / (n_pairs * (n_pairs - 1))

    if D_e == 0:
        return 1.0

    return float(1.0 - D_o / D_e)


def krippendorff_alpha_interval(reliability_data: list[list[Optional[float]]]) -> float:
    """Compute Krippendorff's alpha for interval (continuous) data.

    Args:
        reliability_data: List of items, each a list of coder values (floats). None = missing.

    Returns:
        Alpha coefficient.
    """
    # Flatten all valid pairs
    pairs_observed = []
    pairs_expected_values = []

    for item_values in reliability_data:
        valid = [v for v in item_values if v is not None]
        m = len(valid)
        if m < 2:
            continue
        # Observed: squared differences within this item
        for i in range(m):
            for j in range(i + 1, m):
                pairs_observed.append((valid[i] - valid[j]) ** 2)
        pairs_expected_values.extend(valid)

    if not pairs_observed or len(pairs_expected_values) < 2:
        return 0.0

    D_o = np.mean(pairs_observed)

    # Expected: squared differences across all values
    all_vals = np.array(pairs_expected_values)
    n = len(all_vals)
    # Efficient computation of mean squared difference
    mean_sq = np.mean(all_vals ** 2)
    mean_val = np.mean(all_vals)
    D_e = 2 * (mean_sq - mean_val ** 2) * n / (n - 1)

    if D_e == 0:
        return 1.0

    return float(1.0 - D_o / D_e)


# ── Main Stability Analyzer ──────────────────────────────────────────────

class StabilityAnalyzer:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)

    def get_annotation_groups(self, experiment_id: int) -> list[dict]:
        """Get completed annotation groups for per-chunk stability."""
        query = """
        SELECT
            chunk_id, construct_name, model_name, temperature, prompt_format,
            COUNT(*) AS num_runs,
            ARRAY_AGG(label_kind ORDER BY run_number) AS label_kinds,
            ARRAY_AGG(label_value_text ORDER BY run_number) AS label_texts,
            ARRAY_AGG(label_value_float ORDER BY run_number) AS label_floats,
            ARRAY_AGG(label_bin ORDER BY run_number) AS label_bins
        FROM llm_annotation_runs
        WHERE experiment_id = %s
          AND label_kind NOT IN ('error')
        GROUP BY chunk_id, construct_name, model_name, temperature, prompt_format
        HAVING COUNT(*) >= 3
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (experiment_id,))
            return cur.fetchall()

    def compute_chunk_stability(self, group: dict, experiment_id: int) -> dict:
        """Compute stability metrics for one chunk/construct/condition group."""
        construct = group['construct_name']
        construct_cfg = STABILITY_THRESHOLDS.get(construct, {})
        construct_type = construct_cfg.get('type', 'categorical')
        vocab = LABEL_VOCABULARIES.get(construct)

        kinds = group['label_kinds']
        n = len(kinds)

        # Compute none_rate and unclear_rate
        none_count = sum(1 for k in kinds if k == 'none')
        unclear_count = sum(1 for k in kinds if k == 'unclear')
        none_rate = none_count / n
        unclear_rate = unclear_count / n

        # Filter to valid (usable) labels only
        if construct_type == 'continuous':
            valid_values = [
                group['label_floats'][i]
                for i in range(n)
                if kinds[i] == 'float' and group['label_floats'][i] is not None
            ]
            valid_count = len(valid_values)

            if valid_count >= 2:
                metrics = compute_continuous_stability(
                    valid_values,
                    bins=vocab.get('bins', {}),
                    max_range=construct_cfg.get('max_range', 0.2),
                    max_stdev=construct_cfg.get('max_stdev', 0.10),
                )
            else:
                metrics = {
                    'agreement_ratio': None, 'label_entropy': None,
                    'modal_label': None, 'modal_count': None,
                    'mean_value': None, 'median_value': None,
                    'stdev_value': None, 'iqr_value': None, 'range_value': None,
                    'modal_bin': None, 'bin_agreement_ratio': None,
                    'is_stable': False,
                    'stability_rule': 'insufficient_valid_runs',
                }

            labels_observed = [str(v) for v in valid_values]

        else:  # categorical
            valid_labels = [
                group['label_texts'][i]
                for i in range(n)
                if kinds[i] == 'category' and group['label_texts'][i] is not None
            ]
            valid_count = len(valid_labels)

            if valid_count >= 2:
                threshold = construct_cfg.get('threshold', 0.8)
                metrics = compute_categorical_stability(valid_labels, threshold)
            else:
                metrics = {
                    'agreement_ratio': None, 'label_entropy': None,
                    'modal_label': None, 'modal_count': None,
                    'mean_value': None, 'median_value': None,
                    'stdev_value': None, 'iqr_value': None, 'range_value': None,
                    'modal_bin': None, 'bin_agreement_ratio': None,
                    'is_stable': False,
                    'stability_rule': 'insufficient_valid_runs',
                }

            labels_observed = valid_labels

        # Build distribution
        label_distribution = dict(Counter(labels_observed))

        return {
            'experiment_id': experiment_id,
            'chunk_id': group['chunk_id'],
            'construct_name': construct,
            'num_runs': n,
            'temperature': group['temperature'],
            'prompt_format': group['prompt_format'],
            'model_name': group['model_name'],
            'none_rate': round(none_rate, 4),
            'unclear_rate': round(unclear_rate, 4),
            'valid_run_count': valid_count,
            'construct_type': construct_type,
            'labels_observed': json.dumps(labels_observed),
            'label_distribution': json.dumps(label_distribution),
            **metrics,
        }

    def compute_group_reliability(self, experiment_id: int):
        """Compute group-level reliability metrics (Krippendorff alpha) across chunks."""
        # Get distinct conditions
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT construct_name, model_name, temperature, prompt_format
                FROM annotation_stability_metrics
                WHERE experiment_id = %s
            """, (experiment_id,))
            conditions = cur.fetchall()

        for cond in conditions:
            construct = cond['construct_name']
            construct_cfg = STABILITY_THRESHOLDS.get(construct, {})
            construct_type = construct_cfg.get('type', 'categorical')

            # Fetch all runs for this condition
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT chunk_id, run_number, label_kind,
                           label_value_text, label_value_float, label_bin
                    FROM llm_annotation_runs
                    WHERE experiment_id = %s
                      AND construct_name = %s
                      AND model_name = %s
                      AND temperature = %s
                      AND prompt_format = %s
                      AND label_kind NOT IN ('error')
                    ORDER BY chunk_id, run_number
                """, (experiment_id, construct, cond['model_name'],
                      cond['temperature'], cond['prompt_format']))
                rows = cur.fetchall()

            if not rows:
                continue

            # Organize into reliability matrix: items x coders
            from collections import defaultdict
            items = defaultdict(dict)
            for row in rows:
                items[row['chunk_id']][row['run_number']] = row

            # Get max run number
            all_runs = set()
            for chunk_runs in items.values():
                all_runs.update(chunk_runs.keys())
            run_numbers = sorted(all_runs)

            if construct_type == 'continuous':
                # Build matrix of floats
                reliability_data = []
                for chunk_id in sorted(items.keys()):
                    item_vals = []
                    for rn in run_numbers:
                        row = items[chunk_id].get(rn)
                        if row and row['label_kind'] == 'float' and row['label_value_float'] is not None:
                            item_vals.append(row['label_value_float'])
                        else:
                            item_vals.append(None)
                    reliability_data.append(item_vals)

                alpha = krippendorff_alpha_interval(reliability_data)
                icc = alpha  # Krippendorff alpha interval approximates ICC

                # Also compute mean within-chunk stdev
                stdevs = []
                for item_vals in reliability_data:
                    valid = [v for v in item_vals if v is not None]
                    if len(valid) >= 2:
                        stdevs.append(float(np.std(valid, ddof=1)))
                mean_stdev = float(np.mean(stdevs)) if stdevs else None
                mean_range_val = None
                ranges = []
                for item_vals in reliability_data:
                    valid = [v for v in item_vals if v is not None]
                    if len(valid) >= 2:
                        ranges.append(max(valid) - min(valid))
                mean_range_val = float(np.mean(ranges)) if ranges else None

                group_metrics = {
                    'krippendorff_alpha': round(alpha, 4),
                    'fleiss_kappa': None,
                    'icc_value': round(icc, 4),
                    'mean_stdev': round(mean_stdev, 4) if mean_stdev is not None else None,
                    'mean_range': round(mean_range_val, 4) if mean_range_val is not None else None,
                }

            else:
                # Build matrix of text labels
                reliability_data = []
                for chunk_id in sorted(items.keys()):
                    item_labels = []
                    for rn in run_numbers:
                        row = items[chunk_id].get(rn)
                        if row and row['label_kind'] == 'category' and row['label_value_text']:
                            item_labels.append(row['label_value_text'])
                        else:
                            item_labels.append(None)
                    reliability_data.append(item_labels)

                alpha = krippendorff_alpha_nominal(reliability_data)

                group_metrics = {
                    'krippendorff_alpha': round(alpha, 4),
                    'fleiss_kappa': round(alpha, 4),  # For nominal, alpha ~ kappa
                    'icc_value': None,
                    'mean_stdev': None,
                    'mean_range': None,
                }

            # Get chunk-level summary stats from stability metrics
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) AS num_chunks,
                        SUM(CASE WHEN is_stable THEN 1 ELSE 0 END) AS num_stable,
                        AVG(none_rate) AS avg_none_rate,
                        AVG(unclear_rate) AS avg_unclear_rate,
                        AVG(agreement_ratio) AS avg_agreement
                    FROM annotation_stability_metrics
                    WHERE experiment_id = %s
                      AND construct_name = %s
                      AND model_name = %s
                      AND temperature = %s
                      AND prompt_format = %s
                """, (experiment_id, construct, cond['model_name'],
                      cond['temperature'], cond['prompt_format']))
                summary = cur.fetchone()

            num_chunks = summary['num_chunks'] or 0
            num_stable = summary['num_stable'] or 0
            avg_none = summary['avg_none_rate'] or 0
            avg_unclear = summary['avg_unclear_rate'] or 0

            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO group_reliability_metrics (
                        experiment_id, construct_name, model_name, temperature, prompt_format,
                        num_chunks, num_stable_chunks,
                        coverage_rate, clarity_rate,
                        krippendorff_alpha, fleiss_kappa,
                        icc_value, mean_stdev, mean_range,
                        stability_rate, mean_agreement
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (experiment_id, construct_name, model_name, temperature, prompt_format)
                    DO UPDATE SET
                        num_chunks = EXCLUDED.num_chunks,
                        krippendorff_alpha = EXCLUDED.krippendorff_alpha,
                        stability_rate = EXCLUDED.stability_rate,
                        computed_at = NOW()
                """, (
                    experiment_id, construct, cond['model_name'],
                    cond['temperature'], cond['prompt_format'],
                    num_chunks, num_stable,
                    round(1.0 - avg_none, 4),
                    round(1.0 - avg_unclear, 4) if avg_none < 1.0 else None,
                    group_metrics['krippendorff_alpha'],
                    group_metrics['fleiss_kappa'],
                    group_metrics['icc_value'],
                    group_metrics['mean_stdev'],
                    group_metrics['mean_range'],
                    round(num_stable / num_chunks, 4) if num_chunks else 0,
                    round(float(summary['avg_agreement']), 4) if summary['avg_agreement'] else None,
                ))
                self.conn.commit()

            logger.info(
                f"  {construct} | {cond['model_name']} | T={cond['temperature']}: "
                f"alpha={group_metrics['krippendorff_alpha']}, "
                f"stable={num_stable}/{num_chunks}"
            )

    def run(self, experiment_id: int):
        """Full stability analysis for an experiment."""
        logger.info(f"Computing stability metrics for experiment {experiment_id}...")

        groups = self.get_annotation_groups(experiment_id)
        logger.info(f"Found {len(groups)} annotation groups")

        # Per-chunk stability
        batch = []
        for idx, group in enumerate(groups, 1):
            metrics = self.compute_chunk_stability(group, experiment_id)
            batch.append(metrics)

            if len(batch) >= 500:
                self._insert_chunk_metrics(batch)
                batch = []

            if idx % 1000 == 0:
                logger.info(f"Per-chunk progress: {idx}/{len(groups)}")

        if batch:
            self._insert_chunk_metrics(batch)

        logger.info("Per-chunk stability complete. Computing group-level reliability...")

        # Group-level reliability
        self.compute_group_reliability(experiment_id)

        logger.info("Stability analysis complete!")
        self._show_summary(experiment_id)

    def _insert_chunk_metrics(self, metrics: list[dict]):
        query = """
        INSERT INTO annotation_stability_metrics (
            experiment_id, chunk_id, construct_name,
            num_runs, temperature, prompt_format, model_name,
            none_rate, unclear_rate, valid_run_count,
            agreement_ratio, label_entropy, modal_label, modal_count,
            mean_value, median_value, stdev_value, iqr_value, range_value,
            modal_bin, bin_agreement_ratio,
            is_stable, stability_rule, construct_type,
            labels_observed, label_distribution
        ) VALUES %s
        ON CONFLICT (experiment_id, chunk_id, construct_name, temperature, prompt_format, model_name)
        DO UPDATE SET
            agreement_ratio = EXCLUDED.agreement_ratio,
            is_stable = EXCLUDED.is_stable,
            stdev_value = EXCLUDED.stdev_value,
            computed_at = NOW()
        """
        values = [
            (
                m['experiment_id'], m['chunk_id'], m['construct_name'],
                m['num_runs'], m['temperature'], m['prompt_format'], m['model_name'],
                m['none_rate'], m['unclear_rate'], m['valid_run_count'],
                m['agreement_ratio'], m['label_entropy'], m['modal_label'], m['modal_count'],
                m['mean_value'], m['median_value'], m['stdev_value'], m['iqr_value'], m['range_value'],
                m['modal_bin'], m['bin_agreement_ratio'],
                m['is_stable'], m['stability_rule'], m['construct_type'],
                m['labels_observed'], m['label_distribution'],
            )
            for m in metrics
        ]
        with self.conn.cursor() as cur:
            execute_values(cur, query, values)
            self.conn.commit()

    def _show_summary(self, experiment_id: int):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    construct_name, model_name, temperature,
                    krippendorff_alpha, stability_rate,
                    coverage_rate, clarity_rate,
                    icc_value, mean_stdev
                FROM group_reliability_metrics
                WHERE experiment_id = %s
                ORDER BY construct_name, model_name, temperature
            """, (experiment_id,))
            rows = cur.fetchall()

        logger.info("=== GROUP-LEVEL RELIABILITY SUMMARY ===")
        for r in rows:
            icc_str = f"ICC={r['icc_value']}" if r['icc_value'] is not None else ""
            logger.info(
                f"  {r['construct_name']:25s} | {r['model_name']:15s} | T={r['temperature']} | "
                f"alpha={r['krippendorff_alpha']} | stable={r['stability_rate']} | "
                f"coverage={r['coverage_rate']} {icc_str}"
            )

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    analyzer = StabilityAnalyzer()
    try:
        analyzer.run(args.experiment_id)
    finally:
        analyzer.close()
