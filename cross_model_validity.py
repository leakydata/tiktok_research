"""
Cross-model convergent validity analysis.

Treats each model as an independent "rater" and computes agreement across models
for the same chunk/construct. This is the centerpiece of the "humans aren't needed"
argument: if 7 architecturally different LLMs independently agree, that convergence
is strong validity evidence.

Outputs:
  - cross_model_agreement_matrix.csv  (pairwise model agreement)
  - cross_model_krippendorff.csv      (alpha treating models as raters)
  - cross_model_consensus.csv         (unanimous / majority agreement rates)
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import csv
import logging
from pathlib import Path
from collections import defaultdict
from itertools import combinations

from config import DB_CONFIG, STABILITY_THRESHOLDS
from stability import (
    krippendorff_alpha_nominal,
    krippendorff_alpha_interval,
    bootstrap_krippendorff_nominal,
    bootstrap_krippendorff_interval,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / 'outputs'


class CrossModelValidator:
    def __init__(self, experiment_id: int):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.experiment_id = experiment_id
        self.output_dir = OUTPUT_DIR / f'experiment_{experiment_id}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_final_labels_by_model(self, temperature: float = None):
        """Get stable final labels organized as {(chunk_id, construct): {model: label}}.

        Uses final_label_text for categorical, final_label_bin for continuous
        (so cross-model comparison is on the same scale).
        """
        temp_filter = "AND fa.temperature = %s" if temperature is not None else ""
        params = [self.experiment_id]
        if temperature is not None:
            params.append(temperature)

        query = f"""
            SELECT
                fa.chunk_id, fa.construct_name, fa.model_name,
                fa.final_label_text, fa.final_label_bin, fa.final_label_float,
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
              {temp_filter}
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        # Organize: {(chunk_id, construct_name): {model_name: label}}
        data = defaultdict(dict)
        construct_types = {}
        for row in rows:
            key = (row['chunk_id'], row['construct_name'])
            construct_types[row['construct_name']] = row['construct_type']
            # Use text label for categorical, bin label for continuous
            if row['construct_type'] == 'continuous':
                label = row['final_label_bin']
            else:
                label = row['final_label_text']
            if label:
                data[key][row['model_name']] = label
        return data, construct_types

    def get_final_floats_by_model(self, temperature: float = None):
        """Get stable continuous float values: {(chunk_id, construct): {model: float}}."""
        temp_filter = "AND temperature = %s" if temperature is not None else ""
        params = [self.experiment_id]
        if temperature is not None:
            params.append(temperature)

        query = f"""
            SELECT chunk_id, construct_name, model_name, final_label_float
            FROM final_annotations
            WHERE experiment_id = %s
              AND is_stable = TRUE
              AND final_label_float IS NOT NULL
              {temp_filter}
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        data = defaultdict(dict)
        for row in rows:
            key = (row['chunk_id'], row['construct_name'])
            data[key][row['model_name']] = row['final_label_float']
        return data

    def compute_pairwise_agreement(self, temperature: float = None):
        """Compute pairwise agreement rate between every model pair."""
        data, _ = self.get_final_labels_by_model(temperature)

        # Collect all models
        all_models = set()
        for model_labels in data.values():
            all_models.update(model_labels.keys())
        models = sorted(all_models)

        # Per-construct and overall
        constructs = sorted(set(k[1] for k in data.keys()))
        results = []

        for construct in constructs:
            # Filter to this construct
            construct_data = {k: v for k, v in data.items() if k[1] == construct}

            for m1, m2 in combinations(models, 2):
                agree = 0
                total = 0
                for (chunk_id, _), model_labels in construct_data.items():
                    if m1 in model_labels and m2 in model_labels:
                        total += 1
                        if model_labels[m1] == model_labels[m2]:
                            agree += 1

                if total > 0:
                    results.append({
                        'construct_name': construct,
                        'model_a': m1,
                        'model_b': m2,
                        'agreements': agree,
                        'total_compared': total,
                        'agreement_rate': round(agree / total, 4),
                        'temperature': temperature,
                    })

        return results

    def compute_cross_model_alpha(self, temperature: float = None):
        """Compute Krippendorff alpha treating each model as a rater.

        For each construct, builds a reliability matrix where:
          - Items = chunks
          - Coders = models
          - Values = labels (text for categorical, float for continuous)
        """
        data, construct_types = self.get_final_labels_by_model(temperature)
        float_data = self.get_final_floats_by_model(temperature)

        all_models = set()
        for model_labels in data.values():
            all_models.update(model_labels.keys())
        models = sorted(all_models)

        constructs = sorted(set(k[1] for k in data.keys()))
        results = []

        for construct in constructs:
            ctype = construct_types.get(construct, 'categorical')
            construct_data = {k: v for k, v in data.items() if k[1] == construct}
            chunk_ids = sorted(set(k[0] for k in construct_data.keys()))

            if ctype == 'continuous':
                # Use float values for interval alpha
                construct_floats = {k: v for k, v in float_data.items() if k[1] == construct}
                reliability_matrix = []
                for chunk_id in chunk_ids:
                    key = (chunk_id, construct)
                    model_vals = construct_floats.get(key, {})
                    item = [model_vals.get(m) for m in models]
                    reliability_matrix.append(item)

                if len(reliability_matrix) >= 3:
                    alpha, ci_lo, ci_hi = bootstrap_krippendorff_interval(reliability_matrix)
                else:
                    alpha, ci_lo, ci_hi = 0.0, 0.0, 0.0
            else:
                # Nominal alpha
                reliability_matrix = []
                for chunk_id in chunk_ids:
                    key = (chunk_id, construct)
                    model_labels = construct_data.get(key, {})
                    item = [model_labels.get(m) for m in models]
                    reliability_matrix.append(item)

                if len(reliability_matrix) >= 3:
                    alpha, ci_lo, ci_hi = bootstrap_krippendorff_nominal(reliability_matrix)
                else:
                    alpha, ci_lo, ci_hi = 0.0, 0.0, 0.0

            # Count how many chunks have labels from >= 2 models
            multi_model_chunks = sum(
                1 for item in reliability_matrix
                if sum(1 for v in item if v is not None) >= 2
            )

            results.append({
                'construct_name': construct,
                'construct_type': ctype,
                'num_models': len(models),
                'num_chunks': len(chunk_ids),
                'num_multi_model_chunks': multi_model_chunks,
                'cross_model_alpha': alpha,
                'alpha_ci_lower': ci_lo,
                'alpha_ci_upper': ci_hi,
                'temperature': temperature,
            })

            logger.info(
                f"  {construct}: cross-model alpha={alpha} [{ci_lo}, {ci_hi}] "
                f"({multi_model_chunks} chunks with 2+ models)"
            )

        return results

    def compute_consensus_rates(self, temperature: float = None):
        """Compute unanimous and majority agreement rates per construct."""
        data, _ = self.get_final_labels_by_model(temperature)

        all_models = set()
        for model_labels in data.values():
            all_models.update(model_labels.keys())
        n_models = len(all_models)

        constructs = sorted(set(k[1] for k in data.keys()))
        results = []

        for construct in constructs:
            construct_data = {k: v for k, v in data.items() if k[1] == construct}

            # Only consider chunks annotated by >= 2 models
            multi_model = {k: v for k, v in construct_data.items() if len(v) >= 2}
            total = len(multi_model)
            if total == 0:
                continue

            unanimous = 0
            majority = 0
            for (chunk_id, _), model_labels in multi_model.items():
                labels = list(model_labels.values())
                n = len(labels)
                from collections import Counter
                counts = Counter(labels)
                top_count = counts.most_common(1)[0][1]

                if top_count == n:
                    unanimous += 1
                if top_count > n / 2:
                    majority += 1

            results.append({
                'construct_name': construct,
                'total_chunks': total,
                'unanimous_count': unanimous,
                'unanimous_rate': round(unanimous / total, 4) if total else 0,
                'majority_count': majority,
                'majority_rate': round(majority / total, 4) if total else 0,
                'num_models_available': n_models,
                'temperature': temperature,
            })

        return results

    def run(self):
        """Run full cross-model validity analysis."""
        logger.info(f"Cross-model validity analysis for experiment {self.experiment_id}")

        # Run for each temperature and overall
        temperatures = [None, 0.0, 0.5]  # None = pooled
        all_agreement = []
        all_alpha = []
        all_consensus = []

        for temp in temperatures:
            temp_label = 'pooled' if temp is None else f'T={temp}'
            logger.info(f"\n--- {temp_label} ---")

            logger.info("Pairwise agreement...")
            all_agreement.extend(self.compute_pairwise_agreement(temp))

            logger.info("Cross-model Krippendorff alpha...")
            all_alpha.extend(self.compute_cross_model_alpha(temp))

            logger.info("Consensus rates...")
            all_consensus.extend(self.compute_consensus_rates(temp))

        # Write CSVs
        self._write_csv('cross_model_agreement_matrix.csv', all_agreement)
        self._write_csv('cross_model_krippendorff.csv', all_alpha)
        self._write_csv('cross_model_consensus.csv', all_consensus)

        logger.info(f"\nCross-model validity reports written to {self.output_dir}")

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
    parser = argparse.ArgumentParser(description="Cross-model convergent validity analysis")
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    validator = CrossModelValidator(args.experiment_id)
    try:
        validator.run()
    finally:
        validator.close()
