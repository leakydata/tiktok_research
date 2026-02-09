"""
Step 7: Validate multi-run annotations against existing narrative_elements data.
Maps our constructs to overlapping fields in the existing single-pass extraction:
  - social_proof <-> mentions_online_community, mentions_other_creators
  - medical_authority <-> mentions_professional_diagnosis, cites_medical_sources,
                         claims_expert_knowledge
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import logging

from config import DB_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mapping: (our_construct, our_label) -> (narrative_field, expected_value_logic)
VALIDATION_MAPPINGS = [
    {
        'construct': 'social_proof',
        'narrative_field': 'mentions_online_community',
        'label_to_expected': {
            'present': True,   # If we say social_proof=present, narrative should be True
            'absent': False,
        },
    },
    {
        'construct': 'medical_authority',
        'narrative_field': 'mentions_professional_diagnosis',
        'label_to_expected': {
            'professional': True,
            'mixed': True,
            'self_research': False,
            'none_observed': False,
        },
    },
    {
        'construct': 'medical_authority',
        'narrative_field': 'cites_medical_sources',
        'label_to_expected': {
            'professional': True,
            'self_research': True,
            'mixed': True,
            'none_observed': False,
        },
    },
]


class ValidationComparator:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)

    def run(self, experiment_id: int):
        """Compare final annotations with existing narrative_elements."""
        logger.info(f"Running validation for experiment {experiment_id}...")

        total_comparisons = 0
        total_agreements = 0

        for mapping in VALIDATION_MAPPINGS:
            construct = mapping['construct']
            nar_field = mapping['narrative_field']
            label_map = mapping['label_to_expected']

            # Get final annotations for this construct joined with narrative_elements
            query = f"""
            SELECT
                fa.chunk_id,
                fa.final_label_text,
                fa.is_stable,
                ac.video_id,
                ne.{nar_field} AS narrative_value
            FROM final_annotations fa
            INNER JOIN annotation_chunks ac ON fa.chunk_id = ac.chunk_id
            INNER JOIN narrative_elements ne ON ac.video_id = ne.video_id
            WHERE fa.experiment_id = %s
              AND fa.construct_name = %s
              AND fa.is_stable = TRUE
              AND fa.final_label_text IS NOT NULL
            """

            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (experiment_id, construct))
                rows = cur.fetchall()

            if not rows:
                logger.info(f"  {construct} vs {nar_field}: no data")
                continue

            comparisons = []
            agreements = 0
            total = 0

            for row in rows:
                label = row['final_label_text']
                nar_val = row['narrative_value']

                if label not in label_map:
                    continue

                expected = label_map[label]
                # narrative_elements stores booleans
                actual = bool(nar_val) if nar_val is not None else None
                if actual is None:
                    continue

                agrees = (expected == actual)
                if agrees:
                    agreements += 1
                total += 1

                comparisons.append((
                    experiment_id,
                    row['chunk_id'],
                    row['video_id'],
                    construct,
                    label,
                    row['is_stable'],
                    nar_field,
                    str(nar_val),
                    agrees,
                    None,  # comparison_notes
                ))

            # Insert comparisons
            if comparisons:
                self._insert_comparisons(comparisons)

            agreement_rate = agreements / total if total else 0
            logger.info(
                f"  {construct} vs {nar_field}: "
                f"{agreements}/{total} agree ({agreement_rate:.1%})"
            )

            total_comparisons += total
            total_agreements += agreements

        overall = total_agreements / total_comparisons if total_comparisons else 0
        logger.info(f"Overall validation agreement: {total_agreements}/{total_comparisons} ({overall:.1%})")

    def _insert_comparisons(self, rows):
        query = """
        INSERT INTO validation_comparisons (
            experiment_id, chunk_id, video_id,
            construct_name, annotation_label, annotation_is_stable,
            narrative_field, narrative_value,
            agrees, comparison_notes
        ) VALUES %s
        ON CONFLICT DO NOTHING
        """
        with self.conn.cursor() as cur:
            execute_values(cur, query, rows)
            self.conn.commit()

    def show_summary(self, experiment_id: int):
        """Show validation summary."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    construct_name,
                    narrative_field,
                    COUNT(*) AS total,
                    SUM(CASE WHEN agrees THEN 1 ELSE 0 END) AS agreed,
                    ROUND(
                        SUM(CASE WHEN agrees THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0),
                        3
                    ) AS agreement_rate
                FROM validation_comparisons
                WHERE experiment_id = %s
                GROUP BY construct_name, narrative_field
                ORDER BY construct_name, narrative_field
            """, (experiment_id,))
            rows = cur.fetchall()

        logger.info("=== VALIDATION SUMMARY ===")
        for r in rows:
            logger.info(
                f"  {r['construct_name']:20s} vs {r['narrative_field']:30s}: "
                f"{r['agreed']}/{r['total']} ({r['agreement_rate']})"
            )

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-id', type=int, required=True)
    args = parser.parse_args()

    v = ValidationComparator()
    try:
        v.run(args.experiment_id)
        v.show_summary(args.experiment_id)
    finally:
        v.close()
