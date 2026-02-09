"""
Step 1: Select study cohort from existing tiktok_disorders database.
Uses development/reliability/holdout splits (methods paper framing).
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from config import DB_CONFIG, COHORT_SPLITS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def select_cohort(min_videos: int = 5, min_days_span: int = 180, max_creators: int = 100):
    """Select longitudinal creators for study cohort."""
    conn = psycopg2.connect(**DB_CONFIG)

    # Check if cohort already exists
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM study_cohort")
        existing_count = cur.fetchone()[0]
        if existing_count > 0:
            logger.warning(f"Study cohort already has {existing_count} creators. Skipping.")
            logger.info("To re-select, run: DELETE FROM study_cohort;")
            conn.close()
            return

    logger.info(f"Selecting cohort: >= {min_videos} videos, >= {min_days_span} days span")

    insert_query = """
    INSERT INTO study_cohort (
        creator_username, video_count, transcript_count,
        first_video_date, last_video_date, days_span,
        diagnoses_claimed, primary_diagnosis
    )
    SELECT
        v.author AS creator_username,
        COUNT(DISTINCT v.id) AS video_count,
        COUNT(DISTINCT t.id) AS transcript_count,
        MIN(v.upload_date) AS first_video_date,
        MAX(v.upload_date) AS last_video_date,
        EXTRACT(DAY FROM MAX(v.upload_date) - MIN(v.upload_date))::INTEGER AS days_span,
        ARRAY_AGG(DISTINCT cd.condition_code)
            FILTER (WHERE cd.condition_code IS NOT NULL) AS diagnoses_claimed,
        MODE() WITHIN GROUP (ORDER BY cd.condition_code) AS primary_diagnosis
    FROM videos v
    INNER JOIN transcripts t ON t.video_id = v.id
    LEFT JOIN claimed_diagnoses cd ON cd.video_id = v.id
    WHERE t.text IS NOT NULL
      AND LENGTH(t.text) > 100
    GROUP BY v.author
    HAVING COUNT(DISTINCT v.id) >= %s
       AND EXTRACT(DAY FROM MAX(v.upload_date) - MIN(v.upload_date)) >= %s
    ORDER BY COUNT(DISTINCT v.id) DESC, days_span DESC
    LIMIT %s;
    """

    with conn.cursor() as cur:
        cur.execute(insert_query, (min_videos, min_days_span, max_creators))
        conn.commit()
        logger.info(f"Inserted {cur.rowcount} creators into study_cohort")

    # Assign splits: development / reliability / holdout
    dev_pct = COHORT_SPLITS['development']
    rel_pct = COHORT_SPLITS['development'] + COHORT_SPLITS['reliability']

    split_query = """
    WITH numbered AS (
        SELECT
            cohort_id,
            ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn,
            COUNT(*) OVER () AS total
        FROM study_cohort
    )
    UPDATE study_cohort sc
    SET cohort_split = CASE
        WHEN n.rn <= n.total * %s THEN 'development'
        WHEN n.rn <= n.total * %s THEN 'reliability'
        ELSE 'holdout'
    END
    FROM numbered n
    WHERE sc.cohort_id = n.cohort_id;
    """

    with conn.cursor() as cur:
        cur.execute(split_query, (dev_pct, rel_pct))
        conn.commit()
        logger.info("Assigned development/reliability/holdout splits")

    # Summary
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                cohort_split,
                COUNT(*) AS count,
                ROUND(AVG(video_count), 1) AS avg_videos,
                ROUND(AVG(transcript_count), 1) AS avg_transcripts,
                ROUND(AVG(days_span), 0) AS avg_days
            FROM study_cohort
            GROUP BY cohort_split
            ORDER BY cohort_split
        """)
        for row in cur.fetchall():
            logger.info(
                f"  {row['cohort_split']}: {row['count']} creators, "
                f"avg {row['avg_videos']} videos, {row['avg_transcripts']} transcripts, "
                f"{row['avg_days']} days"
            )

    conn.close()


if __name__ == "__main__":
    select_cohort()
