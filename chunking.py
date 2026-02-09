"""
Step 2: Chunk transcripts using multi-sentence grouping with character budget.
Supports two strategies:
  - multi_sentence: Groups sentences to hit ~150-400 chars (configurable)
  - whole_transcript: Uses entire transcript as one chunk (for short videos)
Records chunking_method and chunking_version for reproducibility.
"""

import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
import re
import json
import logging
from config import DB_CONFIG, CHUNKING_CONFIGS, DEFAULT_CHUNKING_METHOD, CHUNKING_BATCH_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CHUNKING_VERSION = 'v1'

# Regex-based sentence splitter for spoken/transcript text.
# No NLTK dependency — punkt performs poorly on Whisper-generated TikTok
# transcripts anyway (inconsistent punctuation, run-on speech, filler words).

# Pattern splits on:
#   1. Standard sentence endings: .!? followed by space and uppercase or quote
#   2. Ellipsis followed by space and uppercase
#   3. Spoken discourse markers followed by uppercase (captures natural pauses)

_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'])'     # standard: ". The" / "! She" / "? I"
    r'|(?<=\.\.\.)\s+(?=[A-Z])'      # ellipsis: "... So"
    r'|(?<=[.!?])\s+(?=\d)'          # sentence ending before number: ". 3 days"
)


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences using regex. No external data files needed.

    Designed for Whisper-generated spoken transcripts where punctuation is
    inconsistent. Falls back to discourse-marker splitting for unpunctuated text.
    """
    text = text.strip()
    if not text:
        return []

    # First try the standard boundary regex
    parts = _SENTENCE_BOUNDARY.split(text)
    parts = [p.strip() for p in parts if p.strip()]

    # If we got a single mega-chunk (no punctuation at all), try splitting
    # on spoken discourse markers that signal topic/thought boundaries.
    # Uses lookahead (not lookbehind) to avoid variable-length lookbehind error.
    if len(parts) == 1 and len(text) > 300:
        discourse_parts = re.split(
            r'\s+(?=(?:[Ss]o |[Aa]nd then |[Bb]ut |[Oo]kay so |[Aa]nyway |[Bb]asically ))',
            text
        )
        if len(discourse_parts) > 1:
            parts = [p.strip() for p in discourse_parts if p.strip()]

    # Last resort: if still one giant block, split on any comma/semicolon
    # followed by a reasonable clause length
    if len(parts) == 1 and len(text) > 500:
        clause_parts = re.split(r'[,;]\s+', text)
        if len(clause_parts) > 2:
            # Re-merge very short fragments
            merged = []
            current = clause_parts[0]
            for part in clause_parts[1:]:
                if len(current) < 80:
                    current += ', ' + part
                else:
                    merged.append(current)
                    current = part
            merged.append(current)
            parts = [p.strip() for p in merged if p.strip()]

    return parts if parts else [text]


class TranscriptChunker:
    def __init__(self, method: str = DEFAULT_CHUNKING_METHOD):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.method = method
        self.params = CHUNKING_CONFIGS[method]

    def get_cohort_transcripts(self, splits: list[str] = None):
        """Fetch transcripts for study cohort, optionally filtered by split."""
        split_filter = ""
        query_params = []
        if splits:
            placeholders = ','.join(['%s'] * len(splits))
            split_filter = f"AND sc.cohort_split IN ({placeholders})"
            query_params = splits

        query = f"""
        SELECT
            t.id AS transcript_id,
            t.video_id,
            t.text AS transcript_text,
            v.author AS creator_username,
            v.upload_date AS video_date,
            sc.first_video_date,
            EXTRACT(DAY FROM v.upload_date - sc.first_video_date)::INTEGER AS days_since_first,
            (
                SELECT ne.content_type
                FROM narrative_elements ne
                WHERE ne.video_id = v.id
                LIMIT 1
            ) AS content_type
        FROM transcripts t
        INNER JOIN videos v ON t.video_id = v.id
        INNER JOIN study_cohort sc ON v.author = sc.creator_username
        WHERE t.text IS NOT NULL
          AND LENGTH(t.text) > 100
          {split_filter}
        ORDER BY v.author, v.upload_date;
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, query_params)
            return cur.fetchall()

    def chunk_multi_sentence(self, transcript: dict) -> list[tuple]:
        """Group sentences into chunks targeting the character budget."""
        text = transcript['transcript_text']
        sentences = _sentence_split(text)
        min_chars = self.params['min_chars']
        target_chars = self.params['target_chars']
        max_chars = self.params['max_chars']
        context_carry_words = self.params.get('context_carry_words', 15)

        chunks = []
        current_sentences = []
        current_length = 0
        chunk_seq = 0
        prev_chunk_tail = None

        for sentence in sentences:
            s_len = len(sentence)

            # If adding this sentence stays within max, accumulate
            if current_length + s_len + 1 <= max_chars:
                current_sentences.append(sentence)
                current_length += s_len + 1  # +1 for space
            else:
                # Flush current chunk if it meets minimum
                if current_sentences and current_length >= min_chars:
                    chunk_seq += 1
                    chunk_text = ' '.join(current_sentences)
                    chunks.append(self._make_chunk_tuple(
                        transcript, chunk_seq, chunk_text,
                        len(current_sentences), prev_chunk_tail
                    ))
                    # Prepare context carry for next chunk
                    words = chunk_text.split()
                    prev_chunk_tail = ' '.join(words[-context_carry_words:]) if len(words) > context_carry_words else chunk_text
                    current_sentences = [sentence]
                    current_length = s_len
                elif current_sentences:
                    # Below minimum — add sentence anyway to avoid losing content
                    current_sentences.append(sentence)
                    current_length += s_len + 1
                else:
                    # Single sentence exceeds max — take it as-is
                    current_sentences = [sentence]
                    current_length = s_len

        # Flush remaining
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            # If too short, merge with previous chunk
            if chunks and len(chunk_text) < min_chars:
                # Pop previous, merge
                prev = chunks.pop()
                merged_text = prev[3] + ' ' + chunk_text  # prev chunk_text is index 3
                merged_sentences = prev[4] + len(current_sentences)  # prev sentence_count is index 4
                chunks.append(self._make_chunk_tuple(
                    transcript, prev[2], merged_text,
                    merged_sentences, prev[7]  # prev context_carry is index 7
                ))
            else:
                chunk_seq += 1
                chunks.append(self._make_chunk_tuple(
                    transcript, chunk_seq, chunk_text,
                    len(current_sentences), prev_chunk_tail
                ))

        return chunks

    def chunk_whole_transcript(self, transcript: dict) -> list[tuple]:
        """Use entire transcript as one chunk (truncated if needed)."""
        text = transcript['transcript_text']
        max_chars = self.params.get('max_chars', 2000)
        if len(text) > max_chars:
            text = text[:max_chars]

        return [self._make_chunk_tuple(
            transcript, 1, text,
            len(_sentence_split(text)), None
        )]

    def _make_chunk_tuple(self, transcript, seq, text, sentence_count, context_carry):
        """Create a tuple for batch insertion."""
        return (
            transcript['transcript_id'],
            transcript['video_id'],
            seq,
            text,
            sentence_count,
            len(text),
            self.method,
            CHUNKING_VERSION,
            json.dumps(self.params),
            context_carry,
            transcript['video_date'],
            transcript['creator_username'],
            transcript.get('days_since_first') or 0,
            transcript.get('content_type'),
        )

    def insert_chunks(self, chunks: list[tuple]):
        """Batch insert chunks."""
        query = """
        INSERT INTO annotation_chunks (
            transcript_id, video_id, chunk_sequence, chunk_text,
            sentence_count, char_length,
            chunking_method, chunking_version, chunking_params,
            context_carry_text,
            video_date, creator_username, days_since_first_video,
            content_type
        ) VALUES %s
        ON CONFLICT (transcript_id, chunk_sequence, chunking_method, chunking_version)
        DO NOTHING;
        """
        with self.conn.cursor() as cur:
            execute_values(cur, query, chunks)
            self.conn.commit()

    def run(self, splits: list[str] = None):
        """Main execution."""
        logger.info(f"Chunking with method='{self.method}', version='{CHUNKING_VERSION}'")
        logger.info(f"Params: {json.dumps(self.params)}")

        transcripts = self.get_cohort_transcripts(splits)
        logger.info(f"Loaded {len(transcripts)} transcripts")

        chunk_func = (
            self.chunk_multi_sentence if self.method == 'multi_sentence'
            else self.chunk_whole_transcript
        )

        total_chunks = 0
        batch = []

        for idx, transcript in enumerate(transcripts, 1):
            chunks = chunk_func(transcript)
            batch.extend(chunks)
            total_chunks += len(chunks)

            if len(batch) >= CHUNKING_BATCH_SIZE:
                self.insert_chunks(batch)
                logger.info(f"Inserted batch, total chunks so far: {total_chunks}")
                batch = []

            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(transcripts)} transcripts")

        if batch:
            self.insert_chunks(batch)

        logger.info(f"Chunking complete: {total_chunks} chunks from {len(transcripts)} transcripts")
        self._show_stats()
        self.conn.close()

    def _show_stats(self):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    chunking_method,
                    COUNT(*) AS total_chunks,
                    COUNT(DISTINCT creator_username) AS unique_creators,
                    ROUND(AVG(char_length)) AS avg_length,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY char_length) AS median_length,
                    MIN(char_length) AS min_length,
                    MAX(char_length) AS max_length,
                    ROUND(AVG(sentence_count), 1) AS avg_sentences
                FROM annotation_chunks
                GROUP BY chunking_method
            """)
            for row in cur.fetchall():
                logger.info(f"=== {row['chunking_method']} ===")
                for k, v in row.items():
                    if k != 'chunking_method':
                        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default=DEFAULT_CHUNKING_METHOD, choices=list(CHUNKING_CONFIGS.keys()))
    parser.add_argument('--splits', nargs='+', default=None, help='Cohort splits to include (development, reliability, holdout)')
    args = parser.parse_args()

    chunker = TranscriptChunker(method=args.method)
    chunker.run(splits=args.splits)
