"""
Step 3: Prompt templates for health language constructs.
Each prompt explicitly distinguishes 'none' (no health content) from 'unclear'.
"""


class HealthLanguagePrompts:
    """Theory-agnostic health language annotation prompts."""

    @staticmethod
    def certainty_hedging(chunk_text: str, context_carry: str = None) -> str:
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Rate the CERTAINTY/HEDGING level in this health-related statement.

[Scale]
0.0-0.3 = HIGH HEDGING (maybe, possibly, I think, might be, could be)
0.4-0.6 = MODERATE CERTAINTY (probably, seems like, appears to be)
0.7-1.0 = HIGH CERTAINTY (definitely, I know, confirmed, diagnosed, absolutely)

[Guidelines]
- Focus on linguistic certainty markers, not medical accuracy
- Hedging = uncertainty words, qualifiers, doubt expressions
- Certainty = definitive statements, confident assertions
- If there is NO health-related content at all, respond: "none"
- If health content exists but certainty level is ambiguous, respond: "unclear"
- Otherwise respond ONLY with a number between 0.0 and 1.0

[Segment]
{chunk_text}

[Response]
"""

    @staticmethod
    def temporal_orientation(chunk_text: str, context_carry: str = None) -> str:
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Identify the dominant TEMPORAL ORIENTATION in this health-related statement.

[Categories]
- past: Focus on what happened before, past tense dominance, historical narrative
- present: Focus on current state, present tense, ongoing experiences
- future: Focus on predictions, expectations, fears about what will happen
- mixed: No clear dominant temporal frame

[Guidelines]
- Look at verb tenses and temporal markers
- Choose the DOMINANT orientation if multiple are present
- If there is NO health-related content at all, respond: "none"
- If health content exists but temporal frame is ambiguous, respond: "unclear"
- Otherwise respond ONLY with: "past" OR "present" OR "future" OR "mixed"

[Segment]
{chunk_text}

[Response]
"""

    @staticmethod
    def symptom_concreteness(chunk_text: str, context_carry: str = None) -> str:
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Rate the SYMPTOM CONCRETENESS in this health-related statement.

[Scale]
0.0-0.3 = ABSTRACT/VAGUE (not feeling right, off, weird, something's wrong)
0.4-0.6 = MODERATE (pain, tired, dizzy, nausea, sick)
0.7-1.0 = CONCRETE (left knee subluxation, heart rate 140bpm, temperature 101.3F, albumin 2.1)

[Guidelines]
- Concrete = specific body parts, measurements, medical terms, named conditions
- Abstract = vague feelings, general malaise, non-specific complaints
- If there is NO health or symptom content at all, respond: "none"
- If health content exists but concreteness is ambiguous, respond: "unclear"
- Otherwise respond ONLY with a number between 0.0 and 1.0

[Segment]
{chunk_text}

[Response]
"""

    @staticmethod
    def agency_control(chunk_text: str, context_carry: str = None) -> str:
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Classify the AGENCY/CONTROL language in this health-related statement.

[Categories]
- active: Person in control, taking action (I'm managing, I decided, I'm working on, I started)
- passive: Things happening to them, receiving actions (it happened, I was told, doctors say, they put me on)
- helpless: Loss of control, powerlessness (nothing works, can't control, trapped, at its mercy, giving up)
- mixed: Combination of agency levels or unclear

[Guidelines]
- Look for grammatical voice and control language
- Active = subject controls the action
- Passive = subject receives the action
- If there is NO health-related content at all, respond: "none"
- If health content exists but agency is ambiguous, respond: "unclear"
- Otherwise respond ONLY with: "active" OR "passive" OR "helpless" OR "mixed"

[Segment]
{chunk_text}

[Response]
"""

    @staticmethod
    def social_proof(chunk_text: str, context_carry: str = None) -> str:
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Detect SOCIAL PROOF/VALIDATION language in this health-related statement.

[Definition]
Social proof = References to others with similar health experiences, community validation, group belonging

[Examples]
PRESENT: "everyone in my support group", "other patients like me", "my community", "others with EDS", "we all experience this", "so many of you said"
ABSENT: Only discussing own personal experience without reference to others

[Guidelines]
- Must explicitly reference other people with similar health experiences
- Generic "people" or "they" without health context = absent
- If there is NO health-related content at all, respond: "none"
- If health content exists but social proof presence is ambiguous, respond: "unclear"
- Otherwise respond ONLY with: "present" OR "absent"

[Segment]
{chunk_text}

[Response]
"""

    @staticmethod
    def medical_authority(chunk_text: str, context_carry: str = None) -> str:
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Classify MEDICAL AUTHORITY references in this health-related statement.

[Categories]
- professional: References doctors, specialists, medical tests, professional diagnoses (my doctor said, test results showed, neurologist confirmed, ER visit)
- self_research: References own research, online sources, self-investigation (I read that, I researched, according to what I found, TikTok taught me)
- mixed: Both professional and self-directed references present
- none_observed: No authority references of any kind

[Guidelines]
- Look for explicit authority citations or knowledge source references
- If there is NO health-related content at all, respond: "none"
- If health content exists but authority source is ambiguous, respond: "unclear"
- Otherwise respond ONLY with: "professional" OR "self_research" OR "mixed" OR "none_observed"

[Segment]
{chunk_text}

[Response]
"""

    @classmethod
    def get_prompt_func(cls, construct_name: str):
        """Return the prompt function for a given construct."""
        mapping = {
            'certainty_hedging': cls.certainty_hedging,
            'temporal_orientation': cls.temporal_orientation,
            'symptom_concreteness': cls.symptom_concreteness,
            'agency_control': cls.agency_control,
            'social_proof': cls.social_proof,
            'medical_authority': cls.medical_authority,
        }
        if construct_name not in mapping:
            raise ValueError(f"Unknown construct: {construct_name}")
        return mapping[construct_name]

    @classmethod
    def get_all_construct_names(cls) -> list[str]:
        return [
            'certainty_hedging', 'temporal_orientation', 'symptom_concreteness',
            'agency_control', 'social_proof', 'medical_authority',
        ]
