"""
Step 3: Prompt templates for health language constructs.
Each prompt explicitly distinguishes 'none' (no health content) from 'unclear'.

Versioning
----------
v1  Original Phase 1 prompts (all constructs).
v2  Phase 2 tightened operationalizations, applied only to boundary-sensitive
    constructs: agency_control and symptom_concreteness.
    All other constructs (certainty_hedging, temporal_orientation,
    social_proof, medical_authority) remain at v1 only.

Module-level function
---------------------
render_prompt(construct_name, version, chunk_text, context_carry=None)
    Returns (rendered_text, template_name, sha256_hash).
    template_name format: '{construct_name}_{version}', e.g. 'agency_control_v2'.
    prompt_hash is sha256 of the full rendered prompt after chunk insertion,
    providing a tamper-evident record of the exact text sent to each model.
"""

import hashlib


class HealthLanguagePrompts:
    """Theory-agnostic health language annotation prompts."""

    # ── V1 prompts (original Phase 1) ────────────────────────────────────

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
    def symptom_concreteness_v2(chunk_text: str, context_carry: str = None) -> str:
        """Phase 2 tightened version. Fixes: none/0.0 ambiguity, midpoint anchor
        drift, and averaging-across-symptom bias that caused low cross-model alpha."""
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Rate the SYMPTOM CONCRETENESS in this health-related statement.
Concreteness measures how specifically and precisely health symptoms are described.

[Scale]
0.0-0.3 = ABSTRACT/VAGUE
  A health complaint is present but unnamed or non-specific.
  Examples: "not feeling well," "something is off," "I feel awful," "not right"
  Named emotional states without physical description also score here (e.g., "anxious").

0.4-0.6 = NAMED SYMPTOM
  A recognizable symptom or condition is named but not measured or localized.
  Examples: "fatigue," "nausea," "headache," "pain," "dizziness," "brain fog,"
            "shortness of breath," "dissociation," "mood swings"
  The symptom is identifiable but lacks specificity (no body part, no number, no diagnosis).

0.7-1.0 = PRECISE/SPECIFIC
  Symptom includes localization, measurement, clinical terminology, or a named diagnosis.
  Examples: "left hip pain," "heart rate 140 bpm," "POTS," "albumin 2.1 g/dL,"
            "hypermobile EDS," "C-PTSD," "T1 diabetes"
  Score higher within this range for multiple precise descriptors or lab values.

[Rules]
1. Rate by the MOST CONCRETE symptom mentioned in the passage.
   If one symptom is vague but another is precise, score by the most specific one.
2. "none" means NO health content at all. This is different from 0.0-0.1:
   Use "none" ONLY when the passage contains no health-related content.
   Use 0.0-0.1 when health content is present but maximally vague.
3. Use "unclear" ONLY when health content is present but you cannot determine
   whether any symptom is being described (e.g., purely logistical health talk
   with no symptom mention).
4. Otherwise respond ONLY with a number between 0.0 and 1.0.

[Anchor examples]
"I just don't feel like myself lately." -> 0.1
"I've been exhausted and kind of nauseous." -> 0.4
"I've been having severe migraines, about 3-4 times a week." -> 0.6
"My cardiologist confirmed POTS -- resting HR was 110 and I hit 145 on tilt." -> 0.9

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
    def agency_control_v2(chunk_text: str, context_carry: str = None) -> str:
        """Phase 2 tightened version. Fixes: grammatical-voice confusion,
        helpless/passive boundary ambiguity, and mixed as an uncertainty escape
        that caused catastrophically low cross-model alpha (~0.16) in Phase 1."""
        context = f"[Prior context]\n{context_carry}\n\n" if context_carry else ""
        return f"""{context}[Task]
Classify the AGENCY/CONTROL STANCE in this health-related statement.

Agency/control reflects how the speaker POSITIONS THEMSELVES relative to their
health experience: are they directing events, receiving events, or expressing
loss of capacity to change their situation?

IMPORTANT: Grammatical voice is NOT the deciding factor.
"I was diagnosed with POTS" is grammatically passive but may express an ACTIVE
stance if the speaker is driving their own health narrative. Focus on whether
the speaker presents themselves as the agent or as the subject of their health trajectory.

[Categories]
- active: Speaker takes initiative, makes decisions, or manages their health.
  They are the driver of health-related actions, regardless of grammatical form.

- passive: Speaker receives care, follows instructions, or describes health
  events happening to them WITHOUT distress about lacking control.
  Neutral or accepting reception of health events.

- helpless: Speaker explicitly signals that control is LOST or UNAVAILABLE.
  Requires an EVALUATIVE or EMOTIONAL signal of futility, resignation, or
  incapacity -- not just grammatical passivity.
  Key markers: "nothing works," "I can't do anything," "I've given up," "at its mercy."
  Receiving treatment passively without distress = passive, NOT helpless.

- mixed: The SAME TEXT contains clear, simultaneous evidence of BOTH active
  AND helpless/passive stances within the same sentence or directly linked
  clauses. Both "I try to manage it but it just takes over completely" and
  "I try... but some days I give up completely" qualify as mixed.
  Do NOT use for general uncertainty. If stances differ across sentences,
  choose the DOMINANT one.

[Rules]
1. Assign the single label best describing the PRIMARY stance across the passage.
2. "mixed" requires unambiguous dual signals in the same sentence or closely
   linked clauses. Variation across sentences -> choose dominant stance.
3. "helpless" requires explicit futility, resignation, or incapacity language.
4. If NO health-related content, respond: "none"
5. If health content exists but stance is genuinely unresolvable, respond: "unclear"
6. Otherwise respond ONLY with: "active" OR "passive" OR "helpless" OR "mixed"

[Examples]
"I track my symptoms daily and brought a log to my rheumatologist."
-> active  (speaker directs their own management)

"They put me on a beta blocker and I go back in three months."
-> passive  (receiving care, no distress signal)

"I've tried everything. Nothing helps. I don't know what else I can do."
-> helpless  (explicit futility and resignation)

"I finally got diagnosed after years of not knowing -- now I'm building my own plan."
-> active  (grammatically passive "got diagnosed" but the overall stance is directive)

"I try to manage it but honestly some days it just takes over completely."
-> mixed  (active management AND helpless loss-of-control in the same utterance)

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

    # ── Registry and dispatch ─────────────────────────────────────────────

    _REGISTRY = None  # populated lazily below

    @classmethod
    def _get_registry(cls):
        if cls._REGISTRY is None:
            cls._REGISTRY = {
                ('certainty_hedging',    'v1'): cls.certainty_hedging,
                ('temporal_orientation', 'v1'): cls.temporal_orientation,
                ('symptom_concreteness', 'v1'): cls.symptom_concreteness,
                ('symptom_concreteness', 'v2'): cls.symptom_concreteness_v2,
                ('agency_control',       'v1'): cls.agency_control,
                ('agency_control',       'v2'): cls.agency_control_v2,
                ('social_proof',         'v1'): cls.social_proof,
                ('medical_authority',    'v1'): cls.medical_authority,
            }
        return cls._REGISTRY

    @classmethod
    def get_prompt_func(cls, construct_name: str):
        """Return the v1 prompt function (backwards-compatible entry point)."""
        return cls.get_versioned_prompt_func(construct_name, 'v1')

    @classmethod
    def get_versioned_prompt_func(cls, construct_name: str, version: str = 'v1'):
        """Return the prompt function for a given construct and version."""
        key = (construct_name, version)
        func = cls._get_registry().get(key)
        if func is None:
            raise ValueError(
                f"No prompt registered for construct={construct_name!r} version={version!r}. "
                f"Available: {list(cls._get_registry().keys())}"
            )
        return func

    @classmethod
    def get_all_construct_names(cls) -> list[str]:
        return [
            'certainty_hedging', 'temporal_orientation', 'symptom_concreteness',
            'agency_control', 'social_proof', 'medical_authority',
        ]

    @classmethod
    def available_versions(cls, construct_name: str) -> list[str]:
        """Return list of available versions for a construct."""
        return [v for (c, v) in cls._get_registry() if c == construct_name]


# ── Module-level convenience function ────────────────────────────────────────

def render_prompt(
    construct_name: str,
    version: str,
    chunk_text: str,
    context_carry: str = None,
) -> tuple:
    """Render a prompt and return (rendered_text, template_name, sha256_hash).

    Parameters
    ----------
    construct_name : str
        e.g. 'agency_control'
    version : str
        e.g. 'v1' or 'v2'
    chunk_text : str
        The transcript chunk to annotate.
    context_carry : str, optional
        Prior context text (last N words of the preceding chunk).

    Returns
    -------
    rendered_text : str
        The full prompt text sent to the model.
    template_name : str
        '{construct_name}_{version}', e.g. 'agency_control_v2'. Stored in DB
        for human-readable audit without needing to re-hash.
    prompt_hash : str
        SHA-256 hex digest of rendered_text. Provides a tamper-evident record
        of the exact prompt text used for each annotation run.
    """
    func = HealthLanguagePrompts.get_versioned_prompt_func(construct_name, version)
    rendered_text = func(chunk_text, context_carry=context_carry)
    template_name = f"{construct_name}_{version}"
    prompt_hash = hashlib.sha256(rendered_text.encode('utf-8')).hexdigest()
    return rendered_text, template_name, prompt_hash
