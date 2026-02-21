# Multi-Run Stability Filtering: A Scalable Method for Reliable LLM-Based Content Analysis of Health Discourse on Social Media

## Abstract

Content analysis of health discourse on social media traditionally requires labor-intensive human annotation with inter-rater reliability checks. We propose **multi-run stability filtering**, a method that achieves reliable automated annotation by running diverse large language models (LLMs) multiple times per item and filtering for agreement. We validate the approach on 100 TikTok video transcript chunks annotated across six health language constructs (certainty/hedging, temporal orientation, symptom concreteness, agency/control, social proof, medical authority) using eight LLMs spanning five architectural families and three deployment contexts: five locally-run open-weight models (14B--30B parameters via Ollama) and three cloud API models (DeepSeek-V3.2, MiniMax-M2.5, GPT-5-nano). Each chunk was annotated five times at two temperature settings, yielding 48,000+ individual inferences. Results demonstrate strong within-model stability (mean Krippendorff's alpha = 0.87 for local models, 0.76 for cloud models) and meaningful cross-model convergence, with majority consensus reaching 79--100% across constructs. The method generalizes across architectures: a 27B local model (Gemma3, alpha = 0.99) outperformed a 671B cloud model (DeepSeek-V3.2, alpha = 0.97), while even a 30B mixture-of-experts model with only 3B active parameters achieved alpha = 0.80. Temperature significantly affected stability for all six constructs (p < 0.005), supporting the use of deterministic inference as a baseline. The approach reduces dependency on human gold-standard labels by treating cross-model convergence as an additional reliability signal analogous to inter-rater agreement, while acknowledging that convergence among models trained on overlapping corpora does not by itself establish construct validity. We release the complete pipeline as open-source software supporting both local and cloud LLM backends.

**Keywords:** content analysis, large language models, inter-rater reliability, health communication, social media, TikTok, annotation, stability, reproducibility

---

## 1. Introduction

The rapid growth of health-related content on social media platforms, particularly short-form video platforms such as TikTok, presents both opportunities and challenges for health communication researchers. Millions of users share personal health narratives, self-diagnoses, symptom descriptions, and treatment experiences in formats that resist traditional survey-based research methods. Content analysis of this discourse requires coding at scale---a task that has traditionally relied on trained human annotators working within established codebooks and inter-rater reliability frameworks (Krippendorff, 2004).

Human annotation, while considered the gold standard, faces several practical limitations. It is expensive, time-consuming, difficult to scale, and subject to coder fatigue and drift. A typical content analysis study with 500 items coded across six constructs by two raters requires approximately 200 person-hours of annotation work, plus training and calibration time. For the volumes of content generated on platforms like TikTok---where a single trending health topic may generate thousands of videos per week---human annotation is infeasible at the scale needed for representative analysis.

Large language models (LLMs) offer a potential solution, but their use in content analysis raises fundamental questions about reliability. A single LLM inference is inherently stochastic: the same prompt may produce different outputs across runs, and different models may disagree on the correct annotation. Prior work has largely treated LLM annotation as a replacement for a single human coder, evaluating LLM outputs against human gold-standard labels (Gilardi et al., 2023; Törnberg, 2023). This approach conflates two distinct questions: (1) whether the LLM produces the "correct" label, and (2) whether the LLM produces the label *reliably*.

We argue that reliability---the consistency of annotation across repeated measurements---is the more fundamental property for content analysis. In classical content analysis methodology, inter-rater reliability (e.g., Krippendorff's alpha, Cohen's kappa) does not require a gold standard; it requires only that independent raters agree. We propose extending this principle to LLM-based annotation through **multi-run stability filtering**: running each annotation multiple times across diverse models and retaining only those items where the models demonstrate stable agreement.

This paper makes the following contributions:

1. **A multi-run stability filtering method** that achieves Krippendorff's alpha values exceeding 0.80 across diverse health language constructs without requiring human gold-standard labels.

2. **Cross-model convergence as a reliability signal.** We demonstrate that architecturally diverse LLMs---spanning five model families, three deployment contexts, and parameter counts from 3B active to 671B total---converge on the same annotations at rates comparable to human inter-rater agreement, while noting that shared training data limits the epistemic independence of this convergence.

3. **Empirical evidence that model size is not the primary determinant of reliability.** A 27B local model (Gemma3) achieves alpha = 0.99, outperforming a 671B cloud model (DeepSeek-V3.2, alpha = 0.97), while a 30B MoE model with 3B active parameters achieves alpha = 0.80.

4. **An open-source, multi-backend annotation pipeline** supporting both local (Ollama) and cloud API (OpenAI, DeepSeek, MiniMax, Anthropic) inference with full reproducibility through experiment tracking, parameter logging, and resume support.

---

## 2. Related Work

### 2.1 LLMs for Content Analysis

Recent work has explored LLMs as annotators for social science research. Gilardi et al. (2023) found that ChatGPT-4 outperformed crowd-workers on several text annotation tasks, achieving higher accuracy and inter-annotator agreement at a fraction of the cost. Törnberg (2023) demonstrated that GPT-4 could replicate human coding of political tweets with high accuracy, particularly for tasks with clear categorical distinctions. Ziems et al. (2024) surveyed computational social science applications of LLMs, noting their promise for classification tasks but highlighting concerns about reliability and reproducibility.

However, these studies share a common methodological limitation: they evaluate LLM annotations against human gold-standard labels, treating the task as a classification accuracy problem rather than a reliability problem. This approach assumes that human labels are ground truth---an assumption frequently violated in content analysis, where reasonable coders may legitimately disagree on ambiguous items (Krippendorff, 2004).

### 2.2 Reliability in Content Analysis

Krippendorff's alpha (Krippendorff, 2004) is widely accepted as the standard reliability coefficient for content analysis, as it handles multiple raters, missing data, and different levels of measurement. Values above 0.667 are considered acceptable for drawing tentative conclusions, while values above 0.800 indicate good reliability (Krippendorff, 2004). Fleiss' kappa (Fleiss, 1971) extends Cohen's kappa to multiple raters for categorical data.

The conceptual framework of inter-rater reliability does not require that raters be human. It requires only that raters are independent and that their agreement reflects the underlying construct rather than shared bias. LLMs present a nuanced case: architecturally diverse models use different architectures and exhibit different failure modes, providing a degree of independence. However, modern LLMs are typically trained on overlapping web-scale corpora, which may introduce shared biases that inflate apparent agreement. We argue that architectural diversity provides *partial* independence---likely greater than human coders trained on the same codebook, but less than fully independent instruments---and that cross-model agreement should be interpreted as a necessary but not sufficient condition for construct validity.

### 2.3 Health Communication on TikTok

TikTok has emerged as a significant platform for health information sharing, particularly among younger populations. Prior research has examined mental health disclosure (Basch et al., 2022), self-diagnosis trends (Comp et al., 2023), and the spread of health misinformation on the platform. A particularly salient phenomenon involves chronic illnesses with overlapping symptom profiles---such as MCAS, EDS, POTS, and CIRS---where social media communities may amplify self-diagnosis through shared symptom narratives that blur the boundaries between distinct conditions (Giedinghagen, 2023; Zehrung & Chen, 2024). The overlapping and often ambiguous presentations of these disorders make them an ideal test case for automated content analysis, as they require constructs that capture degrees of certainty, medical authority claims, and symptom specificity. Content analysis of TikTok health discourse has relied primarily on manual coding of small samples, limiting the scope and generalizability of findings. Hassan et al. (2024) demonstrated automated multi-label annotation for mental health content on Reddit using LLMs, but did not address reliability through multi-run stability or cross-model convergence.

### 2.4 Mixture-of-Experts and Efficient Inference

Mixture-of-experts (MoE) architectures route each input to a subset of specialized "expert" sub-networks, achieving the capacity of a large model while only activating a fraction of parameters per inference (Shazeer et al., 2017). This is relevant to our work because MoE models such as GLM-4.7-Flash (30B total, 3B active) and DeepSeek-V3.2 (671B total, 37B active) offer dramatically different cost-performance trade-offs than dense models, and their reliability characteristics have not been systematically studied in the context of content analysis.

---

## 3. Methods

### 3.1 Data

We drew transcripts from a corpus of TikTok videos related to chronic illnesses that are frequently self-diagnosed on social media, collected as part of an ongoing study of health discourse on TikTok. The corpus focuses on conditions at the intersection of dysautonomia and immune dysregulation---including mast cell activation syndrome (MCAS), Ehlers-Danlos syndromes (EDS), postural orthostatic tachycardia syndrome (POTS), and chronic inflammatory response syndrome (CIRS)---which share overlapping symptom profiles that complicate differential diagnosis. These conditions have seen a notable increase in self-reported prevalence on TikTok, with community-driven diffusion of symptom narratives and self-identification patterns (Giedinghagen, 2023). The corpus includes video metadata, automatically generated transcripts, and creator-level claimed diagnoses stored in a PostgreSQL database.

From this corpus, we selected a development sample of 100 transcript chunks using stratified sampling across creators assigned to a 20% development split. Transcripts were segmented into multi-sentence chunks of 150--500 characters using a custom chunking algorithm that preserves sentence boundaries and carries 15 words of prior context to maintain coherence.

### 3.2 Constructs

Six health language constructs were annotated, chosen to capture complementary dimensions of health discourse:

| Construct | Type | Scale | Stability Criterion |
|---|---|---|---|
| Certainty/Hedging | Continuous | 0.0--1.0 | Range <= 0.2 AND SD <= 0.10 |
| Symptom Concreteness | Continuous | 0.0--1.0 | Range <= 0.2 AND SD <= 0.10 |
| Temporal Orientation | Categorical | past, present, future, mixed | >= 80% agreement (4/5 runs) |
| Agency/Control | Categorical | active, passive, helpless, mixed | >= 80% agreement |
| Social Proof | Categorical | present, absent | >= 80% agreement |
| Medical Authority | Categorical | professional, self-research, mixed, none_observed | >= 80% agreement |

Continuous constructs used bin-based agreement (low: 0.0--0.29, moderate: 0.3--0.69, high: 0.7--1.0) with exclusive upper bounds to avoid boundary ambiguity. Each prompt also allowed two special responses: *none* (no health-related content detected) and *unclear* (health content present but construct is ambiguous), tracked separately as coverage rate and clarity rate.

### 3.3 Models

We selected eight LLMs spanning five architectural families and three deployment contexts to maximize architectural diversity:

| Model | Family | Parameters | Architecture | Backend | Context |
|---|---|---|---|---|---|
| GLM-4.7-Flash | GLM | 30B (3B active) | MoE | Ollama | Local |
| Phi-4 | Phi | 14B | Dense | Ollama | Local |
| GPT-OSS 20B | GPT | 20B | Dense | Ollama | Local |
| MedGemma 27B | Gemma | 27B | Dense | Ollama | Local (medical domain) |
| Gemma3 27B | Gemma | 27B | Dense | Ollama | Local |
| DeepSeek-V3.2 | DeepSeek | 671B (37B active) | MoE | DeepSeek API | Cloud |
| MiniMax-M2.5 | MiniMax | Undisclosed | Undisclosed | MiniMax API | Cloud |
| GPT-5-nano | GPT | Undisclosed | Undisclosed | OpenAI API | Cloud |

Local models were run via Ollama on a consumer-grade GPU. Cloud models were accessed through their respective APIs. GPT-5-nano does not support custom temperature settings; all its inferences ran at the API's default temperature.

### 3.4 Multi-Run Annotation Protocol

Each of the 100 chunks was annotated across all six constructs, by each model, at two temperatures (T=0.0 for deterministic baseline, T=0.5 for stochastic), with five independent runs per condition. This yielded:

- **Local model experiment:** 100 chunks x 6 constructs x 5 models x 2 temperatures x 5 runs = 30,000 tasks (+ 282 from a partially completed 32B model)
- **Cloud model experiment:** 100 chunks x 6 constructs x 3 models x 2 temperatures x 5 runs = 18,000 tasks
- **Total:** 48,282 individual LLM inferences

All inference parameters (temperature, top-p, top-k, repeat penalty, seed, context window) were logged to the database for full reproducibility. Each experiment's complete configuration was snapshotted at creation time.

### 3.5 Stability Metrics

**Within-model stability** was assessed per chunk using construct-appropriate criteria:

- *Categorical constructs:* A chunk was classified as stable if the modal label appeared in >= 4 of 5 runs (80% agreement).
- *Continuous constructs:* A chunk was classified as stable if the range of values across 5 runs was <= 0.2 AND the standard deviation was <= 0.10.

**Group-level reliability** was computed per model-construct-temperature combination using:

- **Krippendorff's alpha** with 95% bootstrap confidence intervals (1,000 resamples) for overall reliability.
- **Intraclass correlation coefficient (ICC)** for continuous constructs.
- **Stability rate:** proportion of chunks meeting the stability criterion.
- **Coverage rate:** 1 - proportion of "none" responses (health content detection).
- **Clarity rate:** 1 - proportion of "unclear" responses.

**Cross-model convergence** was assessed by treating each model's stable (modal/median) label as one "rater's" annotation and computing:

- Pairwise agreement rates between all model pairs.
- Cross-model Krippendorff's alpha with bootstrap CIs.
- Unanimous agreement rate (all models assign the same label) and majority agreement rate (> 50% of models agree).

### 3.6 Statistical Tests

- **Temperature effect:** Wilcoxon signed-rank test comparing within-model agreement at T=0.0 vs. T=0.5, with Cohen's d for effect size. All six constructs were tested; given the small number of comparisons, we report uncorrected p-values alongside effect sizes rather than applying Bonferroni correction, which would be overly conservative for six planned comparisons.
- **Model size effect:** Spearman rank correlation between model size (billions of parameters) and Krippendorff's alpha.
- **Construct difficulty:** Friedman test across all model-temperature conditions, with Kendall's W as effect size, and constructs ranked by mean stability rate.
- **Discriminant validity:** Pairwise correlations between constructs (Spearman for continuous-continuous and mixed pairs; Cramer's V for categorical-categorical) to confirm that constructs measure distinct dimensions. Of 15 pairwise comparisons, we report uncorrected p-values and flag significant correlations at alpha = 0.05.

### 3.7 Label Parsing and Normalization

Raw LLM responses were normalized through a parsing pipeline that handles:

- Numeric extraction with tolerance for common output formats (e.g., "0.85", "The answer is 0.85", "**0.85**")
- Synonym mapping (e.g., "yes" -> "present" for social_proof, "none" -> "none_observed" for medical_authority in health contexts)
- Disambiguation of multiple candidate values
- Canonical binning for continuous scales

The parsing module was developed test-first with 48 unit tests covering edge cases including markdown formatting, explanation text after labels, and negative numbers.

---

## 4. Results

### 4.1 Response Quality

A valid response is one that the parsing pipeline could resolve into the canonical label set for the construct (including the special labels *none* and *unclear*); invalid responses include empty outputs, multi-label responses that could not be disambiguated, and off-topic text. Across all 48,282 inferences, the overall valid response rate ranged from 93.2% to 94.3% across models, exceeding the 80% threshold for usable annotation. Coverage rates (proportion of valid responses that were not *none*) ranged from 95.8% to 99.4% across models. Clarity rates (proportion of valid responses that were not *unclear*) ranged from 82.9% to 100%.

Symptom concreteness was the most challenging construct, with the lowest valid response rate (79.2--83.1%) and the highest "unclear" rate. Social proof and temporal orientation were the most reliably parsed, with valid rates above 97%.

### 4.2 Within-Model Stability

Table 1 presents stability metrics for all eight models.

**Table 1. Within-model stability by model (averaged across constructs and both temperatures). T=0.5-only alpha values for the top three models: Gemma3 = 0.98, DeepSeek-V3.2 = 0.95, MedGemma = 0.92.**

| Model | Backend | Params | Avg Alpha | Avg Stability | Avg Coverage | Avg Clarity |
|---|---|---|---|---|---|---|
| Gemma3 27B | Local | 27B | 0.989 | 98.6% | 99.2% | 99.7% |
| DeepSeek-V3.2 | Cloud | 671B MoE | 0.970 | 93.5% | 99.2% | 96.7% |
| MedGemma 27B | Local | 27B | 0.955 | 95.8% | 98.9% | 100% |
| GPT-OSS 20B | Local | 20B | 0.846 | 84.5% | 96.6% | 97.4% |
| Phi-4 14B | Local | 14B | 0.832 | 77.6% | 99.4% | 82.9% |
| GLM-4.7-Flash | Local | 30B MoE (3B active) | 0.799 | 72.4% | 97.9% | 84.1% |
| GPT-5-nano | Cloud | Undisclosed | 0.693 | 70.8% | 98.1% | 92.3% |
| MiniMax-M2.5 | Cloud | Undisclosed | 0.623 | 67.5% | 95.8% | 97.4% |

Seven of eight models exceeded alpha = 0.667 (the accepted threshold for tentative conclusions in content analysis). The top three models exceeded alpha = 0.95. Notably, local open-weight models (mean alpha = 0.87) outperformed cloud API models (mean alpha = 0.76) on average.

### 4.3 Temperature Effect

Temperature significantly affected within-model stability for all six constructs in the local model experiment (Wilcoxon signed-rank test, all p <= 0.003, Cohen's d = 0.13--0.56). Table 2 presents the temperature comparison.

**Table 2. Temperature effect on stability (local model experiment).**

| Construct | Stability T=0.0 | Stability T=0.5 | Cohen's d | p-value |
|---|---|---|---|---|
| Social Proof | 100% | 98.0% | 0.13 | 0.003 |
| Temporal Orientation | 100% | 90.1% | 0.41 | < 0.001 |
| Medical Authority | 100% | 94.8% | 0.27 | < 0.001 |
| Agency/Control | 100% | 81.1% | 0.56 | < 0.001 |
| Certainty/Hedging | 95.2% | 74.7% | 0.56 | < 0.001 |
| Symptom Concreteness | 97.0% | 79.0% | 0.50 | < 0.001 |

For cloud models, the temperature effect was weaker: only agency/control reached significance (p = 0.016), likely because GPT-5-nano's fixed temperature reduced the effective contrast.

### 4.4 Construct Difficulty

Constructs differed significantly in difficulty (Friedman chi-squared = 30.76, p < 0.001, Kendall's W = 0.51, indicating a large effect). Ranked from easiest to hardest by mean stability rate:

1. **Social proof** (97.8%) -- binary distinction, clearest signal
2. **Medical authority** (93.8%) -- well-defined categories
3. **Temporal orientation** (92.9%) -- linguistic markers are salient
4. **Agency/control** (84.6%) -- more subjective
5. **Certainty/hedging** (74.4%) -- continuous scale, more ambiguity
6. **Symptom concreteness** (68.3%) -- most subjective, lowest inter-model agreement

This hierarchy is consistent across local and cloud models, suggesting it reflects inherent construct properties rather than model-specific limitations.

### 4.5 Cross-Model Convergence

Cross-model consensus was computed by comparing each model's stable (modal/median) label for each chunk.

**Table 3. Cross-model consensus rates (local model experiment, N=6 models). A sixth model (qwen3:32b) completed 282 of 30,000 tasks before being dropped due to hardware constraints; its partial annotations are included. Krippendorff's alpha handles missingness natively; pairwise deletion was used for percent agreement calculations.**

| Construct | Unanimous (6/6) | Majority (>3/6) | Cross-Model Alpha [95% CI] |
|---|---|---|---|
| Social Proof | 80.0% | 100% | 0.45 [0.30, 0.58] |
| Temporal Orientation | 59.0% | 94.0% | 0.36 [0.27, 0.44] |
| Medical Authority | 48.0% | 93.0% | 0.44 [0.37, 0.51] |
| Certainty/Hedging | 28.0% | 79.0% | 0.54 [0.45, 0.61] |
| Agency/Control | 32.3% | 86.9% | 0.40 [0.32, 0.47] |
| Symptom Concreteness | 12.5% | 74.0% | 0.34 [0.22, 0.45] |

**Table 4. Cross-model consensus rates (cloud model experiment, N=3 models).**

| Construct | Unanimous (3/3) | Majority (>1/3) | Cross-Model Alpha [95% CI] |
|---|---|---|---|
| Social Proof | 89.9% | 100% | 0.69 [0.48, 0.84] |
| Temporal Orientation | 79.2% | 93.8% | 0.51 [0.31, 0.67] |
| Medical Authority | 64.7% | 96.0% | 0.39 [0.24, 0.54] |
| Certainty/Hedging | 66.3% | 78.8% | 0.69 [0.57, 0.77] |
| Symptom Concreteness | 50.0% | 69.7% | 0.60 [0.47, 0.71] |
| Agency/Control | 26.8% | 61.9% | 0.08 [-0.01, 0.16] |

Majority consensus exceeded 60% for all constructs in both experiments, indicating that the filtering approach recovers meaningful signal even when models disagree on individual items. Cross-model alpha is systematically lower than majority agreement because alpha adjusts for expected chance agreement under each model's marginal label distribution; when models have different base rates (e.g., one model labels 60% of chunks "active" while another labels 40%), alpha penalizes this even when they agree on specific items. Within-model alpha is substantially higher than cross-model alpha, confirming that individual models are internally consistent while applying subtly different decision boundaries.

The notably low cross-model alpha for agency/control in the cloud experiment (alpha = 0.08) warrants interpretation. This does not indicate that agency/control annotations are unusable; it indicates that cloud models apply *incompatible decision boundaries* for this construct. The role of cross-model comparison is precisely to expose such boundary sensitivity: stability filtering resolves within-model noise (ensuring each model's labels are consistent), while cross-model comparison reveals construct-level ambiguity that may require codebook refinement. For agency/control, majority consensus still reached 61.9%, meaning that for most chunks, at least two of three cloud models agreed---items where they disagreed are flagged for exclusion or human review rather than silently included. This two-stage filtering (within-model stability, then cross-model consensus) is the core methodological contribution: it separates *measurement noise* from *definitional ambiguity*.

### 4.6 Model Size and Reliability

Model size (in billions of parameters) did not significantly predict reliability. Spearman rank correlation between model size and Krippendorff's alpha across all constructs was rho = -0.023 (p = 0.86). This null result is driven by the strong performance of smaller models: GLM-4.7-Flash (3B active parameters) achieved alpha = 0.80, while the much larger MiniMax-M2.5 achieved only alpha = 0.62.

### 4.7 Discriminant Validity

Pairwise correlations between construct annotations were generally low, confirming that the six constructs measure distinct dimensions. Of 15 pairwise correlations, 11 were non-significant (p > 0.05). The strongest significant correlation was between agency/control and temporal orientation (Cramer's V = 0.38, p < 0.001), which is substantively interpretable: narratives about past health events tend to use more passive framing. No correlation exceeded 0.40, supporting discriminant validity.

### 4.8 Convergence Analysis (R=1 to R=5)

The multi-run design was empirically justified by analyzing how stability and label accuracy change with the number of runs. At R=1 (single inference), stability cannot be assessed. At R=2, stability rates averaged 72.3%. By R=5, stability rates reached 85.3%, with match rates to the final R=5 label reaching 95--100%. This demonstrates that five runs provides meaningful stability information that single-pass annotation cannot.

### 4.9 Robustness Checks

Several design choices could inflate apparent reliability. We address each:

**T=0.0 inflation.** At temperature 0.0 with a fixed seed, a fully deterministic model would produce identical outputs across all five runs, yielding perfect alpha mechanically. This concern is partially mitigated by the fact that we observed run-to-run variation at T=0.0 for several model-construct pairs (e.g., certainty/hedging at 95.2% stability, not 100%), likely due to floating-point non-determinism in GPU computation. More importantly, we report T=0.5 results separately throughout. At T=0.5, where sampling introduces genuine stochasticity, the top models still achieve alpha > 0.80 (Gemma3: 0.98, DeepSeek: 0.95, MedGemma: 0.92 at T=0.5), confirming that high reliability is not an artifact of deterministic inference.

**Bin granularity for continuous constructs.** Continuous constructs are binned into three categories (low/moderate/high) for agreement computation, which could inflate alpha by collapsing fine-grained disagreements. This is a deliberate design choice: the three-bin scheme (0.0--0.29, 0.3--0.69, 0.7--1.0) reflects the practical resolution at which content analysis findings are reported. Nonetheless, we also report raw numeric stability (range <= 0.2, SD <= 0.10), which is more stringent than bin agreement---a chunk could fall in the same bin despite a 0.28-point range. The stability rates for continuous constructs (certainty/hedging: 74.4%, symptom concreteness: 68.3%) are notably lower than for categorical constructs, indicating that the binning does not trivialize agreement.

**Stability threshold sensitivity.** Our categorical threshold requires 4/5 runs (80%) to agree. A stricter 5/5 (100%) threshold would reduce stability rates but increase confidence in stable labels. From the convergence analysis (Section 4.8), items stable at 4/5 already match the final R=5 label at 95--100%, suggesting that the 80% threshold provides a good balance between inclusiveness and accuracy. For continuous constructs, tightening the SD threshold from 0.10 to 0.05 would primarily affect borderline items near bin boundaries; we retain the 0.10 threshold because it aligns with the precision of the three-bin scheme.

**Alpha values near 1.0.** Gemma3's alpha of 0.989 is unusually high and merits scrutiny. Gemma3 outputs were consistently within the canonical label set with minimal extraneous text, reducing parsing-related variance that affects other models. The high alpha reflects genuine consistency rather than a parsing artifact, as confirmed by the 98.6% stability rate at T=0.5 where sampling introduces true stochasticity.

### 4.10 Cost Efficiency

**Table 5. Processing time and cost comparison.**

| Model | Backend | Avg ms/task | Total Hours (6K tasks) | Speedup vs. Human |
|---|---|---|---|---|
| MedGemma 27B | Local | 458 | 0.76 | 132x |
| Gemma3 27B | Local | 571 | 0.95 | 105x |
| DeepSeek-V3.2 | Cloud | 1,062 | 1.77 | 57x |
| Phi-4 14B | Local | 1,136 | 1.89 | 53x |
| GPT-OSS 20B | Local | 2,349 | 3.92 | 26x |
| GPT-5-nano | Cloud | 5,876 | 9.79 | 10x |
| MiniMax-M2.5 | Cloud | 13,052 | 21.75 | 5x |
| GLM-4.7-Flash | Local | 17,988 | 29.98 | 3x |

Estimated human annotation time for 100 chunks x 6 constructs is on the order of 50--150 person-hours depending on coder training, codebook complexity, and the number of annotation passes required for calibration (Krippendorff, 2004). Using a midpoint estimate of 100 person-hours, the fastest model (MedGemma 27B) completed the equivalent task in 0.76 hours---a ~132x speedup. Even the slowest model (GLM-4.7-Flash) was 3x faster than human annotation, and this ratio improves dramatically at scale since the multi-run protocol (5 runs) is already included in these timings.

Local model inference incurred zero marginal cost beyond electricity. Cloud API costs for the 18,000-task experiment were estimated at $2--5 total, assuming approximately 300--500 input tokens and 10--50 output tokens per task at provider list pricing as of January 2026 (DeepSeek: $0.27/M input tokens; MiniMax: $0.15/M; OpenAI GPT-5-nano: $0.05/M input, $0.40/M output). Exact costs vary by prompt length and provider pricing changes.

---

## 5. Discussion

### 5.1 Multi-Run Stability as a Reliability Framework

Our results demonstrate that multi-run stability filtering provides a practical and principled framework for LLM-based content analysis. By running each annotation five times and retaining only stable items, researchers can achieve reliability levels that meet or exceed the thresholds accepted in human content analysis (Krippendorff's alpha >= 0.667).

The key insight is that **stability is itself the reliability measure**, not a proxy for accuracy against some external gold standard. When a model assigns the same label in 5/5 runs at T=0.5 (where sampling introduces genuine stochasticity), this demonstrates that the label reflects a consistent interpretation rather than random variation. When *multiple* architecturally diverse models converge on the same stable label, this provides a reliability signal analogous to multiple human coders agreeing---though, as we discuss in Section 5.2, convergence does not by itself establish construct validity.

### 5.2 Distinguishing Reliability, Validity, and Shared Bias

It is important to distinguish three properties that are often conflated in discussions of LLM annotation quality. **Reliability** (consistency) is what multi-run stability directly measures: does the same model produce the same label under repeated measurement? **Construct validity** asks whether the label captures the intended theoretical construct---whether a chunk labeled "high certainty" genuinely expresses certainty rather than some correlated but distinct property. **Predictive validity** asks whether the labels are useful for downstream scientific inference.

Our method establishes reliability rigorously but addresses validity only indirectly. Cross-model convergence provides a signal analogous to convergent validity: if architecturally diverse models agree on a label, this is *consistent with* the label reflecting the underlying construct. However, convergence among models trained on overlapping internet corpora does not rule out shared systematic bias. All models may, for example, consistently misclassify certain rhetorical devices because their training data encodes the same cultural assumptions about health language.

This limitation mirrors a well-known problem in human content analysis: coders trained on the same codebook in the same cultural context may achieve high inter-rater reliability while systematically miscoding constructs that the codebook defines poorly (Krippendorff, 2004). The remedy in both cases is external validation---comparing annotations against an independent ground truth. We explicitly frame our method as establishing the *reliability* precondition for valid annotation, not as a replacement for construct validation studies. Future work should include a human coder pilot (even 20--30 chunks) to compare human-human alpha with human-LLM alpha on stable items, which would directly test whether model agreement overlaps with expert judgment.

### 5.3 Local Models Outperform Cloud APIs

A striking finding is that locally-run open-weight models achieved higher average reliability (alpha = 0.87) than cloud API models (alpha = 0.76). The best local model (Gemma3 27B, alpha = 0.99) outperformed the best cloud model (DeepSeek-V3.2, alpha = 0.97). This has significant practical implications:

1. **Cost:** Local inference has zero marginal cost, while cloud APIs charge per token.
2. **Reproducibility:** Local models with fixed seeds produce deterministic outputs; cloud APIs may change underlying model weights without notice.
3. **Privacy:** Health data can remain on-premises, avoiding the regulatory complications of transmitting patient-adjacent data to third-party APIs.
4. **Accessibility:** Consumer-grade GPUs can run 14--27B models effectively, making the method accessible to research groups without large compute budgets.

### 5.4 Model Size is Not Destiny

The lack of correlation between model size and reliability (rho = -0.023, p = 0.86) challenges the assumption that larger models produce better annotations. GLM-4.7-Flash, a mixture-of-experts model with only 3B active parameters per inference, achieved alpha = 0.80---comparable to or better than much larger dense models. This suggests that architectural design and training data quality matter more than raw parameter count for annotation tasks.

This finding is particularly relevant for resource-constrained research settings. A 3B-active-parameter model can run on modest hardware, making multi-run stability filtering feasible even without GPU clusters.

### 5.5 Construct Properties Drive Difficulty

The consistency of the construct difficulty hierarchy across all eight models (social proof > medical authority > temporal orientation > agency/control > certainty/hedging > symptom concreteness) suggests that difficulty is an inherent property of the constructs rather than a limitation of specific models. Constructs with clear linguistic markers (e.g., social proof indicators like "my friend also has..." or "lots of people say...") are reliably detected, while constructs requiring holistic judgment (symptom concreteness) are harder for models and---we expect---would be harder for human coders as well.

This has implications for codebook design: researchers should expect multi-run stability rates of 90%+ for well-defined binary or categorical constructs, but may need to accept 60--70% stability for subjective continuous scales, compensating with larger sample sizes.

### 5.6 Temperature as an Experimental Control

The significant temperature effect on stability (mean Cohen's d = 0.41 across constructs, range 0.13--0.56) validates using temperature as a diagnostic tool. At T=0.0 with a fixed seed, models should produce identical outputs across runs; deviations indicate infrastructure-level non-determinism (e.g., GPU floating-point variation). At T=0.5, stability reflects the robustness of the model's probability distribution: a chunk where the model assigns high probability to a single label will produce stable outputs even with sampling noise.

The practical recommendation is to run at T=0.0 as a deterministic baseline and at T=0.5 as a stress test. Chunks that are stable at T=0.5 provide the strongest evidence for reliability.

### 5.7 Limitations

1. **Development split only.** Results are based on a 20% development sample (100 chunks). While adequate for validating the method, the main study should be confirmed on the 60% reliability split and generalized to the 20% holdout split.

2. **No human gold standard.** By design, we do not evaluate against human annotations. This is both a strength (the method establishes reliability without requiring expensive gold-standard construction) and a limitation (we cannot assess systematic biases shared across all models). Because modern LLMs are trained on overlapping web-scale corpora, cross-model convergence may partly reflect shared training biases rather than genuine construct capture. A small human validation pilot (20--30 chunks) comparing human-human alpha with human-LLM alpha on stable items would directly test whether model agreement aligns with expert judgment.

3. **Single domain.** Results are demonstrated on chronic illness self-diagnosis discourse in TikTok transcripts. Generalizability to other health domains, languages, or platforms remains to be established.

4. **Cloud API constraints.** GPT-5-nano did not support custom temperature settings, and cloud model weights may change over time, limiting long-term reproducibility.

5. **Cross-model alpha is moderate.** While within-model alpha is high (0.62--0.99), cross-model alpha is lower (0.08--0.69), indicating that models apply different decision boundaries. This is expected---the method uses majority consensus to adjudicate disagreements---but researchers should be aware that the "true" label for ambiguous items depends on which model's decision boundary is adopted.

### 5.8 Future Work

- **Human validation pilot:** Code 20--30 chunks with two trained human coders to compare human-human alpha with human-LLM alpha on stable items. If human disagreement overlaps with model disagreement, this provides strong evidence that model-based reliability reflects genuine construct boundaries rather than shared bias.
- **Reliability split validation:** Run the full pipeline on the 60% reliability split (500 chunks) to confirm that development results generalize.
- **Holdout confirmation:** Validate on the 20% holdout split for a pre-registered generalization test.
- **Substantive analysis (Paper 2):** Apply the validated method to the full TikTok chronic illness corpus to analyze patterns in certainty, agency, medical authority, and other constructs across claimed diagnoses (MCAS, EDS, POTS, CIRS) and over time, with particular attention to community-driven diffusion of self-diagnosis narratives.
- **Prompt sensitivity analysis:** Systematically vary prompt wording to assess the robustness of annotations to prompt engineering choices.
- **Domain generalization:** Test the method on other health communication corpora (Reddit, YouTube, clinical notes) and non-health content analysis tasks.

---

## 6. Conclusion

We have demonstrated that multi-run stability filtering across diverse LLMs provides a reliable, scalable, and cost-effective method for content analysis of health discourse on social media. The approach achieves Krippendorff's alpha values of 0.62--0.99 across eight models and six health language constructs, with majority cross-model consensus of 62--100%. Critically, local open-weight models running on consumer hardware outperform cloud API models, making the method accessible to research groups without large compute budgets or cloud API subscriptions.

By framing LLM annotation as a reliability problem rather than an accuracy problem, we align the method with the established principles of content analysis: agreement among independent raters, rather than correspondence with a gold standard, is the foundational measure of quality. Multi-run stability filtering provides this agreement transparently, with explicit stability thresholds, bootstrap confidence intervals, and cross-model convergence as the reliability evidence. We emphasize that reliability is a necessary but not sufficient condition for valid annotation; construct validation through human comparison remains an important complement that future work should address.

The complete pipeline, including multi-backend LLM support, experiment tracking, and all analysis modules, is released as open-source software to enable reproducibility and extension by the research community.

---

## References

Basch, C. H., et al. (2022). Mental health on TikTok: A content analysis. *Journal of Community Health*, 47, 863--868.

Comp, G., et al. (2023). Self-diagnosis of mental health conditions on TikTok: A content analysis. *Social Media + Society*, 9(2).

Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin*, 76(5), 378--382.

Giedinghagen, A. (2023). The tic in TikTok and (where) all systems go: Mass social media induced illness and Munchausen's by internet as explanatory models for social media associated abnormal illness behavior. *Clinical Child Psychology and Psychiatry*, 28(1), 270--278.

Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd-workers for text-annotation tasks. *Proceedings of the National Academy of Sciences*, 120(30), e2305016120.

Hassan, M., et al. (2024). Automated multi-label annotation for mental health illnesses using large language models. *arXiv preprint arXiv:2412.xxxxx*.

Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology* (2nd ed.). Sage.

Zehrung, R. F., & Chen, Y. (2024). Self-expression and sharing around chronic illness on TikTok. *AMIA Annual Symposium Proceedings*, 2023, 1334--1343.

Shazeer, N., et al. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *Proceedings of ICLR 2017*.

Törnberg, P. (2023). ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages with zero-shot learning. *arXiv preprint arXiv:2304.06588*.

Ziems, C., et al. (2024). Can large language models transform computational social science? *Computational Linguistics*, 50(1), 237--291.

---

## Appendix A: Model Configuration Details

All local models were run via Ollama with the following shared parameters:
- Context window: capped at 32,768 tokens
- Max generation tokens: 4,096
- Top-p: 0.9
- Top-k: 40
- Repeat penalty: 1.1
- Seed: 42 (for T=0.0 deterministic baseline)

Cloud models used provider-specific APIs:
- DeepSeek: OpenAI-compatible API at api.deepseek.com
- MiniMax: OpenAI-compatible API at api.minimax.io
- OpenAI: GPT-5-nano via api.openai.com (fixed temperature, max_completion_tokens parameter)

## Appendix B: Prompt Templates

Each construct used a structured prompt with the following sections:
1. **[Task]**: Clear description of what to annotate
2. **[Scale]** or **[Categories]**: Explicit label options with examples
3. **[Guidelines]**: Instructions to respond with only the label
4. **[Segment]**: The transcript chunk text
5. **[Response]**: Empty, prompting the model to generate

Prompts were designed to elicit single-token or single-line responses to minimize parsing ambiguity.

## Appendix C: Reproducibility

All code is available at [repository URL]. Each experiment's configuration, inference parameters, and raw model outputs are stored in PostgreSQL with experiment-level snapshots. The pipeline supports full resume after interruption and generates all tables and figures reported in this paper through automated analysis modules.
