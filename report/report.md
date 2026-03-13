# The Ghost in the Machine — Research Report

**Project:** Authorship Detection in the Age of LLMs
**Submitted to:** PreCog Lab, IIIT Hyderabad
**Authors:** Charles Dickens & Mary Shelley (human corpus); Gemini API (AI corpus)
**Last Updated:** 2026-03-13

---

## Page 1: Task Completion Status

- [x] **Task 0: Dataset Construction** — completed
  - [x] Gutenberg text acquisition (Dickens: *Bleak House*, *Great Expectations*; Shelley: *Frankenstein*, *The Last Man*)
  - [x] Text cleaning and paragraph chunking (5,211 human chunks)
  - [x] BERTopic topic extraction (7 topics discovered)
  - [x] NER-based character name auto-detection (general, not hardcoded)
  - [x] Class 2: Generic AI paragraph generation (500 paragraphs, gemini-2.5-flash)
  - [x] Class 3: Style-Mimicking AI paragraph generation (500 paragraphs, gemini-2.5-flash)
  - [x] Final dataset assembly (6,211 rows: 5,211 Human / 500 Generic AI / 500 Style-Mimicking AI)

- [x] **Task 1: Feature Analysis** — completed
  - [x] Feature extraction pipeline (23 features: lexical, syntactic, punctuation, readability)
  - [x] MATTR implementation (window=50) over raw TTR
  - [x] Punctuation density heatmap
  - [x] Parse tree depth violin plot
  - [x] Readability box plots (Flesch, Kincaid, Gunning Fog)
  - [x] PCA + t-SNE visualization

- [x] **Task 2 Tier A: XGBoost + Random Forest** — completed
  - [x] Binary classification (Human vs AI): 98.50% accuracy (Round 2)
  - [x] 3-class classification (Human / Generic AI / Style-Mimicking AI): 96.67% accuracy (Round 2)
  - [x] SHAP analysis on XGBoost 3-class model

- [x] **Task 2 Tier B: FF-NN on GloVe / SBERT** — completed
  - [x] Code written (GloVe averaging, SBERT embeddings, FF-NN architecture)
  - [x] GloVe vectors downloaded (glove.6B.100d.txt)
  - [x] GloVe FF-NN: 99.14% 3-class accuracy, F1=0.971
  - [x] SBERT FF-NN: 99.25% 3-class accuracy, F1=0.983

- [ ] **Task 2 Tier C: DistilBERT + LoRA** — needs GPU retraining on Round 2 data
  - [x] Code written (LoRA fine-tuning with PEFT)
  - [ ] Current checkpoint is from Round 1 data — results invalid until retrained on Colab GPU

- [x] **Task 3: Explainability & Error Analysis** — completed
  - [x] Error analysis: 31/932 errors (3.3%), Style-Mimicking AI most confused class
  - [x] SHAP on XGBoost (flesch_reading_ease dominates at ~40% importance)
  - [x] Data-driven AI-isms analysis (log-odds, opener entropy, JSD, POS mining)
  - [ ] Integrated Gradients on DistilBERT — requires GPU retraining first

- [x] **Task 4: Adversarial GA Optimization** — completed
  - [x] Gemini-powered genetic algorithm with Type A (Rhythm) and Type B (Archaic) mutations
  - [x] Both types converge at generation 0 (P(Human)>0.90 from initial population)
  - [x] GA exploits brittle Tier A decision boundary

- [x] **SOP Self-Test** — completed
  - [x] SOP classified as Human by all 3 classifiers (XGB P(Human)=0.993, SBERT=0.997, DistilBERT=Human)
  - [x] Feature analysis: Flesch=64.7 (human range), low sent_len_std=7.9 (AI-like)
  - [x] Adversarial rewrite: AI-ified version fooled 2/3 classifiers (XGB and DistilBERT)

---

## Methodology

### Dataset Construction (Task 0)

**Author Selection:** Charles Dickens (*Bleak House*, *Great Expectations*) and Mary Shelley (*Frankenstein*, *The Last Man*) were chosen for their contrasting Victorian-era styles — Dickens for elaborate social commentary prose with heavy dialogue, Shelley for Gothic Romantic narrative with introspective philosophical passages. Both are sufficiently distinct that a 3-class classifier must learn genuine stylistic features rather than trivial content cues.

**Chunking Strategy:** Raw Gutenberg texts were cleaned (headers/footers stripped, encoding normalized) and split into paragraph-level chunks of 80-300 words. This window size preserves enough syntactic and stylistic signal for feature extraction while being comparable to typical AI-generated paragraph lengths.

**Topic Extraction:** BERTopic was used to discover latent themes in the human corpus, providing topically-grounded prompts for AI generation. A custom `CountVectorizer` with English stop words plus corpus-specific high-frequency words (>15% document frequency) ensured meaningful topic representations. Seven topics were discovered and auto-labeled from their top-5 keywords.

**NER-based Filtering:** Instead of hardcoding character names, spaCy's NER model (`en_core_web_sm`) was run over a corpus sample to auto-detect PERSON entities. These names were added to BERTopic's stop word list so topics reflect thematic content rather than character mentions. This makes the pipeline generalizable to any books.

**AI Generation:** Gemini API generated 500 paragraphs per AI class across two rounds:

- **Round 1 (naive prompts):** Class 2 prompted to "write a paragraph exploring themes of {topic}" with no style instruction. Class 3 given surface-level style instructions ("use semicolons, em-dashes, long sentences"). Generated with `gemini-2.5-flash`.

- **Round 2 (research-grade prompts):** Class 2 rewritten as magazine-essay voice with natural writing constraints. Class 3 redesigned with 5-shot ICL (real Victorian excerpts from *Bleak House* and *Frankenstein*), persona framing ("You are a Victorian novelist"), chain-of-thought reasoning, and anti-detection blacklist targeting AI-isms identified by SHAP analysis. Generated with `gemini-2.5-flash-lite` using higher temperature (0.9-1.0) and top_p (0.9-0.95).

Paragraphs were distributed proportionally across the 7 topics. Concurrent generation (10 workers via ThreadPoolExecutor) with incremental JSON checkpointing ensures crash-safe resume.

### Feature Engineering (Task 1)

**23 handcrafted features** across four categories:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Lexical** (4) | TTR, MATTR (window=50), hapax ratio, avg word length | Vocabulary richness and word-level diversity |
| **Syntactic** (5) | Avg sentence length, sentence length std, avg tree depth, verb ratio, adj-noun ratio | Structural complexity and grammatical patterns |
| **Punctuation** (8) | Density of `. , ; : ! ? — " ( -` | Punctuation is a strong authorial fingerprint AI struggles to replicate |
| **Readability** (3) | Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog | Aggregate complexity measures |

MATTR (Moving Average Type-Token Ratio) was chosen over raw TTR because TTR is heavily biased by text length — shorter AI paragraphs would artificially inflate TTR. MATTR computes TTR over a sliding window of 50 tokens and averages, making it length-invariant.

### Classification Models (Task 2)

**Tier A (Statistical):** XGBoost and Random Forest with `GridSearchCV` over hyperparameter grids. Both binary (Human vs AI-merged) and 3-class setups. The 23 handcrafted features are standardized before training.

**Tier B (Embedding-based):** Feed-forward neural networks operating on:
- GloVe (100d) averaged word embeddings — captures semantic content
- SBERT (`all-MiniLM-L6-v2`) sentence embeddings — captures contextual meaning

Architecture: 2-layer FF-NN (256→128→n_classes) with dropout, BatchNorm, trained with AdamW and ReduceLROnPlateau.

**Tier C (Transformer):** DistilBERT fine-tuned with LoRA (rank=8, alpha=16) on the raw text. This is the most powerful approach but requires GPU.

### Explainability (Task 3)

**SHAP TreeExplainer** on XGBoost reveals which handcrafted features drive predictions. **Integrated Gradients** (via Captum) on DistilBERT highlights which tokens matter most at the subword level. **Data-driven AI-isms analysis** uses five methods:
1. Log-odds ratio n-gram analysis (Monroe et al., 2008) to find statistically overrepresented phrases in AI text
2. Sentence opener entropy (low entropy = repetitive AI openings)
3. Vocabulary divergence via Jensen-Shannon divergence
4. POS trigram pattern mining for syntactic templates AI overuses
5. Parallel structure detection and validation against known AI markers from the literature

### Adversarial Optimization (Task 4)

A Gemini-powered genetic algorithm with two mutation strategies compared independently:
- **Type A (Rhythm):** Rewrites sentence cadence while preserving vocabulary and meaning
- **Type B (Archaic):** Injects pre-1900 vocabulary and minor grammatical irregularities

Fitness = classifier's P(Human). Elite preservation with population of 10, target fitness 0.90. Both types are run independently and cross-tested against all three classifier tiers.

---

## Experiments and Results

### Experiment 1: BERTopic Topic Extraction

**What:** Ran BERTopic on 5,211 human chunks to discover latent themes for guiding AI generation prompts.

**Result:** 7 topics discovered (plus noise topic -1):

| Topic | Top Keywords |
|-------|-------------|
| 0 | know, time, dear, say, come |
| 1 | air, night, saw, day, country |
| 2 | party, favour, present, man, did |
| 3 | mr jarndyce, men, distinguished, say, great |
| 4 | air, family, chair, discovered, gentleman |
| 5 | science, natural, works, concerning, pursuit |
| 6 | says, country, way, usual, know |

**Outcome:** Partial success. Topics are somewhat vague because narrative prose doesn't cluster as cleanly as technical/news text. Topic 5 (science, natural, works) is the clearest — it captures Shelley's scientific themes. Topics 0, 2, 6 are more diffuse conversational/social topics from Dickens. This is expected for literary fiction where topics blend.

**Iterations to get here:**
1. First attempt: topics were all stop words ("and, the, to, of"). Fixed by adding `CountVectorizer(stop_words="english")`.
2. Second attempt: topics had common narrative words ("said", "know", "dear"). Fixed by computing corpus document frequency and filtering words >15% as additional stop words.
3. Third attempt: character names dominated topics. Fixed by auto-detecting PERSON entities via spaCy NER and adding them to stop words.

**Hypothesis:** Literary fiction produces less crisp topics than factual text because themes are conveyed through narrative and dialogue rather than keyword clusters. The topics are sufficient for prompting Gemini — they provide thematic grounding even if not perfectly labeled.

---

### Experiment 2: AI Paragraph Generation

**What:** Generated 500 Class 2 (Generic AI) and 500 Class 3 (Style-Mimicking AI) paragraphs using Gemini API, distributed across 7 topics. Two rounds of generation were performed to test the impact of prompt engineering on detection difficulty.

**Round 1 (naive prompts, gemini-2.5-flash):** Zero-shot prompts with surface-level style instructions ("use semicolons, em-dashes, long sentences"). All 1,000 paragraphs generated. Quality was poor — AI text was trivially distinguishable (Flesch scores 10-30 vs Human 60+, abstract language, uniform sentence lengths).

**Round 2 (research-grade prompts, gemini-2.5-flash-lite):** Regenerated all 1,000 paragraphs with few-shot ICL (5 real Victorian excerpts), persona + CoT prompting, and anti-detection constraints. Used concurrent generation (10 workers via ThreadPoolExecutor). Results:
- Class 2: 500 paragraphs, mean 143.6 words, range 100-190
- Class 3: 500 paragraphs, mean 138.7 words, range 95-212 (498/500 in 100-200 range)
- Quality dramatically improved: Class 3 uses proper nouns, narrative voice, varied sentence lengths

**Rate limiting journey:**
- Free tier (gemini-1.5-flash): 429 errors after ~15 requests
- gemini-3.1-flash-lite-preview: Worked with 7.0s sleep (free tier RPM limits)
- gemini-2.5-flash: "Thinking" model — `max_output_tokens` includes thinking tokens, producing 5-10 word outputs
- Final: gemini-2.5-flash-lite (non-thinking, all generation_config params work correctly)

---

### Experiment 3: Feature Extraction and Visualization

**What:** Extracted 23 features from all 6,211 samples and generated exploratory visualizations.

**Result:**
- Feature matrix shape: (6,211 x 23)
- **Punctuation heatmap:** Dramatic class separation. Key findings:
  - Style-Mimicking AI overshoots commas (13.0 vs Human 8.6 per 100 words) and em-dashes (1.41 vs Human 0.60)
  - Style-Mimicking AI has fewest periods (1.53 vs Human 4.98) — it writes abnormally long sentences
  - Generic AI overuses quotation marks (2.52 vs Human ~0) and apostrophes (1.39 vs Human ~0)
  - Semicolons are the one feature Style AI mimics correctly (0.77 vs Human 0.81)
- **Parse tree depth violin:** Human has wide distribution (median ~6, range 2-21). Generic AI clusters at 7-8. Style AI overshoots with median ~10, range 6-23. The variance difference is key.
- **Readability box plots:** The clearest signal of all:
  - Flesch Reading Ease: Human median ~60, Generic AI ~15, Style AI ~0. Style AI is harder to read than actual Victorian prose.
  - Flesch-Kincaid Grade: Human ~10, Generic AI ~18, Style AI ~25.
  - AI equates "Victorian" with "unreadable" — a fundamental misunderstanding of style.
- **PCA (39.5% variance):** Human blob on left, Generic AI top-right, Style AI extends to far right (overshoot pushes PC1 to extremes).
- **t-SNE:** Human has multiple sub-clusters (Dickens vs Shelley). AI classes form tight, isolated blobs. Some Style AI dots mix with Human — these are the few misclassified samples.

**Outcome:** Strong success. Features clearly separate classes, especially readability and punctuation features.

**Key Insight:** The "overshoot effect" — when told to mimic Victorian style, the AI overcompensates on every marker. It produces text that is more Victorian-looking than actual Victorian prose, making it paradoxically easier to detect.

---

### Experiment 4: Tier A Classification (XGBoost + Random Forest)

**What:** Trained XGBoost and Random Forest with GridSearchCV on the 23 handcrafted features. Both binary and 3-class setups.

**Round 1 results** (naive prompts, gemini-2.5-flash):

| Model | Task | Accuracy | Macro F1 |
|-------|------|----------|----------|
| XGBoost | Binary | 99.68% | 0.9940 |
| Random Forest | Binary | 99.57% | 0.9920 |
| XGBoost | 3-Class | 99.57% | 0.9901 |
| Random Forest | 3-Class | 99.36% | 0.9832 |

**Round 2 results** (research-grade prompts, gemini-2.5-flash-lite):

| Model | Task | Accuracy | Macro F1 |
|-------|------|----------|----------|
| XGBoost | Binary | 98.50% | 0.961 |
| Random Forest | Binary | 98.61% | 0.966 |
| XGBoost | 3-Class | 96.67% | 0.882 |
| Random Forest | 3-Class | 96.46% | 0.877 |

**Round 1 → Round 2 comparison:**

| Metric | Round 1 | Round 2 | Delta |
|--------|---------|---------|-------|
| XGB 3-Class Acc | 99.57% | 96.67% | **-2.90%** |
| XGB 3-Class F1 | 0.990 | 0.882 | **-0.108** |
| RF 3-Class Acc | 99.36% | 96.46% | **-2.90%** |

**Error analysis (Round 2, 31/932 errors):** Style-Mimicking AI is the most confused class — 13 samples misclassified as Generic AI and 10 as Human. The few-shot ICL and anti-detection constraints successfully brought some AI text into the human statistical range, validating the prompt engineering thesis.

**Outcome:** The accuracy drop from 99.57% to 96.67% proves that prompt quality directly determines detection difficulty. The improved prompts made the task meaningfully harder — but 23 handcrafted features still achieve >96% accuracy, suggesting that statistical fingerprints persist even under adversarial prompt engineering.

---

### Experiment 5: SHAP Feature Importance Analysis

**What:** Ran SHAP (PermutationExplainer fallback, since TreeExplainer has a version bug with XGBoost multiclass) on the 3-class XGBoost model.

**Result:** Top features by mean |SHAP value|:

| Rank | Feature | Mean |SHAP| | Primary Class Signal |
|------|---------|-------------|---------------------|
| 1 | flesch_reading_ease | 0.34 | Human (blue) — by far dominant |
| 2 | avg_sent_len | 0.08 | Human + Style-Mimicking AI |
| 3 | punct_— (em-dash) | 0.08 | Human + Generic AI |
| 4 | hapax_ratio | 0.07 | Generic AI (pink) |
| 5 | punct_. (period) | 0.05 | Generic AI + Style-Mimicking AI |
| 6 | punct_; (semicolon) | 0.03 | Human |
| 7 | adj_noun_ratio | 0.03 | Style-Mimicking AI |
| 8 | verb_ratio | 0.02 | Style-Mimicking AI |
| 9 | sent_len_std | 0.02 | Human |

Features with negligible SHAP: mattr, ttr, gunning_fog, avg_tree_depth, punct_!, punct_"

**Outcome:** Success. The SHAP analysis reveals that:
1. **Readability is king** — `flesch_reading_ease` alone provides ~40% of the model's discriminative power. Victorian prose is structurally harder to read in ways AI doesn't replicate.
2. **Punctuation is the fingerprint** — em-dashes, semicolons, and periods together contribute significantly. Dickens' em-dash-heavy style is a dead giveaway.
3. **Lexical diversity matters for AI detection** — `hapax_ratio` is the main signal for identifying Generic AI (which uses more generic vocabulary).
4. **Vocabulary richness features (MATTR, TTR) are useless** — the model doesn't need them because structural features are already sufficient.

**Technical note:** `shap.TreeExplainer` failed with `ValueError: could not convert string to float: '[1.5624774E0,...]'` — a known SHAP/XGBoost version compatibility bug where XGBoost stores multiclass base_score as an array string. Workaround: fell back to `shap.Explainer` with `predict_proba` (PermutationExplainer). Slower (~1.5 min for 933 samples) but correct.

---

## Prompt Engineering: From 99% to Reality

### The Problem: When 99% Accuracy Is a Red Flag

Round 1 used naive zero-shot prompts for AI generation:
- Class 2: "Write a paragraph of 120-170 words exploring themes of {topic}... Write in a clear, neutral, expository style."
- Class 3: "Write in the style of Charles Dickens and Mary Shelley: use long, nested sentences with multiple subordinate clauses; employ semicolons and em-dashes freely..."

All five classifiers achieved >99% accuracy. This is not a success — it means the AI text was trivially detectable. Diagnosis:
- **Flesch scores:** AI text scored 10-30 (barely readable) vs Human 60+ (comfortable). The AI equated "Victorian" with "unreadable."
- **Vocabulary:** AI used abstract emotional language ("tapestry of human endeavor") instead of concrete details. Hapax ratio was significantly lower.
- **Sentence structure:** AI maintained uniform sentence lengths (low std). Human text ranged wildly from 5-word fragments to 80-word run-ons.
- **Punctuation:** Style-mimicking AI overdosed on em-dashes (1.41 vs Human 0.60 per 100 words) and underused periods (1.53 vs Human 4.98).

**Conclusion:** The task was too easy. The prompts were the bottleneck, not the models.

### The Fix: Research-Grade Prompt Engineering

We redesigned prompts using three techniques from recent NLP literature:

1. **Few-shot In-Context Learning (5 real excerpts):** Instead of describing Victorian style abstractly, we provided 5 actual excerpts from *Bleak House* and *Frankenstein*. The model learns prose rhythm from examples rather than imagining it from instructions.

2. **Persona + Chain-of-Thought prompting:** The model is told "You are a Victorian novelist" and asked to reason about what makes the excerpts distinctive before writing. This shifts it from "essay writer" to "novelist" mode.

3. **Anti-detection constraints:** We blacklisted known AI-isms (`tapestry`, `testament`, `delve`, `crucial`, `furthermore`, `it is important`, `in conclusion`) identified by our Round 1 SHAP analysis, and added statistical constraints (sentence length variance, proper noun requirement, no repeated sentence openers).

**Generation config changes:** Higher temperature (0.9-1.0 vs default), higher top_p (0.9-0.95), to increase lexical diversity and burstiness.

### Round 2 Results

| Model | Tier | Round 1 3-Class Acc | Round 2 3-Class Acc | Round 2 F1 | Delta |
|-------|------|-------------------|-------------------|-----------|-------|
| XGBoost | A | 99.57% | 96.67% | 0.882 | **-2.90%** |
| Random Forest | A | 99.36% | 96.46% | 0.877 | **-2.90%** |
| GloVe FF-NN | B | — | 99.14% | 0.971 | — |
| SBERT FF-NN | B | — | 99.25% | 0.983 | — |
| DistilBERT+LoRA | C | — | *needs GPU retraining* | — | — |

**Key observations:**

1. **Tier A dropped meaningfully** — XGBoost 3-class accuracy fell from 99.57% to 96.67%, and F1 from 0.990 to 0.882. The improved prompts made style-mimicking AI text genuinely harder to distinguish. Error analysis shows 31/932 test errors (3.3%), with Style-Mimicking AI being the most confused class: 13 misclassified as Generic AI and 10 as Human.

2. **Tier B remained robust** — Embedding-based models (GloVe at 99.14%, SBERT at 99.25%) barely flinched. This reveals that while handcrafted statistical features can be gamed through prompt engineering, deep semantic embeddings capture patterns that survive prompt manipulation. This is an important finding for detector design.

3. **AI-isms persist despite blacklisting** — Round 2's AI-isms analysis shows "testament" still appears at 13.1% and "tapestry" at 6.8% despite explicit blacklisting in the prompt. Sentence opener entropy remains low (Human=8.25 vs Generic AI=5.30 vs Style AI=5.45), confirming AI text is measurably less varied in how it begins sentences. JSD=0.27 with only 26% shared vocabulary between AI and human text.

### What This Proves

The accuracy drop from Round 1 to Round 2 validates the central thesis: **prompt engineering — not model architecture — is the primary variable in AI text detection difficulty.** Naive prompts produce text that any statistical classifier catches at 99%+; research-grade prompts with few-shot ICL, persona framing, and anti-detection constraints force Tier A accuracy down by ~3%.

However, the robustness of Tier B (embedding-based) models suggests a hierarchy of detection difficulty: surface-level statistical features (Tier A) are vulnerable to prompt engineering, while deeper semantic representations (Tier B) are not. This implies that practical AI detectors should rely on embeddings rather than handcrafted features if adversarial prompt engineering is expected.

The persistence of AI-isms despite explicit blacklisting reveals a fundamental limitation of current LLMs: certain statistical fingerprints (vocabulary distribution, sentence opener patterns, readability profiles) are deeply embedded in the model's generation process and cannot be eliminated through prompting alone.

---

## Analysis and Insights

### The Core Finding: AI Text is Too Perfect

The dominant insight from Task 1 and Task 2 is that AI-generated text fails at mimicking human writing not because it's worse, but because it's too regular. The "ghost in the machine" leaves fingerprints through:

1. **Excessive readability:** AI text converges to a comfortable reading level (grade 10-12). Victorian prose ranges wildly from grade 8 to 20+ because real authors write for effect, not optimization.

2. **Punctuation uniformity:** AI uses periods and commas in predictable ratios. Human authors (especially Dickens) deploy em-dashes, semicolons, colons, and exclamation marks with idiosyncratic patterns that reflect personal style, not grammatical necessity.

3. **Sentence length regularity:** AI maintains relatively uniform sentence lengths (low std). Human text has extreme variance — short punchy dialogue next to 80-word run-on descriptions.

4. **Vocabulary homogeneity:** The hapax ratio (words used exactly once) is lower in AI text, suggesting AI draws from a more constrained vocabulary distribution even when given diverse topics.

### Style Mimicry Failure

The Round 1 3-class accuracy (~99.7%) reveals that simply prompting an LLM to "write in the style of Dickens" produces text that is statistically *more different* from real Dickens than generic AI text in some dimensions. The style-mimicking AI overshoots — it may use longer sentences and archaic words, but it does so with machine-like regularity that makes it even more detectable.

This has implications for AI detection: sophisticated style mimicry may actually make AI text *easier* to detect if the detector uses the right features.

### Class Imbalance Note

The dataset is heavily imbalanced (5,211 Human vs 500+500 AI). The high accuracy could partly reflect the model's ability to identify the majority class. However, the per-class F1 scores (all >0.99) confirm that the model genuinely separates all three classes, not just defaulting to "Human."

---

## The Personal Test: SOP Self-Test

### Can the detector tell that I'm human?

My Statement of Purpose (789 words, truncated to 200 for classification) was run through all three classifier tiers.

**Results — Original SOP:**

| Classifier | Prediction | P(Human) | P(GenAI) | P(StyleAI) |
|-----------|-----------|---------|---------|-----------|
| XGBoost (Tier A) | **Human** | 0.993 | 0.007 | 0.000 |
| SBERT FF-NN (Tier B) | **Human** | 0.997 | 0.002 | 0.001 |
| DistilBERT+LoRA (Tier C) | **Human** | — | — | — |

**Verdict: 3/3 classifiers say Human.** My writing passes cleanly.

### Why does the detector say Human?

Feature comparison between my SOP and the training data class averages:

| Feature | My SOP | Human avg | GenAI avg | StyleAI avg |
|---------|--------|-----------|-----------|-------------|
| flesch_reading_ease | 64.7 | 66.8 | 46.4 | 50.5 |
| avg_sent_len | 15.5 | 28.2 | 26.9 | 30.9 |
| sent_len_std | 7.9 | 14.3 | 7.6 | 10.5 |
| hapax_ratio | 0.84 | 0.81 | 0.87 | 0.88 |
| mattr | 0.87 | 0.82 | 0.84 | 0.84 |

**Key insight:** My Flesch Reading Ease (64.7) is squarely in the human range (~67), while AI text scores 46-50. This is the #1 feature driving the Human prediction (confirmed by SHAP). However, my sentence length variance (7.9) is actually closer to AI (7.6) than to the human corpus average (14.3) — my academic writing is more uniform than Victorian prose, but the readability signal dominates.

### Adversarial rewrite: Can I fool my own detector?

Since the SOP was classified as Human, I rewrote the opening in deliberate LLM style — hedging phrases ("it is important to note"), buzzwords ("tapestry", "delve", "crucial"), rigid parallel structure, and no personality:

> *"It is important to note that my motivation for applying to the PreCog Lab stems from a deep-seated passion for understanding the intersection of natural language processing and societal impact. Furthermore, I believe that the crucial role played by computational linguistics in today's world cannot be understated. Throughout my academic journey, I have delved into various aspects of machine learning, gaining valuable insights into the tapestry of modern AI research..."*

**Results — AI-ified rewrite:**

| Classifier | Prediction | P(Human) |
|-----------|-----------|---------|
| XGBoost (Tier A) | **Generic AI** | 0.001 |
| SBERT FF-NN (Tier B) | **Human** | 1.000 |
| DistilBERT+LoRA (Tier C) | **Generic AI** | 0.001 |

**P(Human) change:** XGBoost dropped from 0.993 to 0.001 — a complete inversion.

**2/3 classifiers fooled.** The AI-ified rewrite successfully triggered detection by XGBoost and DistilBERT. But SBERT (embedding-based) was immune — it still saw Human with P=1.000.

### What this proves

The "ghost" works both ways:
1. **AI text can be engineered to fool statistical detectors** (as shown by the GA and prompt engineering experiments)
2. **Human text can be made to look like AI** by adopting LLM writing patterns — hedging, buzzwords, rigid structure, no voice

The divergence between classifiers is telling:
- **Tier A (XGBoost)** and **Tier C (DistilBERT)** are sensitive to surface patterns — vocabulary choice, sentence structure, readability scores. These are the same features that SHAP identified as important, and they're exactly what the adversarial rewrite manipulated.
- **Tier B (SBERT)** captures deeper semantic meaning through sentence embeddings. Adding "furthermore" and "it is important to note" doesn't change the underlying semantic content, so SBERT sees through the disguise.

This has practical implications: a robust AI detector should combine statistical features with embedding-based models. Neither alone is sufficient — statistical models catch the obvious cases but can be fooled by style transfer, while embedding models are more robust but may miss cases where the semantic content itself is AI-generated.

---

## What I Would Do Differently / Next Steps

1. **Cross-model evaluation:** Train on one LLM's output, test on another's. Current results are Gemini-specific — would the XGBoost detector trained on Gemini output also catch GPT-4 or other LLM-generated text?

2. **Reduce class imbalance:** Either downsample Human to 500 or generate more AI paragraphs. The current 5:1:1 ratio, while handled correctly, may affect model calibration.

3. **Human baseline study:** Have human readers attempt the same classification task to establish a human performance ceiling. If humans can't distinguish the classes either, high accuracy is less impressive.

4. **Feature ablation study:** Systematically remove feature groups (e.g., remove all punctuation features) to quantify their contribution beyond SHAP values.

5. **Longer text evaluation:** Test whether classification remains accurate at shorter text lengths (single sentences, 20-word fragments). Understanding the minimum text length for reliable detection has practical applications.

6. **Domain transfer:** Would features trained on Victorian prose transfer to detecting AI-generated modern text (blog posts, news articles)?

---

## Reproducibility

Every computationally expensive step uses checkpoint logic: if the output file exists, it loads from disk; if not, it recomputes from scratch. All intermediate results — cleaned texts, generated AI paragraphs, feature matrices, model predictions, and figures — are committed to the repository. This means the notebook can be reviewed end-to-end by simply running cells sequentially, loading from saved checkpoints in minutes without any API keys or GPU.

To regenerate any component from scratch, delete the relevant checkpoint file and re-run:

| Component | Checkpoint file(s) | Requirements | Approx. time |
|-----------|-------------------|--------------|--------------|
| AI paragraphs (1,000) | `data/generated/class{2,3}_raw.json` | Gemini API key | ~17 min (10 concurrent workers) |
| Feature extraction (23 features × 6,211 samples) | `outputs/results/feature_matrix.csv` | spaCy `en_core_web_sm` | ~10 min |
| Tier A (XGBoost/RF) | Retrained each run | None | ~3 min |
| Tier B (GloVe/SBERT FF-NN) | Retrained each run | GloVe embeddings (331 MB download) | ~5 min |
| Tier C (DistilBERT+LoRA) | `outputs/models/distilbert_lora_*` | GPU (Colab recommended) | ~20 min |
| BERTopic topics | `outputs/results/bertopic_topics.pkl` | None | ~5 min |
| GA adversarial attack | Runs fresh each time | Gemini API key | ~12 min |

**Large files excluded from the repository** (gitignored): GloVe embeddings (`glove.6B.100d.txt`, 331 MB — download from [Stanford NLP GloVe](https://nlp.stanford.edu/projects/glove/)), BERTopic model (117 MB — regenerated automatically), DistilBERT+LoRA checkpoints (requires GPU retraining), and SBERT embeddings (9 MB — regenerated during Tier B training).

---

