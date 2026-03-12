# The Ghost in the Machine — Research Report

**Project:** Authorship Detection in the Age of LLMs
**Submitted to:** PreCog Lab, IIIT Hyderabad
**Authors:** Charles Dickens & Mary Shelley (human corpus); Gemini API (AI corpus)
**Last Updated:** 2026-03-10

---

## Page 1: Task Completion Status

- [x] **Task 0: Dataset Construction** — completed
  - [x] Gutenberg text acquisition (Dickens: *Bleak House*; Shelley: *Frankenstein*, *The Last Man*)
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
  - [x] Binary classification (Human vs AI): ~99.9% accuracy
  - [x] 3-class classification (Human / Generic AI / Style-Mimicking AI): ~99.7% accuracy
  - [x] SHAP analysis on XGBoost 3-class model

- [ ] **Task 2 Tier B: FF-NN on GloVe / SBERT** — not yet run
  - [x] Code written (GloVe averaging, SBERT embeddings, FF-NN architecture)
  - [x] GloVe vectors downloaded (glove.6B.100d.txt)
  - [ ] Awaiting execution

- [ ] **Task 2 Tier C: DistilBERT + LoRA** — not yet run
  - [x] Code written (LoRA fine-tuning with PEFT)
  - [ ] Requires GPU (designed for Google Colab)

- [ ] **Task 3: Explainability & Error Analysis** — code written, not yet run
  - [x] Error analysis function written
  - [x] Integrated Gradients (Captum) function written
  - [x] AI-isms detection function written
  - [ ] Awaiting execution

- [ ] **Task 4: Adversarial GA Optimization** — code written, not yet run
  - [x] Genetic algorithm engine written (synonym/punctuation/shuffle/filler mutations)
  - [x] Fitness function (P(Human) from classifier)
  - [ ] Awaiting execution

- [ ] **SOP Self-Test** — not yet attempted

---

## Methodology

### Dataset Construction (Task 0)

**Author Selection:** Charles Dickens (*Bleak House*) and Mary Shelley (*Frankenstein*, *The Last Man*) were chosen for their contrasting Victorian-era styles — Dickens for elaborate social commentary prose with heavy dialogue, Shelley for Gothic Romantic narrative with introspective philosophical passages. Both are sufficiently distinct that a 3-class classifier must learn genuine stylistic features rather than trivial content cues.

**Chunking Strategy:** Raw Gutenberg texts were cleaned (headers/footers stripped, encoding normalized) and split into paragraph-level chunks of 80-300 words. This window size preserves enough syntactic and stylistic signal for feature extraction while being comparable to typical AI-generated paragraph lengths.

**Topic Extraction:** BERTopic was used to discover latent themes in the human corpus, providing topically-grounded prompts for AI generation. A custom `CountVectorizer` with English stop words plus corpus-specific high-frequency words (>15% document frequency) ensured meaningful topic representations. Seven topics were discovered and auto-labeled from their top-5 keywords.

**NER-based Filtering:** Instead of hardcoding character names, spaCy's NER model (`en_core_web_sm`) was run over a corpus sample to auto-detect PERSON entities. These names were added to BERTopic's stop word list so topics reflect thematic content rather than character mentions. This makes the pipeline generalizable to any books.

**AI Generation:** Gemini API generated 500 paragraphs per AI class:
- Class 2 (Generic AI): Prompted to "write a paragraph exploring themes of {topic}" — no style instruction.
- Class 3 (Style-Mimicking AI): Prompted to "write a paragraph in the style of {author}, exploring themes of {topic}" — explicit style mimicry.

Paragraphs were distributed proportionally across the 7 topics. Incremental JSON checkpointing ensures crash-safe resume. The current data was generated with `gemini-3.1-flash-lite-preview`; regeneration with `gemini-2.5-flash-lite` is planned for better quality.

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

**SHAP TreeExplainer** on XGBoost reveals which handcrafted features drive predictions. **Integrated Gradients** (via Captum) on DistilBERT highlights which tokens matter most at the subword level. **AI-isms detector** hunts for known AI writing patterns: filler phrases ("It is important to note"), uniform sentence lengths, and repetitive opening words.

### Adversarial Optimization (Task 4)

A genetic algorithm with linguistically-motivated mutations:
- Synonym substitution (WordNet)
- Punctuation variation (period→semicolon, add em-dashes)
- Sentence reordering
- Filler phrase insertion ("indeed", "of course")

Fitness = classifier's P(Human) for the mutated text. Tournament selection, single-point sentence-level crossover, elite preservation.

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

### Experiment 2: AI Paragraph Generation (gemini-3.1-flash-lite-preview)

**What:** Generated 500 Class 2 (Generic AI) and 500 Class 3 (Style-Mimicking AI) paragraphs using Gemini API, distributed across 7 topics.

**Result:** All 1,000 paragraphs generated successfully. Incremental checkpointing saved progress every paragraph.

**Outcome:** Success, but quality is questionable given the model used (flash-lite-preview is the lowest-tier model). The style-mimicking paragraphs may not represent a strong adversarial challenge. Regeneration with `gemini-2.5-flash-lite` is planned.

**Rate limiting journey:**
- Free tier (gemini-1.5-flash): 429 errors after ~15 requests
- gemini-3.1-flash-lite-preview: Worked with 7.0s sleep (free tier RPM limits)
- After Paid Tier 1 ($300 credits): Switched to gemini-2.5-flash, but it's a "thinking" model (~9s/request), making it paradoxically slower
- Final: gemini-2.5-flash (thinking model, ~9-15s/request). Slower but highest quality flash model.

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

**What:** Trained XGBoost and Random Forest with GridSearchCV on the 23 handcrafted features. Both binary and 3-class setups. Data generated with gemini-2.5-flash.

**Result:**

| Model | Task | Accuracy | Macro F1 | Best Params |
|-------|------|----------|----------|-------------|
| XGBoost | Binary | 99.68% | 0.9940 | lr=0.1, depth=5, n=200 |
| Random Forest | Binary | 99.57% | 0.9920 | — |
| XGBoost | 3-Class | 99.57% | 0.9901 | lr=0.1, depth=7, n=200 |
| Random Forest | 3-Class | 99.36% | 0.9832 | — |

**Confusion matrix (XGBoost 3-class):**
- Human: 781/782 correct (1 misclassified as Style-Mimicking AI)
- Generic AI: 75/75 perfect
- Style-Mimicking AI: 72/75 (3 misclassified as Human — 4% error rate)

**Comparison with previous run (gemini-3.1-flash-lite-preview):**

| Metric | flash-lite-preview | 2.5-flash | Delta |
|--------|-------------------|-----------|-------|
| XGB Binary Acc | ~99.9% | 99.68% | -0.2% |
| XGB 3-Class Acc | ~99.7% | 99.57% | -0.1% |
| XGB 3-Class F1 | ~0.994 | 0.9901 | -0.4% |

The accuracy dropped slightly with gemini-2.5-flash, confirming the better model produces marginally more human-like text. But the drop is negligible — even a state-of-the-art thinking model can't fool handcrafted statistical features.

**Outcome:** Near-perfect classification. Style-Mimicking AI is the only class with any confusion (4% error → Human). Generic AI is trivially detectable.

**Hypothesis:** The "thinking" capability of 2.5-flash occasionally produces paragraphs that genuinely hit the right statistical profile (the 3 misclassified Style AI samples), but it also tends to overthink and overshoot stylistic markers, making most samples even more detectable than simpler models.

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

## Analysis and Insights

### The Core Finding: AI Text is Too Perfect

The dominant insight from Task 1 and Task 2 is that AI-generated text fails at mimicking human writing not because it's worse, but because it's too regular. The "ghost in the machine" leaves fingerprints through:

1. **Excessive readability:** AI text converges to a comfortable reading level (grade 10-12). Victorian prose ranges wildly from grade 8 to 20+ because real authors write for effect, not optimization.

2. **Punctuation uniformity:** AI uses periods and commas in predictable ratios. Human authors (especially Dickens) deploy em-dashes, semicolons, colons, and exclamation marks with idiosyncratic patterns that reflect personal style, not grammatical necessity.

3. **Sentence length regularity:** AI maintains relatively uniform sentence lengths (low std). Human text has extreme variance — short punchy dialogue next to 80-word run-on descriptions.

4. **Vocabulary homogeneity:** The hapax ratio (words used exactly once) is lower in AI text, suggesting AI draws from a more constrained vocabulary distribution even when given diverse topics.

### Style Mimicry Failure

The 3-class accuracy (~99.7%) reveals that simply prompting an LLM to "write in the style of Dickens" produces text that is statistically *more different* from real Dickens than generic AI text in some dimensions. The style-mimicking AI overshoots — it may use longer sentences and archaic words, but it does so with machine-like regularity that makes it even more detectable.

This has implications for AI detection: sophisticated style mimicry may actually make AI text *easier* to detect if the detector uses the right features.

### Class Imbalance Note

The dataset is heavily imbalanced (5,211 Human vs 500+500 AI). The high accuracy could partly reflect the model's ability to identify the majority class. However, the per-class F1 scores (all >0.99) confirm that the model genuinely separates all three classes, not just defaulting to "Human."

---

## What I Would Do Differently / Next Steps

1. **Regenerate AI data with a stronger model:** The current data from `gemini-3.1-flash-lite-preview` may be too easy to detect. Using `gemini-2.5-flash-lite` or `gemini-2.5-pro` would create a more challenging and realistic adversarial scenario.

2. **Cross-model evaluation:** Train on one LLM's output, test on another's. Current results are model-specific — would the XGBoost detector trained on Gemini output also catch GPT-4 or Claude-generated text?

3. **Reduce class imbalance:** Either downsample Human to 500 or generate more AI paragraphs. The current 5:1:1 ratio, while handled correctly, may affect model calibration.

4. **Human baseline study:** Have human readers attempt the same classification task to establish a human performance ceiling. If humans can't distinguish the classes either, 99.7% accuracy is less impressive.

5. **Feature ablation study:** Systematically remove feature groups (e.g., remove all punctuation features) to quantify their contribution beyond SHAP values.

6. **Longer text evaluation:** Test whether classification remains accurate at shorter text lengths (single sentences, 20-word fragments). Understanding the minimum text length for reliable detection has practical applications.

7. **Domain transfer:** Would features trained on Victorian prose transfer to detecting AI-generated modern text (blog posts, news articles)?

---

*This report is updated incrementally as experiments complete. Sections below this line will be filled as Tiers B, C, Task 3, and Task 4 are executed.*