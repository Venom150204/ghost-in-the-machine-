# The Ghost in the Machine: Authorship Detection in the Age of LLMs

Detecting whether a piece of prose was written by a human author, a generic AI, or an AI deliberately mimicking a literary style — using multi-tiered NLP classifiers with explainability and adversarial testing.

## Directory Structure

```
Precog-IIIT_Hybd/
├── data/
│   ├── raw/                    # Raw Gutenberg .txt files
│   ├── cleaned/                # Cleaned author text chunks
│   ├── generated/              # Gemini-generated paragraphs (Class 2 and Class 3)
│   └── final/                  # Final merged dataset as CSV
├── notebooks/
│   └── ghost_in_the_machine.ipynb   # Main notebook — all experiments
├── src/
│   ├── data_utils.py           # Cleaning, chunking, dataset assembly
│   ├── feature_extraction.py   # Task 1 feature functions
│   ├── models.py               # Tier A and Tier B model code
│   ├── genetic_algorithm.py    # Task 4 GA logic
│   └── explainability.py       # Task 3 saliency/SHAP utilities
├── outputs/
│   ├── figures/                # All plots and heatmaps
│   ├── models/                 # Saved model checkpoints
│   └── results/                # CSVs of predictions and metrics
├── sop/
│   └── my_sop.txt              # Your Statement of Purpose
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

### 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if you don't have it
pip install uv

# Create venv and install all dependencies
uv sync
```

### 2. User-provided files

Place these manually before running the notebook:

| File | Description |
|------|-------------|
| `data/raw/BLEAK_HOUSE.txt` | Project Gutenberg #1023 (Plain Text UTF-8) |
| `data/raw/GREAT_EXPECTATIONS.txt` | Project Gutenberg #1400 (Plain Text UTF-8) |
| `data/raw/FRANKENSTEIN.txt` | Project Gutenberg #84 (Plain Text UTF-8) |
| `data/raw/THE_LAST_MAN.txt` | Project Gutenberg #18247 (Plain Text UTF-8) |
| `sop/my_sop.txt` | Your Statement of Purpose or personal essay |

### 3. API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

### 4. Download NLP models

```bash
uv run python -m spacy download en_core_web_sm
uv run python -m nltk.downloader punkt averaged_perceptron_tagger stopwords
```

## How to Run

Open `notebooks/ghost_in_the_machine.ipynb` and run cells sequentially. All results, plots, and metrics are displayed inline. Helper modules in `src/` are imported automatically.

## Tasks and Approach

- **Task 0 — Dataset Construction:** Clean Gutenberg texts, extract topics via BERTopic, generate AI paragraphs with Gemini (generic + style-mimicking). Three classes: Human, Generic AI, Style-Mimicking AI.
- **Task 1 — Feature Analysis:** Compute lexical (MATTR, hapax), syntactic (POS ratios, parse tree depth), punctuation density, and readability features. Visualize with PCA/t-SNE.
- **Task 2 — Classification:** Three tiers — XGBoost/RF on handcrafted features, feedforward NNs on GloVe/SBERT embeddings, DistilBERT+LoRA fine-tuning. Binary and 3-class evaluation.
- **Task 3 — Explainability:** SHAP on tree models, Integrated Gradients on DistilBERT via Captum. Error analysis on misclassified human paragraphs.
- **Task 4 — Adversarial Testing:** Genetic algorithm to evolve AI text past the detector. SOP self-analysis with the trained models.

## Key Findings

- **All tiers achieve >99% accuracy** on both binary and 3-class tasks — current AI-generated prose is distinguishable from 19th-century literary text across multiple representation levels.
- **Handcrafted features are surprisingly competitive** — XGBoost reaches 99.57% 3-class accuracy using only 23 interpretable features. AI text has measurable statistical fingerprints.
- **SHAP reveals key discriminators:** MATTR (lexical diversity), semicolon density, and Flesch-Kincaid grade are the top features. Human Victorian prose uses more varied vocabulary and complex punctuation.
- **DistilBERT+LoRA achieves perfect classification** (100% on both tasks) while training only 1.09% of parameters, showing authorship signals live in a narrow subspace of transformer representations.
- **GA adversarial attacks partially succeed against XGBoost** — 2 of 3 AI samples crossed the 0.95 P(Human) threshold through Gemini-powered mutations, but deeper models remain robust.
- **Style mimicry overshoots** — Gemini's "Victorian style" text is statistically *more extreme* than real Victorian prose, making it paradoxically easier to detect.

## Limitations and Future Work

- **Dataset imbalance:** 5,211 human vs 500+500 AI paragraphs. Stratified splitting mitigates but doesn't eliminate the effect.
- **Temporal specificity:** Human corpus is exclusively 19th-century British literature. Modern human writing would be harder to distinguish.
- **Single AI source:** All AI text from Gemini. Different LLMs have different fingerprints — cross-model generalization is untested.
- **GA attacks only target Tier A:** Gradient-based attacks on the transformer would be the natural next step.
- **Future directions:** (1) Multi-source AI text from diverse LLMs, (2) gradient-based adversarial attacks on Tier C, (3) modern human text to test temporal robustness, (4) sentence-level detection, (5) ensemble of all tiers with confidence calibration.

## References

- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian Knot: The Moving-Average Type-Token Ratio (MATTR). *Journal of Quantitative Linguistics*.
- Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. *NAACL*.
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv:2203.05794*.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
- Dickens, C. *Bleak House*. Project Gutenberg #1023.
- Dickens, C. *Great Expectations*. Project Gutenberg #1400.
- Shelley, M. *Frankenstein*. Project Gutenberg #84.
- Shelley, M. *The Last Man*. Project Gutenberg #18247.
