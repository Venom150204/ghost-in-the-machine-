"""
Explainability utilities for authorship detection.

Three levels of explainability:
1. SHAP (handled in notebook via shap library directly)
2. Integrated Gradients on DistilBERT (token-level attribution)
3. Data-driven AI-isms detection:
   - Statistical n-gram analysis (log-odds ratio)
   - POS pattern mining for syntactic templates
   - Sentence opener entropy
   - Vocabulary distribution divergence (KL divergence)
   - Validation against known AI-isms from the literature
"""

import re
import numpy as np
import pandas as pd
import torch
import spacy
from collections import Counter
from captum.attr import IntegratedGradients

# Load spaCy model once (POS-only variant for efficiency)
_nlp_pos = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def get_misclassified(y_true, y_pred, texts, class_names):
    """Returns a DataFrame of misclassified samples with their true/predicted labels."""
    mask = y_true != y_pred
    errors = pd.DataFrame({
        "text": np.array(texts)[mask],
        "true_label": y_true[mask],
        "pred_label": y_pred[mask],
        "true_class": [class_names[y] for y in y_true[mask]],
        "pred_class": [class_names[y] for y in y_pred[mask]],
    })
    return errors


def error_analysis_summary(y_true, y_pred, texts, class_names):
    """Computes error breakdown with confusion pair counts."""
    errors = get_misclassified(y_true, y_pred, texts, class_names)
    if len(errors) == 0:
        return errors, "No misclassifications found."

    pair_counts = errors.groupby(["true_class", "pred_class"]).size().sort_values(ascending=False)
    summary_lines = [f"Total errors: {len(errors)} / {len(y_true)} ({len(errors)/len(y_true):.1%})\n"]
    summary_lines.append("Most confused pairs:")
    for (tc, pc), count in pair_counts.items():
        summary_lines.append(f"  {tc} -> {pc}: {count}")

    return errors, "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Integrated Gradients on DistilBERT
# ---------------------------------------------------------------------------

def compute_integrated_gradients_distilbert(model, tokenizer, text, target_class, device="cpu"):
    """
    Computes Integrated Gradients attributions for a DistilBERT model.
    Returns (tokens, attribution_scores) for non-padding tokens.
    """
    model.eval()
    model.to(device)

    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Forward through embeddings -> classifier
    # Handle both raw DistilBert and PEFT-wrapped models
    distilbert = getattr(model, "distilbert", None)
    classifier = getattr(model, "classifier", None)
    if distilbert is None:
        # PEFT-wrapped model: model.base_model.model.distilbert
        distilbert = model.base_model.model.distilbert
        classifier = model.base_model.model.classifier

    def forward_func(inputs_embeds):
        outputs = distilbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    embeddings = distilbert.embeddings(input_ids)
    embeddings.requires_grad_(True)
    baseline = torch.zeros_like(embeddings)

    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(
        embeddings, baselines=baseline, target=target_class, n_steps=50,
    )

    token_attrs = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

    mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)
    tokens = [t for t, m in zip(tokens, mask) if m]
    token_attrs = token_attrs[mask]

    return tokens, token_attrs


# ---------------------------------------------------------------------------
# Data-driven AI-isms detection
# ---------------------------------------------------------------------------

def _tokenize_simple(text):
    """Lowercase tokenization preserving hyphenated words."""
    return re.findall(r"\b[a-z](?:[a-z'-]*[a-z])?\b", text.lower())


def _get_ngrams(tokens, n):
    """Extract n-grams from token list."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_log_odds_ngrams(ai_texts, human_texts, ns=(1, 2, 3), top_k=30, min_count=3):
    """
    Finds n-grams statistically overrepresented in AI text vs human text
    using the log-odds ratio with an informative Dirichlet prior
    (Monroe, Colaresi & Quinn, 2008).

    This is the core data-driven method: instead of guessing which words
    are "AI-isms", we let the data tell us. The log-odds ratio measures
    how much more likely an n-gram is in AI text vs human text, smoothed
    by a prior to avoid division-by-zero on rare terms.

    Returns:
        DataFrame with columns: ngram, n, ai_count, human_count, log_odds, z_score
        Sorted by z_score descending (most AI-distinctive first).
    """
    # Build frequency dictionaries per class (tokenize each text once)
    ai_freq = Counter()
    human_freq = Counter()

    for text in ai_texts:
        tokens = _tokenize_simple(text)
        for n in ns:
            ai_freq.update(_get_ngrams(tokens, n))

    for text in human_texts:
        tokens = _tokenize_simple(text)
        for n in ns:
            human_freq.update(_get_ngrams(tokens, n))

    # Total counts
    n_ai = sum(ai_freq.values())
    n_human = sum(human_freq.values())
    vocab = set(ai_freq.keys()) | set(human_freq.keys())

    # Compute log-odds with Dirichlet prior (alpha = 0.01 per term)
    alpha = 0.01
    n_vocab = len(vocab)

    rows = []
    for ngram in vocab:
        ai_c = ai_freq.get(ngram, 0)
        human_c = human_freq.get(ngram, 0)

        if ai_c + human_c < min_count:
            continue

        # Log-odds ratio with smoothing
        ai_rate = (ai_c + alpha) / (n_ai + alpha * n_vocab)
        human_rate = (human_c + alpha) / (n_human + alpha * n_vocab)
        log_odds = np.log(ai_rate) - np.log(human_rate)

        # Approximate variance for z-score
        variance = 1.0 / (ai_c + alpha) + 1.0 / (human_c + alpha)
        z_score = log_odds / np.sqrt(variance)

        n_size = len(ngram.split())
        rows.append({
            "ngram": ngram,
            "n": n_size,
            "ai_count": ai_c,
            "human_count": human_c,
            "log_odds": round(log_odds, 4),
            "z_score": round(z_score, 4),
        })

    df = pd.DataFrame(rows).sort_values("z_score", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def compute_sentence_opener_entropy(texts, class_labels, class_names):
    """
    Measures Shannon entropy of sentence-opening words per class.

    Low entropy = repetitive openings = AI fingerprint.
    Human writers vary their sentence starts more, producing higher entropy.

    Returns:
        dict mapping class_name -> (entropy, top_5_openers)
    """
    results = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = class_labels == cls_idx
        openers = []
        for text in texts[mask]:
            sentences = re.split(r'(?<=[.!?])\s+', str(text))
            for sent in sentences:
                words = sent.strip().split()
                if words:
                    openers.append(words[0].lower())

        if not openers:
            results[cls_name] = (0.0, [])
            continue

        counter = Counter(openers)
        total = sum(counter.values())
        probs = np.array([c / total for c in counter.values()])
        entropy = -np.sum(probs * np.log2(probs + 1e-12))

        results[cls_name] = (round(entropy, 3), counter.most_common(5))

    return results


def compute_vocab_divergence(ai_texts, human_texts):
    """
    Computes symmetrized KL divergence (Jensen-Shannon divergence) between
    the word frequency distributions of AI and human text.

    Higher JSD = more different vocabularies. This quantifies the overall
    "vocabulary gap" between classes without picking specific words.

    Returns:
        dict with jsd, ai_vocab_size, human_vocab_size, shared_vocab_fraction
    """
    ai_freq = Counter()
    human_freq = Counter()

    for text in ai_texts:
        ai_freq.update(_tokenize_simple(text))
    for text in human_texts:
        human_freq.update(_tokenize_simple(text))

    vocab = set(ai_freq.keys()) | set(human_freq.keys())
    shared = set(ai_freq.keys()) & set(human_freq.keys())

    n_ai = sum(ai_freq.values())
    n_human = sum(human_freq.values())

    # Build probability distributions over shared vocabulary
    p = np.array([ai_freq.get(w, 0) / n_ai for w in vocab])
    q = np.array([human_freq.get(w, 0) / n_human for w in vocab])

    # Add smoothing to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    # Jensen-Shannon divergence
    m = 0.5 * (p + q)
    jsd = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))

    return {
        "jsd": round(float(jsd), 4),
        "ai_vocab_size": len(ai_freq),
        "human_vocab_size": len(human_freq),
        "shared_vocab_pct": round(100 * len(shared) / len(vocab), 1),
    }


def detect_pos_patterns(texts, class_labels, class_names, top_k=10):
    """
    Mines POS tag trigram patterns that are overrepresented in AI text.
    Uses spaCy for POS tagging.

    This catches syntactic templates AI overuses — e.g., repeated
    "DET NOUN VERB" or "ADJ NOUN ADP" patterns — without needing to
    know the specific words.

    Returns:
        DataFrame of POS trigrams with per-class frequencies and log-odds.
    """
    nlp = _nlp_pos

    pos_freqs = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = class_labels == cls_idx
        counter = Counter()
        sample_texts = texts[mask]
        # Sample to keep runtime reasonable
        if len(sample_texts) > 200:
            indices = np.random.choice(len(sample_texts), 200, replace=False)
            sample_texts = sample_texts[indices]

        for text in sample_texts:
            doc = nlp(str(text))
            pos_tags = [token.pos_ for token in doc if not token.is_space]
            trigrams = [" ".join(pos_tags[i:i+3]) for i in range(len(pos_tags)-2)]
            counter.update(trigrams)

        pos_freqs[cls_name] = counter

    # Compare AI (merged class 1+2) vs Human (class 0)
    human_name = class_names[0]
    ai_names = class_names[1:]

    human_counts = pos_freqs[human_name]
    ai_counts = Counter()
    for name in ai_names:
        ai_counts.update(pos_freqs.get(name, Counter()))

    n_human = sum(human_counts.values()) or 1
    n_ai = sum(ai_counts.values()) or 1
    alpha = 0.01

    rows = []
    all_patterns = set(human_counts.keys()) | set(ai_counts.keys())
    for pat in all_patterns:
        hc = human_counts.get(pat, 0)
        ac = ai_counts.get(pat, 0)
        if hc + ac < 5:
            continue
        ai_rate = (ac + alpha) / (n_ai + alpha * len(all_patterns))
        h_rate = (hc + alpha) / (n_human + alpha * len(all_patterns))
        log_odds = np.log(ai_rate) - np.log(h_rate)
        rows.append({
            "pos_pattern": pat,
            "ai_count": ac,
            "human_count": hc,
            "ai_pct": round(100 * ac / n_ai, 2),
            "human_pct": round(100 * hc / n_human, 2),
            "log_odds_ai": round(log_odds, 3),
        })

    df = pd.DataFrame(rows).sort_values("log_odds_ai", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def detect_parallel_structures(texts, class_labels, class_names):
    """
    Detects syntactic parallelism — repeated structural patterns within
    a single paragraph that AI tends to overuse.

    Checks for:
    - "not only X but also Y" constructions
    - Lists with repeated POS patterns (e.g., three ADJ+NOUN phrases)
    - Repeated sentence-opening patterns within a paragraph

    Returns:
        dict mapping class_name -> {parallel_count, total, fraction}
    """
    # Pattern: "not only ... but also" or "both ... and" or "either ... or"
    parallel_patterns = [
        r"not only\b.{5,60}\bbut also\b",
        r"both\b.{5,40}\band\b",
        r"either\b.{5,40}\bor\b",
        r"neither\b.{5,40}\bnor\b",
    ]
    compiled = [re.compile(p, re.IGNORECASE) for p in parallel_patterns]

    results = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = class_labels == cls_idx
        cls_texts = texts[mask]
        parallel_count = 0
        repeated_opener_count = 0

        for text in cls_texts:
            text_str = str(text).lower()
            # Check explicit parallel constructions
            for pat in compiled:
                if pat.search(text_str):
                    parallel_count += 1
                    break

            # Check repeated sentence openers (3+ sentences starting same way)
            sentences = re.split(r'(?<=[.!?])\s+', str(text))
            if len(sentences) >= 3:
                openers = [s.split()[0].lower() for s in sentences if s.split()]
                opener_counts = Counter(openers)
                if opener_counts and opener_counts.most_common(1)[0][1] >= 3:
                    repeated_opener_count += 1

        total = int(mask.sum())
        results[cls_name] = {
            "explicit_parallel": parallel_count,
            "repeated_openers": repeated_opener_count,
            "total": total,
            "parallel_pct": round(100 * parallel_count / total, 1) if total else 0,
            "repeated_opener_pct": round(100 * repeated_opener_count / total, 1) if total else 0,
        }

    return results


def validate_known_ai_isms(ai_texts, human_texts):
    """
    Validation step: checks whether known AI-isms from the literature
    actually appear in our dataset, and at what rates.

    These terms are from published AI detection research and the project brief.
    The point is not to use these as features — the log-odds analysis above
    discovers markers from data. This is a sanity check: do known markers
    appear in our specific AI output?

    Returns:
        DataFrame with phrase, ai_rate, human_rate, ratio
    """
    known_markers = [
        "delve", "tapestry", "testament", "underscore", "one might",
        "it is important to note", "it is worth noting", "furthermore",
        "moreover", "in conclusion", "it is essential", "in the realm of",
        "plays a crucial role", "not only", "in today's world",
    ]

    ai_lower = [t.lower() for t in ai_texts]
    human_lower = [t.lower() for t in human_texts]
    n_ai = len(ai_lower)
    n_human = len(human_lower)

    rows = []
    for phrase in known_markers:
        ai_hits = sum(1 for t in ai_lower if phrase in t)
        human_hits = sum(1 for t in human_lower if phrase in t)
        ai_rate = ai_hits / n_ai if n_ai else 0
        human_rate = human_hits / n_human if n_human else 0
        ratio = ai_rate / human_rate if human_rate > 0 else (float("inf") if ai_rate > 0 else 0)

        rows.append({
            "phrase": phrase,
            "ai_hits": ai_hits,
            "human_hits": human_hits,
            "ai_rate_pct": round(100 * ai_rate, 2),
            "human_rate_pct": round(100 * human_rate, 2),
            "ratio": round(ratio, 1) if ratio != float("inf") else "inf",
        })

    return pd.DataFrame(rows).sort_values("ai_rate_pct", ascending=False).reset_index(drop=True)


def run_full_ai_isms_analysis(texts, class_labels, feature_df, class_names, verbose=True):
    """
    Master function: runs all seven AI-isms detection methods and returns
    a structured results dict.

    This is the main entry point called from the notebook.
    """
    ai_mask = class_labels > 0
    human_mask = class_labels == 0
    ai_texts = texts[ai_mask]
    human_texts = texts[human_mask]

    results = {}

    # 1. Statistical n-gram analysis
    if verbose:
        print("1/7  Computing log-odds n-gram analysis...")
    results["log_odds_ngrams"] = compute_log_odds_ngrams(
        ai_texts, human_texts, ns=(1, 2, 3), top_k=30, min_count=3
    )
    if verbose:
        print(f"     Found {len(results['log_odds_ngrams'])} significant n-grams")

    # 2. Sentence opener entropy
    if verbose:
        print("2/7  Computing sentence opener entropy...")
    results["opener_entropy"] = compute_sentence_opener_entropy(
        texts, class_labels, class_names
    )
    if verbose:
        for cls, (ent, top5) in results["opener_entropy"].items():
            print(f"     {cls}: entropy={ent:.3f}, top opener='{top5[0][0] if top5 else 'N/A'}'")

    # 3. Vocabulary divergence
    if verbose:
        print("3/7  Computing vocabulary divergence (JSD)...")
    results["vocab_divergence"] = compute_vocab_divergence(ai_texts, human_texts)
    if verbose:
        jsd = results["vocab_divergence"]["jsd"]
        shared = results["vocab_divergence"]["shared_vocab_pct"]
        print(f"     JSD={jsd:.4f}, shared vocab={shared}%")

    # 4. POS pattern mining
    if verbose:
        print("4/7  Mining POS tag patterns (sampling 200 per class)...")
    results["pos_patterns"] = detect_pos_patterns(
        texts, class_labels, class_names, top_k=10
    )
    if verbose:
        top_pat = results["pos_patterns"].iloc[0]["pos_pattern"] if len(results["pos_patterns"]) > 0 else "N/A"
        print(f"     Top AI-distinctive POS pattern: {top_pat}")

    # 5. Parallel structure detection
    if verbose:
        print("5/7  Detecting parallel structures...")
    results["parallel_structures"] = detect_parallel_structures(
        texts, class_labels, class_names
    )

    # 6. Validation against known markers
    if verbose:
        print("6/7  Validating against known AI-isms from literature...")
    results["known_markers"] = validate_known_ai_isms(ai_texts, human_texts)
    if verbose:
        hits = (results["known_markers"]["ai_hits"] > 0).sum()
        print(f"     {hits}/{len(results['known_markers'])} known markers found in AI text")

    # 7. Sentence length variability (from pre-computed features)
    if verbose:
        print("7/7  Computing sentence length variability...")
    if "sent_len_std" in feature_df.columns:
        results["sent_len_variability"] = {
            "human_mean_std": round(feature_df.loc[human_mask, "sent_len_std"].mean(), 2),
            "ai_mean_std": round(feature_df.loc[ai_mask, "sent_len_std"].mean(), 2),
        }

    if verbose:
        print("\nAI-isms analysis complete.")

    return results