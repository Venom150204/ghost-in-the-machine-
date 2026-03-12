"""
Feature extraction functions for authorship detection.
Each function takes a text string and returns a dictionary of numerical features.
All features are used both for analysis (Task 1) and as input to Tier A classifiers (Task 2).
"""

import re
from collections import Counter
import numpy as np
import spacy
import textstat

# Load spaCy model once at module level
nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Lexical Richness
# ---------------------------------------------------------------------------

def compute_ttr(text: str) -> float:
    """Raw Type-Token Ratio: unique words / total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_mattr(text: str, window: int = 50) -> float:
    """
    Moving Average Type-Token Ratio (Covington & McFall, 2010).
    Computes TTR over a sliding window and averages, correcting for
    the text-length bias of raw TTR. Longer texts mechanically produce
    lower raw TTR because repeated function words accumulate; MATTR
    avoids this by never looking at more than `window` tokens at once.
    """
    words = text.lower().split()
    if len(words) < window:
        return compute_ttr(text)

    ttrs = []
    for i in range(len(words) - window + 1):
        w = words[i:i + window]
        ttrs.append(len(set(w)) / len(w))
    return float(np.mean(ttrs))


def compute_hapax_ratio(text: str) -> float:
    """
    Hapax Legomena Ratio: words appearing exactly once / total unique words.
    Hypothesis: human authors take more lexical risks, producing more
    single-use words. AI text tends to recycle a narrower vocabulary.
    """
    words = text.lower().split()
    if not words:
        return 0.0
    freq = Counter(words)
    hapax = sum(1 for w, c in freq.items() if c == 1)
    unique = len(freq)
    return hapax / unique if unique > 0 else 0.0


# ---------------------------------------------------------------------------
# Syntactic Complexity
# ---------------------------------------------------------------------------

def compute_syntactic_features(text: str) -> dict:
    """
    Uses spaCy to extract POS-based ratios, sentence length stats,
    and parse tree depth.
    """
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]

    if not tokens:
        return {
            "adj_noun_ratio": 0.0,
            "verb_ratio": 0.0,
            "avg_sent_len": 0.0,
            "sent_len_std": 0.0,
            "avg_tree_depth": 0.0,
        }

    # POS counts
    pos_counts = Counter(t.pos_ for t in tokens)
    n_adj = pos_counts.get("ADJ", 0)
    n_noun = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
    n_verb = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)

    adj_noun_ratio = n_adj / n_noun if n_noun > 0 else 0.0
    verb_ratio = n_verb / len(tokens) if tokens else 0.0

    # Sentence length statistics
    sent_lengths = [len([t for t in s if not t.is_space]) for s in doc.sents]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    # Sentence length std dev is crucial — AI text has notably lower variance
    sent_len_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0

    # Parse tree depth
    def get_tree_depth(token):
        children = list(token.children)
        if children:
            return 1 + max(get_tree_depth(child) for child in children)
        return 1

    depths = []
    for sent in doc.sents:
        depths.append(get_tree_depth(sent.root))
    avg_tree_depth = float(np.mean(depths)) if depths else 0.0

    return {
        "adj_noun_ratio": adj_noun_ratio,
        "verb_ratio": verb_ratio,
        "avg_sent_len": avg_sent_len,
        "sent_len_std": sent_len_std,
        "avg_tree_depth": avg_tree_depth,
    }


# ---------------------------------------------------------------------------
# Punctuation Density
# ---------------------------------------------------------------------------

PUNCTUATION_CHARS = ['.', ',', ';', ':', '!', '?', '-', '—', '"', "'", '(', ')']


def compute_punctuation_density(text: str) -> dict:
    """
    Counts each punctuation character and normalizes by word count
    to get density per 100 words.
    """
    word_count = len(text.split())
    if word_count == 0:
        return {f"punct_{p}": 0.0 for p in PUNCTUATION_CHARS}

    densities = {}
    for p in PUNCTUATION_CHARS:
        count = text.count(p)
        densities[f"punct_{p}"] = (count / word_count) * 100
    return densities


# ---------------------------------------------------------------------------
# Readability Indices
# ---------------------------------------------------------------------------

def compute_readability(text: str) -> dict:
    """
    Uses the textstat library for three readability metrics.
    These capture how "difficult" the prose is at a surface level.
    Hypothesis: Victorian prose scores higher grade level than AI text.
    """
    return {
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
        "flesch_reading_ease": textstat.flesch_reading_ease(text),
        "gunning_fog": textstat.gunning_fog(text),
    }


# ---------------------------------------------------------------------------
# Master Feature Extractor
# ---------------------------------------------------------------------------

def extract_all_features(text: str) -> dict:
    """
    Extracts the complete feature vector for a single paragraph.
    Returns a flat dictionary of all numerical features.
    """
    features = {}

    # Lexical
    features["ttr"] = compute_ttr(text)
    features["mattr"] = compute_mattr(text, window=50)
    features["hapax_ratio"] = compute_hapax_ratio(text)

    # Syntactic
    features.update(compute_syntactic_features(text))

    # Punctuation
    features.update(compute_punctuation_density(text))

    # Readability
    features.update(compute_readability(text))

    return features