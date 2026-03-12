"""
Explainability utilities: SHAP wrappers, Integrated Gradients helpers, error analysis.
"""

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients


def get_misclassified(y_true, y_pred, texts, class_names):
    """
    Returns a DataFrame of misclassified samples with their true/predicted labels and text.
    """
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
    """
    Computes error breakdown: which class pairs are most confused, with examples.
    """
    errors = get_misclassified(y_true, y_pred, texts, class_names)
    if len(errors) == 0:
        return errors, "No misclassifications found."

    # Confusion pair counts
    pair_counts = errors.groupby(["true_class", "pred_class"]).size().sort_values(ascending=False)

    summary_lines = [f"Total errors: {len(errors)} / {len(y_true)} ({len(errors)/len(y_true):.1%})\n"]
    summary_lines.append("Most confused pairs:")
    for (tc, pc), count in pair_counts.items():
        summary_lines.append(f"  {tc} -> {pc}: {count}")

    return errors, "\n".join(summary_lines)


def compute_integrated_gradients_distilbert(model, tokenizer, text, target_class, device="cpu"):
    """
    Computes Integrated Gradients attributions for a DistilBERT model on a single text.
    Returns token strings and their attribution scores.
    """
    model.eval()
    model.to(device)

    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get embedding layer
    def forward_func(inputs_embeds):
        outputs = model.distilbert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = model.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    # Get embeddings for the input
    embeddings = model.distilbert.embeddings(input_ids)
    embeddings.requires_grad_(True)

    # Baseline: zero embedding
    baseline = torch.zeros_like(embeddings)

    ig = IntegratedGradients(forward_func)
    attributions = ig.attribute(
        embeddings,
        baselines=baseline,
        target=target_class,
        n_steps=50,
    )

    # Sum attributions across embedding dimensions to get per-token score
    token_attrs = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

    # Filter out padding
    mask = attention_mask.squeeze(0).cpu().numpy().astype(bool)
    tokens = [t for t, m in zip(tokens, mask) if m]
    token_attrs = token_attrs[mask]

    return tokens, token_attrs


def find_ai_isms(texts, class_labels, feature_df, class_names):
    """
    Hunts for recurring AI patterns: repetitive phrases, low punctuation variance,
    suspiciously uniform sentence lengths.
    """
    import re
    from collections import Counter

    ai_mask = class_labels > 0
    human_mask = class_labels == 0

    findings = []

    # 1. Check for common AI filler phrases
    ai_fillers = [
        "it is important to note",
        "in conclusion",
        "furthermore",
        "moreover",
        "it is worth noting",
        "in today's world",
        "plays a crucial role",
        "it is essential",
        "in this regard",
        "on the other hand",
    ]

    ai_texts = [t.lower() for t in texts[ai_mask]]
    human_texts = [t.lower() for t in texts[human_mask]]

    findings.append("=== AI Filler Phrase Frequency ===")
    for phrase in ai_fillers:
        ai_count = sum(1 for t in ai_texts if phrase in t)
        human_count = sum(1 for t in human_texts if phrase in t)
        if ai_count > 0 or human_count > 0:
            findings.append(f"  '{phrase}': AI={ai_count}, Human={human_count}")

    # 2. Sentence length std deviation comparison
    if "sent_len_std" in feature_df.columns:
        ai_std = feature_df.loc[ai_mask, "sent_len_std"].mean()
        human_std = feature_df.loc[human_mask, "sent_len_std"].mean()
        findings.append(f"\n=== Sentence Length Variability ===")
        findings.append(f"  Human avg sent_len_std: {human_std:.2f}")
        findings.append(f"  AI avg sent_len_std: {ai_std:.2f}")
        findings.append(f"  Ratio (Human/AI): {human_std/ai_std:.2f}" if ai_std > 0 else "  AI std is 0")

    # 3. Unique opening words (first word of each text)
    ai_openers = Counter(t.split()[0] if t.split() else "" for t in ai_texts)
    human_openers = Counter(t.split()[0] if t.split() else "" for t in human_texts)
    findings.append(f"\n=== Most Common Opening Words ===")
    findings.append(f"  AI top 5: {ai_openers.most_common(5)}")
    findings.append(f"  Human top 5: {human_openers.most_common(5)}")

    return "\n".join(findings)
