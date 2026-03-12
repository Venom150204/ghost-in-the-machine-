"""
Classification models for Tier A (XGBoost/RF), Tier B (FF-NN), and Tier C (DistilBERT+LoRA).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, class_names=None):
    """Returns a dict of standard classification metrics."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    if class_names:
        results["report"] = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )
    return results


def print_metrics(results, title=""):
    """Pretty-print classification metrics."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision_macro']:.4f} (macro)")
    print(f"  Recall:    {results['recall_macro']:.4f} (macro)")
    print(f"  F1:        {results['f1_macro']:.4f} (macro)")
    print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    if "report" in results:
        print(f"\n{results['report']}")


# ---------------------------------------------------------------------------
# Tier A — XGBoost + Random Forest on handcrafted features
# ---------------------------------------------------------------------------

def train_tier_a(X_train, y_train, X_val, y_val, seed=42):
    """
    Trains XGBoost (with grid search) and Random Forest on the feature matrix.
    Returns both fitted models and their validation predictions.
    """
    # XGBoost with hyperparameter tuning
    xgb_param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
    }
    xgb_base = XGBClassifier(
        random_state=seed,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )
    xgb_cv = GridSearchCV(
        xgb_base, xgb_param_grid,
        cv=5, scoring="f1_macro", n_jobs=-1, verbose=1,
    )
    xgb_cv.fit(X_train, y_train)
    xgb_best = xgb_cv.best_estimator_
    xgb_val_pred = xgb_best.predict(X_val)

    print(f"XGBoost best params: {xgb_cv.best_params_}")
    print(f"XGBoost best CV F1: {xgb_cv.best_score_:.4f}")

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_val_pred = rf.predict(X_val)

    return {
        "xgboost": {"model": xgb_best, "val_pred": xgb_val_pred},
        "random_forest": {"model": rf, "val_pred": rf_val_pred},
    }


# ---------------------------------------------------------------------------
# Tier B — Feedforward Neural Network
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """Simple dataset for embedding vectors + labels."""
    def __init__(self, embeddings, labels):
        self.X = torch.FloatTensor(embeddings)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeedForwardNN(nn.Module):
    """
    Simple 3-layer feedforward network for classification.
    Architecture chosen to be expressive enough for ~400-dim input
    without overfitting on ~1500 samples (dropout=0.3 helps).
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_ffnn(
    X_train, y_train, X_val, y_val,
    input_dim, num_classes,
    lr=1e-3, batch_size=32, epochs=20, patience=5, seed=42,
):
    """
    Trains a feedforward NN with early stopping on validation loss.
    Returns the trained model and validation predictions.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = EmbeddingDataset(X_train, y_train)
    val_ds = EmbeddingDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FeedForwardNN(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
        val_loss /= len(val_ds)

        print(f"  Epoch {epoch+1:2d}/{epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and predict
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return model, np.array(all_preds)


# ---------------------------------------------------------------------------
# Embedding helpers for Tier B
# ---------------------------------------------------------------------------

def compute_glove_embeddings(texts, glove_path, dim=100):
    """
    Loads GloVe vectors and computes averaged word embeddings per text.
    This loses word order — a known limitation documented in the notebook.
    """
    # Load GloVe
    print(f"Loading GloVe vectors from {glove_path}...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if len(vec) == dim:
                glove[word] = vec
    print(f"  Loaded {len(glove)} word vectors")

    embeddings = []
    for text in texts:
        words = text.lower().split()
        vecs = [glove[w] for w in words if w in glove]
        if vecs:
            embeddings.append(np.mean(vecs, axis=0))
        else:
            embeddings.append(np.zeros(dim))
    return np.array(embeddings)


def compute_sbert_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Uses sentence-transformers to encode paragraphs into 384-dim vectors.
    Preserves semantic meaning at the sentence level — unlike GloVe averaging.
    """
    from sentence_transformers import SentenceTransformer
    print(f"Encoding with SBERT ({model_name})...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results_dict):
    """
    Builds a summary DataFrame from a dict of {model_name: {binary: metrics, three_class: metrics}}.
    """
    rows = []
    for name, res in results_dict.items():
        row = {"Model": name}
        if "binary" in res:
            row["Binary Acc"] = f"{res['binary']['accuracy']:.4f}"
            row["Binary F1"] = f"{res['binary']['f1_macro']:.4f}"
        if "three_class" in res:
            row["3-Class Acc"] = f"{res['three_class']['accuracy']:.4f}"
            row["3-Class F1"] = f"{res['three_class']['f1_macro']:.4f}"
            # Class 0 vs Class 2 confusion: how many Class 0 predicted as Class 2
            cm = res["three_class"]["confusion_matrix"]
            if cm.shape[0] >= 3:
                row["C0-C2 Confusion"] = int(cm[0, 2])
            else:
                row["C0-C2 Confusion"] = "N/A"
        rows.append(row)
    return pd.DataFrame(rows)