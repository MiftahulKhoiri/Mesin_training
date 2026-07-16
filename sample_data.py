# ============================================================
# sample_data.py  (FILE BARU)
# ============================================================
"""Data sintetis reusable untuk training.py dan demo.py."""

import numpy as np


def generate_mlp_classification_data(n_samples=500, n_features=4, seed=42):
    """Data klasifikasi biner sintetis: label=1 jika jumlah fitur positif."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


# --- Toy character-level corpus untuk demo SequenceModel ---
# Sengaja pakai pola berulang sederhana (bukan random murni) supaya model
# punya sesuatu yang bisa benar-benar "dipelajari" dalam training singkat,
# dan hasil generate bisa dibandingkan terhadap pola aslinya.
_TOY_SENTENCE = "the cat sat on the mat and the dog ran to the tree. "

CHAR_VOCAB = sorted(set(_TOY_SENTENCE))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)


def encode_text(text):
    """String -> numpy array token id (float32, 1 x N), untuk konsumsi Matrix di C++."""
    ids = [CHAR_TO_ID[ch] for ch in text if ch in CHAR_TO_ID]
    return np.array(ids, dtype=np.float32).reshape(1, -1)


def decode_tokens(ids):
    """Array/list token id -> string."""
    return "".join(ID_TO_CHAR[int(round(i))] for i in np.ravel(ids))


def generate_toy_token_stream(repeat=200):
    """Corpus panjang untuk training SequenceModel: pola kalimat sederhana diulang."""
    text = _TOY_SENTENCE * repeat
    return encode_text(text)