# ============================================================
# sample_data.py  (LENGKAP)
# ============================================================
import numpy as np


def generate_mlp_classification_data(n_samples=500, n_features=4, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


# Toy corpus untuk demo SequenceModel + BPETokenizer. Pola berulang sederhana
# supaya model kecil bisa belajar sesuatu dalam training singkat.
_TOY_SENTENCE = "the cat sat on the mat and the dog ran to the tree. "


def generate_toy_corpus_text(repeat=200):
    """Kembalikan teks mentah (string) — dipakai untuk melatih BPETokenizer
    dan sebagai sumber token_stream setelah di-encode."""
    return _TOY_SENTENCE * repeat