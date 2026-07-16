# ============================================================
# sample_data.py  (LENGKAP)
# ============================================================
import numpy as np


def generate_mlp_classification_data(n_samples=500, n_features=4, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


_TOY_SENTENCE = "the cat sat on the mat and the dog ran to the tree. "

CHAR_VOCAB = sorted(set(_TOY_SENTENCE))
CHAR_TO_ID = {ch: i for i, ch in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: ch for i, ch in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)


def encode_text(text):
    ids = [CHAR_TO_ID[ch] for ch in text if ch in CHAR_TO_ID]
    return np.array(ids, dtype=np.float32).reshape(1, -1)


def decode_tokens(ids):
    return "".join(ID_TO_CHAR[int(round(i))] for i in np.ravel(ids))


def generate_toy_token_stream(repeat=200):
    text = _TOY_SENTENCE * repeat
    return encode_text(text)