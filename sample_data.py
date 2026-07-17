# ============================================================
# data/sample_data.py  (LENGKAP)
# ============================================================
import os
import numpy as np

# Path relatif ke lokasi file ini sendiri (bukan ke cwd saat dijalankan) —
# jadi hasil generate selalu masuk ke folder data/ ini, apapun direktori
# tempat training.py/demo.py dijalankan.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_mlp_classification_data(n_samples=500, n_features=4, seed=42, save=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)

    if save:
        path = os.path.join(DATA_DIR, "mlp_data.npz")
        np.savez(path, X=X, y=y)
        print(f"  data MLP disimpan: {path}")

    return X, y


def load_mlp_classification_data():
    """Muat data MLP yang sudah pernah disimpan (hasil generate_mlp_classification_data)."""
    path = os.path.join(DATA_DIR, "mlp_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} tidak ditemukan — panggil generate_mlp_classification_data() dulu")
    npz = np.load(path)
    return npz["X"], npz["y"]


# Toy corpus untuk demo SequenceModel + BPETokenizer. Pola berulang sederhana
# supaya model kecil bisa belajar sesuatu dalam training singkat.
_TOY_SENTENCE = "the cat sat on the mat and the dog ran to the tree. "


def generate_toy_corpus_text(repeat=200, save=True):
    """Kembalikan teks mentah (string) — dipakai untuk melatih BPETokenizer
    dan sebagai sumber token_stream setelah di-encode."""
    text = _TOY_SENTENCE * repeat

    if save:
        path = os.path.join(DATA_DIR, "corpus.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  korpus disimpan: {path}")

    return text


def load_toy_corpus_text():
    """Muat korpus yang sudah pernah disimpan (hasil generate_toy_corpus_text)."""
    path = os.path.join(DATA_DIR, "corpus.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} tidak ditemukan — panggil generate_toy_corpus_text() dulu")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    print("Generate & simpan data contoh ke folder data/ ...")
    generate_mlp_classification_data()
    generate_toy_corpus_text()