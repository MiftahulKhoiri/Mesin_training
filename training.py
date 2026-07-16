# ============================================================
# training.py  (LENGKAP)
# ============================================================
import sys
sys.path.append("build")  # sesuaikan kalau struktur build beda

import numpy as np
import ml_manual_cpp as mlc


def train_mlp_example():
    """Contoh training MLP untuk klasifikasi biner sederhana (data sintetis)."""
    rng = np.random.default_rng(42)
    n_samples = 500
    X = rng.normal(size=(n_samples, 4)).astype(np.float32)
    # label: 1 kalau jumlah fitur positif, 0 kalau tidak (target biner)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)

    model = mlc.NeuralNetwork(mlc.LossType.BinaryCrossEntropy)
    model.add_dense_layer(4, 16, mlc.ActivationType.ReLU)
    model.add_batch_norm_layer()
    model.add_dense_layer(16, 1, mlc.ActivationType.Sigmoid)

    config = mlc.TrainConfig()
    config.epochs = 20
    config.batch_size = 32
    config.learning_rate = 0.05
    config.validation_split = 0.2

    trainer = mlc.Trainer(model, config)
    history = trainer.fit(X, y)

    for log in history:
        val_str = f", val_loss={log.val_loss:.4f}" if log.val_loss >= 0 else ""
        print(f"[MLP] epoch {log.epoch + 1}: train_loss={log.train_loss:.4f}{val_str}")


def train_sequence_example():
    """Contoh training SequenceModel (mini transformer) untuk next-token prediction
    pada token stream sintetis (ganti dengan hasil tokenizer BPE asli untuk kasus nyata)."""
    rng = np.random.default_rng(7)
    vocab_size = 50
    token_stream = rng.integers(0, vocab_size, size=(1, 2000)).astype(np.float32)

    model = mlc.SequenceModel(
        vocab_size=vocab_size,
        embed_dim=32,
        num_heads=4,
        ff_hidden_dim=64,
        num_layers=2,
        max_seq_len=64,
        causal_mask=True,
    )

    config = mlc.SequenceTrainConfig()
    config.epochs = 3
    config.batch_size = 4
    config.seq_len = 32
    config.learning_rate = 3e-4
    config.validation_split = 0.1
    config.log_every_n_steps = 5

    trainer = mlc.SequenceTrainer(model, config)
    history = trainer.fit(token_stream)

    for log in history:
        val_str = f", val_loss={log.val_loss:.4f}" if log.val_loss >= 0 else ""
        print(f"[Sequence] epoch {log.epoch + 1}: train_loss={log.train_loss:.4f}{val_str}")


if __name__ == "__main__":
    print("=== Training MLP ===")
    train_mlp_example()

    print("\n=== Training SequenceModel ===")
    train_sequence_example()