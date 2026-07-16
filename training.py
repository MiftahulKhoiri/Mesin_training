# ============================================================
# training.py  (LENGKAP)
# ============================================================
import sys
sys.path.append("build")

import numpy as np
import ml_manual_cpp as mlc
from sample_data import generate_mlp_classification_data, generate_toy_token_stream, VOCAB_SIZE


def train_mlp_example():
    X, y = generate_mlp_classification_data(n_samples=500, seed=42)

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

    model.save_checkpoint("mlp_checkpoint.bin")
    print("  checkpoint disimpan: mlp_checkpoint.bin")


def train_sequence_example():
    token_stream = generate_toy_token_stream(repeat=200)

    model = mlc.SequenceModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=32,
        num_heads=4,
        ff_hidden_dim=64,
        num_layers=2,
        max_seq_len=32,
        causal_mask=True,
    )

    config = mlc.SequenceTrainConfig()
    config.epochs = 5
    config.batch_size = 8
    config.seq_len = 32
    config.learning_rate = 3e-4
    config.validation_split = 0.1
    config.log_every_n_steps = 20

    trainer = mlc.SequenceTrainer(model, config)
    history = trainer.fit(token_stream)

    for log in history:
        val_str = f", val_loss={log.val_loss:.4f}" if log.val_loss >= 0 else ""
        print(f"[Sequence] epoch {log.epoch + 1}: train_loss={log.train_loss:.4f}{val_str}")

    model.save_checkpoint("sequence_checkpoint.bin")
    print("  checkpoint disimpan: sequence_checkpoint.bin")


if __name__ == "__main__":
    print("=== Training MLP ===")
    train_mlp_example()

    print("\n=== Training SequenceModel ===")
    train_sequence_example()