# ============================================================
# training.py  (LENGKAP — checkpoint sekarang ke folder training/)
# ============================================================
import os
import sys
sys.path.append("build")
sys.path.append("data")

import numpy as np
import ml_manual_cpp as mlc
from sample_data import generate_mlp_classification_data, generate_toy_corpus_text

TRAINING_DIR = "training"


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

    checkpoint_path = os.path.join(TRAINING_DIR, "mlp_checkpoint.bin")
    model.save_checkpoint(checkpoint_path)
    print(f"  checkpoint disimpan: {checkpoint_path}")


def train_sequence_example():
    corpus_text = generate_toy_corpus_text(repeat=200)

    tokenizer = mlc.BPETokenizer()
    tokenizer.train(corpus_text, target_vocab_size=300, min_frequency=2)
    tokenizer_path = os.path.join(TRAINING_DIR, "tokenizer_checkpoint.bin")
    tokenizer.save_checkpoint(tokenizer_path)
    print(f"  tokenizer dilatih: vocab_size={tokenizer.vocab_size()}, checkpoint disimpan: {tokenizer_path}")

    token_ids = tokenizer.encode(corpus_text)
    token_stream = np.array(token_ids, dtype=np.float32).reshape(1, -1)

    model = mlc.SequenceModel(
        vocab_size=tokenizer.vocab_size(),
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

    checkpoint_path = os.path.join(TRAINING_DIR, "sequence_checkpoint.bin")
    model.save_checkpoint(checkpoint_path)
    print(f"  checkpoint disimpan: {checkpoint_path}")


if __name__ == "__main__":
    os.makedirs(TRAINING_DIR, exist_ok=True)

    print("=== Training MLP ===")
    train_mlp_example()

    print("\n=== Training SequenceModel ===")
    train_sequence_example()