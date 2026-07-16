# ============================================================
# demo.py  (FILE BARU)
# ============================================================
import sys
sys.path.append("build")

import numpy as np
import ml_manual_cpp as mlc
from sample_data import (
    generate_mlp_classification_data,
    generate_toy_token_stream,
    encode_text,
    decode_tokens,
    VOCAB_SIZE,
)


def demo_mlp():
    """Latih MLP singkat, lalu tampilkan beberapa prediksi contoh."""
    print("=== Demo MLP ===")
    X, y = generate_mlp_classification_data(n_samples=300, seed=1)

    model = mlc.NeuralNetwork(mlc.LossType.BinaryCrossEntropy)
    model.add_dense_layer(4, 16, mlc.ActivationType.ReLU)
    model.add_dense_layer(16, 1, mlc.ActivationType.Sigmoid)

    config = mlc.TrainConfig()
    config.epochs = 15
    config.batch_size = 32
    config.learning_rate = 0.05
    config.verbose = False

    trainer = mlc.Trainer(model, config)
    trainer.fit(X, y)

    sample = X[:5]
    predictions = model.forward(sample, False)  # training=False -> mode inferensi
    for i in range(5):
        print(f"  input={sample[i]} -> predicted={predictions[i][0]:.3f}, actual={y[i][0]:.0f}")


def sample_next_token(logits_last_position, temperature=0.8):
    """logits_last_position: array 1D (vocab_size,). Sampling dengan temperature scaling."""
    scaled = logits_last_position / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled)  # stabilitas numerik
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)


def generate(model, prompt_ids, max_new_tokens, max_seq_len, temperature=0.8):
    """Generasi autoregresif: prediksi token berikutnya berulang kali, append ke sequence.

    prompt_ids: numpy array (1, N) float32 — hasil encode_text().
    """
    tokens = prompt_ids.copy()

    for _ in range(max_new_tokens):
        # Ambil context terakhir (maksimal max_seq_len token) — model tidak bisa
        # melihat lebih jauh dari itu (dibatasi positional encoding saat training)
        context = tokens[:, -max_seq_len:]
        logits = model.forward(context)          # (1, seq_len, vocab_size)
        last_logits = logits[0, -1, :]            # logits posisi terakhir saja
        next_id = sample_next_token(last_logits, temperature)
        tokens = np.concatenate([tokens, np.array([[next_id]], dtype=np.float32)], axis=1)

    return tokens


def demo_sequence():
    """Latih SequenceModel singkat pada toy corpus, lalu generate teks baru."""
    print("\n=== Demo SequenceModel (text generation) ===")
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
    config.verbose = True
    config.log_every_n_steps = 20

    trainer = mlc.SequenceTrainer(model, config)
    trainer.fit(token_stream)

    prompt = encode_text("the cat")
    generated = generate(model, prompt, max_new_tokens=60, max_seq_len=32, temperature=0.7)

    print("\nHasil generate:")
    print(f"  prompt   : {decode_tokens(prompt)!r}")
    print(f"  generated: {decode_tokens(generated)!r}")


if __name__ == "__main__":
    demo_mlp()
    demo_sequence()