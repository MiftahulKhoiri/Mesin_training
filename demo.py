# ============================================================
# demo.py  (DIPERBARUI — load checkpoint, tidak training ulang)
# ============================================================
import os
import sys
sys.path.append("build")

import numpy as np
import ml_manual_cpp as mlc
from sample_data import encode_text, decode_tokens, VOCAB_SIZE

MLP_CHECKPOINT = "mlp_checkpoint.bin"
SEQUENCE_CHECKPOINT = "sequence_checkpoint.bin"


def demo_mlp():
    print("=== Demo MLP ===")
    if not os.path.exists(MLP_CHECKPOINT):
        print(f"  Checkpoint {MLP_CHECKPOINT} tidak ditemukan — jalankan training.py dulu.")
        return

    model = mlc.NeuralNetwork.load_checkpoint(MLP_CHECKPOINT)

    from sample_data import generate_mlp_classification_data
    X, y = generate_mlp_classification_data(n_samples=5, seed=99)
    predictions = model.forward(X, False)  # training=False -> mode inferensi

    for i in range(5):
        print(f"  input={X[i]} -> predicted={predictions[i][0]:.3f}, actual={y[i][0]:.0f}")


def sample_next_token(logits_last_position, temperature=0.8):
    scaled = logits_last_position / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)


def generate(model, prompt_ids, max_new_tokens, max_seq_len, temperature=0.8):
    tokens = prompt_ids.copy()
    for _ in range(max_new_tokens):
        context = tokens[:, -max_seq_len:]
        logits = model.forward(context)
        next_id = sample_next_token(logits[0, -1, :], temperature)
        tokens = np.concatenate([tokens, np.array([[next_id]], dtype=np.float32)], axis=1)
    return tokens


def demo_sequence():
    print("\n=== Demo SequenceModel (text generation) ===")
    if not os.path.exists(SEQUENCE_CHECKPOINT):
        print(f"  Checkpoint {SEQUENCE_CHECKPOINT} tidak ditemukan — jalankan training.py dulu.")
        return

    model = mlc.SequenceModel.load_checkpoint(SEQUENCE_CHECKPOINT)

    prompt = encode_text("the cat")
    generated = generate(model, prompt, max_new_tokens=60, max_seq_len=32, temperature=0.7)

    print(f"  prompt   : {decode_tokens(prompt)!r}")
    print(f"  generated: {decode_tokens(generated)!r}")


if __name__ == "__main__":
    demo_mlp()
    demo_sequence()