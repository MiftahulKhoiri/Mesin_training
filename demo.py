# ============================================================
# demo.py  (LENGKAP)
# ============================================================
import os
import sys
sys.path.append("build")

import numpy as np
import ml_manual_cpp as mlc
from sample_data import generate_mlp_classification_data

MLP_CHECKPOINT = "mlp_checkpoint.bin"
SEQUENCE_CHECKPOINT = "sequence_checkpoint.bin"
TOKENIZER_CHECKPOINT = "tokenizer_checkpoint.bin"


def demo_mlp():
    print("=== Demo MLP ===")
    if not os.path.exists(MLP_CHECKPOINT):
        print(f"  Checkpoint {MLP_CHECKPOINT} tidak ditemukan — jalankan training.py dulu.")
        return

    model = mlc.NeuralNetwork.load_checkpoint(MLP_CHECKPOINT)
    X, y = generate_mlp_classification_data(n_samples=5, seed=99)
    predictions = model.forward(X, False)

    for i in range(5):
        print(f"  input={X[i]} -> predicted={predictions[i][0]:.3f}, actual={y[i][0]:.0f}")


def sample_next_token(logits_last_position, temperature=0.8):
    scaled = logits_last_position / max(temperature, 1e-6)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)


def generate(model, prompt_ids, max_new_tokens, max_seq_len, temperature=0.8):
    """prompt_ids: numpy array (1, N) float32."""
    tokens = prompt_ids.copy()
    for _ in range(max_new_tokens):
        context = tokens[:, -max_seq_len:]
        logits = model.forward(context)
        next_id = sample_next_token(logits[0, -1, :], temperature)
        tokens = np.concatenate([tokens, np.array([[next_id]], dtype=np.float32)], axis=1)
    return tokens


def demo_sequence():
    print("\n=== Demo SequenceModel (text generation) ===")
    if not os.path.exists(SEQUENCE_CHECKPOINT) or not os.path.exists(TOKENIZER_CHECKPOINT):
        print("  Checkpoint model/tokenizer tidak ditemukan — jalankan training.py dulu.")
        return

    tokenizer = mlc.BPETokenizer.load_checkpoint(TOKENIZER_CHECKPOINT)
    model = mlc.SequenceModel.load_checkpoint(SEQUENCE_CHECKPOINT)

    prompt_text = "the cat"
    prompt_ids = tokenizer.encode(prompt_text)  # list[int]
    prompt_array = np.array(prompt_ids, dtype=np.float32).reshape(1, -1)

    generated_array = generate(model, prompt_array, max_new_tokens=60, max_seq_len=32, temperature=0.7)
    generated_ids = [int(round(x)) for x in generated_array[0]]

    print(f"  prompt   : {prompt_text!r}")
    print(f"  generated: {tokenizer.decode(generated_ids)!r}")


if __name__ == "__main__":
    demo_mlp()
    demo_sequence()