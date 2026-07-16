// ============================================================
// include/sequence_model.h  (DIPERBARUI — weight tying + backward gabungan)
// ============================================================
#pragma once
#include <vector>
#include "matrix_ops.h"
#include "tensor3d.h"
#include "embedding_layer.h"
#include "transformer_block.h"
#include "layer_norm_tensor.h"

class SequenceModel {
public:
    SequenceModel(size_t vocab_size, size_t embed_dim, size_t num_heads,
                  size_t ff_hidden_dim, size_t num_layers, size_t max_seq_len,
                  bool causal_mask = true, unsigned seed = 42);

    // token_ids: batch x seq_len -> logits: batch x seq_len x vocab_size
    // Weight tying: proyeksi ke vocab pakai embedding_table_ langsung (tanpa matriks terpisah)
    Tensor3D forward(const Matrix& token_ids);

    // Hanya untuk evaluasi/validasi TANPA backward (mis. saat tidak butuh update parameter).
    // Kalau backward() akan dipanggil juga, jangan panggil ini dulu — pakai loss yang
    // dikembalikan backward() supaya tidak menghitung softmax dua kali.
    Scalar compute_loss(const Tensor3D& logits, const Matrix& target_ids) const;

    // Backward penuh + mengembalikan loss yang dihitung dalam proses yang sama
    // (satu pass softmax per token, bukan dua kali seperti compute_loss+backward terpisah).
    Scalar backward(const Tensor3D& logits, const Matrix& target_ids, const Matrix& token_ids);

    void update(Scalar learning_rate);

private:
    size_t vocab_size_, embed_dim_;

    EmbeddingLayer embedding_;
    std::vector<TransformerBlock> blocks_;
    LayerNormTensor final_norm_;

    Tensor3D last_hidden_cache_; // hasil final_norm_, sebelum proyeksi ke vocab
};