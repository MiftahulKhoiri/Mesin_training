// ============================================================
// include/sequence_model.h  (LENGKAP)
// ============================================================
#pragma once
#include <vector>
#include <string>
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

    // Weight tying: proyeksi ke vocab pakai embedding_table_ langsung, tanpa matriks terpisah
    Tensor3D forward(const Matrix& token_ids);

    // Hanya untuk evaluasi/validasi TANPA backward. Kalau backward() akan dipanggil juga,
    // pakai loss yang dikembalikan backward() supaya tidak double softmax.
    Scalar compute_loss(const Tensor3D& logits, const Matrix& target_ids) const;

    // Backward penuh + loss dari satu pass softmax yang sama
    Scalar backward(const Tensor3D& logits, const Matrix& target_ids, const Matrix& token_ids);

    void update(Scalar learning_rate);

    void save_checkpoint(const std::string& path) const;
    static SequenceModel load_checkpoint(const std::string& path);

private:
    size_t vocab_size_, embed_dim_, num_heads_, ff_hidden_dim_, max_seq_len_;
    bool causal_mask_;

    EmbeddingLayer embedding_;
    std::vector<TransformerBlock> blocks_;
    LayerNormTensor final_norm_;

    Tensor3D last_hidden_cache_;
};