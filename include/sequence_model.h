// ============================================================
// include/sequence_model.h  (FILE BARU)
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
    Tensor3D forward(const Matrix& token_ids);

    // target_ids: batch x seq_len, biasanya token_ids digeser satu posisi
    // (next-token prediction). Loss = rata-rata cross-entropy semua token.
    Scalar compute_loss(const Tensor3D& logits, const Matrix& target_ids) const;

    // Backward penuh dari loss sampai ke embedding table.
    void backward(const Tensor3D& logits, const Matrix& target_ids, const Matrix& token_ids);

    void update(Scalar learning_rate);

private:
    size_t vocab_size_, embed_dim_;

    EmbeddingLayer embedding_;
    std::vector<TransformerBlock> blocks_;
    LayerNormTensor final_norm_;

    Matrix output_projection_;       // embed_dim x vocab_size
    Matrix grad_output_projection_;

    Tensor3D last_hidden_cache_;     // hasil final_norm_, sebelum output_projection_
};