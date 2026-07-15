// ============================================================
// include/attention_layer.h  (FILE BARU)
// ============================================================
#pragma once
#include <vector>
#include "matrix_ops.h"
#include "tensor3d.h"
#include "activations.h"

class AttentionLayer {
public:
    // causal_mask=true untuk autoregressive/GPT-style (posisi hanya lihat ke belakang);
    // false untuk encoder-style (lihat seluruh sequence)
    AttentionLayer(size_t embed_dim, size_t num_heads, bool causal_mask, unsigned seed = 42);

    Tensor3D forward(const Tensor3D& input);   // batch x seq x embed_dim -> sama
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

    size_t embed_dim() const { return embed_dim_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

private:
    size_t embed_dim_, num_heads_, head_dim_;
    bool causal_mask_;

    Matrix W_q_, W_k_, W_v_, W_o_;             // masing-masing embed_dim x embed_dim
    Matrix grad_W_q_, grad_W_k_, grad_W_v_, grad_W_o_;

    // Cache untuk backward
    Tensor3D input_cache_;
    Tensor3D Q_cache_, K_cache_, V_cache_;      // sebelum split per-head
    Tensor3D concat_output_cache_;              // hasil gabungan semua head, sebelum W_o
    std::vector<std::vector<Matrix>> attn_weights_cache_; // [batch][head] -> seq x seq

    static Matrix init_projection_weight(size_t dim, unsigned seed);
    static Matrix extract_head_cols(const Matrix& full, size_t head_idx, size_t head_dim);
    static void write_head_cols(Matrix& full, size_t head_idx, size_t head_dim, const Matrix& head_data);
};