// ============================================================
// include/attention_layer.h  (LENGKAP)
// ============================================================
#pragma once
#include <vector>
#include <ostream>
#include <istream>
#include "matrix_ops.h"
#include "tensor3d.h"
#include "activations.h"

class AttentionLayer {
public:
    AttentionLayer(size_t embed_dim, size_t num_heads, bool causal_mask, unsigned seed = 42);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

    size_t embed_dim() const { return embed_dim_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

    void save(std::ostream& os) const;
    void load_weights(std::istream& is);

private:
    size_t embed_dim_, num_heads_, head_dim_;
    bool causal_mask_;

    Matrix W_q_, W_k_, W_v_, W_o_;
    Matrix grad_W_q_, grad_W_k_, grad_W_v_, grad_W_o_;

    // attn_weights TIDAK di-cache — direkomputasi saat backward (hemat memori O(batch*heads*seq^2))
    Tensor3D input_cache_;
    Tensor3D Q_cache_, K_cache_, V_cache_;
    Tensor3D concat_output_cache_;

    static Matrix init_projection_weight(size_t dim, unsigned seed);
    static Matrix extract_head_cols(const Matrix& full, size_t head_idx, size_t head_dim);
    static void write_head_cols(Matrix& full, size_t head_idx, size_t head_dim, const Matrix& head_data);
    Matrix compute_attn_weights(const Matrix& Qh, const Matrix& Kh, size_t seq_len) const;
};