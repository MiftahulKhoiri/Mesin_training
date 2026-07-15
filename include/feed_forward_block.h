// ============================================================
// include/feed_forward_block.h  (FILE BARU)
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "tensor3d.h"

// Position-wise feed-forward: Linear(embed_dim -> hidden_dim) -> GELU -> Linear(hidden_dim -> embed_dim)
// Diterapkan identik & independen ke tiap posisi token (khas Transformer FFN).
class FeedForwardBlock {
public:
    FeedForwardBlock(size_t embed_dim, size_t hidden_dim, unsigned seed = 42);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

private:
    size_t embed_dim_, hidden_dim_;

    Matrix W1_, b1_; // embed_dim x hidden_dim, 1 x hidden_dim
    Matrix W2_, b2_; // hidden_dim x embed_dim, 1 x embed_dim
    Matrix grad_W1_, grad_b1_, grad_W2_, grad_b2_;

    Tensor3D input_cache_;
    Tensor3D pre_activation_cache_;  // z1, sebelum GELU
    Tensor3D post_activation_cache_; // a1 = GELU(z1)

    static Scalar gelu_scalar(Scalar x);
    static Scalar gelu_derivative_scalar(Scalar x);
};