// ============================================================
// include/feed_forward_block.h  (LENGKAP)
// ============================================================
#pragma once
#include <ostream>
#include <istream>
#include "matrix_ops.h"
#include "tensor3d.h"

class FeedForwardBlock {
public:
    FeedForwardBlock(size_t embed_dim, size_t hidden_dim, unsigned seed = 42);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

    void save(std::ostream& os) const;
    void load_weights(std::istream& is);

private:
    size_t embed_dim_, hidden_dim_;

    Matrix W1_, b1_;
    Matrix W2_, b2_;
    Matrix grad_W1_, grad_b1_, grad_W2_, grad_b2_;

    Tensor3D input_cache_;
    Tensor3D pre_activation_cache_;
    Tensor3D post_activation_cache_;

    static Scalar gelu_scalar(Scalar x);
    static Scalar gelu_derivative_scalar(Scalar x);
};