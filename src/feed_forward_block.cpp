// ============================================================
// src/feed_forward_block.cpp  (LENGKAP)
// ============================================================
#include "feed_forward_block.h"
#include <cmath>

Scalar FeedForwardBlock::gelu_scalar(Scalar x) {
    const Scalar c = 0.7978845608f;
    Scalar x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x3)));
}

Scalar FeedForwardBlock::gelu_derivative_scalar(Scalar x) {
    const Scalar c = 0.7978845608f;
    Scalar x3 = x * x * x;
    Scalar inner = c * (x + 0.044715f * x3);
    Scalar t = std::tanh(inner);
    Scalar dinner = c * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * dinner;
}

FeedForwardBlock::FeedForwardBlock(size_t embed_dim, size_t hidden_dim, unsigned seed)
    : embed_dim_(embed_dim), hidden_dim_(hidden_dim),
      W1_(embed_dim, hidden_dim), b1_(1, hidden_dim, 0.0f),
      W2_(hidden_dim, embed_dim), b2_(1, embed_dim, 0.0f),
      grad_W1_(embed_dim, hidden_dim, 0.0f), grad_b1_(1, hidden_dim, 0.0f),
      grad_W2_(hidden_dim, embed_dim, 0.0f), grad_b2_(1, embed_dim, 0.0f)
{
    Scalar std1 = std::sqrt(2.0f / static_cast<Scalar>(embed_dim));
    Scalar std2 = std::sqrt(2.0f / static_cast<Scalar>(hidden_dim));
    W1_ = Matrix::random_normal(embed_dim, hidden_dim, 0.0f, std1, seed);
    W2_ = Matrix::random_normal(hidden_dim, embed_dim, 0.0f, std2, seed + 1);
}

Tensor3D FeedForwardBlock::forward(const Tensor3D& input) {
    input_cache_ = input;

    Tensor3D z1 = input.batched_matmul(W1_).add_bias(b1_);
    pre_activation_cache_ = z1;

    Tensor3D a1(z1.batch(), z1.seq_len(), z1.features());
    size_t n = z1.batch() * z1.seq_len() * z1.features();
    const Scalar* zp = z1.data();
    Scalar* ap = a1.data();
    for (size_t i = 0; i < n; ++i) ap[i] = gelu_scalar(zp[i]);
    post_activation_cache_ = a1;

    return a1.batched_matmul(W2_).add_bias(b2_);
}

Tensor3D FeedForwardBlock::backward(const Tensor3D& grad_output) {
    size_t batch = grad_output.batch();
    Scalar batch_scalar = static_cast<Scalar>(batch);

    grad_W2_ = Matrix(hidden_dim_, embed_dim_, 0.0f);
    grad_b2_ = Matrix(1, embed_dim_, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        Matrix a1_b = post_activation_cache_.slice_batch(b);
        Matrix go_b = grad_output.slice_batch(b);
        grad_W2_.add_inplace(a1_b.transpose() * go_b);
        grad_b2_.add_inplace(go_b.sum_rows());
    }
    grad_W2_ = grad_W2_.scale(1.0f / batch_scalar);
    grad_b2_ = grad_b2_.scale(1.0f / batch_scalar);

    Matrix W2_t = W2_.transpose();
    Tensor3D grad_a1 = grad_output.batched_matmul(W2_t);

    Tensor3D grad_z1(grad_a1.batch(), grad_a1.seq_len(), grad_a1.features());
    size_t n = grad_a1.batch() * grad_a1.seq_len() * grad_a1.features();
    const Scalar* ga1p = grad_a1.data();
    const Scalar* z1p = pre_activation_cache_.data();
    Scalar* gz1p = grad_z1.data();
    for (size_t i = 0; i < n; ++i) gz1p[i] = ga1p[i] * gelu_derivative_scalar(z1p[i]);

    grad_W1_ = Matrix(embed_dim_, hidden_dim_, 0.0f);
    grad_b1_ = Matrix(1, hidden_dim_, 0.0f);
    for (size_t b = 0; b < batch; ++b) {
        Matrix input_b = input_cache_.slice_batch(b);
        Matrix gz1_b = grad_z1.slice_batch(b);
        grad_W1_.add_inplace(input_b.transpose() * gz1_b);
        grad_b1_.add_inplace(gz1_b.sum_rows());
    }
    grad_W1_ = grad_W1_.scale(1.0f / batch_scalar);
    grad_b1_ = grad_b1_.scale(1.0f / batch_scalar);

    Matrix W1_t = W1_.transpose();
    return grad_z1.batched_matmul(W1_t);
}

void FeedForwardBlock::update(Scalar learning_rate) {
    W1_.sub_inplace(grad_W1_.scale(learning_rate));
    b1_.sub_inplace(grad_b1_.scale(learning_rate));
    W2_.sub_inplace(grad_W2_.scale(learning_rate));
    b2_.sub_inplace(grad_b2_.scale(learning_rate));
}

void FeedForwardBlock::save(std::ostream& os) const {
    W1_.save(os);
    b1_.save(os);
    W2_.save(os);
    b2_.save(os);
}

void FeedForwardBlock::load_weights(std::istream& is) {
    W1_ = Matrix::load(is);
    b1_ = Matrix::load(is);
    W2_ = Matrix::load(is);
    b2_ = Matrix::load(is);
}