// ============================================================
// src/transformer_block.cpp  (LENGKAP)
// ============================================================
#include "transformer_block.h"

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_hidden_dim,
                                    bool causal_mask, unsigned seed)
    : attention_(embed_dim, num_heads, causal_mask, seed),
      norm1_(embed_dim),
      feed_forward_(embed_dim, ff_hidden_dim, seed + 10),
      norm2_(embed_dim) {}

Tensor3D TransformerBlock::forward(const Tensor3D& input) {
    Tensor3D x1 = norm1_.forward(input);
    Tensor3D attn_out = attention_.forward(x1);
    Tensor3D residual1 = input + attn_out;

    Tensor3D x2 = norm2_.forward(residual1);
    Tensor3D ff_out = feed_forward_.forward(x2);
    return residual1 + ff_out;
}

Tensor3D TransformerBlock::backward(const Tensor3D& grad_output) {
    Tensor3D grad_ff_out = grad_output;
    Tensor3D grad_residual1 = grad_output;

    Tensor3D grad_x2 = feed_forward_.backward(grad_ff_out);
    Tensor3D grad_residual1_from_norm2 = norm2_.backward(grad_x2);
    grad_residual1 = grad_residual1 + grad_residual1_from_norm2;

    Tensor3D grad_attn_out = grad_residual1;
    Tensor3D grad_input = grad_residual1;

    Tensor3D grad_x1 = attention_.backward(grad_attn_out);
    Tensor3D grad_input_from_norm1 = norm1_.backward(grad_x1);
    grad_input = grad_input + grad_input_from_norm1;

    return grad_input;
}

void TransformerBlock::update(Scalar learning_rate) {
    attention_.update(learning_rate);
    norm1_.update(learning_rate);
    feed_forward_.update(learning_rate);
    norm2_.update(learning_rate);
}

void TransformerBlock::save(std::ostream& os) const {
    attention_.save(os);
    norm1_.save(os);
    feed_forward_.save(os);
    norm2_.save(os);
}

void TransformerBlock::load_weights(std::istream& is) {
    attention_.load_weights(is);
    norm1_.load_weights(is);
    feed_forward_.load_weights(is);
    norm2_.load_weights(is);
}