// ============================================================
// include/transformer_block.h  (FILE BARU)
// ============================================================
#pragma once
#include "tensor3d.h"
#include "attention_layer.h"
#include "layer_norm_tensor.h"
#include "feed_forward_block.h"

// Pre-norm Transformer block, gaya GPT-2:
//   x1 = LayerNorm(input)
//   attn_out = Attention(x1)
//   residual1 = input + attn_out
//   x2 = LayerNorm(residual1)
//   ff_out = FeedForward(x2)
//   output = residual1 + ff_out
class TransformerBlock {
public:
    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_hidden_dim,
                      bool causal_mask, unsigned seed = 42);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

private:
    AttentionLayer attention_;
    LayerNormTensor norm1_;
    FeedForwardBlock feed_forward_;
    LayerNormTensor norm2_;
};