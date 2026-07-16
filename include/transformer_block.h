// ============================================================
// include/transformer_block.h  (LENGKAP)
// ============================================================
#pragma once
#include <ostream>
#include <istream>
#include "tensor3d.h"
#include "attention_layer.h"
#include "layer_norm_tensor.h"
#include "feed_forward_block.h"

class TransformerBlock {
public:
    TransformerBlock(size_t embed_dim, size_t num_heads, size_t ff_hidden_dim,
                      bool causal_mask, unsigned seed = 42);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

    void save(std::ostream& os) const;
    void load_weights(std::istream& is);

private:
    AttentionLayer attention_;
    LayerNormTensor norm1_;
    FeedForwardBlock feed_forward_;
    LayerNormTensor norm2_;
};