// ============================================================
// include/layer_norm_tensor.h  (FILE BARU)
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "tensor3d.h"

// Beda dari BatchNormLayer: normalisasi di sini dilakukan PER TOKEN
// (lintas dimensi embed_dim untuk tiap (batch, posisi)), bukan lintas
// batch. Ini yang dipakai standar di arsitektur Transformer.
class LayerNormTensor {
public:
    explicit LayerNormTensor(size_t embed_dim, Scalar epsilon = 1e-5f);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

    size_t embed_dim() const { return embed_dim_; }

private:
    size_t embed_dim_;
    Scalar epsilon_;

    Matrix gamma_, beta_;           // 1 x embed_dim (learnable, sama untuk semua token)
    Matrix grad_gamma_, grad_beta_;

    Tensor3D input_cache_;
    Tensor3D normalized_cache_;     // x_hat
    Matrix mean_cache_;             // batch x seq_len
    Matrix var_cache_;              // batch x seq_len
};