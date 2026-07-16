// ============================================================
// include/layer_norm_tensor.h  (LENGKAP)
// ============================================================
#pragma once
#include <ostream>
#include <istream>
#include "matrix_ops.h"
#include "tensor3d.h"

class LayerNormTensor {
public:
    explicit LayerNormTensor(size_t embed_dim, Scalar epsilon = 1e-5f);

    Tensor3D forward(const Tensor3D& input);
    Tensor3D backward(const Tensor3D& grad_output);
    void update(Scalar learning_rate);

    size_t embed_dim() const { return embed_dim_; }

    void save(std::ostream& os) const;
    void load_weights(std::istream& is);

private:
    size_t embed_dim_;
    Scalar epsilon_;

    Matrix gamma_, beta_;
    Matrix grad_gamma_, grad_beta_;

    Tensor3D input_cache_;
    Tensor3D normalized_cache_;
    Matrix mean_cache_;
    Matrix var_cache_;
};