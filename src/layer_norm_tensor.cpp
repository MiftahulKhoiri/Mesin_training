// ============================================================
// src/layer_norm_tensor.cpp  (LENGKAP)
// ============================================================
#include "layer_norm_tensor.h"
#include <cmath>

LayerNormTensor::LayerNormTensor(size_t embed_dim, Scalar epsilon)
    : embed_dim_(embed_dim), epsilon_(epsilon),
      gamma_(1, embed_dim, 1.0f), beta_(1, embed_dim, 0.0f),
      grad_gamma_(1, embed_dim, 0.0f), grad_beta_(1, embed_dim, 0.0f) {}

Tensor3D LayerNormTensor::forward(const Tensor3D& input) {
    input_cache_ = input;
    size_t batch = input.batch(), seq_len = input.seq_len();

    mean_cache_ = Matrix(batch, seq_len, 0.0f);
    var_cache_ = Matrix(batch, seq_len, 0.0f);
    normalized_cache_ = Tensor3D(batch, seq_len, embed_dim_);
    Tensor3D output(batch, seq_len, embed_dim_);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            Scalar sum = 0.0f;
            for (size_t f = 0; f < embed_dim_; ++f) sum += input.at(b, s, f);
            Scalar mean = sum / static_cast<Scalar>(embed_dim_);

            Scalar sq_sum = 0.0f;
            for (size_t f = 0; f < embed_dim_; ++f) {
                Scalar diff = input.at(b, s, f) - mean;
                sq_sum += diff * diff;
            }
            Scalar var = sq_sum / static_cast<Scalar>(embed_dim_);

            mean_cache_.at(b, s) = mean;
            var_cache_.at(b, s) = var;

            Scalar denom = std::sqrt(var + epsilon_);
            for (size_t f = 0; f < embed_dim_; ++f) {
                Scalar x_hat = (input.at(b, s, f) - mean) / denom;
                normalized_cache_.at(b, s, f) = x_hat;
                output.at(b, s, f) = gamma_.at(0, f) * x_hat + beta_.at(0, f);
            }
        }
    }
    return output;
}

Tensor3D LayerNormTensor::backward(const Tensor3D& grad_output) {
    size_t batch = input_cache_.batch(), seq_len = input_cache_.seq_len();
    Scalar D = static_cast<Scalar>(embed_dim_);

    grad_gamma_ = Matrix(1, embed_dim_, 0.0f);
    grad_beta_ = Matrix(1, embed_dim_, 0.0f);
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq_len; ++s)
            for (size_t f = 0; f < embed_dim_; ++f) {
                grad_gamma_.at(0, f) += grad_output.at(b, s, f) * normalized_cache_.at(b, s, f);
                grad_beta_.at(0, f) += grad_output.at(b, s, f);
            }

    Tensor3D grad_input(batch, seq_len, embed_dim_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            Scalar std_inv = 1.0f / std::sqrt(var_cache_.at(b, s) + epsilon_);

            Scalar sum_dxhat = 0.0f, sum_dxhat_xhat = 0.0f;
            for (size_t f = 0; f < embed_dim_; ++f) {
                Scalar dxhat = grad_output.at(b, s, f) * gamma_.at(0, f);
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * normalized_cache_.at(b, s, f);
            }

            for (size_t f = 0; f < embed_dim_; ++f) {
                Scalar dxhat = grad_output.at(b, s, f) * gamma_.at(0, f);
                Scalar x_hat = normalized_cache_.at(b, s, f);
                grad_input.at(b, s, f) = std_inv * (dxhat - sum_dxhat / D - x_hat * sum_dxhat_xhat / D);
            }
        }
    }
    return grad_input;
}

void LayerNormTensor::update(Scalar learning_rate) {
    gamma_.sub_inplace(grad_gamma_.scale(learning_rate));
    beta_.sub_inplace(grad_beta_.scale(learning_rate));
}

void LayerNormTensor::save(std::ostream& os) const {
    gamma_.save(os);
    beta_.save(os);
}

void LayerNormTensor::load_weights(std::istream& is) {
    gamma_ = Matrix::load(is);
    beta_ = Matrix::load(is);
}