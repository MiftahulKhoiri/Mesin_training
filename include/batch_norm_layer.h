// ============================================================
// include/batch_norm_layer.h
// ============================================================
#pragma once
#include "matrix_ops.h"

class BatchNormLayer {
public:
    explicit BatchNormLayer(size_t num_features, Scalar momentum = 0.9f, Scalar epsilon = 1e-5f);

    // training=true: hitung mean/var dari batch saat ini & update running stats.
    // training=false (mode inferensi): pakai running_mean_/running_var_ yang sudah terkumpul.
    Matrix forward(const Matrix& input, bool training);

    // Backward hanya valid dipanggil setelah forward(..., training=true)
    Matrix backward(const Matrix& grad_output);

    void update(Scalar learning_rate);

    size_t num_features() const { return num_features_; }

    Matrix& gamma() { return gamma_; }
    Matrix& beta() { return beta_; }
    const Matrix& gamma_grad() const { return grad_gamma_; }
    const Matrix& beta_grad() const { return grad_beta_; }

private:
    size_t num_features_;
    Scalar momentum_;
    Scalar epsilon_;

    Matrix gamma_;         // 1 x num_features, scale (learnable)
    Matrix beta_;          // 1 x num_features, shift (learnable)

    Matrix running_mean_;  // 1 x num_features, dipakai saat training=false
    Matrix running_var_;   // 1 x num_features

    // Cache untuk backward (diisi ulang tiap forward saat training=true)
    Matrix input_cache_;
    Matrix normalized_cache_; // x_hat
    Matrix batch_mean_cache_;
    Matrix batch_var_cache_;

    Matrix grad_gamma_;
    Matrix grad_beta_;
};