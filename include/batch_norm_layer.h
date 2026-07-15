// ============================================================
// include/batch_norm_layer.h  (DIPERBARUI)
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "layer_base.h"

class BatchNormLayer : public LayerBase {
public:
    explicit BatchNormLayer(size_t num_features, Scalar momentum = 0.9f, Scalar epsilon = 1e-5f);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& grad_output, bool combined_with_loss) override;
    void update(Scalar learning_rate) override;

    size_t input_size() const override { return num_features_; }
    size_t output_size() const override { return num_features_; }

    Matrix& gamma() { return gamma_; }
    Matrix& beta() { return beta_; }
    const Matrix& gamma_grad() const { return grad_gamma_; }
    const Matrix& beta_grad() const { return grad_beta_; }

private:
    size_t num_features_;
    Scalar momentum_;
    Scalar epsilon_;

    Matrix gamma_;
    Matrix beta_;
    Matrix running_mean_;
    Matrix running_var_;

    Matrix input_cache_;
    Matrix normalized_cache_;
    Matrix batch_mean_cache_;
    Matrix batch_var_cache_;

    Matrix grad_gamma_;
    Matrix grad_beta_;
};