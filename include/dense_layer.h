// ============================================================
// include/dense_layer.h  (DIPERBARUI)
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "activations.h"
#include "layer_base.h"

class DenseLayer : public LayerBase {
public:
    DenseLayer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed = 42);

    Matrix forward(const Matrix& input, bool training) override;
    Matrix backward(const Matrix& grad_output, bool combined_with_loss) override;
    void update(Scalar learning_rate) override;

    size_t input_size() const override { return weights_.rows(); }
    size_t output_size() const override { return weights_.cols(); }
    ActivationType activation_type() const override { return activation_; }

    Matrix& weights() { return weights_; }
    Matrix& bias() { return bias_; }
    const Matrix& weight_grad() const { return grad_weights_; }
    const Matrix& bias_grad() const { return grad_bias_; }

private:
    Matrix weights_;
    Matrix bias_;
    ActivationType activation_;

    Matrix input_cache_;
    Matrix z_cache_;

    Matrix grad_weights_;
    Matrix grad_bias_;
};