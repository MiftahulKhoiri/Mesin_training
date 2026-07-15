// ============================================================
// src/dense_layer.cpp
// ============================================================
#include "dense_layer.h"
#include <cmath>
#include <stdexcept>

DenseLayer::DenseLayer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed)
    : weights_(input_size, output_size),
      bias_(1, output_size, 0.0f),
      activation_(activation),
      grad_weights_(input_size, output_size, 0.0f),
      grad_bias_(1, output_size, 0.0f)
{
    Scalar stddev;
    if (activation == ActivationType::ReLU || activation == ActivationType::LeakyReLU) {
        stddev = std::sqrt(2.0f / static_cast<Scalar>(input_size)); // He init
    } else {
        stddev = std::sqrt(1.0f / static_cast<Scalar>(input_size)); // Xavier
    }
    weights_ = Matrix::random_normal(input_size, output_size, 0.0f, stddev, seed);
}

Matrix DenseLayer::forward(const Matrix& input) {
    input_cache_ = input;

    Matrix z = input * weights_;
    z = z.add_row_vector(bias_);
    z_cache_ = z;

    return Activation::forward(z, activation_);
}

Matrix DenseLayer::backward(const Matrix& grad_output, bool combined_with_loss) {
    Matrix grad_z;

    if (combined_with_loss) {
        // Turunan aktivasi sudah tergabung di grad_output oleh loss function
        // (kontrak ini harus dijaga konsisten oleh pemanggil, mis. trainer.cpp)
        grad_z = grad_output;
    } else if (activation_ == ActivationType::Softmax) {
        throw std::logic_error(
            "DenseLayer::backward: Softmax berdiri sendiri (bukan combined_with_loss) "
            "belum didukung — Jacobian softmax penuh belum diimplementasikan. "
            "Pasangkan dengan loss yang sesuai (mis. SoftmaxCrossEntropy) dan "
            "panggil backward(grad_output, /*combined_with_loss=*/true).");
    } else {
        Matrix act_deriv = Activation::derivative(z_cache_, activation_);
        grad_z = grad_output.hadamard(act_deriv);
    }

    Scalar batch_size = static_cast<Scalar>(input_cache_.rows());
    Matrix input_t = input_cache_.transpose();
    grad_weights_ = (input_t * grad_z).scale(1.0f / batch_size);
    grad_bias_ = grad_z.sum_rows().scale(1.0f / batch_size);

    Matrix weights_t = weights_.transpose();
    return grad_z * weights_t;
}

void DenseLayer::update(Scalar learning_rate) {
    weights_.sub_inplace(grad_weights_.scale(learning_rate));
    bias_.sub_inplace(grad_bias_.scale(learning_rate));
}