// ============================================================
// src/dense_layer.cpp
// ============================================================
#include "dense_layer.h"
#include <cmath>

DenseLayer::DenseLayer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed)
    : weights_(input_size, output_size),
      bias_(1, output_size, 0.0f),
      activation_(activation),
      grad_weights_(input_size, output_size, 0.0f),
      grad_bias_(1, output_size, 0.0f)
{
    // Inisialisasi bobot: He untuk ReLU/LeakyReLU, Xavier untuk sisanya —
    // penting untuk stabilitas gradien, terutama di jaringan yang lebih dalam
    Scalar stddev;
    if (activation == ActivationType::ReLU || activation == ActivationType::LeakyReLU) {
        stddev = std::sqrt(2.0f / static_cast<Scalar>(input_size)); // He init
    } else {
        stddev = std::sqrt(1.0f / static_cast<Scalar>(input_size)); // Xavier (versi sederhana)
    }
    weights_ = Matrix::random_normal(input_size, output_size, 0.0f, stddev, seed);
}

Matrix DenseLayer::forward(const Matrix& input) {
    input_cache_ = input; // batch x input_size

    Matrix z = input * weights_;           // batch x output_size
    z = z.add_row_vector(bias_);            // broadcast bias ke tiap baris
    z_cache_ = z;

    return Activation::forward(z, activation_);
}

Matrix DenseLayer::backward(const Matrix& grad_output) {
    // grad_output: d_loss/d_output, shape batch x output_size

    Matrix grad_z;
    if (activation_ == ActivationType::Softmax) {
        // Kasus khusus: kalau layer ini output softmax, asumsikan grad_output
        // SUDAH berupa (probs - targets) dari SoftmaxCrossEntropy::derivative,
        // jadi tidak perlu dikalikan turunan softmax lagi di sini.
        grad_z = grad_output;
    } else {
        Matrix act_deriv = Activation::derivative(z_cache_, activation_);
        grad_z = grad_output.hadamard(act_deriv); // batch x output_size
    }

    // Gradien weight: input^T * grad_z, dinormalisasi per batch
    Scalar batch_size = static_cast<Scalar>(input_cache_.rows());
    Matrix input_t = input_cache_.transpose();      // input_size x batch
    grad_weights_ = (input_t * grad_z).scale(1.0f / batch_size);

    // Gradien bias: jumlah grad_z per kolom, dinormalisasi per batch
    grad_bias_ = grad_z.sum_rows().scale(1.0f / batch_size);

    // Gradien untuk diteruskan ke layer sebelumnya: grad_z * weight^T
    Matrix weights_t = weights_.transpose();        // output_size x input_size
    return grad_z * weights_t;                      // batch x input_size
}

void DenseLayer::update(Scalar learning_rate) {
    weights_.sub_inplace(grad_weights_.scale(learning_rate));
    bias_.sub_inplace(grad_bias_.scale(learning_rate));
}