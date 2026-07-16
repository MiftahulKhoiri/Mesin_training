// ============================================================
// src/dense_layer.cpp  (LENGKAP)
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
        stddev = std::sqrt(2.0f / static_cast<Scalar>(input_size));
    } else {
        stddev = std::sqrt(1.0f / static_cast<Scalar>(input_size));
    }
    weights_ = Matrix::random_normal(input_size, output_size, 0.0f, stddev, seed);
}

Matrix DenseLayer::forward(const Matrix& input, bool /*training*/) {
    input_cache_ = input;
    Matrix z = input * weights_;
    z = z.add_row_vector(bias_);
    z_cache_ = z;
    return Activation::forward(z, activation_);
}

Matrix DenseLayer::backward(const Matrix& grad_output, bool combined_with_loss) {
    Matrix grad_z;

    if (combined_with_loss) {
        grad_z = grad_output;
    } else if (activation_ == ActivationType::Softmax) {
        throw std::logic_error(
            "DenseLayer::backward: Softmax berdiri sendiri (bukan combined_with_loss) "
            "belum didukung — pasangkan dengan loss yang sesuai dan set combined_with_loss=true.");
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

void DenseLayer::save(std::ostream& os) const {
    int act = static_cast<int>(activation_);
    os.write(reinterpret_cast<const char*>(&act), sizeof(act));
    weights_.save(os);
    bias_.save(os);
}

std::unique_ptr<DenseLayer> DenseLayer::load(std::istream& is) {
    int act;
    is.read(reinterpret_cast<char*>(&act), sizeof(act));
    Matrix w = Matrix::load(is);
    Matrix b = Matrix::load(is);
    auto layer = std::make_unique<DenseLayer>(w.rows(), w.cols(), static_cast<ActivationType>(act), 0u);
    layer->weights() = w;
    layer->bias() = b;
    return layer;
}