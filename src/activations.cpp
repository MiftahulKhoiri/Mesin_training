// ============================================================
// src/activations.cpp
// ============================================================
#include "activations.h"
#include <algorithm>
#include <limits>

Matrix Activation::forward(const Matrix& z, ActivationType type, Scalar leaky_alpha) {
    switch (type) {
        case ActivationType::ReLU:       return relu(z);
        case ActivationType::LeakyReLU:  return leaky_relu(z, leaky_alpha);
        case ActivationType::Sigmoid:    return sigmoid(z);
        case ActivationType::Tanh:       return tanh_activation(z);
        case ActivationType::Softmax:    return softmax(z);
        case ActivationType::Linear:     return z;
    }
    throw std::invalid_argument("Activation::forward: tipe tidak dikenal");
}

Matrix Activation::derivative(const Matrix& z, ActivationType type, Scalar leaky_alpha) {
    switch (type) {
        case ActivationType::ReLU:       return relu_derivative(z);
        case ActivationType::LeakyReLU:  return leaky_relu_derivative(z, leaky_alpha);
        case ActivationType::Sigmoid:    return sigmoid_derivative(z);
        case ActivationType::Tanh:       return tanh_derivative(z);
        case ActivationType::Linear: {
            Matrix ones(z.rows(), z.cols(), 1.0f);
            return ones;
        }
        case ActivationType::Softmax:
            throw std::logic_error(
                "Softmax::derivative tidak dipakai langsung — gabungkan dengan cross-entropy di losses.cpp");
    }
    throw std::invalid_argument("Activation::derivative: tipe tidak dikenal");
}

// --- ReLU ---
Matrix Activation::relu(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    const Scalar* zp = z.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) rp[i] = zp[i] > 0.0f ? zp[i] : 0.0f;
    return result;
}

Matrix Activation::relu_derivative(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    const Scalar* zp = z.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) rp[i] = zp[i] > 0.0f ? 1.0f : 0.0f;
    return result;
}

// --- Leaky ReLU ---
Matrix Activation::leaky_relu(const Matrix& z, Scalar alpha) {
    Matrix result(z.rows(), z.cols());
    const Scalar* zp = z.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) rp[i] = zp[i] > 0.0f ? zp[i] : alpha * zp[i];
    return result;
}

Matrix Activation::leaky_relu_derivative(const Matrix& z, Scalar alpha) {
    Matrix result(z.rows(), z.cols());
    const Scalar* zp = z.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) rp[i] = zp[i] > 0.0f ? 1.0f : alpha;
    return result;
}

// --- Sigmoid ---
Matrix Activation::sigmoid(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    const Scalar* zp = z.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) {
        // clamp untuk cegah overflow expf pada nilai sangat negatif/positif
        Scalar x = std::max(-60.0f, std::min(60.0f, zp[i]));
        rp[i] = 1.0f / (1.0f + std::exp(-x));
    }
    return result;
}

Matrix Activation::sigmoid_derivative(const Matrix& z) {
    Matrix s = sigmoid(z);
    Matrix result(z.rows(), z.cols());
    const Scalar* sp = s.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) rp[i] = sp[i] * (1.0f - sp[i]);
    return result;
}

// --- Tanh ---
Matrix Activation::tanh_activation(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    const Scalar* zp = z.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) rp[i] = std::tanh(zp[i]);
    return result;
}

Matrix Activation::tanh_derivative(const Matrix& z) {
    Matrix t = tanh_activation(z);
    Matrix result(z.rows(), z.cols());
    const Scalar* tp = t.data();
    Scalar* rp = result.data();
    size_t n = z.rows() * z.cols();
    for (size_t i = 0; i < n; ++i) rp[i] = 1.0f - tp[i] * tp[i];
    return result;
}

// --- Softmax (per baris, dengan max-subtraction untuk stabilitas numerik) ---
Matrix Activation::softmax(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    for (size_t i = 0; i < z.rows(); ++i) {
        Scalar row_max = -std::numeric_limits<Scalar>::infinity();
        for (size_t j = 0; j < z.cols(); ++j) row_max = std::max(row_max, z.at(i, j));

        Scalar sum = 0.0f;
        for (size_t j = 0; j < z.cols(); ++j) {
            Scalar e = std::exp(z.at(i, j) - row_max);
            result.at(i, j) = e;
            sum += e;
        }
        // hindari pembagian nol pada baris degenerate
        Scalar inv_sum = 1.0f / (sum > 1e-12f ? sum : 1e-12f);
        for (size_t j = 0; j < z.cols(); ++j) result.at(i, j) *= inv_sum;
    }
    return result;
}