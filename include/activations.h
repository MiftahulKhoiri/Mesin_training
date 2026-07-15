// ============================================================
// include/activations.h
// ============================================================
#pragma once
#include "matrix_ops.h"

enum class ActivationType {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear   // identitas, untuk output layer regresi
};

class Activation {
public:
    // Forward: hitung aktivasi dari pre-activation z
    static Matrix forward(const Matrix& z, ActivationType type, Scalar leaky_alpha = 0.01f);

    // Derivative: turunan terhadap z (dibutuhkan saat backprop non-softmax)
    // Untuk Softmax, turunan ditangani khusus di losses.cpp (biasanya digabung dengan cross-entropy)
    static Matrix derivative(const Matrix& z, ActivationType type, Scalar leaky_alpha = 0.01f);

private:
    static Matrix relu(const Matrix& z);
    static Matrix relu_derivative(const Matrix& z);

    static Matrix leaky_relu(const Matrix& z, Scalar alpha);
    static Matrix leaky_relu_derivative(const Matrix& z, Scalar alpha);

    static Matrix sigmoid(const Matrix& z);
    static Matrix sigmoid_derivative(const Matrix& z);

    static Matrix tanh_activation(const Matrix& z);
    static Matrix tanh_derivative(const Matrix& z);

    // Softmax dihitung per baris (setiap baris = satu sample)
    static Matrix softmax(const Matrix& z);
};