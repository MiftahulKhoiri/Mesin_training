// ============================================================
// include/dense_layer.h
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "activations.h"

class DenseLayer {
public:
    DenseLayer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed = 42);

    Matrix forward(const Matrix& input);

    // Backward pass: terima gradien dari layer berikutnya / loss (d_loss/d_output).
    //
    // combined_with_loss: set true jika grad_output SUDAH merupakan gradien
    // gabungan aktivasi+loss (mis. hasil SoftmaxCrossEntropy::derivative yang
    // sudah = probs - targets, atau kombinasi Sigmoid+BCE serupa). Dalam kasus
    // ini turunan aktivasi TIDAK dihitung ulang di sini.
    //
    // Jika false, turunan aktivasi dihitung normal lewat Activation::derivative.
    // Softmax TIDAK mendukung mode ini (Jacobian penuh belum diimplementasikan) —
    // jika activation_ == Softmax dan combined_with_loss == false, akan throw.
    Matrix backward(const Matrix& grad_output, bool combined_with_loss = false);

    void update(Scalar learning_rate);

    size_t input_size() const { return weights_.rows(); }
    size_t output_size() const { return weights_.cols(); }
// Tambahkan di bagian public DenseLayer (dense_layer.h), setelah output_size():
    ActivationType activation_type() const { return activation_; }

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