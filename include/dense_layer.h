// ============================================================
// include/dense_layer.h
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "activations.h"

class DenseLayer {
public:
    DenseLayer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed = 42);

    // Forward pass: simpan input & z (pre-activation) untuk backprop
    Matrix forward(const Matrix& input);

    // Backward pass: terima gradien dari layer berikutnya (d_loss/d_output),
    // hitung & simpan gradien weight/bias secara internal, kembalikan
    // gradien untuk diteruskan ke layer sebelumnya (d_loss/d_input)
    Matrix backward(const Matrix& grad_output);

    // Update parameter pakai gradien yang sudah dihitung backward()
    void update(Scalar learning_rate);

    size_t input_size() const { return weights_.rows(); }
    size_t output_size() const { return weights_.cols(); }

    // Akses untuk keperluan save/load checkpoint & optimizer custom
    Matrix& weights() { return weights_; }
    Matrix& bias() { return bias_; }
    const Matrix& weight_grad() const { return grad_weights_; }
    const Matrix& bias_grad() const { return grad_bias_; }

private:
    Matrix weights_;       // input_size x output_size
    Matrix bias_;          // 1 x output_size
    ActivationType activation_;

    // Cache untuk backprop (diisi ulang tiap forward)
    Matrix input_cache_;   // batch x input_size
    Matrix z_cache_;       // batch x output_size (pre-activation)

    // Gradien hasil backward, siap dipakai update() atau optimizer eksternal
    Matrix grad_weights_;
    Matrix grad_bias_;
};