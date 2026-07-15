// ============================================================
// include/losses.h
// ============================================================
#pragma once
#include "matrix_ops.h"

enum class LossType {
    MSE,                  // Mean Squared Error, untuk regresi
    BinaryCrossEntropy,    // untuk klasifikasi biner, input = probabilitas (setelah sigmoid)
    SoftmaxCrossEntropy    // untuk klasifikasi multi-kelas, input = logits mentah (softmax dihitung di dalam)
};

class Loss {
public:
    // Forward: hitung loss rata-rata per batch (1 nilai skalar)
    // predictions: output model (lihat catatan tiap LossType di atas)
    // targets: label (one-hot untuk SoftmaxCrossEntropy, 0/1 untuk BCE, nilai riil untuk MSE)
    static Scalar forward(const Matrix& predictions, const Matrix& targets, LossType type);

    // Derivative: gradien terhadap input predictions/logits, sudah dinormalisasi per batch
    // (dibagi jumlah sample), siap langsung dipakai backprop ke layer sebelumnya
    static Matrix derivative(const Matrix& predictions, const Matrix& targets, LossType type);

private:
    static Scalar mse_forward(const Matrix& p, const Matrix& t);
    static Matrix mse_derivative(const Matrix& p, const Matrix& t);

    static Scalar bce_forward(const Matrix& p, const Matrix& t);
    static Matrix bce_derivative(const Matrix& p, const Matrix& t);

    // Softmax + cross-entropy digabung jadi satu untuk stabilitas numerik
    // dan gradien sederhana: d(loss)/d(logits) = probs - targets
    static Scalar softmax_ce_forward(const Matrix& logits, const Matrix& targets);
    static Matrix softmax_ce_derivative(const Matrix& logits, const Matrix& targets);
};