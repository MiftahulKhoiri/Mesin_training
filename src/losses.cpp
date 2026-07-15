// ============================================================
// src/losses.cpp
// ============================================================
#include "losses.h"
#include "activations.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

static void check_shape(const Matrix& p, const Matrix& t, const char* op) {
    if (p.rows() != t.rows() || p.cols() != t.cols()) {
        throw std::invalid_argument(std::string("Loss shape mismatch on ") + op);
    }
}

Scalar Loss::forward(const Matrix& predictions, const Matrix& targets, LossType type) {
    switch (type) {
        case LossType::MSE:                return mse_forward(predictions, targets);
        case LossType::BinaryCrossEntropy: return bce_forward(predictions, targets);
        case LossType::SoftmaxCrossEntropy: return softmax_ce_forward(predictions, targets);
    }
    throw std::invalid_argument("Loss::forward: tipe tidak dikenal");
}

Matrix Loss::derivative(const Matrix& predictions, const Matrix& targets, LossType type) {
    switch (type) {
        case LossType::MSE:                return mse_derivative(predictions, targets);
        case LossType::BinaryCrossEntropy: return bce_derivative(predictions, targets);
        case LossType::SoftmaxCrossEntropy: return softmax_ce_derivative(predictions, targets);
    }
    throw std::invalid_argument("Loss::derivative: tipe tidak dikenal");
}

// --- MSE ---
Scalar Loss::mse_forward(const Matrix& p, const Matrix& t) {
    check_shape(p, t, "MSE forward");
    const Scalar* pp = p.data();
    const Scalar* tp = t.data();
    size_t n = p.rows() * p.cols();
    Scalar sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        Scalar diff = pp[i] - tp[i];
        sum += diff * diff;
    }
    // rata-rata per sample (bukan per elemen), konsisten dengan batch size = rows()
    return sum / static_cast<Scalar>(p.rows());
}

Matrix Loss::mse_derivative(const Matrix& p, const Matrix& t) {
    check_shape(p, t, "MSE derivative");
    Matrix result(p.rows(), p.cols());
    const Scalar* pp = p.data();
    const Scalar* tp = t.data();
    Scalar* rp = result.data();
    size_t n = p.rows() * p.cols();
    Scalar batch_size = static_cast<Scalar>(p.rows());
    for (size_t i = 0; i < n; ++i) {
        rp[i] = 2.0f * (pp[i] - tp[i]) / batch_size;
    }
    return result;
}

// --- Binary Cross-Entropy ---
// p diasumsikan probabilitas (0,1), hasil sigmoid
Scalar Loss::bce_forward(const Matrix& p, const Matrix& t) {
    check_shape(p, t, "BCE forward");
    const Scalar* pp = p.data();
    const Scalar* tp = t.data();
    size_t n = p.rows() * p.cols();
    const Scalar eps = 1e-7f; // cegah log(0)
    Scalar sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        Scalar pc = std::min(std::max(pp[i], eps), 1.0f - eps);
        sum += -(tp[i] * std::log(pc) + (1.0f - tp[i]) * std::log(1.0f - pc));
    }
    return sum / static_cast<Scalar>(p.rows());
}

Matrix Loss::bce_derivative(const Matrix& p, const Matrix& t) {
    check_shape(p, t, "BCE derivative");
    Matrix result(p.rows(), p.cols());
    const Scalar* pp = p.data();
    const Scalar* tp = t.data();
    Scalar* rp = result.data();
    size_t n = p.rows() * p.cols();
    const Scalar eps = 1e-7f;
    Scalar batch_size = static_cast<Scalar>(p.rows());
    for (size_t i = 0; i < n; ++i) {
        Scalar pc = std::min(std::max(pp[i], eps), 1.0f - eps);
        rp[i] = (-(tp[i] / pc) + (1.0f - tp[i]) / (1.0f - pc)) / batch_size;
    }
    return result;
}

// --- Softmax + Cross-Entropy (gabungan, input = logits mentah) ---
Scalar Loss::softmax_ce_forward(const Matrix& logits, const Matrix& targets) {
    check_shape(logits, targets, "SoftmaxCE forward");
    Matrix probs = Activation::forward(logits, ActivationType::Softmax);
    const Scalar eps = 1e-9f;
    Scalar sum = 0.0f;
    for (size_t i = 0; i < probs.rows(); ++i) {
        for (size_t j = 0; j < probs.cols(); ++j) {
            Scalar t = targets.at(i, j);
            if (t != 0.0f) { // hanya kelas target yang berkontribusi (one-hot)
                sum += -t * std::log(std::max(probs.at(i, j), eps));
            }
        }
    }
    return sum / static_cast<Scalar>(logits.rows());
}

Matrix Loss::softmax_ce_derivative(const Matrix& logits, const Matrix& targets) {
    check_shape(logits, targets, "SoftmaxCE derivative");
    Matrix probs = Activation::forward(logits, ActivationType::Softmax);
    Matrix result(logits.rows(), logits.cols());
    Scalar batch_size = static_cast<Scalar>(logits.rows());
    for (size_t i = 0; i < probs.rows(); ++i) {
        for (size_t j = 0; j < probs.cols(); ++j) {
            result.at(i, j) = (probs.at(i, j) - targets.at(i, j)) / batch_size;
        }
    }
    return result;
}