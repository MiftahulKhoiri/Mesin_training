// ============================================================
// src/batch_norm_layer.cpp
// ============================================================
#include "batch_norm_layer.h"
#include <cmath>
#include <stdexcept>

BatchNormLayer::BatchNormLayer(size_t num_features, Scalar momentum, Scalar epsilon)
    : num_features_(num_features), momentum_(momentum), epsilon_(epsilon),
      gamma_(1, num_features, 1.0f),
      beta_(1, num_features, 0.0f),
      running_mean_(1, num_features, 0.0f),
      running_var_(1, num_features, 1.0f),
      grad_gamma_(1, num_features, 0.0f),
      grad_beta_(1, num_features, 0.0f)
{}

Matrix BatchNormLayer::forward(const Matrix& input, bool training) {
    if (input.cols() != num_features_) {
        throw std::invalid_argument("BatchNormLayer::forward: jumlah kolom input tidak cocok num_features");
    }

    size_t N = input.rows();
    Matrix output(input.rows(), input.cols());

    if (training) {
        input_cache_ = input;

        // Mean & variance per kolom (per feature), dihitung lintas baris (batch)
        Matrix mean(1, num_features_, 0.0f);
        Matrix var(1, num_features_, 0.0f);

        for (size_t j = 0; j < num_features_; ++j) {
            Scalar sum = 0.0f;
            for (size_t i = 0; i < N; ++i) sum += input.at(i, j);
            mean.at(0, j) = sum / static_cast<Scalar>(N);
        }
        for (size_t j = 0; j < num_features_; ++j) {
            Scalar sq_sum = 0.0f;
            for (size_t i = 0; i < N; ++i) {
                Scalar diff = input.at(i, j) - mean.at(0, j);
                sq_sum += diff * diff;
            }
            var.at(0, j) = sq_sum / static_cast<Scalar>(N);
        }

        batch_mean_cache_ = mean;
        batch_var_cache_ = var;

        Matrix x_hat(input.rows(), input.cols());
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < num_features_; ++j) {
                Scalar denom = std::sqrt(var.at(0, j) + epsilon_);
                x_hat.at(i, j) = (input.at(i, j) - mean.at(0, j)) / denom;
                output.at(i, j) = gamma_.at(0, j) * x_hat.at(i, j) + beta_.at(0, j);
            }
        }
        normalized_cache_ = x_hat;

        // Update running stats (dipakai nanti saat training=false / inferensi)
        for (size_t j = 0; j < num_features_; ++j) {
            running_mean_.at(0, j) = momentum_ * running_mean_.at(0, j) + (1.0f - momentum_) * mean.at(0, j);
            running_var_.at(0, j) = momentum_ * running_var_.at(0, j) + (1.0f - momentum_) * var.at(0, j);
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < num_features_; ++j) {
                Scalar denom = std::sqrt(running_var_.at(0, j) + epsilon_);
                Scalar x_hat = (input.at(i, j) - running_mean_.at(0, j)) / denom;
                output.at(i, j) = gamma_.at(0, j) * x_hat + beta_.at(0, j);
            }
        }
    }

    return output;
}

Matrix BatchNormLayer::backward(const Matrix& grad_output) {
    if (input_cache_.rows() == 0) {
        throw std::logic_error("BatchNormLayer::backward: dipanggil tanpa forward(training=true) sebelumnya");
    }

    size_t N = input_cache_.rows();
    Scalar n = static_cast<Scalar>(N);

    // Gradien gamma & beta: jumlah lintas batch
    grad_gamma_ = Matrix(1, num_features_, 0.0f);
    grad_beta_ = Matrix(1, num_features_, 0.0f);
    for (size_t j = 0; j < num_features_; ++j) {
        Scalar sum_gg = 0.0f, sum_gb = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            sum_gg += grad_output.at(i, j) * normalized_cache_.at(i, j);
            sum_gb += grad_output.at(i, j);
        }
        grad_gamma_.at(0, j) = sum_gg;
        grad_beta_.at(0, j) = sum_gb;
    }

    // Backprop standar batch norm (per fitur/kolom)
    Matrix grad_input(N, num_features_);
    for (size_t j = 0; j < num_features_; ++j) {
        Scalar var_j = batch_var_cache_.at(0, j);
        Scalar mean_j = batch_mean_cache_.at(0, j);
        Scalar std_inv = 1.0f / std::sqrt(var_j + epsilon_);
        Scalar gamma_j = gamma_.at(0, j);

        Scalar sum_dxhat = 0.0f;
        Scalar sum_dxhat_xmu = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            Scalar dxhat = grad_output.at(i, j) * gamma_j;
            sum_dxhat += dxhat;
            sum_dxhat_xmu += dxhat * (input_cache_.at(i, j) - mean_j);
        }

        for (size_t i = 0; i < N; ++i) {
            Scalar dxhat = grad_output.at(i, j) * gamma_j;
            Scalar x_mu = input_cache_.at(i, j) - mean_j;
            grad_input.at(i, j) = (1.0f / n) * std_inv *
                (n * dxhat - sum_dxhat - x_mu * std_inv * std_inv * sum_dxhat_xmu);
        }
    }

    return grad_input;
}

void BatchNormLayer::update(Scalar learning_rate) {
    for (size_t j = 0; j < num_features_; ++j) {
        gamma_.at(0, j) -= learning_rate * grad_gamma_.at(0, j);
        beta_.at(0, j) -= learning_rate * grad_beta_.at(0, j);
    }
}