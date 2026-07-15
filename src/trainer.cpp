// ============================================================
// src/trainer.cpp
// ============================================================
#include "trainer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <stdexcept>

Trainer::Trainer(NeuralNetwork& model, const TrainConfig& config)
    : model_(model), config_(config) {}

std::vector<size_t> Trainer::make_shuffled_indices(size_t n, unsigned seed) const {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(idx.begin(), idx.end(), gen);
    return idx;
}

Matrix Trainer::gather_rows(const Matrix& source, const std::vector<size_t>& indices,
                             size_t start, size_t count) {
    size_t n = std::min(count, indices.size() - start);
    Matrix result(n, source.cols());
    for (size_t i = 0; i < n; ++i) {
        size_t src_row = indices[start + i];
        for (size_t j = 0; j < source.cols(); ++j) {
            result.at(i, j) = source.at(src_row, j);
        }
    }
    return result;
}

void Trainer::split_data(const Matrix& inputs, const Matrix& targets,
                          Matrix& train_x, Matrix& train_y,
                          Matrix& val_x, Matrix& val_y) const {
    size_t n = inputs.rows();
    size_t val_count = static_cast<size_t>(static_cast<Scalar>(n) * config_.validation_split);
    size_t train_count = n - val_count;

    // Shuffle sekali sebelum split, supaya val set tidak bias urutan data asli
    std::vector<size_t> idx = make_shuffled_indices(n, config_.shuffle_seed);

    train_x = gather_rows(inputs, idx, 0, train_count);
    train_y = gather_rows(targets, idx, 0, train_count);
    if (val_count > 0) {
        val_x = gather_rows(inputs, idx, train_count, val_count);
        val_y = gather_rows(targets, idx, train_count, val_count);
    }
}

Scalar Trainer::evaluate_loss(const Matrix& inputs, const Matrix& targets) {
    Matrix predictions = model_.forward(inputs);
    return model_.compute_loss(predictions, targets);
}

std::vector<EpochLog> Trainer::fit(const Matrix& inputs, const Matrix& targets) {
    if (inputs.rows() != targets.rows()) {
        throw std::invalid_argument("Trainer::fit: jumlah baris inputs dan targets tidak sama");
    }

    Matrix train_x, train_y, val_x, val_y;
    bool has_val = config_.validation_split > 0.0f;

    if (has_val) {
        split_data(inputs, targets, train_x, train_y, val_x, val_y);
    } else {
        train_x = inputs;
        train_y = targets;
    }

    std::vector<EpochLog> history;
    history.reserve(config_.epochs);

    for (size_t epoch = 0; epoch < config_.epochs; ++epoch) {
        std::vector<size_t> idx = config_.shuffle_each_epoch
            ? make_shuffled_indices(train_x.rows(), config_.shuffle_seed + static_cast<unsigned>(epoch))
            : [&]{ std::vector<size_t> seq(train_x.rows()); std::iota(seq.begin(), seq.end(), 0); return seq; }();

        Scalar epoch_loss_sum = 0.0f;
        size_t num_batches = 0;

        for (size_t start = 0; start < idx.size(); start += config_.batch_size) {
            Matrix batch_x = gather_rows(train_x, idx, start, config_.batch_size);
            Matrix batch_y = gather_rows(train_y, idx, start, config_.batch_size);

            Matrix predictions = model_.forward(batch_x);
            epoch_loss_sum += model_.compute_loss(predictions, batch_y);
            num_batches++;

            model_.backward(predictions, batch_y);
            model_.update(config_.learning_rate);
        }

        Scalar avg_train_loss = epoch_loss_sum / static_cast<Scalar>(num_batches);
        Scalar val_loss = -1.0f;
        if (has_val) {
            val_loss = evaluate_loss(val_x, val_y);
        }

        history.push_back({epoch, avg_train_loss, val_loss});

        if (config_.verbose) {
            std::cout << "Epoch " << (epoch + 1) << "/" << config_.epochs
                       << " - train_loss: " << avg_train_loss;
            if (has_val) std::cout << " - val_loss: " << val_loss;
            std::cout << "\n";
        }
    }

    return history;
}