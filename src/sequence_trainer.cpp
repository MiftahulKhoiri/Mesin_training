// ============================================================
// src/sequence_trainer.cpp  (FILE BARU)
// ============================================================
#include "sequence_trainer.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <stdexcept>

SequenceTrainer::SequenceTrainer(SequenceModel& model, const SequenceTrainConfig& config)
    : model_(model), config_(config) {}

std::vector<size_t> SequenceTrainer::build_example_starts(size_t total_tokens) const {
    std::vector<size_t> starts;
    // +1 karena target butuh satu token setelah akhir window input (next-token prediction)
    if (total_tokens <= config_.seq_len) {
        throw std::invalid_argument("SequenceTrainer: token_stream lebih pendek dari seq_len");
    }
    for (size_t start = 0; start + config_.seq_len + 1 <= total_tokens; start += config_.seq_len) {
        starts.push_back(start);
    }
    return starts;
}

std::vector<size_t> SequenceTrainer::make_shuffled_indices(size_t n, unsigned seed) const {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 gen(seed);
    std::shuffle(idx.begin(), idx.end(), gen);
    return idx;
}

void SequenceTrainer::make_batch(const Matrix& token_stream, const std::vector<size_t>& starts,
                                  size_t batch_begin, size_t count,
                                  Matrix& batch_input, Matrix& batch_target) const {
    size_t actual_count = std::min(count, starts.size() - batch_begin);
    batch_input = Matrix(actual_count, config_.seq_len);
    batch_target = Matrix(actual_count, config_.seq_len);

    for (size_t i = 0; i < actual_count; ++i) {
        size_t s = starts[batch_begin + i];
        for (size_t t = 0; t < config_.seq_len; ++t) {
            batch_input.at(i, t) = token_stream.at(0, s + t);
            batch_target.at(i, t) = token_stream.at(0, s + t + 1); // digeser 1 (next-token)
        }
    }
}

Scalar SequenceTrainer::evaluate_loss(const Matrix& token_stream, const std::vector<size_t>& starts) const {
    if (starts.empty()) return -1.0f;

    Scalar total_loss = 0.0f;
    size_t num_batches = 0;

    for (size_t begin = 0; begin < starts.size(); begin += config_.batch_size) {
        Matrix batch_input, batch_target;
        make_batch(token_stream, starts, begin, config_.batch_size, batch_input, batch_target);

        Tensor3D logits = model_.forward(batch_input);
        // Hanya compute_loss (tanpa backward) — sesuai desain SequenceModel untuk evaluasi murni
        total_loss += model_.compute_loss(logits, batch_target);
        num_batches++;
    }
    return total_loss / static_cast<Scalar>(num_batches);
}

std::vector<SequenceEpochLog> SequenceTrainer::fit(const Matrix& token_stream) {
    if (token_stream.rows() != 1) {
        throw std::invalid_argument("SequenceTrainer::fit: token_stream harus berbentuk 1 x N");
    }

    std::vector<size_t> all_starts = build_example_starts(token_stream.cols());

    std::vector<size_t> train_starts, val_starts;
    bool has_val = config_.validation_split > 0.0f;

    if (has_val) {
        std::vector<size_t> shuffled = make_shuffled_indices(all_starts.size(), config_.shuffle_seed);
        size_t val_count = static_cast<size_t>(static_cast<Scalar>(all_starts.size()) * config_.validation_split);
        size_t train_count = all_starts.size() - val_count;

        train_starts.reserve(train_count);
        for (size_t i = 0; i < train_count; ++i) train_starts.push_back(all_starts[shuffled[i]]);

        val_starts.reserve(val_count);
        for (size_t i = train_count; i < all_starts.size(); ++i) val_starts.push_back(all_starts[shuffled[i]]);
    } else {
        train_starts = all_starts;
    }

    std::vector<SequenceEpochLog> history;
    history.reserve(config_.epochs);

    for (size_t epoch = 0; epoch < config_.epochs; ++epoch) {
        std::vector<size_t> order = config_.shuffle_each_epoch
            ? make_shuffled_indices(train_starts.size(), config_.shuffle_seed + static_cast<unsigned>(epoch))
            : [&]{ std::vector<size_t> seq(train_starts.size()); std::iota(seq.begin(), seq.end(), 0); return seq; }();

        std::vector<size_t> epoch_starts(train_starts.size());
        for (size_t i = 0; i < order.size(); ++i) epoch_starts[i] = train_starts[order[i]];

        Scalar epoch_loss_sum = 0.0f;
        size_t num_batches = 0;
        size_t step = 0;

        for (size_t begin = 0; begin < epoch_starts.size(); begin += config_.batch_size) {
            Matrix batch_input, batch_target;
            make_batch(token_stream, epoch_starts, begin, config_.batch_size, batch_input, batch_target);

            Tensor3D logits = model_.forward(batch_input);
            // backward() SequenceModel mengembalikan loss langsung — satu pass softmax,
            // tidak perlu compute_loss terpisah untuk step training (lihat catatan sequence_model)
            Scalar step_loss = model_.backward(logits, batch_target, batch_input);
            model_.update(config_.learning_rate);

            epoch_loss_sum += step_loss;
            num_batches++;
            step++;

            if (config_.verbose && config_.log_every_n_steps > 0 && step % config_.log_every_n_steps == 0) {
                std::cout << "  epoch " << (epoch + 1) << " step " << step
                           << " - loss: " << step_loss << "\n";
            }
        }

        Scalar avg_train_loss = epoch_loss_sum / static_cast<Scalar>(num_batches);
        Scalar val_loss = has_val ? evaluate_loss(token_stream, val_starts) : -1.0f;

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