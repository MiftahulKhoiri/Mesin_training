// ============================================================
// src/sequence_model.cpp  (LENGKAP)
// ============================================================
#include "sequence_model.h"
#include <cmath>
#include <limits>
#include <fstream>
#include <stdexcept>

SequenceModel::SequenceModel(size_t vocab_size, size_t embed_dim, size_t num_heads,
                              size_t ff_hidden_dim, size_t num_layers, size_t max_seq_len,
                              bool causal_mask, unsigned seed)
    : vocab_size_(vocab_size), embed_dim_(embed_dim), num_heads_(num_heads),
      ff_hidden_dim_(ff_hidden_dim), max_seq_len_(max_seq_len), causal_mask_(causal_mask),
      embedding_(vocab_size, embed_dim, max_seq_len, seed),
      final_norm_(embed_dim)
{
    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(embed_dim, num_heads, ff_hidden_dim, causal_mask,
                              seed + 100 * static_cast<unsigned>(i + 1));
    }
}

Tensor3D SequenceModel::forward(const Matrix& token_ids) {
    Tensor3D x = embedding_.forward(token_ids);
    for (auto& block : blocks_) x = block.forward(x);

    Tensor3D normed = final_norm_.forward(x);
    last_hidden_cache_ = normed;

    Matrix tied_weight = embedding_.embedding_table().transpose(); // embed_dim x vocab_size
    return normed.batched_matmul(tied_weight);
}

Scalar SequenceModel::compute_loss(const Tensor3D& logits, const Matrix& target_ids) const {
    size_t batch = logits.batch(), seq_len = logits.seq_len(), vocab = logits.features();
    Scalar total_loss = 0.0f;
    size_t count = 0;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            Scalar max_logit = -std::numeric_limits<Scalar>::infinity();
            for (size_t v = 0; v < vocab; ++v) max_logit = std::max(max_logit, logits.at(b, s, v));

            Scalar sum_exp = 0.0f;
            for (size_t v = 0; v < vocab; ++v) sum_exp += std::exp(logits.at(b, s, v) - max_logit);

            size_t target = static_cast<size_t>(std::round(target_ids.at(b, s)));
            Scalar log_prob = (logits.at(b, s, target) - max_logit) - std::log(sum_exp);
            total_loss += -log_prob;
            count++;
        }
    }
    return total_loss / static_cast<Scalar>(count);
}

Scalar SequenceModel::backward(const Tensor3D& logits, const Matrix& target_ids, const Matrix& token_ids) {
    size_t batch = logits.batch(), seq_len = logits.seq_len(), vocab = logits.features();
    Scalar total_tokens = static_cast<Scalar>(batch * seq_len);

    Tensor3D grad_logits(batch, seq_len, vocab);
    Scalar total_loss = 0.0f;

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            Scalar max_logit = -std::numeric_limits<Scalar>::infinity();
            for (size_t v = 0; v < vocab; ++v) max_logit = std::max(max_logit, logits.at(b, s, v));

            Scalar sum_exp = 0.0f;
            for (size_t v = 0; v < vocab; ++v) sum_exp += std::exp(logits.at(b, s, v) - max_logit);

            size_t target = static_cast<size_t>(std::round(target_ids.at(b, s)));
            Scalar log_sum_exp = std::log(sum_exp);

            for (size_t v = 0; v < vocab; ++v) {
                Scalar prob = std::exp(logits.at(b, s, v) - max_logit) / sum_exp;
                grad_logits.at(b, s, v) = prob / total_tokens;
            }
            grad_logits.at(b, s, target) -= 1.0f / total_tokens;

            Scalar log_prob = (logits.at(b, s, target) - max_logit) - log_sum_exp;
            total_loss += -log_prob;
        }
    }
    Scalar loss = total_loss / total_tokens;

    Scalar batch_scalar = static_cast<Scalar>(batch);
    Matrix embed_table = embedding_.embedding_table();
    Matrix grad_tied_weight(embed_dim_, vocab_size_, 0.0f);
    Tensor3D grad_last_hidden(batch, seq_len, embed_dim_);

    for (size_t b = 0; b < batch; ++b) {
        Matrix hidden_b = last_hidden_cache_.slice_batch(b);
        Matrix gl_b = grad_logits.slice_batch(b);
        grad_tied_weight.add_inplace(hidden_b.transpose() * gl_b);
        grad_last_hidden.set_batch(b, gl_b * embed_table);
    }
    grad_tied_weight = grad_tied_weight.scale(1.0f / batch_scalar);
    Matrix grad_tied_weight_t = grad_tied_weight.transpose();

    Tensor3D grad = final_norm_.backward(grad_last_hidden);
    for (size_t i = blocks_.size(); i-- > 0;) {
        grad = blocks_[i].backward(grad);
    }

    embedding_.backward(grad, token_ids);
    embedding_.accumulate_external_grad(grad_tied_weight_t);

    return loss;
}

void SequenceModel::update(Scalar learning_rate) {
    embedding_.update(learning_rate);
    for (auto& block : blocks_) block.update(learning_rate);
    final_norm_.update(learning_rate);
}

void SequenceModel::save_checkpoint(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("SequenceModel::save_checkpoint: gagal membuka file: " + path);

    size_t num_layers = blocks_.size();
    ofs.write(reinterpret_cast<const char*>(&vocab_size_), sizeof(vocab_size_));
    ofs.write(reinterpret_cast<const char*>(&embed_dim_), sizeof(embed_dim_));
    ofs.write(reinterpret_cast<const char*>(&num_heads_), sizeof(num_heads_));
    ofs.write(reinterpret_cast<const char*>(&ff_hidden_dim_), sizeof(ff_hidden_dim_));
    ofs.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    ofs.write(reinterpret_cast<const char*>(&max_seq_len_), sizeof(max_seq_len_));
    ofs.write(reinterpret_cast<const char*>(&causal_mask_), sizeof(causal_mask_));

    embedding_.save(ofs);
    for (const auto& block : blocks_) block.save(ofs);
    final_norm_.save(ofs);
}

SequenceModel SequenceModel::load_checkpoint(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("SequenceModel::load_checkpoint: gagal membuka file: " + path);

    size_t vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len;
    bool causal_mask;
    ifs.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    ifs.read(reinterpret_cast<char*>(&embed_dim), sizeof(embed_dim));
    ifs.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    ifs.read(reinterpret_cast<char*>(&ff_hidden_dim), sizeof(ff_hidden_dim));
    ifs.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    ifs.read(reinterpret_cast<char*>(&max_seq_len), sizeof(max_seq_len));
    ifs.read(reinterpret_cast<char*>(&causal_mask), sizeof(causal_mask));

    SequenceModel model(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers,
                         max_seq_len, causal_mask, 0u);

    model.embedding_.load_weights(ifs);
    for (auto& block : model.blocks_) block.load_weights(ifs);
    model.final_norm_.load_weights(ifs);

    return model;
}