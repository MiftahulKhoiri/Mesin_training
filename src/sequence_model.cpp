// ============================================================
// src/sequence_model.cpp  (FILE BARU)
// ============================================================
#include "sequence_model.h"
#include <cmath>
#include <limits>
#include <stdexcept>

SequenceModel::SequenceModel(size_t vocab_size, size_t embed_dim, size_t num_heads,
                              size_t ff_hidden_dim, size_t num_layers, size_t max_seq_len,
                              bool causal_mask, unsigned seed)
    : vocab_size_(vocab_size), embed_dim_(embed_dim),
      embedding_(vocab_size, embed_dim, max_seq_len, seed),
      final_norm_(embed_dim),
      output_projection_(embed_dim, vocab_size),
      grad_output_projection_(embed_dim, vocab_size, 0.0f)
{
    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(embed_dim, num_heads, ff_hidden_dim, causal_mask,
                              seed + 100 * static_cast<unsigned>(i + 1));
    }
    // Skala kecil (0.02), sama seperti embedding_table_ — konvensi umum di transformer
    output_projection_ = Matrix::random_normal(embed_dim, vocab_size, 0.0f, 0.02f, seed + 9999);
}

Tensor3D SequenceModel::forward(const Matrix& token_ids) {
    Tensor3D x = embedding_.forward(token_ids);
    for (auto& block : blocks_) {
        x = block.forward(x);
    }
    Tensor3D normed = final_norm_.forward(x);
    last_hidden_cache_ = normed;
    return normed.batched_matmul(output_projection_); // batch x seq x vocab
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

void SequenceModel::backward(const Tensor3D& logits, const Matrix& target_ids, const Matrix& token_ids) {
    size_t batch = logits.batch(), seq_len = logits.seq_len(), vocab = logits.features();
    Scalar total_tokens = static_cast<Scalar>(batch * seq_len);

    // --- Gradien logits: softmax(logits) - one_hot(target), tanpa materialisasi one-hot ---
    Tensor3D grad_logits(batch, seq_len, vocab);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            Scalar max_logit = -std::numeric_limits<Scalar>::infinity();
            for (size_t v = 0; v < vocab; ++v) max_logit = std::max(max_logit, logits.at(b, s, v));

            Scalar sum_exp = 0.0f;
            for (size_t v = 0; v < vocab; ++v) sum_exp += std::exp(logits.at(b, s, v) - max_logit);

            size_t target = static_cast<size_t>(std::round(target_ids.at(b, s)));
            for (size_t v = 0; v < vocab; ++v) {
                Scalar prob = std::exp(logits.at(b, s, v) - max_logit) / sum_exp;
                grad_logits.at(b, s, v) = prob / total_tokens;
            }
            grad_logits.at(b, s, target) -= 1.0f / total_tokens;
        }
    }

    // --- Backward lewat output_projection_ ---
    Scalar batch_scalar = static_cast<Scalar>(batch);
    Matrix proj_t = output_projection_.transpose();
    grad_output_projection_ = Matrix(embed_dim_, vocab_size_, 0.0f);
    Tensor3D grad_last_hidden(batch, seq_len, embed_dim_);

    for (size_t b = 0; b < batch; ++b) {
        Matrix hidden_b = last_hidden_cache_.slice_batch(b);  // seq x embed_dim
        Matrix gl_b = grad_logits.slice_batch(b);              // seq x vocab
        grad_output_projection_.add_inplace(hidden_b.transpose() * gl_b);
        grad_last_hidden.set_batch(b, gl_b * proj_t);
    }
    grad_output_projection_ = grad_output_projection_.scale(1.0f / batch_scalar);

    // --- Backward lewat final_norm_ dan tumpukan TransformerBlock (urutan terbalik) ---
    Tensor3D grad = final_norm_.backward(grad_last_hidden);
    for (size_t i = blocks_.size(); i-- > 0;) {
        grad = blocks_[i].backward(grad);
    }

    // --- Backward lewat embedding ---
    embedding_.backward(grad, token_ids);
}

void SequenceModel::update(Scalar learning_rate) {
    embedding_.update(learning_rate);
    for (auto& block : blocks_) {
        block.update(learning_rate);
    }
    final_norm_.update(learning_rate);
    output_projection_.sub_inplace(grad_output_projection_.scale(learning_rate));
}