// ============================================================
// src/sequence_model.cpp  (DIPERBARUI)
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
      final_norm_(embed_dim)
{
    blocks_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        blocks_.emplace_back(embed_dim, num_heads, ff_hidden_dim, causal_mask,
                              seed + 100 * static_cast<unsigned>(i + 1));
    }
    // Tidak ada lagi output_projection_ terpisah — weight tying dengan embedding_table_
}

Tensor3D SequenceModel::forward(const Matrix& token_ids) {
    Tensor3D x = embedding_.forward(token_ids);
    for (auto& block : blocks_) {
        x = block.forward(x);
    }
    Tensor3D normed = final_norm_.forward(x);
    last_hidden_cache_ = normed;

    // Weight tying: proyeksi embed_dim -> vocab pakai transpose embedding_table_
    // (vocab_size x embed_dim) -> (embed_dim x vocab_size), bukan matriks belajar terpisah
    Matrix tied_weight = embedding_.embedding_table().transpose();
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

    // --- Satu pass softmax: hitung loss DAN grad_logits sekaligus ---
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

    // --- Backward lewat proyeksi (tied weight = embedding_table_ langsung, tanpa transpose lagi) ---
    Scalar batch_scalar = static_cast<Scalar>(batch);
    Matrix embed_table = embedding_.embedding_table(); // vocab_size x embed_dim
    Matrix grad_tied_weight(embed_dim_, vocab_size_, 0.0f); // embed_dim x vocab_size
    Tensor3D grad_last_hidden(batch, seq_len, embed_dim_);

    for (size_t b = 0; b < batch; ++b) {
        Matrix hidden_b = last_hidden_cache_.slice_batch(b); // seq x embed_dim
        Matrix gl_b = grad_logits.slice_batch(b);             // seq x vocab
        grad_tied_weight.add_inplace(hidden_b.transpose() * gl_b);
        grad_last_hidden.set_batch(b, gl_b * embed_table);     // seq x vocab * vocab x embed_dim
    }
    grad_tied_weight = grad_tied_weight.scale(1.0f / batch_scalar);
    Matrix grad_tied_weight_t = grad_tied_weight.transpose(); // vocab_size x embed_dim

    // --- Backward lewat final_norm_ dan tumpukan TransformerBlock ---
    Tensor3D grad = final_norm_.backward(grad_last_hidden);
    for (size_t i = blocks_.size(); i-- > 0;) {
        grad = blocks_[i].backward(grad);
    }

    // --- Backward lewat embedding: reset+isi gradien lookup, lalu tambah kontribusi tied weight ---
    embedding_.backward(grad, token_ids);
    embedding_.accumulate_external_grad(grad_tied_weight_t);

    return loss;
}

void SequenceModel::update(Scalar learning_rate) {
    embedding_.update(learning_rate);
    for (auto& block : blocks_) {
        block.update(learning_rate);
    }
    final_norm_.update(learning_rate);
    // Tidak ada update output_projection_ terpisah — sudah tercakup di embedding_.update()
}