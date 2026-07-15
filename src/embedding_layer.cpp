// ============================================================
// src/embedding_layer.cpp  (FILE BARU)
// ============================================================
#include "embedding_layer.h"
#include <cmath>
#include <stdexcept>

EmbeddingLayer::EmbeddingLayer(size_t vocab_size, size_t embed_dim, size_t max_seq_len, unsigned seed)
    : vocab_size_(vocab_size), embed_dim_(embed_dim), max_seq_len_(max_seq_len),
      embedding_table_(vocab_size, embed_dim),
      grad_embedding_table_(vocab_size, embed_dim, 0.0f)
{
    // Skala inisialisasi kecil (0.02), umum dipakai untuk embedding di model transformer —
    // membantu stabilitas awal training dibanding He/Xavier standar
    embedding_table_ = Matrix::random_normal(vocab_size, embed_dim, 0.0f, 0.02f, seed);
    positional_encoding_ = compute_sinusoidal_positional_encoding(max_seq_len, embed_dim);
}

Matrix EmbeddingLayer::compute_sinusoidal_positional_encoding(size_t max_seq_len, size_t embed_dim) {
    Matrix pe(max_seq_len, embed_dim);
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t i = 0; i < embed_dim; ++i) {
            Scalar exponent = static_cast<Scalar>(2 * (i / 2)) / static_cast<Scalar>(embed_dim);
            Scalar div_term = std::pow(10000.0f, exponent);
            Scalar angle = static_cast<Scalar>(pos) / div_term;
            pe.at(pos, i) = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
    }
    return pe;
}

Tensor3D EmbeddingLayer::forward(const Matrix& token_ids) {
    size_t batch = token_ids.rows();
    size_t seq_len = token_ids.cols();

    if (seq_len > max_seq_len_) {
        throw std::invalid_argument("EmbeddingLayer::forward: seq_len melebihi max_seq_len");
    }

    Tensor3D output(batch, seq_len, embed_dim_);
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t token_id = static_cast<size_t>(std::round(token_ids.at(b, s)));
            if (token_id >= vocab_size_) {
                throw std::out_of_range("EmbeddingLayer::forward: token_id di luar jangkauan vocab_size");
            }
            for (size_t f = 0; f < embed_dim_; ++f) {
                output.at(b, s, f) = embedding_table_.at(token_id, f) + positional_encoding_.at(s, f);
            }
        }
    }
    return output;
}

void EmbeddingLayer::backward(const Tensor3D& grad_output, const Matrix& token_ids) {
    size_t batch = token_ids.rows();
    size_t seq_len = token_ids.cols();

    grad_embedding_table_.fill(0.0f);

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t token_id = static_cast<size_t>(std::round(token_ids.at(b, s)));
            // Scatter-add: kalau token yang sama muncul berkali-kali (di sample lain
            // atau posisi lain), gradiennya diakumulasi, bukan ditimpa
            for (size_t f = 0; f < embed_dim_; ++f) {
                grad_embedding_table_.at(token_id, f) += grad_output.at(b, s, f);
            }
        }
    }
    // positional_encoding_ tidak diupdate — fixed/non-learnable
}

void EmbeddingLayer::update(Scalar learning_rate) {
    embedding_table_.sub_inplace(grad_embedding_table_.scale(learning_rate));
}