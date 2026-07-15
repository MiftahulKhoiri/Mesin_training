// ============================================================
// include/embedding_layer.h  (FILE BARU)
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "tensor3d.h"

class EmbeddingLayer {
public:
    EmbeddingLayer(size_t vocab_size, size_t embed_dim, size_t max_seq_len, unsigned seed = 42);

    // token_ids: Matrix batch x seq_len, tiap elemen adalah ID token (disimpan
    // sebagai Scalar/float, dibulatkan ke integer saat lookup)
    // Output: Tensor3D batch x seq_len x embed_dim = embedding lookup + positional encoding
    Tensor3D forward(const Matrix& token_ids);

    // grad_output: gradien dari layer berikutnya, shape sama dengan output forward().
    // token_ids: token ID yang sama dipakai saat forward (dibutuhkan untuk scatter-add
    // gradien ke baris embedding_table_ yang tepat).
    // Positional encoding tidak punya gradien (fixed/non-learnable).
    void backward(const Tensor3D& grad_output, const Matrix& token_ids);

    void update(Scalar learning_rate);

    size_t vocab_size() const { return vocab_size_; }
    size_t embed_dim() const { return embed_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }

    Matrix& embedding_table() { return embedding_table_; }
    const Matrix& embedding_table_grad() const { return grad_embedding_table_; }

private:
    size_t vocab_size_, embed_dim_, max_seq_len_;

    Matrix embedding_table_;       // vocab_size x embed_dim (learnable)
    Matrix positional_encoding_;   // max_seq_len x embed_dim (fixed, precomputed)
    Matrix grad_embedding_table_;  // vocab_size x embed_dim (akumulasi gradien)

    static Matrix compute_sinusoidal_positional_encoding(size_t max_seq_len, size_t embed_dim);
};