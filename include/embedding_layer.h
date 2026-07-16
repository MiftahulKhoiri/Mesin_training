// ============================================================
// include/embedding_layer.h  (LENGKAP)
// ============================================================
#pragma once
#include <ostream>
#include <istream>
#include "matrix_ops.h"
#include "tensor3d.h"

class EmbeddingLayer {
public:
    EmbeddingLayer(size_t vocab_size, size_t embed_dim, size_t max_seq_len, unsigned seed = 42);

    Tensor3D forward(const Matrix& token_ids);
    void backward(const Tensor3D& grad_output, const Matrix& token_ids);
    void update(Scalar learning_rate);

    size_t vocab_size() const { return vocab_size_; }
    size_t embed_dim() const { return embed_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }

    Matrix& embedding_table() { return embedding_table_; }
    const Matrix& embedding_table_grad() const { return grad_embedding_table_; }

    // Tambahkan gradien eksternal ke grad_embedding_table_ (dipakai untuk weight tying).
    // WAJIB dipanggil SETELAH backward() (yang reset+isi gradien lookup), SEBELUM update().
    void accumulate_external_grad(const Matrix& grad); // grad: vocab_size x embed_dim

    // Checkpoint I/O — hanya embedding_table_ (positional_encoding_ deterministik, dihitung ulang)
    void save(std::ostream& os) const;
    void load_weights(std::istream& is);

private:
    size_t vocab_size_, embed_dim_, max_seq_len_;

    Matrix embedding_table_;
    Matrix positional_encoding_;
    Matrix grad_embedding_table_;

    static Matrix compute_sinusoidal_positional_encoding(size_t max_seq_len, size_t embed_dim);
};