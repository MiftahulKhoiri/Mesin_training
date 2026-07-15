// ============================================================
// include/tensor3d.h  (FILE BARU)
// ============================================================
#pragma once
#include <vector>
#include <stdexcept>
#include "matrix_ops.h"

// Layout: batch x seq_len x features, row-major, contiguous.
// Penting: setiap "batch slab" (seq_len x features) tersimpan contiguous,
// jadi bisa disalin ke/dari Matrix 2D untuk dipakai operasi yang sudah
// dioptimasi (matmul blocked, dsb.) tanpa reimplementasi dari nol.
class Tensor3D {
public:
    Tensor3D() : batch_(0), seq_len_(0), features_(0) {}
    Tensor3D(size_t batch, size_t seq_len, size_t features, Scalar init_val = 0.0f);

    Scalar& at(size_t b, size_t s, size_t f);
    const Scalar& at(size_t b, size_t s, size_t f) const;

    size_t batch() const { return batch_; }
    size_t seq_len() const { return seq_len_; }
    size_t features() const { return features_; }

    Scalar* data() { return data_.data(); }
    const Scalar* data() const { return data_.data(); }

    // Ambil satu elemen batch sebagai Matrix (seq_len x features), dipakai
    // untuk operasi per-sample seperti attention (Q*K^T, dll.)
    Matrix slice_batch(size_t b) const;
    void set_batch(size_t b, const Matrix& m);

    // Ambil satu timestep lintas semua batch sebagai Matrix (batch x features),
    // dipakai untuk layer yang beroperasi per-timestep (mis. RNN cell)
    Matrix slice_timestep(size_t s) const;
    void set_timestep(size_t s, const Matrix& m);

    // Proyeksi linear bersama di semua timestep & batch: (b,s,:) * weight -> (b,s,:out)
    // weight: features x out_features, sama untuk semua batch/timestep (khas Q/K/V/FFN projection)
    Tensor3D batched_matmul(const Matrix& weight) const;

    // Broadcast tambah bias (1 x features) ke setiap (b, s)
    Tensor3D add_bias(const Matrix& bias_row) const;

    Tensor3D operator+(const Tensor3D& other) const;

    static Tensor3D zeros(size_t batch, size_t seq_len, size_t features);
    void fill(Scalar val);

private:
    size_t batch_, seq_len_, features_;
    std::vector<Scalar> data_;

    inline size_t idx(size_t b, size_t s, size_t f) const {
        return (b * seq_len_ + s) * features_ + f;
    }
    void check_same_shape(const Tensor3D& other, const char* op) const;
};