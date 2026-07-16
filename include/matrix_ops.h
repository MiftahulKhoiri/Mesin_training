// ============================================================
// include/matrix_ops.h
// ============================================================
#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <cstring>
#include <ostream>
#include <istream>

#ifdef USE_OPENBLAS
extern "C" {
#include <cblas.h>
}
#endif

using Scalar = float; // ganti ke double di satu tempat ini kalau perlu presisi lebih tinggi

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, Scalar init_val = 0.0f);

    Scalar& at(size_t r, size_t c);
    const Scalar& at(size_t r, size_t c) const;

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Akses pointer mentah untuk BLAS / SIMD
    Scalar* data() { return data_.data(); }
    const Scalar* data() const { return data_.data(); }

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;      // matrix multiply (blocked / BLAS)
    Matrix hadamard(const Matrix& other) const;
    Matrix transpose() const;
    Matrix scale(Scalar scalar) const;

    void add_inplace(const Matrix& other);
    void sub_inplace(const Matrix& other);
    void save(std::ostream& os) const;
static Matrix load(std::istream& is);
    Matrix add_row_vector(const Matrix& row_vec) const;

    Matrix sum_rows() const;
    Scalar sum_all() const;

    static Matrix random_uniform(size_t rows, size_t cols, Scalar low, Scalar high, unsigned seed);
    static Matrix random_normal(size_t rows, size_t cols, Scalar mean, Scalar stddev, unsigned seed);
    static Matrix identity(size_t n);

    void fill(Scalar val);
    void print() const;

private:
    size_t rows_, cols_;
    std::vector<Scalar> data_; // row-major, contiguous (siap untuk BLAS/SIMD)

    inline size_t idx(size_t r, size_t c) const { return r * cols_ + c; }
    void check_same_shape(const Matrix& other, const char* op) const;

    // Matmul naif dengan cache-blocking (dipakai jika USE_OPENBLAS tidak aktif)
    static Matrix matmul_blocked(const Matrix& a, const Matrix& b);
};