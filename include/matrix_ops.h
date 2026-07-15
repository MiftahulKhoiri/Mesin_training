// ============================================================
// include/matrix_ops.h
// ============================================================
#pragma once
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(size_t rows, size_t cols, double init_val = 0.0);

    // Akses elemen
    double& at(size_t r, size_t c);
    const double& at(size_t r, size_t c) const;

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Operasi dasar
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;      // matrix multiply
    Matrix hadamard(const Matrix& other) const;        // element-wise multiply
    Matrix transpose() const;
    Matrix scale(double scalar) const;

    // Operasi in-place (hindari copy berlebih saat training)
    void add_inplace(const Matrix& other);
    void sub_inplace(const Matrix& other);

    // Broadcasting: tambah vector bias ke setiap baris
    Matrix add_row_vector(const Matrix& row_vec) const;

    // Reduksi
    Matrix sum_rows() const;   // hasil: 1 x cols
    double sum_all() const;

    // Utilitas
    static Matrix random_uniform(size_t rows, size_t cols, double low, double high, unsigned seed);
    static Matrix random_normal(size_t rows, size_t cols, double mean, double stddev, unsigned seed);
    static Matrix identity(size_t n);

    void fill(double val);
    void print() const;

private:
    size_t rows_, cols_;
    std::vector<double> data_; // row-major

    inline size_t idx(size_t r, size_t c) const { return r * cols_ + c; }
    void check_same_shape(const Matrix& other, const char* op) const;
};