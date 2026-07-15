// ============================================================
// src/matrix_ops.cpp
// ============================================================
#include "matrix_ops.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

// Ukuran blok disetel untuk L1 cache umum (~32KB). Sesuaikan lewat benchmark
// jika target device (mis. Raspberry Pi) punya cache berbeda.
static constexpr size_t BLOCK_SIZE = 64;

Matrix::Matrix(size_t rows, size_t cols, Scalar init_val)
    : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

Scalar& Matrix::at(size_t r, size_t c) {
    if (r >= rows_ || c >= cols_) throw std::out_of_range("Matrix::at index out of range");
    return data_[idx(r, c)];
}

const Scalar& Matrix::at(size_t r, size_t c) const {
    if (r >= rows_ || c >= cols_) throw std::out_of_range("Matrix::at index out of range");
    return data_[idx(r, c)];
}

void Matrix::check_same_shape(const Matrix& other, const char* op) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(std::string("Matrix shape mismatch on ") + op);
    }
}

Matrix Matrix::operator+(const Matrix& other) const {
    check_same_shape(other, "operator+");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) result.data_[i] = data_[i] + other.data_[i];
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    check_same_shape(other, "operator-");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) result.data_[i] = data_[i] - other.data_[i];
    return result;
}

// --- Matmul: cache-blocked naif, atau OpenBLAS jika tersedia ---
Matrix Matrix::matmul_blocked(const Matrix& a, const Matrix& b) {
    Matrix result(a.rows_, b.cols_, 0.0f);
    const size_t N = a.rows_, K = a.cols_, M = b.cols_;

    for (size_t ii = 0; ii < N; ii += BLOCK_SIZE) {
        size_t i_max = std::min(ii + BLOCK_SIZE, N);
        for (size_t kk = 0; kk < K; kk += BLOCK_SIZE) {
            size_t k_max = std::min(kk + BLOCK_SIZE, K);
            for (size_t jj = 0; jj < M; jj += BLOCK_SIZE) {
                size_t j_max = std::min(jj + BLOCK_SIZE, M);

                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        Scalar a_ik = a.at(i, k);
                        if (a_ik == 0.0f) continue;
                        const Scalar* b_row = &b.data_[b.idx(k, jj)];
                        Scalar* r_row = &result.data_[result.idx(i, jj)];
                        size_t len = j_max - jj;
                        for (size_t j = 0; j < len; ++j) {
                            r_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix multiply: inner dimensions mismatch");
    }

#ifdef USE_OPENBLAS
    Matrix result(rows_, other.cols_, 0.0f);
    // sgemm: C = alpha * A*B + beta * C, row-major
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)rows_, (int)other.cols_, (int)cols_,
                1.0f, data_.data(), (int)cols_,
                other.data_.data(), (int)other.cols_,
                0.0f, result.data(), (int)other.cols_);
    return result;
#else
    return matmul_blocked(*this, other);
#endif
}

Matrix Matrix::hadamard(const Matrix& other) const {
    check_same_shape(other, "hadamard");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) result.data_[i] = data_[i] * other.data_[i];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    // blocked transpose supaya lebih cache-friendly untuk matriks besar
    for (size_t ii = 0; ii < rows_; ii += BLOCK_SIZE) {
        size_t i_max = std::min(ii + BLOCK_SIZE, rows_);
        for (size_t jj = 0; jj < cols_; jj += BLOCK_SIZE) {
            size_t j_max = std::min(jj + BLOCK_SIZE, cols_);
            for (size_t i = ii; i < i_max; ++i)
                for (size_t j = jj; j < j_max; ++j)
                    result.at(j, i) = at(i, j);
        }
    }
    return result;
}

Matrix Matrix::scale(Scalar scalar) const {
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) result.data_[i] = data_[i] * scalar;
    return result;
}

void Matrix::add_inplace(const Matrix& other) {
    check_same_shape(other, "add_inplace");
    for (size_t i = 0; i < data_.size(); ++i) data_[i] += other.data_[i];
}

void Matrix::sub_inplace(const Matrix& other) {
    check_same_shape(other, "sub_inplace");
    for (size_t i = 0; i < data_.size(); ++i) data_[i] -= other.data_[i];
}

Matrix Matrix::add_row_vector(const Matrix& row_vec) const {
    if (row_vec.rows_ != 1 || row_vec.cols_ != cols_) {
        throw std::invalid_argument("add_row_vector: shape mismatch, expected 1 x cols");
    }
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            result.at(i, j) = at(i, j) + row_vec.at(0, j);
    return result;
}

Matrix Matrix::sum_rows() const {
    Matrix result(1, cols_, 0.0f);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            result.at(0, j) += at(i, j);
    return result;
}

Scalar Matrix::sum_all() const {
    Scalar total = 0.0f;
    for (Scalar v : data_) total += v;
    return total;
}

Matrix Matrix::random_uniform(size_t rows, size_t cols, Scalar low, Scalar high, unsigned seed) {
    Matrix result(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(low, high);
    for (size_t i = 0; i < rows * cols; ++i) result.data_[i] = dist(gen);
    return result;
}

Matrix Matrix::random_normal(size_t rows, size_t cols, Scalar mean, Scalar stddev, unsigned seed) {
    Matrix result(rows, cols);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < rows * cols; ++i) result.data_[i] = dist(gen);
    return result;
}

Matrix Matrix::identity(size_t n) {
    Matrix result(n, n, 0.0f);
    for (size_t i = 0; i < n; ++i) result.at(i, i) = 1.0f;
    return result;
}

void Matrix::fill(Scalar val) {
    std::fill(data_.begin(), data_.end(), val);
}

void Matrix::print() const {
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << std::fixed << std::setprecision(4) << at(i, j) << " ";
        }
        std::cout << "\n";
    }
}