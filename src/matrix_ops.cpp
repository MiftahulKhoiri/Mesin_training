// ============================================================
// src/matrix_ops.cpp
// ============================================================
#include "matrix_ops.h"
#include <iostream>
#include <iomanip>

Matrix::Matrix(size_t rows, size_t cols, double init_val)
    : rows_(rows), cols_(cols), data_(rows * cols, init_val) {}

double& Matrix::at(size_t r, size_t c) {
    if (r >= rows_ || c >= cols_) throw std::out_of_range("Matrix::at index out of range");
    return data_[idx(r, c)];
}

const double& Matrix::at(size_t r, size_t c) const {
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

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix multiply: inner dimensions mismatch");
    }
    Matrix result(rows_, other.cols_, 0.0);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t k = 0; k < cols_; ++k) {
            double a_ik = at(i, k);
            if (a_ik == 0.0) continue; // sedikit optimasi untuk matriks jarang
            for (size_t j = 0; j < other.cols_; ++j) {
                result.at(i, j) += a_ik * other.at(k, j);
            }
        }
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    check_same_shape(other, "hadamard");
    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) result.data_[i] = data_[i] * other.data_[i];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            result.at(j, i) = at(i, j);
    return result;
}

Matrix Matrix::scale(double scalar) const {
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
    Matrix result(1, cols_, 0.0);
    for (size_t i = 0; i < rows_; ++i)
        for (size_t j = 0; j < cols_; ++j)
            result.at(0, j) += at(i, j);
    return result;
}

double Matrix::sum_all() const {
    double total = 0.0;
    for (double v : data_) total += v;
    return total;
}

Matrix Matrix::random_uniform(size_t rows, size_t cols, double low, double high, unsigned seed) {
    Matrix result(rows, cols);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(low, high);
    for (size_t i = 0; i < rows * cols; ++i) result.data_[i] = dist(gen);
    return result;
}

Matrix Matrix::random_normal(size_t rows, size_t cols, double mean, double stddev, unsigned seed) {
    Matrix result(rows, cols);
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < rows * cols; ++i) result.data_[i] = dist(gen);
    return result;
}

Matrix Matrix::identity(size_t n) {
    Matrix result(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) result.at(i, i) = 1.0;
    return result;
}

void Matrix::fill(double val) {
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