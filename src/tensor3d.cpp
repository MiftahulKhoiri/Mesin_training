// ============================================================
// src/tensor3d.cpp  (FILE BARU)
// ============================================================
#include "tensor3d.h"

Tensor3D::Tensor3D(size_t batch, size_t seq_len, size_t features, Scalar init_val)
    : batch_(batch), seq_len_(seq_len), features_(features),
      data_(batch * seq_len * features, init_val) {}

Scalar& Tensor3D::at(size_t b, size_t s, size_t f) {
    if (b >= batch_ || s >= seq_len_ || f >= features_)
        throw std::out_of_range("Tensor3D::at index out of range");
    return data_[idx(b, s, f)];
}

const Scalar& Tensor3D::at(size_t b, size_t s, size_t f) const {
    if (b >= batch_ || s >= seq_len_ || f >= features_)
        throw std::out_of_range("Tensor3D::at index out of range");
    return data_[idx(b, s, f)];
}

void Tensor3D::check_same_shape(const Tensor3D& other, const char* op) const {
    if (batch_ != other.batch_ || seq_len_ != other.seq_len_ || features_ != other.features_) {
        throw std::invalid_argument(std::string("Tensor3D shape mismatch on ") + op);
    }
}

Matrix Tensor3D::slice_batch(size_t b) const {
    if (b >= batch_) throw std::out_of_range("Tensor3D::slice_batch: index out of range");
    Matrix result(seq_len_, features_);
    // Slab (seq_len x features) untuk batch b bersifat contiguous di data_,
    // jadi ini satu memcpy logis, bukan loop acak-akses.
    const Scalar* src = &data_[b * seq_len_ * features_];
    std::copy(src, src + seq_len_ * features_, result.data());
    return result;
}

void Tensor3D::set_batch(size_t b, const Matrix& m) {
    if (b >= batch_) throw std::out_of_range("Tensor3D::set_batch: index out of range");
    if (m.rows() != seq_len_ || m.cols() != features_)
        throw std::invalid_argument("Tensor3D::set_batch: shape Matrix tidak cocok (seq_len x features)");
    Scalar* dst = &data_[b * seq_len_ * features_];
    std::copy(m.data(), m.data() + seq_len_ * features_, dst);
}

Matrix Tensor3D::slice_timestep(size_t s) const {
    if (s >= seq_len_) throw std::out_of_range("Tensor3D::slice_timestep: index out of range");
    Matrix result(batch_, features_);
    // Timestep TIDAK contiguous lintas batch (strided), jadi memang perlu loop
    for (size_t b = 0; b < batch_; ++b) {
        for (size_t f = 0; f < features_; ++f) {
            result.at(b, f) = at(b, s, f);
        }
    }
    return result;
}

void Tensor3D::set_timestep(size_t s, const Matrix& m) {
    if (s >= seq_len_) throw std::out_of_range("Tensor3D::set_timestep: index out of range");
    if (m.rows() != batch_ || m.cols() != features_)
        throw std::invalid_argument("Tensor3D::set_timestep: shape Matrix tidak cocok (batch x features)");
    for (size_t b = 0; b < batch_; ++b) {
        for (size_t f = 0; f < features_; ++f) {
            at(b, s, f) = m.at(b, f);
        }
    }
}

Tensor3D Tensor3D::batched_matmul(const Matrix& weight) const {
    if (features_ != weight.rows()) {
        throw std::invalid_argument("Tensor3D::batched_matmul: features tidak cocok dengan weight.rows()");
    }
    Tensor3D result(batch_, seq_len_, weight.cols());
    for (size_t b = 0; b < batch_; ++b) {
        // Setiap batch: (seq_len x features) * (features x out_features), pakai
        // Matrix::operator* yang sudah cache-blocked — tidak reimplementasi manual
        Matrix batch_slice = slice_batch(b);
        Matrix projected = batch_slice * weight; // seq_len x out_features
        result.set_batch(b, projected);
    }
    return result;
}

Tensor3D Tensor3D::add_bias(const Matrix& bias_row) const {
    if (bias_row.rows() != 1 || bias_row.cols() != features_) {
        throw std::invalid_argument("Tensor3D::add_bias: bias harus berbentuk 1 x features");
    }
    Tensor3D result(batch_, seq_len_, features_);
    for (size_t b = 0; b < batch_; ++b) {
        for (size_t s = 0; s < seq_len_; ++s) {
            for (size_t f = 0; f < features_; ++f) {
                result.at(b, s, f) = at(b, s, f) + bias_row.at(0, f);
            }
        }
    }
    return result;
}

Tensor3D Tensor3D::operator+(const Tensor3D& other) const {
    check_same_shape(other, "operator+");
    Tensor3D result(batch_, seq_len_, features_);
    for (size_t i = 0; i < data_.size(); ++i) result.data_[i] = data_[i] + other.data_[i];
    return result;
}

Tensor3D Tensor3D::zeros(size_t batch, size_t seq_len, size_t features) {
    return Tensor3D(batch, seq_len, features, 0.0f);
}

void Tensor3D::fill(Scalar val) {
    std::fill(data_.begin(), data_.end(), val);
}