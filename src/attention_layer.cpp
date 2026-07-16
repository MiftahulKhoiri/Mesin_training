// ============================================================
// src/attention_layer.cpp  (DIPERBARUI)
// ============================================================
#include "attention_layer.h"
#include <cmath>
#include <stdexcept>

Matrix AttentionLayer::init_projection_weight(size_t dim, unsigned seed) {
    Scalar stddev = std::sqrt(1.0f / static_cast<Scalar>(dim));
    return Matrix::random_normal(dim, dim, 0.0f, stddev, seed);
}

Matrix AttentionLayer::extract_head_cols(const Matrix& full, size_t head_idx, size_t head_dim) {
    Matrix result(full.rows(), head_dim);
    size_t offset = head_idx * head_dim;
    for (size_t r = 0; r < full.rows(); ++r)
        for (size_t c = 0; c < head_dim; ++c)
            result.at(r, c) = full.at(r, offset + c);
    return result;
}

void AttentionLayer::write_head_cols(Matrix& full, size_t head_idx, size_t head_dim, const Matrix& head_data) {
    size_t offset = head_idx * head_dim;
    for (size_t r = 0; r < head_data.rows(); ++r)
        for (size_t c = 0; c < head_dim; ++c)
            full.at(r, offset + c) = head_data.at(r, c);
}

AttentionLayer::AttentionLayer(size_t embed_dim, size_t num_heads, bool causal_mask, unsigned seed)
    : embed_dim_(embed_dim), num_heads_(num_heads), causal_mask_(causal_mask)
{
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("AttentionLayer: embed_dim harus habis dibagi num_heads");
    }
    head_dim_ = embed_dim / num_heads;

    W_q_ = init_projection_weight(embed_dim, seed);
    W_k_ = init_projection_weight(embed_dim, seed + 1);
    W_v_ = init_projection_weight(embed_dim, seed + 2);
    W_o_ = init_projection_weight(embed_dim, seed + 3);

    grad_W_q_ = Matrix(embed_dim, embed_dim, 0.0f);
    grad_W_k_ = Matrix(embed_dim, embed_dim, 0.0f);
    grad_W_v_ = Matrix(embed_dim, embed_dim, 0.0f);
    grad_W_o_ = Matrix(embed_dim, embed_dim, 0.0f);
}

Matrix AttentionLayer::compute_attn_weights(const Matrix& Qh, const Matrix& Kh, size_t seq_len) const {
    Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(head_dim_));
    Matrix scores = (Qh * Kh.transpose()).scale(scale);

    if (causal_mask_) {
        for (size_t i = 0; i < seq_len; ++i)
            for (size_t j = i + 1; j < seq_len; ++j)
                scores.at(i, j) = -1e9f;
    }
    return Activation::forward(scores, ActivationType::Softmax);
}

Tensor3D AttentionLayer::forward(const Tensor3D& input) {
    input_cache_ = input;
    Q_cache_ = input.batched_matmul(W_q_);
    K_cache_ = input.batched_matmul(W_k_);
    V_cache_ = input.batched_matmul(W_v_);

    size_t batch = input.batch();
    size_t seq_len = input.seq_len();
    concat_output_cache_ = Tensor3D(batch, seq_len, embed_dim_);

    for (size_t b = 0; b < batch; ++b) {
        Matrix Qb = Q_cache_.slice_batch(b);
        Matrix Kb = K_cache_.slice_batch(b);
        Matrix Vb = V_cache_.slice_batch(b);
        Matrix concat(seq_len, embed_dim_);

        for (size_t h = 0; h < num_heads_; ++h) {
            Matrix Qh = extract_head_cols(Qb, h, head_dim_);
            Matrix Kh = extract_head_cols(Kb, h, head_dim_);
            Matrix Vh = extract_head_cols(Vb, h, head_dim_);

            Matrix weights = compute_attn_weights(Qh, Kh, seq_len); // dipakai lokal, TIDAK disimpan
            Matrix head_out = weights * Vh;
            write_head_cols(concat, h, head_dim_, head_out);
        }
        concat_output_cache_.set_batch(b, concat);
    }

    return concat_output_cache_.batched_matmul(W_o_);
}

Tensor3D AttentionLayer::backward(const Tensor3D& grad_output) {
    size_t batch = grad_output.batch();
    size_t seq_len = grad_output.seq_len();
    Scalar batch_scalar = static_cast<Scalar>(batch);
    Scalar scale = 1.0f / std::sqrt(static_cast<Scalar>(head_dim_));

    Matrix W_o_t = W_o_.transpose();
    grad_W_o_ = Matrix(embed_dim_, embed_dim_, 0.0f);
    Tensor3D grad_concat(batch, seq_len, embed_dim_);

    for (size_t b = 0; b < batch; ++b) {
        Matrix go_b = grad_output.slice_batch(b);
        Matrix concat_b = concat_output_cache_.slice_batch(b);
        grad_W_o_.add_inplace(concat_b.transpose() * go_b);
        grad_concat.set_batch(b, go_b * W_o_t);
    }
    grad_W_o_ = grad_W_o_.scale(1.0f / batch_scalar);

    grad_W_q_ = Matrix(embed_dim_, embed_dim_, 0.0f);
    grad_W_k_ = Matrix(embed_dim_, embed_dim_, 0.0f);
    grad_W_v_ = Matrix(embed_dim_, embed_dim_, 0.0f);
    Tensor3D grad_input(batch, seq_len, embed_dim_);

    for (size_t b = 0; b < batch; ++b) {
        Matrix Qb = Q_cache_.slice_batch(b);
        Matrix Kb = K_cache_.slice_batch(b);
        Matrix Vb = V_cache_.slice_batch(b);
        Matrix input_b = input_cache_.slice_batch(b);
        Matrix grad_concat_b = grad_concat.slice_batch(b);

        Matrix dQb(seq_len, embed_dim_, 0.0f);
        Matrix dKb(seq_len, embed_dim_, 0.0f);
        Matrix dVb(seq_len, embed_dim_, 0.0f);

        for (size_t h = 0; h < num_heads_; ++h) {
            Matrix Qh = extract_head_cols(Qb, h, head_dim_);
            Matrix Kh = extract_head_cols(Kb, h, head_dim_);
            Matrix Vh = extract_head_cols(Vb, h, head_dim_);

            // Rekomputasi weights (bukan baca dari cache) — trade-off memori vs compute
            Matrix weights = compute_attn_weights(Qh, Kh, seq_len);

            Matrix d_head_out = extract_head_cols(grad_concat_b, h, head_dim_);

            Matrix dWeights = d_head_out * Vh.transpose();
            Matrix dVh = weights.transpose() * d_head_out;

            Matrix dScores(seq_len, seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                Scalar dot = 0.0f;
                for (size_t j = 0; j < seq_len; ++j) dot += dWeights.at(i, j) * weights.at(i, j);
                for (size_t j = 0; j < seq_len; ++j)
                    dScores.at(i, j) = weights.at(i, j) * (dWeights.at(i, j) - dot);
            }
            dScores = dScores.scale(scale);

            Matrix dQh = dScores * Kh;
            Matrix dKh = dScores.transpose() * Qh;

            write_head_cols(dQb, h, head_dim_, dQh);
            write_head_cols(dKb, h, head_dim_, dKh);
            write_head_cols(dVb, h, head_dim_, dVh);
        }

        grad_W_q_.add_inplace(input_b.transpose() * dQb);
        grad_W_k_.add_inplace(input_b.transpose() * dKb);
        grad_W_v_.add_inplace(input_b.transpose() * dVb);

        Matrix grad_input_b = dQb * W_q_.transpose() + dKb * W_k_.transpose() + dVb * W_v_.transpose();
        grad_input.set_batch(b, grad_input_b);
    }

    grad_W_q_ = grad_W_q_.scale(1.0f / batch_scalar);
    grad_W_k_ = grad_W_k_.scale(1.0f / batch_scalar);
    grad_W_v_ = grad_W_v_.scale(1.0f / batch_scalar);

    return grad_input;
}

void AttentionLayer::update(Scalar learning_rate) {
    W_q_.sub_inplace(grad_W_q_.scale(learning_rate));
    W_k_.sub_inplace(grad_W_k_.scale(learning_rate));
    W_v_.sub_inplace(grad_W_v_.scale(learning_rate));
    W_o_.sub_inplace(grad_W_o_.scale(learning_rate));
}