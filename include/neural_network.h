// ============================================================
// include/neural_network.h
// ============================================================
#pragma once
#include <vector>
#include <memory>
#include "matrix_ops.h"
#include "activations.h"
#include "losses.h"
#include "dense_layer.h"

class NeuralNetwork {
public:
    explicit NeuralNetwork(LossType loss_type);

    // Tambah layer baru; input_size layer pertama harus ditentukan eksplisit,
    // layer selanjutnya otomatis dicek harus sama dengan output_size layer sebelumnya.
    void add_layer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed = 42);

    // Forward penuh melalui semua layer
    Matrix forward(const Matrix& input);

    // Hitung nilai loss (untuk monitoring/logging), tidak mengubah state
    Scalar compute_loss(const Matrix& predictions, const Matrix& targets) const;

    // Backward penuh: hitung gradien loss terhadap output, lalu propagasi mundur
    // melalui semua layer. Gradien tiap layer tersimpan internal (siap update()).
    void backward(const Matrix& predictions, const Matrix& targets);

    // Update semua layer dengan gradien yang sudah dihitung backward()
    void update(Scalar learning_rate);

    size_t num_layers() const { return layers_.size(); }
    DenseLayer& layer(size_t idx) { return layers_.at(idx); }
    const DenseLayer& layer(size_t idx) const { return layers_.at(idx); }

private:
    std::vector<DenseLayer> layers_;
    LossType loss_type_;
    size_t next_input_size_; // validasi kecocokan shape antar layer

    // True jika pasangan (activation, loss_type_) sudah menggabungkan turunan
    // aktivasi ke dalam loss derivative (saat ini hanya Softmax + SoftmaxCrossEntropy)
    static bool is_combined_pair(ActivationType act, LossType loss);
};