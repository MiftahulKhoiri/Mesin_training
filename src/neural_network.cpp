// ============================================================
// src/neural_network.cpp
// ============================================================
#include "neural_network.h"
#include <stdexcept>

NeuralNetwork::NeuralNetwork(LossType loss_type)
    : loss_type_(loss_type), next_input_size_(0) {}

void NeuralNetwork::add_layer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed) {
    if (!layers_.empty() && input_size != next_input_size_) {
        throw std::invalid_argument(
            "NeuralNetwork::add_layer: input_size (" + std::to_string(input_size) +
            ") tidak cocok dengan output_size layer sebelumnya (" +
            std::to_string(next_input_size_) + ")");
    }
    layers_.emplace_back(input_size, output_size, activation, seed);
    next_input_size_ = output_size;
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    if (layers_.empty()) {
        throw std::logic_error("NeuralNetwork::forward: belum ada layer, panggil add_layer() dulu");
    }
    Matrix current = input;
    for (auto& l : layers_) {
        current = l.forward(current);
    }
    return current;
}

Scalar NeuralNetwork::compute_loss(const Matrix& predictions, const Matrix& targets) const {
    return Loss::forward(predictions, targets, loss_type_);
}

bool NeuralNetwork::is_combined_pair(ActivationType act, LossType loss) {
    // Saat ini hanya kombinasi ini yang gradiennya sudah digabung di losses.cpp
    // (softmax_ce_derivative langsung menghasilkan probs - targets).
    // Kalau nanti bce_derivative diubah jadi gabungan dengan sigmoid, tambahkan
    // pasangan (Sigmoid, BinaryCrossEntropy) di sini juga.
    return act == ActivationType::Softmax && loss == LossType::SoftmaxCrossEntropy;
}

void NeuralNetwork::backward(const Matrix& predictions, const Matrix& targets) {
    if (layers_.empty()) {
        throw std::logic_error("NeuralNetwork::backward: belum ada layer");
    }

    Matrix grad = Loss::derivative(predictions, targets, loss_type_);

    for (size_t i = layers_.size(); i-- > 0;) {
        bool is_last = (i == layers_.size() - 1);
        bool combined = is_last && is_combined_pair(layers_[i].activation_type(), loss_type_);
        grad = layers_[i].backward(grad, combined);
    }
}

void NeuralNetwork::update(Scalar learning_rate) {
    for (auto& l : layers_) {
        l.update(learning_rate);
    }
}