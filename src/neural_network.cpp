// ============================================================
// src/neural_network.cpp  (LENGKAP)
// ============================================================
#include "neural_network.h"
#include <fstream>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(LossType loss_type)
    : loss_type_(loss_type), next_input_size_(0) {}

void NeuralNetwork::add_dense_layer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed) {
    if (!layers_.empty() && input_size != next_input_size_) {
        throw std::invalid_argument(
            "NeuralNetwork::add_dense_layer: input_size (" + std::to_string(input_size) +
            ") tidak cocok dengan output_size layer sebelumnya (" +
            std::to_string(next_input_size_) + ")");
    }
    layers_.push_back(std::make_unique<DenseLayer>(input_size, output_size, activation, seed));
    next_input_size_ = output_size;
}

void NeuralNetwork::add_batch_norm_layer(Scalar momentum, Scalar epsilon) {
    if (layers_.empty()) {
        throw std::logic_error("NeuralNetwork::add_batch_norm_layer: tidak bisa jadi layer pertama");
    }
    layers_.push_back(std::make_unique<BatchNormLayer>(next_input_size_, momentum, epsilon));
}

Matrix NeuralNetwork::forward(const Matrix& input, bool training) {
    if (layers_.empty()) {
        throw std::logic_error("NeuralNetwork::forward: belum ada layer, panggil add_dense_layer() dulu");
    }
    Matrix current = input;
    for (auto& l : layers_) current = l->forward(current, training);
    return current;
}

Scalar NeuralNetwork::compute_loss(const Matrix& predictions, const Matrix& targets) const {
    return Loss::forward(predictions, targets, loss_type_);
}

bool NeuralNetwork::is_combined_pair(ActivationType act, LossType loss) {
    return act == ActivationType::Softmax && loss == LossType::SoftmaxCrossEntropy;
}

void NeuralNetwork::backward(const Matrix& predictions, const Matrix& targets) {
    if (layers_.empty()) {
        throw std::logic_error("NeuralNetwork::backward: belum ada layer");
    }
    Matrix grad = Loss::derivative(predictions, targets, loss_type_);
    for (size_t i = layers_.size(); i-- > 0;) {
        bool is_last = (i == layers_.size() - 1);
        bool combined = is_last && is_combined_pair(layers_[i]->activation_type(), loss_type_);
        grad = layers_[i]->backward(grad, combined);
    }
}

void NeuralNetwork::update(Scalar learning_rate) {
    for (auto& l : layers_) l->update(learning_rate);
}

void NeuralNetwork::save_checkpoint(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("NeuralNetwork::save_checkpoint: gagal membuka file: " + path);

    int loss = static_cast<int>(loss_type_);
    size_t n = layers_.size();
    ofs.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));

    for (const auto& l : layers_) {
        std::string type = l->layer_type();
        size_t len = type.size();
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(type.data(), static_cast<std::streamsize>(len));
        l->save(ofs);
    }
}

NeuralNetwork NeuralNetwork::load_checkpoint(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("NeuralNetwork::load_checkpoint: gagal membuka file: " + path);

    int loss_int;
    size_t n;
    ifs.read(reinterpret_cast<char*>(&loss_int), sizeof(loss_int));
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));

    NeuralNetwork net(static_cast<LossType>(loss_int));
    for (size_t i = 0; i < n; ++i) {
        size_t len;
        ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string type(len, ' ');
        ifs.read(&type[0], static_cast<std::streamsize>(len));

        if (type == "Dense") {
            auto layer = DenseLayer::load(ifs);
            net.next_input_size_ = layer->output_size();
            net.layers_.push_back(std::move(layer));
        } else if (type == "BatchNorm") {
            net.layers_.push_back(BatchNormLayer::load(ifs));
        } else {
            throw std::runtime_error("NeuralNetwork::load_checkpoint: tipe layer tidak dikenal: " + type);
        }
    }
    return net;
}