// ============================================================
// include/neural_network.h  (DIPERBARUI — layers_ jadi polymorphic)
// ============================================================
#pragma once
#include <vector>
#include <memory>
#include "matrix_ops.h"
#include "activations.h"
#include "losses.h"
#include "layer_base.h"
#include "dense_layer.h"
#include "batch_norm_layer.h"

class NeuralNetwork {
public:
    explicit NeuralNetwork(LossType loss_type);

    void add_dense_layer(size_t input_size, size_t output_size, ActivationType activation, unsigned seed = 42);
    void add_batch_norm_layer(Scalar momentum = 0.9f, Scalar epsilon = 1e-5f);

    // training: diteruskan ke tiap layer (BatchNorm pakai batch stats vs running stats)
    Matrix forward(const Matrix& input, bool training = true);

    Scalar compute_loss(const Matrix& predictions, const Matrix& targets) const;

    void backward(const Matrix& predictions, const Matrix& targets);

    void update(Scalar learning_rate);

    size_t num_layers() const { return layers_.size(); }
    LayerBase& layer(size_t idx) { return *layers_.at(idx); }
    const LayerBase& layer(size_t idx) const { return *layers_.at(idx); }

private:
    std::vector<std::unique_ptr<LayerBase>> layers_;
    LossType loss_type_;
    size_t next_input_size_;

    static bool is_combined_pair(ActivationType act, LossType loss);
};