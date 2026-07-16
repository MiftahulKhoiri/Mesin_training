// ============================================================
// bindings/pybind_module.cpp  (LENGKAP)
// ============================================================
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

#include "matrix_ops.h"
#include "tensor3d.h"
#include "activations.h"
#include "losses.h"
#include "neural_network.h"
#include "trainer.h"
#include "sequence_model.h"
#include "sequence_trainer.h"

namespace py = pybind11;

using NpFloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

// --- Konversi numpy (float32, C-contiguous) <-> Matrix / Tensor3D ---
static Matrix numpy_to_matrix(NpFloatArray arr) {
    if (arr.ndim() != 2) throw std::invalid_argument("Diharapkan array numpy 2D untuk Matrix");
    size_t rows = static_cast<size_t>(arr.shape(0));
    size_t cols = static_cast<size_t>(arr.shape(1));
    Matrix m(rows, cols);
    auto buf = arr.unchecked<2>();
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            m.at(i, j) = buf(i, j);
    return m;
}

static py::array_t<float> matrix_to_numpy(const Matrix& m) {
    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(m.rows()), static_cast<py::ssize_t>(m.cols())};
    py::array_t<float> result(shape);
    auto buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j)
            buf(i, j) = m.at(i, j);
    return result;
}

static py::array_t<float> tensor3d_to_numpy(const Tensor3D& t) {
    std::vector<py::ssize_t> shape{
        static_cast<py::ssize_t>(t.batch()),
        static_cast<py::ssize_t>(t.seq_len()),
        static_cast<py::ssize_t>(t.features())
    };
    py::array_t<float> result(shape);
    auto buf = result.mutable_unchecked<3>();
    for (size_t b = 0; b < t.batch(); ++b)
        for (size_t s = 0; s < t.seq_len(); ++s)
            for (size_t f = 0; f < t.features(); ++f)
                buf(b, s, f) = t.at(b, s, f);
    return result;
}

PYBIND11_MODULE(ml_manual_cpp, m) {
    m.doc() = "ml_manual_cpp: neural network dari nol di C++ (MLP + Transformer), binding Python via pybind11";

    // --- Enums ---
    py::enum_<ActivationType>(m, "ActivationType")
        .value("ReLU", ActivationType::ReLU)
        .value("LeakyReLU", ActivationType::LeakyReLU)
        .value("Sigmoid", ActivationType::Sigmoid)
        .value("Tanh", ActivationType::Tanh)
        .value("Softmax", ActivationType::Softmax)
        .value("Linear", ActivationType::Linear);

    py::enum_<LossType>(m, "LossType")
        .value("MSE", LossType::MSE)
        .value("BinaryCrossEntropy", LossType::BinaryCrossEntropy)
        .value("SoftmaxCrossEntropy", LossType::SoftmaxCrossEntropy);

    // --- NeuralNetwork (MLP) ---
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<LossType>())
        .def("add_dense_layer", &NeuralNetwork::add_dense_layer,
             py::arg("input_size"), py::arg("output_size"), py::arg("activation"), py::arg("seed") = 42)
        .def("add_batch_norm_layer", &NeuralNetwork::add_batch_norm_layer,
             py::arg("momentum") = 0.9f, py::arg("epsilon") = 1e-5f)
        .def("forward", [](NeuralNetwork& self, NpFloatArray input, bool training) {
            return matrix_to_numpy(self.forward(numpy_to_matrix(input), training));
        }, py::arg("input"), py::arg("training") = true)
        .def("compute_loss", [](NeuralNetwork& self, NpFloatArray predictions, NpFloatArray targets) {
            return self.compute_loss(numpy_to_matrix(predictions), numpy_to_matrix(targets));
        })
        .def("backward", [](NeuralNetwork& self, NpFloatArray predictions, NpFloatArray targets) {
            self.backward(numpy_to_matrix(predictions), numpy_to_matrix(targets));
        })
        .def("update", &NeuralNetwork::update)
        .def("num_layers", &NeuralNetwork::num_layers);

    // --- Trainer (MLP) ---
    py::class_<TrainConfig>(m, "TrainConfig")
        .def(py::init<>())
        .def_readwrite("epochs", &TrainConfig::epochs)
        .def_readwrite("batch_size", &TrainConfig::batch_size)
        .def_readwrite("learning_rate", &TrainConfig::learning_rate)
        .def_readwrite("shuffle_each_epoch", &TrainConfig::shuffle_each_epoch)
        .def_readwrite("shuffle_seed", &TrainConfig::shuffle_seed)
        .def_readwrite("validation_split", &TrainConfig::validation_split)
        .def_readwrite("verbose", &TrainConfig::verbose);

    py::class_<EpochLog>(m, "EpochLog")
        .def_readonly("epoch", &EpochLog::epoch)
        .def_readonly("train_loss", &EpochLog::train_loss)
        .def_readonly("val_loss", &EpochLog::val_loss);

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<NeuralNetwork&, const TrainConfig&>())
        .def("fit", [](Trainer& self, NpFloatArray inputs, NpFloatArray targets) {
            return self.fit(numpy_to_matrix(inputs), numpy_to_matrix(targets));
        });

    // --- SequenceModel (Transformer) ---
    py::class_<SequenceModel>(m, "SequenceModel")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, bool, unsigned>(),
             py::arg("vocab_size"), py::arg("embed_dim"), py::arg("num_heads"),
             py::arg("ff_hidden_dim"), py::arg("num_layers"), py::arg("max_seq_len"),
             py::arg("causal_mask") = true, py::arg("seed") = 42)
        .def("forward", [](SequenceModel& self, NpFloatArray token_ids) {
            return tensor3d_to_numpy(self.forward(numpy_to_matrix(token_ids)));
        })
        .def("update", &SequenceModel::update);
        // compute_loss/backward SequenceModel sengaja tidak diekspos langsung —
        // dipakai lewat SequenceTrainer.fit() supaya API Python tetap simpel.

    // --- SequenceTrainer ---
    py::class_<SequenceTrainConfig>(m, "SequenceTrainConfig")
        .def(py::init<>())
        .def_readwrite("epochs", &SequenceTrainConfig::epochs)
        .def_readwrite("batch_size", &SequenceTrainConfig::batch_size)
        .def_readwrite("seq_len", &SequenceTrainConfig::seq_len)
        .def_readwrite("learning_rate", &SequenceTrainConfig::learning_rate)
        .def_readwrite("shuffle_each_epoch", &SequenceTrainConfig::shuffle_each_epoch)
        .def_readwrite("shuffle_seed", &SequenceTrainConfig::shuffle_seed)
        .def_readwrite("validation_split", &SequenceTrainConfig::validation_split)
        .def_readwrite("verbose", &SequenceTrainConfig::verbose)
        .def_readwrite("log_every_n_steps", &SequenceTrainConfig::log_every_n_steps);

    py::class_<SequenceEpochLog>(m, "SequenceEpochLog")
        .def_readonly("epoch", &SequenceEpochLog::epoch)
        .def_readonly("train_loss", &SequenceEpochLog::train_loss)
        .def_readonly("val_loss", &SequenceEpochLog::val_loss);

    py::class_<SequenceTrainer>(m, "SequenceTrainer")
        .def(py::init<SequenceModel&, const SequenceTrainConfig&>())
        .def("fit", [](SequenceTrainer& self, NpFloatArray token_stream) {
            return self.fit(numpy_to_matrix(token_stream));
        });
}