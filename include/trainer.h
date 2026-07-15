// ============================================================
// include/trainer.h
// ============================================================
#pragma once
#include <vector>
#include <string>
#include "matrix_ops.h"
#include "neural_network.h"

struct TrainConfig {
    size_t epochs = 10;
    size_t batch_size = 32;
    Scalar learning_rate = 0.01f;
    bool shuffle_each_epoch = true;
    unsigned shuffle_seed = 123;
    Scalar validation_split = 0.0f; // 0.0 = tanpa validasi, mis. 0.2 = 20% data untuk validasi
    bool verbose = true;
};

struct EpochLog {
    size_t epoch;
    Scalar train_loss;
    Scalar val_loss; // -1.0f jika validation_split == 0
};

class Trainer {
public:
    Trainer(NeuralNetwork& model, const TrainConfig& config);

    // Latih model dengan seluruh dataset (inputs/targets = batch penuh, bukan per-batch).
    // Trainer yang urus split validasi, shuffle, dan mini-batching secara internal.
    // Mengembalikan riwayat loss per epoch untuk dianalisis/diplot di Python nanti.
    std::vector<EpochLog> fit(const Matrix& inputs, const Matrix& targets);

private:
    NeuralNetwork& model_;
    TrainConfig config_;

    // Pisahkan data jadi train/val sesuai validation_split (tanpa shuffle di sini —
    // shuffle dilakukan terpisah sebelum split supaya val set representatif)
    void split_data(const Matrix& inputs, const Matrix& targets,
                     Matrix& train_x, Matrix& train_y,
                     Matrix& val_x, Matrix& val_y) const;

    // Hasilkan urutan indeks baris teracak untuk satu epoch
    std::vector<size_t> make_shuffled_indices(size_t n, unsigned seed) const;

    // Ambil submatrix berdasarkan daftar indeks baris (untuk membentuk mini-batch)
    static Matrix gather_rows(const Matrix& source, const std::vector<size_t>& indices,
                               size_t start, size_t count);

    // Evaluasi loss di suatu dataset tanpa update parameter (dipakai untuk validasi)
    Scalar evaluate_loss(const Matrix& inputs, const Matrix& targets);
};