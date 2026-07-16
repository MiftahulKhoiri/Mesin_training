// ============================================================
// include/sequence_trainer.h  (FILE BARU)
// ============================================================
#pragma once
#include <vector>
#include "matrix_ops.h"
#include "sequence_model.h"

struct SequenceTrainConfig {
    size_t epochs = 10;
    size_t batch_size = 8;
    size_t seq_len = 128;           // panjang context per contoh training
    Scalar learning_rate = 0.0003f; // umumnya lebih kecil dari MLP untuk transformer
    bool shuffle_each_epoch = true;
    unsigned shuffle_seed = 123;
    Scalar validation_split = 0.0f;
    bool verbose = true;
    size_t log_every_n_steps = 50;  // sequence training bisa punya banyak step per epoch
};

struct SequenceEpochLog {
    size_t epoch;
    Scalar train_loss;
    Scalar val_loss; // -1.0f jika validation_split == 0
};

class SequenceTrainer {
public:
    SequenceTrainer(SequenceModel& model, const SequenceTrainConfig& config);

    // token_stream: Matrix 1 x N — satu barisan token panjang (mis. seluruh korpus
    // yang sudah di-tokenize & digabung jadi satu array). Trainer yang memotongnya
    // jadi banyak contoh (input, target) sepanjang seq_len secara internal.
    std::vector<SequenceEpochLog> fit(const Matrix& token_stream);

private:
    SequenceModel& model_;
    SequenceTrainConfig config_;

    // Bikin daftar start-index tiap contoh non-overlapping: 0, seq_len, 2*seq_len, ...
    // (lihat catatan di bawah soal trade-off non-overlapping vs sliding window)
    std::vector<size_t> build_example_starts(size_t total_tokens) const;

    // Ambil satu mini-batch contoh berdasarkan start-index terpilih.
    // batch_input/batch_target: count x seq_len, target = input digeser 1 posisi.
    void make_batch(const Matrix& token_stream, const std::vector<size_t>& starts,
                     size_t batch_begin, size_t count,
                     Matrix& batch_input, Matrix& batch_target) const;

    std::vector<size_t> make_shuffled_indices(size_t n, unsigned seed) const;

    // Evaluasi loss di sekumpulan start-index TANPA backward (dipakai untuk validasi)
    Scalar evaluate_loss(const Matrix& token_stream, const std::vector<size_t>& starts) const;
};