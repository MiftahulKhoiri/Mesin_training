// ============================================================
// include/layer_base.h  (FILE BARU)
// ============================================================
#pragma once
#include "matrix_ops.h"
#include "activations.h"

class LayerBase {
public:
    virtual ~LayerBase() = default;

    // training: true saat pelatihan (relevan untuk BatchNorm/Dropout nanti),
    // false saat inferensi/evaluasi (pakai running stats, dsb.)
    virtual Matrix forward(const Matrix& input, bool training) = 0;

    // combined_with_loss: true jika grad_output sudah gradien gabungan
    // aktivasi+loss (lihat penjelasan di DenseLayer). Layer tanpa konsep
    // aktivasi (mis. BatchNorm) cukup abaikan parameter ini.
    virtual Matrix backward(const Matrix& grad_output, bool combined_with_loss) = 0;

    virtual void update(Scalar learning_rate) = 0;

    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

    // Default: layer ini tidak punya aktivasi yang relevan untuk pengecekan
    // combined_with_loss otomatis di NeuralNetwork. DenseLayer meng-override ini.
    virtual ActivationType activation_type() const { return ActivationType::Linear; }
};