// ============================================================
// include/layer_base.h  (LENGKAP)
// ============================================================
#pragma once
#include <string>
#include <ostream>
#include "matrix_ops.h"
#include "activations.h"

class LayerBase {
public:
    virtual ~LayerBase() = default;

    virtual Matrix forward(const Matrix& input, bool training) = 0;
    virtual Matrix backward(const Matrix& grad_output, bool combined_with_loss) = 0;
    virtual void update(Scalar learning_rate) = 0;

    virtual size_t input_size() const = 0;
    virtual size_t output_size() const = 0;

    virtual ActivationType activation_type() const { return ActivationType::Linear; }

    // Checkpoint I/O
    virtual std::string layer_type() const = 0;
    virtual void save(std::ostream& os) const = 0;
};