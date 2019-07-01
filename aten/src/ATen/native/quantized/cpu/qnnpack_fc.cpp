#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/quantized/Quantizer.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKFullyConnected final : public torch::OperatorKernel {
 public:
  Tensor operator()(
      at::Tensor input,
      at::Tensor weight,
      at::Tensor bias,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(input.dim() >= 2, "Input tensor rank should be >= 2");
    TORCH_CHECK(weight.dim() == 2, "Weight tensor rank should be == 2");
    Tensor input_contig = input.contiguous();

    // C(output) = A(input_contig) x B(weight), where C, A, B are M x N, M x K,
    // N x K matrices, respectively.
    int64_t rows_a = 1;//input_contig.size(0);
    int64_t cols_a = input_contig.size(input_contig.dim() - 1);
    for (size_t i = 0; i < input_contig.dim()-1; ++i) {
      rows_a *= input_contig.size(i);
    }

    int64_t rows_b = weight.size(0);

    TORCH_CHECK(
        cols_a == weight.size(1),
        "qnnpack_linear(): input size does not match weight dimension 1 size: got ",
        cols_a,
        " but expected ",
        weight.size(1));

    TORCH_CHECK(
        !bias.defined() || (bias.ndimension() == 1 && bias.size(0) == rows_b),
        "qnnpack_linear(): Given weight of size ",
        weight.sizes(),
        ", expected bias to be 1-dimensional with ",
        rows_b,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");

    initQNNPACK();
    qnnp_operator_t qnnpackOperator_{nullptr};

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
        {rows_a, rows_b},
        input.options(),
        output_scale,
        output_zero_point);

    // QNNPACK expects both weights and inputs to be uint8
    const qnnp_status createStatus = qnnp_create_fully_connected_nc_q8(
        cols_a /* input channels */,
        rows_b /* output channels */,
        input_contig.q_zero_point() /* input zero_point */,
        input_contig.q_scale() /* input scale */,
        weight.q_zero_point() /* kernel zero_point */,
        weight.q_scale() /* kernel scale */,
        (uint8_t*)weight.data_ptr() /* kernel data */,
        (int32_t*)bias.data_ptr() /* bias data */,
        output.q_zero_point() /* output zero_point */,
        output.q_scale() /* output scale */,
        std::numeric_limits<uint8_t>::min() /* output_min */,
        std::numeric_limits<uint8_t>::max() /* output_max */,
        0 /* flags */,
        &qnnpackOperator_);

    TORCH_INTERNAL_ASSERT(
        createStatus == qnnp_status_success,
        "failed to create QNNPACK FullyConnected operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    const qnnp_status setupStatus = qnnp_setup_fully_connected_nc_q8(
        qnnpackOperator_ /* convolution */,
        rows_a /* batch_size */,
        (uint8_t*)input_contig.data_ptr() /* input */,
        cols_a /* input stride */,
        (uint8_t*)output.data_ptr() /* output */,
        rows_b /* output stride */);
    TORCH_INTERNAL_ASSERT(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK fully connected operator");
    const qnnp_status runStatus = qnnp_run_operator(
        qnnpackOperator_, qnnpack_threadpool() /* thread pool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == qnnp_status_success, "failed to run QNNPACK operator");

    return output;
  }
};

static auto registry = torch::RegisterOperators().op(
    "quantized::qnnpack_linear(Tensor X, Tensor W, Tensor b, float Y_scale, int Y_zero_point) -> Tensor",
    torch::RegisterOperators::options().kernel<QNNPACKFullyConnected>(
        QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
