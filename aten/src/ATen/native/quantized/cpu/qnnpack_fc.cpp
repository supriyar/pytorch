#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Config.h>
#include "init_qnnpack.h"

#include <stdio.h>

namespace at { namespace native {
namespace {

class QNNPACKFullyConnected final : public c10::OperatorKernel {
 public:
  Tensor operator()(at::Tensor input,
      at::Tensor weight,
      at::Tensor bias,
      double output_scale,
      int64_t output_zero_point) {

    Tensor input_contig = input.contiguous();

    AT_ASSERT(input_contig.dim() >= 2);

    // C(output) = A(input_contig) x B(weight), where C, A, B are M x N, M x K, K x N
    // matrices, respectively.
    int64_t M = 1;
    for (size_t i = 0; i < input_contig.dim() - 1; ++i) {
      M *= input_contig.size(i);
    }
    int64_t K = input_contig.size(input_contig.dim() - 1);
    int64_t N = weight.size(0);
    std::cout << " Input size " << input_contig.sizes();
    std::cout << " weight size " << weight.sizes();

    TORCH_CHECK(K == weight.size(1),
        "qnnpack_fc(): input size does not match weight dimension 1 size: got ",
        K, " but expected ", weight.size(1));

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == N),
        "qnnpack_fc(): Given weight of size ", weight.sizes(),
        ", expected bias to be 1-dimensional with ", N, " elements",
        ", but got bias of size ", bias.sizes(), " instead");

    initQNNPACK();
    qnnp_operator_t qnnpackOperator_{nullptr};

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
      {M, N},
      at::device(kCPU).dtype(kQUInt8),
      output_scale,
      output_zero_point);

    // QNNPACK expects both weights and inputs to be uint8
    const qnnp_status createStatus = qnnp_create_fully_connected_nc_q8(
        K,
        N,
        input_contig.q_zero_point().toInt(),
        input_contig.q_scale().toFloat(),
        weight.q_zero_point().toInt(),
        weight.q_scale().toFloat(),
        (uint8_t*)weight.data_ptr(),
        (int32_t*)bias.data_ptr(),
        output.q_zero_point().toInt(),
        output.q_scale().toFloat(),
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        0 /* flags */,
        &qnnpackOperator_);
    TORCH_INTERNAL_ASSERT(createStatus == qnnp_status_success,
        "failed to create QNNPACK FullyConnected operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    const qnnp_status setupStatus = qnnp_setup_fully_connected_nc_q8(
        qnnpackOperator_,
        M,
        (uint8_t*)input_contig.data_ptr(),
        K /* input stride */,
        (uint8_t*)output.data_ptr(),
        N /* output stride */);
    TORCH_INTERNAL_ASSERT(setupStatus == qnnp_status_success,
        "failed to setup QNNPACK fully connected operator");
    const qnnp_status runStatus =
        qnnp_run_operator(qnnpackOperator_, nullptr /* thread pool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == qnnp_status_success, "failed to run QNNPACK operator");

    return output;
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::qnnpack_fc(Tensor X, Tensor W, Tensor b, float Y_scale, int Y_zero_point) -> Tensor",
    c10::RegisterOperators::options()
      .kernel<QNNPACKFullyConnected>(QuantizedCPUTensorId()));
}  // namespace
}}  // namespace at::native
