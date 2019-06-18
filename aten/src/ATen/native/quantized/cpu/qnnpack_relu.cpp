#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include "init_qnnpack.h"

namespace at { namespace native {
namespace {

class QNNPACKRelu final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor input) {
    Tensor qy;

    Tensor input_contig = input.contiguous();

    const auto zero_point = input_contig.q_zero_point().toInt();

    initQNNPACK();

    TORCH_CHECK(input_contig.ndimension() > 0,
        "qnnpack_relu(): Got empty input tensor");

    size_t volume = 1;
    for (int i = 0; i < input_contig.ndimension(); ++i) {
      volume *= input_contig.size(i);
    }
    size_t channels_x = volume / input_contig.size(0);
    qnnp_operator_t qnnpackOperator_{nullptr};

    const qnnp_status createStatus = qnnp_create_clamp_nc_u8(
        channels_x /* channels */,
        zero_point /* output min */,
        255 /* output max */,
        0 /* flags */,
        &qnnpackOperator_);
    TORCH_INTERNAL_ASSERT(createStatus == qnnp_status_success,
        "failed to create QNNPACK Relu operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    qy = at::_empty_affine_quantized(input_contig.sizes(),
                                     at::device(kCPU).dtype(kQUInt8),
                                     input_contig.q_scale().toDouble(),
                                     input_contig.q_zero_point().toLong());

    size_t channels_y = volume / qy.size(0);

    const qnnp_status setupStatus = qnnp_setup_clamp_nc_u8(
        qnnpackOperator_,
        input_contig.size(0) /* batch size */,
        (uint8_t*)input_contig.data_ptr(),
        channels_x /* X stride */,
        (uint8_t*)qy.data_ptr(),
        channels_y /* Y stride */);
    TORCH_INTERNAL_ASSERT(setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Relu operator");

    const qnnp_status runStatus =
        qnnp_run_operator(qnnpackOperator_, nullptr /* thread pool */);

    TORCH_INTERNAL_ASSERT( runStatus == qnnp_status_success,
       "failed to run QNNPACK Relu operator");

    return qy;
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::qnnpack_relu",
    //"quantized::qnnpack_relu(Tensor input) -> Tensor",
    c10::RegisterOperators::options()
      .kernel<QNNPACKRelu>(QuantizedCPUTensorId()));
}  // namespace
}}  // namespace at::native
