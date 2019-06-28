#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKRelu final : public torch::OperatorKernel {
 public:
  Tensor operator()(Tensor input) {
    Tensor qy;

    TORCH_CHECK(
        input.ndimension() > 0, "qnnpack_relu(): Got empty input tensor");

    Tensor input_contig = input.contiguous();

    const auto zero_point = input_contig.q_zero_point();

    initQNNPACK();

    size_t volume = 1;
    for (int i = 0; i < input_contig.ndimension(); ++i) {
      volume *= input_contig.size(i);
    }
    size_t num_elems_x = volume / input_contig.size(0);
    qnnp_operator_t qnnpackOperator_{nullptr};

    const qnnp_status createStatus = qnnp_create_clamp_nc_u8(
        num_elems_x /* channels */,
        zero_point /* output min */,
        std::numeric_limits<uint8_t>::max() /* output max */,
        0 /* flags */,
        &qnnpackOperator_);
    TORCH_INTERNAL_ASSERT(
        createStatus == qnnp_status_success,
        "failed to create QNNPACK Relu operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    qy = at::_empty_affine_quantized(
        input_contig.sizes(),
        at::device(kCPU).dtype(kQUInt8),
        input_contig.q_scale(),
        input_contig.q_zero_point());

    size_t num_elems_y = volume / qy.size(0);

    const qnnp_status setupStatus = qnnp_setup_clamp_nc_u8(
        qnnpackOperator_, /* clamp */
        input_contig.size(0) /* batch size */,
        (uint8_t*)input_contig.data_ptr() /* input data */,
        num_elems_x /* input stride */,
        (uint8_t*)qy.data_ptr() /* output data */,
        num_elems_y /* output stride */);
    TORCH_INTERNAL_ASSERT(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Relu operator");

    const qnnp_status runStatus = qnnp_run_operator(
        qnnpackOperator_, qnnpack_threadpool() /* thread pool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == qnnp_status_success,
        "failed to run QNNPACK Relu operator");

    return qy;
  }
};

static auto registry = torch::RegisterOperators().op(
    "quantized::qnnpack_relu(Tensor input) -> Tensor",
    torch::RegisterOperators::options().kernel<QNNPACKRelu>(
        QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
