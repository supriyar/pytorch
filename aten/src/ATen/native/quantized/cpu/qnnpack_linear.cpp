#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/quantized/Quantizer.h>

#include "init_qnnpack.h"
#include "qnnpack_func.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKLinearOp final : public torch::OperatorKernel {
 public:
#ifdef USE_QNNPACK
  Tensor operator()(
      at::Tensor input,
      at::Tensor packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(
        input.dim() >= 2, "qnnpack_linear(): Input tensor rank should be >= 2");
    auto input_contig = input.contiguous();

    auto& pack_ptr = cpp_custom_type_hack::cast<PackedFCWeights>(packed_weight);
    auto packB = pack_ptr.w.get();
    auto kernel_zp = pack_ptr.w_zp;
    auto kernel_scale = pack_ptr.w_scale;

    size_t rows_input = 1;
    size_t cols_input = input_contig.size(input_contig.dim() - 1);
    for (size_t i = 0; i < input_contig.dim() - 1; ++i) {
      rows_input *= input_contig.size(i);
    }

    size_t rows_w = packB->get_output_channels();
    size_t cols_w = packB->get_input_channels();
    TORCH_CHECK(
        cols_input == cols_w,
        "qnnpack_linear(): input size does not match weight dimension 1 size: \
        got ",
        cols_input,
        " but expected ",
        cols_w);

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
        {static_cast<long>(rows_input), static_cast<long>(rows_w)},
        input.options(),
        output_scale,
        output_zero_point);

    const qnnp_status runStatus = qnnpack::qnnpackLinear(
        rows_input /* batch_size */,
        cols_input /* input_channels */,
        rows_w /* output_channels */,
        input_contig.q_zero_point(),
        input_contig.q_scale(),
        kernel_zp,
        kernel_scale,
        output_zero_point,
        output_scale,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        (uint8_t*)input_contig.data<c10::quint8>(),
        cols_input /* input_stride */,
        packB->get_packed_weights(),
        (uint8_t*)output.data<c10::quint8>(),
        rows_w /* output_stride */,
        nullptr /* threadpool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == qnnp_status_success,
        "failed to run QNNPACK Linear operator");
    return output;
  }
#else
  Tensor operator()(
      at::Tensor /* input */,
      at::Tensor /* weight */,
      double /* output_scale */,
      int64_t /* output_zero_point */) {
    TORCH_CHECK(
        false,
        "This PyTorch installation was not built "
        "with QNNPACK operators");
  }
#endif
};

static auto registry = torch::RegisterOperators().op(
    "quantized::qnnpack_linear_op",
    torch::RegisterOperators::options().kernel<QNNPACKLinearOp>(
        QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
