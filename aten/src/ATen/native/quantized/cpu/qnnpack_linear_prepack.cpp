#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/quantized/Quantizer.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace caffe2 {
#ifdef USE_QNNPACK
// Required for cpp_custom_type_hack to work
CAFFE_KNOWN_TYPE(PackedFCWeights);
#endif // USE_QNNPACK
} // namespace caffe2

namespace at {
namespace native {
namespace {

class QNNPACKPrepackLinear final : public torch::OperatorKernel {
 public:
#ifdef USE_QNNPACK
  Tensor operator()(at::Tensor weight, at::Tensor bias) {
    TORCH_CHECK(
        weight.dim() == 2,
        "qnnpack_linear(): Weight tensor rank should be == 2");

    int64_t rows_w = weight.size(0);
    int64_t cols_w = weight.size(1);

    TORCH_CHECK(
        !bias.defined() || (bias.ndimension() == 1 && bias.size(0) == rows_w),
        "qnnpack_linear(): Given weight of size ",
        weight.sizes(),
        ", expected bias to be 1-dimensional with ",
        rows_w,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");

    Tensor weight_contig = weight.contiguous();
    Tensor bias_contig = bias.contiguous();

    initQNNPACK();

    auto wt_ptr = guts::make_unique<PackedFCWeights>(
        PackedFCWeights{guts::make_unique<qnnpack::PackBMatrix>(
                            cols_w /* input_channels */,
                            rows_w /* output_channels */,
                            weight.q_zero_point(),
                            weight.q_scale(),
                            (uint8_t*)weight_contig.data<c10::quint8>(),
                            (int32_t*)bias_contig.data<c10::qint32>()),
                        weight.q_scale(),
                        weight.q_zero_point()});

    return cpp_custom_type_hack::create(std::move(wt_ptr), weight.options());
  }
#else
  Tensor operator()(at::Tensor /* weight */, at::Tensor /* bias */) {
    TORCH_CHECK(
        false,
        "This PyTorch installation was not built "
        "with QNNPACK operators");
  }
#endif
};

static auto registry = torch::RegisterOperators().op(
    "quantized::qnnpack_linear_prepack",
    torch::RegisterOperators::options().kernel<QNNPACKPrepackLinear>(
        QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
