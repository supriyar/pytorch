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
CAFFE_KNOWN_TYPE(PackedConvWeights);
#endif // USE_QNNPACK
} // namespace caffe2

namespace at {
namespace native {
namespace {

class QNNPACKPrepackConv final : public torch::OperatorKernel {
 public:
#ifdef USE_QNNPACK
  Tensor operator()(
      Tensor weight,
      Tensor bias,
      torch::List<int64_t> stride,
      torch::List<int64_t> padding,
      torch::List<int64_t> dilation,
      int64_t groups) {
    TORCH_CHECK(
        weight.ndimension() == 4,
        "qnnpack_prepack_conv(): Weights are expected to have 4 dimensions");
    TORCH_CHECK(
        stride.size() == 2, "qnnpack_prepack_conv(): 2D convolution only");
    TORCH_CHECK(
        padding.size() == 2,
        "qnnpack_prepack_conv(): Specify top/left padding only. \
       bottom/right padding assumed to be equal to top/left");
    TORCH_CHECK(
        dilation.size() == 2, " qnnpack_prepack_conv(): 2D convolution only");

    initQNNPACK();

    // QNNPACK expects weights to be of the format {out_c, kH, kW, in_c/groups}
    const size_t out_ch = weight.size(0);
    const uint32_t kernel_h = weight.size(1);
    const uint32_t kernel_w = weight.size(2);
    const size_t in_ch = weight.size(3) * groups;

    uint32_t stride_h = stride[0];
    uint32_t stride_w = stride[1];
    uint32_t pad_t = padding[0];
    uint32_t pad_l = padding[1];
    uint32_t dilation_h = dilation[0];
    uint32_t dilation_w = dilation[1];

    qnnpack::conv_param_t conv_p(
        {kernel_w, kernel_h},
        {stride_w, stride_h},
        {dilation_w, dilation_h},
        {pad_t, pad_l, pad_t, pad_l},
        groups,
        in_ch,
        out_ch,
        weight.q_zero_point(),
        weight.q_scale(),
        0,
        255);

    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto wt_ptr = guts::make_unique<PackedConvWeights>(
        PackedConvWeights{guts::make_unique<qnnpack::PrePackConvWeights>(
                              conv_p,
                              (uint8_t*)weight_contig.data<c10::quint8>(),
                              (int32_t*)bias_contig.data<c10::qint32>()),
                          {kernel_h, kernel_w},
                          weight.q_scale(),
                          weight.q_zero_point()});
    return cpp_custom_type_hack::create(std::move(wt_ptr), weight.options());
  }
#else
  Tensor operator()(
      at::Tensor /* input */,
      at::Tensor /* weight */,
      at::Tensor /* bias */,
      torch::List<int64_t> /* stride */,
      torch::List<int64_t> /* padding */,
      torch::List<int64_t> /* dilation */,
      int64_t /* groups */) {
    TORCH_CHECK(
        false,
        "This PyTorch installation was not built "
        "with QNNPACK operators");
  }
#endif
};

static auto registry = torch::RegisterOperators().op(
    "quantized::qnnpack_prepack_conv",
    torch::RegisterOperators::options().kernel<QNNPACKPrepackConv>(
        QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
