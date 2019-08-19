#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/quantized/Quantizer.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {
SmallVector<int64_t, 4> convOutputShape(
    int N, // mini-batch
    int H, // input height
    int W, // input width
    int K, // output channels
    const std::vector<int64_t>& kernel,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& dilation) {
  SmallVector<int64_t, 4> out_shape;
  out_shape.push_back(N);

  int H_out = std::floor(
      (H + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1);
  int W_out = std::floor(
      (W + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1);

  out_shape.push_back(H_out);
  out_shape.push_back(W_out);
  out_shape.push_back(K);

  return out_shape;
}

template <Activation Ac>
class QNNPACKConv final : public torch::OperatorKernel {
 public:
#ifdef USE_QNNPACK
  Tensor operator()(
      Tensor act,
      Tensor packed_weight,
      Tensor bias,
      const std::vector<int64_t>& stride, // {stride_height, stride_width}
      const std::vector<int64_t>& padding, // {padding_top, padding_left}
      const std::vector<int64_t>& dilation, // {dilation_height, dilation_width}
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {
    TORCH_CHECK(
        act.ndimension() == 4,
        "qnnpack_conv2d(): Expected activation tensor to be 4-dimensional");
    TORCH_CHECK(stride.size() == 2, "qnnpack_conv2d(): 2D convolution only");
    TORCH_CHECK(
        padding.size() == 2,
        "qnnpack_conv2d(): Specify top/left padding only. \
        bottom/right padding assumed to be equal to top/left");
    TORCH_CHECK(dilation.size() == 2, "qnnpack_conv2d(): 2D convolution only");

    PackedConvWeights& pack_ptr =
        cpp_custom_type_hack::cast<PackedConvWeights>(packed_weight);
    auto packB = pack_ptr.w.get();
    auto& kernel = pack_ptr.kernel;
    auto kernel_zp = pack_ptr.w_zp;
    auto kernel_scale = pack_ptr.w_scale;

    const uint32_t kernel_h = kernel[0];
    const uint32_t kernel_w = kernel[1];
    const auto out_ch = packB->get_output_channels();

    TORCH_CHECK(
        !bias.defined() ||
            (bias.ndimension() == 1 && bias.size(0) == out_ch),
        "qnnpack_conv2d(): Given weight of size ",
        kernel,
        ", expected bias to be 1-dimensional with ",
        out_ch,
        " elements",
        ", but got bias of size ",
        bias.sizes(),
        " instead");

    // inputs are in NHWC format
    Tensor input_contig = act.contiguous();
    int N = input_contig.size(0);
    int H = input_contig.size(1);
    int W = input_contig.size(2);
    int in_ch = input_contig.size(3);
    int K = bias.size(0); // output channels

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
        kernel_zp,
        kernel_scale,
        activationLimits(output_scale, output_zero_point, Ac).first, /* output min */
        activationLimits(output_scale, output_zero_point, Ac).second /* output max */
        );
    auto outShape =
        convOutputShape(N, H, W, K, kernel, stride, padding, dilation);
    TORCH_CHECK(
        std::all_of(
            outShape.begin(), outShape.end(), [](int64_t i) { return i > 0; }),
        "qnnpack_conv2d(): each dimension of output tensor should be greater "
        "than 0")
    TORCH_CHECK(
        (outShape[3] == out_ch),
        "qnnpack_conv2d(): Number of filters must be equal to number of "
        "output channels")

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
        outShape,
        at::device(kCPU).dtype(kQUInt8),
        output_scale,
        output_zero_point);

    const qnnp_status runStatus = qnnpack::qnnpackConv(
        conv_p,
        packB->get_packed_weights(),
        N,
        H,
        W,
        input_contig.q_scale(),
        input_contig.q_zero_point(),
        (uint8_t*)input_contig.data<c10::quint8>(),
        output.q_scale(),
        output.q_zero_point(),
        (uint8_t*)output.data<c10::quint8>(),
        nullptr);

    TORCH_INTERNAL_ASSERT(
        runStatus == qnnp_status_success,
        "failed to run QNNPACK Conv operator");

    return output;
  }
#else
  Tensor operator()(
      Tensor /* act */,
      Tensor /* packed_weight */,
      Tensor /* bias */,
      const std::vector<int64_t>& /* stride */,
      const std::vector<int64_t>& /* padding */,
      const std::vector<int64_t>& /* dilation */,
      int64_t /* groups */,
      double /* output_scale */,
      int64_t /* output_zero_point */) {
    TORCH_CHECK(
        false,
        "This PyTorch installation was not built "
        "with QNNPACK operators");
  }
#endif
};

static auto registry = c10::RegisterOperators()
    .op("quantized::qnnpack_conv2d",
        c10::RegisterOperators::options()
        .kernel<QNNPACKConv<Activation::NONE>>(QuantizedCPUTensorId()))
    .op("quantized::qnnpack_conv2d_relu",
        c10::RegisterOperators::options()
        .kernel<QNNPACKConv<Activation::RELU>>(QuantizedCPUTensorId()));

} // namespace
} // namespace native
} // namespace at
