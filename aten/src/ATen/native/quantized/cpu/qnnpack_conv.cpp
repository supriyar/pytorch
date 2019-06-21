#include "init_qnnpack.h"

#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Config.h>

namespace at { namespace native {
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

class QNNPACKConv2d final : public c10::OperatorKernel {
 public:
  Tensor operator()(
      Tensor act,
      Tensor weight,
      Tensor bias,
      const std::vector<int64_t>& stride,
      const std::vector<int64_t>& padding,
      const std::vector<int64_t>& dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point) {

    TORCH_CHECK(act.ndimension() == 4,
        "qnnpack_conv2d(): Expected activation tensor to be 4-dimensional");

    // QNNPACK expects weights to be of the format {out_c, kH, kW, in_c}
    const auto out_ch = weight.size(0);
    const auto kH = weight.size(1);
    const auto kW = weight.size(2);

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == out_ch * groups),
        "qnnpack_conv2d(): Given weight of size ", weight.sizes(),
        ", expected bias to be 1-dimensional with ", out_ch * groups, " elements",
        ", but got bias of size ", bias.sizes(), " instead");

    // inputs are in NHWC format
    int N = act.size(0);
    int H = act.size(1);
    int W = act.size(2);
    int in_ch = act.size(3);
    int K = bias.size(0); // output channels

    std::vector<int64_t> kernel{kH, kW};

    initQNNPACK();

    qnnp_operator_t qnnpackOperator_{nullptr};

    auto outShape =
        convOutputShape(N, H, W, K, kernel, stride, padding, dilation);
    TORCH_CHECK(
        std::all_of(
            outShape.begin(), outShape.end(), [](int64_t i) { return i > 0; }),
        "qnnpack_conv2d(): each dimension of output tensor should be greater than 0")
    TORCH_CHECK((outShape[3] == out_ch),
        "qnnpack_conv2d(): Number of filters must be equal to number of output channels")

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
      outShape,
      at::device(kCPU).dtype(kQUInt8),
      output_scale,
      output_zero_point);

    TORCH_CHECK(
        in_ch % groups == 0,
        "qnnpack_conv2d(): number of input channels must be divisible by groups count");
    TORCH_CHECK(
        out_ch % groups == 0,
        "qnnpack_conv2d(): number of output channels must be divisible by groups count");

    int padL = padding[0];
    int padT = padding[1];
    int strideH = stride[0];
    int strideW = stride[1];
    int dilationH = dilation[0];
    int dilationW = dilation[1];

    float outScale = output.q_scale().toDouble();
    int outZeroPoint = output.q_zero_point().toInt();

    // QNNPACK expects both weights and inputs to be uint8
    const qnnp_status createStatus = qnnp_create_convolution2d_nhwc_q8(
            padT, /* padding top */
            padL, /* padding right */
            padT, /* padding bottom */
            padL, /* padding left */
            kH,   /* kernel height */
            kW,   /* kernel width */
            strideH, /* stride height */
            strideW, /* stride width */
            dilationH, /* dilation height */
            dilationW, /* dilation width */
            groups, /* groups */
            in_ch / groups, /* group input_channels */
            out_ch / groups, /* group output_channels */
            act.q_zero_point().toInt(), /* input zero_point */
            act.q_scale().toDouble(),   /* input scale */
            weight.q_zero_point().toInt(), /* kernel zero_point */
            weight.q_scale().toDouble(),  /* kernel scale */
            (uint8_t*)weight.data_ptr(),  /* kernel data */
            (int32_t*)bias.data_ptr(),   /* bias data */
            outZeroPoint, /* output zero_point */
            outScale, /* output scale */
            std::numeric_limits<uint8_t>::min(), /* output min */
            std::numeric_limits<uint8_t>::max(), /* output max */
            0 /* flags */,
            &qnnpackOperator_);

    TORCH_INTERNAL_ASSERT(createStatus == qnnp_status_success,
        "failed to create QNNPACK Conv2D operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    const qnnp_status setupStatus = qnnp_setup_convolution2d_nhwc_q8(
            qnnpackOperator_ /* convolution */,
            N /* batch_size */,
            H /* input height */,
            W /* input width */,
            (uint8_t*)act.data_ptr(), /* input data */
            in_ch /* input pixel stride */,
            (uint8_t*)output.data_ptr(), /* output data */
            out_ch /* output pixel stride */,
            nullptr /* threadpool */);

    TORCH_INTERNAL_ASSERT(setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Conv2D operator");

    const qnnp_status runStatus =
        qnnp_run_operator(qnnpackOperator_, nullptr /* thread pool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == qnnp_status_success, "failed to run QNNPACK Conv operator");

    return output;
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::qnnpack_conv2d",
    c10::RegisterOperators::options()
      .kernel<QNNPACKConv2d>(QuantizedCPUTensorId()));
}  // namespace
}}  // namespace at::native
