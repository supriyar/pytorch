#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <THNN/generic/pooling_shape.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

namespace at {
namespace native {
namespace {

class QNNPACKMaxPool final : public c10::OperatorKernel {
 public:
  Tensor operator()(
      Tensor input,
      std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride,
      std::vector<int64_t> dilation,
      std::vector<int64_t> padding) {
    Tensor qy;

    TORCH_CHECK(
        input.ndimension() == 4,
        "qnnpack_maxpool(): Expected input to be 4-dimensional: got ",
        input.ndimension());

    Tensor input_contig = input.contiguous();

    initQNNPACK();

    const auto scale = input_contig.q_scale();
    const auto zero_point = input_contig.q_zero_point();
    qnnp_operator_t qnnpackOperator_{nullptr};

    int64_t padL = padding[0];
    int64_t padT = padding[1];
    int64_t kH = kernel_size[0];
    int64_t kW = kernel_size[1];
    int64_t strideH = stride[0];
    int64_t strideW = stride[1];
    int64_t dilationH = dilation[0];
    int64_t dilationW = dilation[1];

    TORCH_CHECK(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
    TORCH_CHECK(
        strideH > 0 && strideW > 0, "strides should be greater than zero.");

    // Input is in NHWC format
    int64_t batch_size = input_contig.size(0);
    int64_t inH = input_contig.size(1);
    int64_t inW = input_contig.size(2);
    int64_t inC = input_contig.size(3);

    const qnnp_status createStatus = qnnp_create_max_pooling2d_nhwc_u8(
        padT /* input_padding_top */,
        padL /* input_padding_right */,
        padT /* input_padding_bottom */,
        padL /* input_padding_left */,
        kH /* pooling height */,
        kW /* pooling width */,
        strideH /* stride height */,
        strideW /* stride width */,
        dilationH /* dilation height */,
        dilationW /* dilation width */,
        inC /* input channels */,
        std::numeric_limits<uint8_t>::min() /* output min */,
        std::numeric_limits<uint8_t>::max() /* output max */,
        0 /* flags */,
        &qnnpackOperator_);
    TORCH_INTERNAL_ASSERT(
        createStatus == qnnp_status_success,
        "failed to create QNNPACK MaxPool operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    int64_t outC = inC;
    int64_t outH =
        pooling_output_shape(inH, kH, padT, strideH, dilationH, false);
    int64_t outW =
        pooling_output_shape(inW, kW, padL, strideW, dilationW, false);

    TORCH_CHECK(
        outH > 0 && outW > 0,
        "the resulting output Tensor size should be >= 0");

    // NHWC
    std::vector<int64_t> outSizes{batch_size, outH, outW, outC};
    qy = at::_empty_affine_quantized(
        outSizes, at::device(kCPU).dtype(kQUInt8), scale, zero_point);

    const qnnp_status setupStatus = qnnp_setup_max_pooling2d_nhwc_u8(
        qnnpackOperator_ /* max pooling */,
        batch_size /* batch size */,
        inH /* input height */,
        inW /* input width */,
        (uint8_t*)input_contig.data_ptr() /* input */,
        inC /* input_pixel_stride */,
        (uint8_t*)qy.data_ptr() /* output data */,
        outC /* output_pixel_stride */,
        qnnpack_threadpool() /* thread pool */);

    TORCH_INTERNAL_ASSERT(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK MaxPool operator");

    const qnnp_status runStatus =
        qnnp_run_operator(qnnpackOperator_, qnnpack_threadpool() /* thread pool */);

    TORCH_INTERNAL_ASSERT(
        runStatus == qnnp_status_success,
        "failed to run QNNPACK MaxPool operator");
    return qy;
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::qnnpack_max_pool2d(Tensor input, int[] kernel_size, int[] stride, int[] dilation, int[] padding) -> Tensor",
    c10::RegisterOperators::options().kernel<QNNPACKMaxPool>(
        QuantizedCPUTensorId()));
} // namespace
} // namespace native
} // namespace at
