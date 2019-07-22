#include "init_qnnpack.h"

#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Config.h>

#include "init_qnnpack.h"
#include "qnnpack_utils.h"

#include <chrono>
using namespace std::chrono;
namespace at {
namespace native {
namespace {

class QNNPACKFullyConnected final : public c10::OperatorKernel {
 public:
  Tensor operator()(at::Tensor input,
      at::Tensor weight,
      at::Tensor bias,
      double output_scale,
      int64_t output_zero_point) {

    TORCH_CHECK(input.dim() >= 2, "Input tensor rank should be >= 2");

    Tensor input_contig = input.contiguous();

    // C(output) = A(input_contig) x B(weight), where C, A, B are M x N, M x K, N x K
    // matrices, respectively.
    int64_t rows_a = input_contig.size(0);
    int64_t cols_a = 1;
    for (size_t i = 1; i < input_contig.dim(); ++i) {
      cols_a *= input_contig.size(i);
    }
    int64_t rows_b = weight.size(0);

    TORCH_CHECK(cols_a == weight.size(1),
        "qnnpack_fc(): input size does not match weight dimension 1 size: got ",
        cols_a, " but expected ", weight.size(1));

    TORCH_CHECK(!bias.defined() || (bias.ndimension() == 1 && bias.size(0) == rows_b),
        "qnnpack_fc(): Given weight of size ", weight.sizes(),
        ", expected bias to be 1-dimensional with ", rows_b, " elements",
        ", but got bias of size ", bias.sizes(), " instead");

    initQNNPACK();
    qnnp_operator_t qnnpackOperator_{nullptr};

    // Allocate output Tensor and a buffer for QNNPACK to use
    Tensor output = at::_empty_affine_quantized(
      {rows_a, rows_b},
      at::device(kCPU).dtype(kQUInt8),
      output_scale,
      output_zero_point);

    auto start = high_resolution_clock::now();
    // QNNPACK expects both weights and inputs to be uint8
    const qnnp_status createStatus = qnnp_create_fully_connected_nc_q8(
        cols_a /* input channels */,
        rows_b /* output channels */,
        input_contig.q_zero_point().toInt() /* input zero_point */,
        input_contig.q_scale().toFloat() /* input scale */,
        weight.q_zero_point().toInt() /* kernel zero_point */,
        weight.q_scale().toFloat() /* kernel scale */,
        (uint8_t*)weight.data_ptr() /* kernel data */,
        (int32_t*)bias.data_ptr() /* bias data */,
        output.q_zero_point().toInt() /* output zero_point */,
        output.q_scale().toFloat() /* output scale */,
        std::numeric_limits<uint8_t>::min() /* output_min */,
        std::numeric_limits<uint8_t>::max() /* output_max */,
        0 /* flags */,
        &qnnpackOperator_);

    TORCH_INTERNAL_ASSERT(createStatus == qnnp_status_success,
        "failed to create QNNPACK FullyConnected operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::string op_name = "qnnpack_fc_create" + to_string(cols_a);
    std::cout << "Caffe2Observer {\"type\": \""<< op_name << "\", \"metric\": \"latency\", \"unit\": \"ms_per_iter\", \"value\": " << duration.count() << "}" << std::endl;

    start = high_resolution_clock::now();

    const qnnp_status setupStatus = qnnp_setup_fully_connected_nc_q8(
        qnnpackOperator_ /* convolution */,
        rows_a /* batch_size */,
        (uint8_t*)input_contig.data_ptr() /* input */,
        cols_a /* input stride */,
        (uint8_t*)output.data_ptr() /* output */,
        rows_b/* output stride */);
    TORCH_INTERNAL_ASSERT(setupStatus == qnnp_status_success,
        "failed to setup QNNPACK fully connected operator");
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    op_name = "qnnpack_fc_setup" + to_string(cols_a);
    std::cout << "Caffe2Observer {\"type\": \""<< op_name << "\", \"metric\": \"latency\", \"unit\": \"ms_per_iter\", \"value\": " << duration.count() << "}" << std::endl;

    start = high_resolution_clock::now();
    const qnnp_status runStatus =
        qnnp_run_operator(qnnpackOperator_, nullptr /*qnnpack_threadpool() /* thread pool */);
     stop = high_resolution_clock::now();
     duration = duration_cast<milliseconds>(stop - start);
    op_name = "qnnpack_fc_run" + to_string(cols_a);
    std::cout << "Caffe2Observer {\"type\": \""<< op_name << "\", \"metric\": \"latency\", \"unit\": \"ms_per_iter\", \"value\": " << duration.count() << "}" << std::endl;

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
