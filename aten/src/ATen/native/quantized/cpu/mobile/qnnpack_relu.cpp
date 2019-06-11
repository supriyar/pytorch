#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/quantized/Quantizer.h>
#include <ATen/Config.h>
#include "init_qnnpack.h"
#include <algorithm>
#include <stdio.h>
namespace at { namespace native {
namespace {

class QNNPACKRelu final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qx) {
    Tensor qy;
    const auto zero_point = qx.q_zero_point().toInt();

    initQNNPACK();
    qnnp_operator_t qnnpackOperator_{nullptr};
    const qnnp_status createStatus = qnnp_create_clamp_nc_u8(
        qx.size(1) /* channels */,
        zero_point /* output min */,
        255 /* output max */,
        0 /* flags */,
        &qnnpackOperator_);
      TORCH_CHECK(
          createStatus == qnnp_status_success,
          "failed to create QNNPACK Clamp operator");
      TORCH_CHECK(qnnpackOperator_ != nullptr);

      qy = at::_empty_affine_quantized(qx.sizes(),
                                       at::device(kCPU).dtype(kQUInt8),
                                       qx.q_scale().toDouble(),
                                       qx.q_zero_point().toLong());

      const qnnp_status setupStatus = qnnp_setup_clamp_nc_u8(
        qnnpackOperator_,
        qx.size(0) /* batch size */,
        (uint8_t*)qx.data_ptr(),
        qx.size(1) /* X stride */,
        (uint8_t*)qy.data_ptr(),
        qy.size(1) /* Y stride */);
      TORCH_CHECK(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Clamp operator");

      const qnnp_status runStatus =
        qnnp_run_operator(qnnpackOperator_, nullptr /* thread pool */);
        std::cout << " Output " << qy.sizes();
        qy.toString();

      TORCH_CHECK( runStatus == qnnp_status_success,
       "failed to run QNNPACK Clamp operator");
    std::cout << " Quantized output tensor scale - " << qy.q_scale().toDouble()
    << " zero_point: " << qy.q_zero_point().toLong() << std::endl;
    return qy;
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::qnnpack_relu(Tensor qx) -> Tensor",
    c10::RegisterOperators::options()
      .kernel<QNNPACKRelu>(QuantizedCPUTensorId()));
}  // namespace
}}  // namespace at::native
