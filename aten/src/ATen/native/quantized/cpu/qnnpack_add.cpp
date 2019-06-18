#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>
#include "init_qnnpack.h"

namespace at { namespace native {
namespace {

class QNNPACKAdd final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Tensor qb,
      double scale, int64_t zero_point) {

    TORCH_CHECK(
        qa.numel() == qb.numel(),
        "qnnpack_add(): Add operands must be the same size!");
    TORCH_CHECK(qa.scalar_type() == qb.scalar_type(),
        "qnnpack_add(): Add operands should have same data type.");

    Tensor qa_contig = qa.contiguous();
    Tensor qb_contig = qb.contiguous();

    const auto a_zero_point = qa_contig.q_zero_point().toInt();
    const auto b_zero_point = qb_contig.q_zero_point().toInt();
    const auto a_scale = qa_contig.q_scale().toDouble();
    const auto b_scale = qb_contig.q_scale().toDouble();

    Tensor qy = at::_empty_affine_quantized(qa_contig.sizes(),
                                            at::device(kCPU).dtype(kQUInt8),
                                            scale,
                                            zero_point);
    initQNNPACK();

    qnnp_operator_t qnnpackOperator_{nullptr};

    TORCH_CHECK(qa_contig.ndimension() > 0,
        "qnnpack_add(): Got empty input tensor");

    size_t volume = 1;
    for (int i = 0; i < qa_contig.ndimension(); ++i) {
      volume *= qa_contig.size(i);
    }

    size_t channels = volume / qa_contig.size(0);

    const qnnp_status createStatus = qnnp_create_add_nc_q8(
        channels /* channels */,
        a_zero_point, a_scale,
        b_zero_point, b_scale,
        static_cast<uint8_t>(zero_point), scale,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        0 /* flags */,
        &qnnpackOperator_);
    TORCH_INTERNAL_ASSERT(createStatus == qnnp_status_success,
        "failed to create QNNPACK Add operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    const qnnp_status setupStatus = qnnp_setup_add_nc_q8(
        qnnpackOperator_,
        qa_contig.size(0) /* batch size */,
        (uint8_t*)qa_contig.data_ptr(),
        channels /* A stride */,
        (uint8_t*)qb_contig.data_ptr(),
        channels /* B stride */,
        (uint8_t*)qy.data_ptr(),
        channels /* Y stride */);
    TORCH_INTERNAL_ASSERT(setupStatus == qnnp_status_success,
        "failed to setup QNNPACK Add operator");

    const qnnp_status runStatus =
        qnnp_run_operator(qnnpackOperator_, nullptr /* thread pool */);

    TORCH_INTERNAL_ASSERT( runStatus == qnnp_status_success,
       "failed to run QNNPACK Add operator");

    return qy;
  }
};

static auto registry = c10::RegisterOperators().op(
    "quantized::qnnpack_add",
    c10::RegisterOperators::options()
      .kernel<QNNPACKAdd>(QuantizedCPUTensorId()));
}  // namespace
}}  // namespace at::native
