#include "init_qnnpack.h"

#include <ATen/ATen.h>
#include <ATen/core/Type.h>
#include <ATen/core/op_registration/op_registration.h>

namespace at { namespace native {
namespace {

class QNNPACKAdd final : public c10::OperatorKernel {
 public:
  Tensor operator()(Tensor qa, Tensor qb,
      double scale, int64_t zero_point) {

    TORCH_CHECK(qa.ndimension() > 0,
        "qnnpack_add(): Got empty input tensor");

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

    size_t volume = 1;
    for (int i = 0; i < qa_contig.ndimension(); ++i) {
      volume *= qa_contig.size(i);
    }

    size_t num_elems = volume / qa_contig.size(0);

    const qnnp_status createStatus = qnnp_create_add_nc_q8(
        num_elems /* channels */,
        a_zero_point /* a zero_point */,
        a_scale /* a scale */,
        b_zero_point /* b zero_point */,
        b_scale /* b scale */,
        static_cast<uint8_t>(zero_point) /* sum zero_point */,
        scale /* sum scale */,
        std::numeric_limits<uint8_t>::min() /* output min */,
        std::numeric_limits<uint8_t>::max() /* output max */,
        0 /* flags */,
        &qnnpackOperator_);

    TORCH_INTERNAL_ASSERT(createStatus == qnnp_status_success,
        "failed to create QNNPACK Add operator");
    TORCH_INTERNAL_ASSERT(qnnpackOperator_ != nullptr);

    const qnnp_status setupStatus = qnnp_setup_add_nc_q8(
        qnnpackOperator_ /* add op */,
        qa_contig.size(0) /* batch size */,
        (uint8_t*)qa_contig.data_ptr() /* a data */,
        num_elems /* A stride */,
        (uint8_t*)qb_contig.data_ptr(),
        num_elems /* B stride */,
        (uint8_t*)qy.data_ptr(),
        num_elems /* sum stride */);
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
