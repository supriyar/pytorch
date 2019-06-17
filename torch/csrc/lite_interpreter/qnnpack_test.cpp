#include <aten/src/ATen/core/dispatch/Dispatcher.h>
#include <torch/csrc/lite_interpreter/import_instructions.h>
#include <torch/csrc/jit/generic_instruction.h>
#include <torch/csrc/lite_interpreter/instruction_executor.h>
#include <torch/script.h> // One-stop header.
#include <c10/core/ScalarType.h>
#include <vector>

int main() {
  auto fc = c10::Dispatcher::singleton().findSchema("quantized::qnnpack_relu(Tensor input) -> Tensor", "");
  assert(fc.has_value());

  auto input_f = at::zeros({1, 1, 28, 28}, at::dtype(at::kFloat));
  ((float*)input_f.data<float>())[0] = -1.1;
  ((float*)input_f.data<float>())[1] = 2.2;
  auto input = at::quantize_linear(input_f, 1., 0, c10::ScalarType::QUInt8);
  std::vector<torch::jit::IValue> inputs{input};
  auto kernel = c10::Dispatcher::singleton().lookup(fc.value(), &inputs);
  kernel.call(&inputs);
  auto output = inputs[0].toTensor().dequantize();
  std::cout << output << std::endl;
}
