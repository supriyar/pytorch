#include <torch/csrc/lite_interpreter/import_instructions.h>
#include <torch/csrc/jit/generic_instruction.h>
#include <torch/csrc/lite_interpreter/instruction_executor.h>
#include <torch/script.h> // One-stop header.

#include "predictor.h"

static at::Tensor input;
static at::Tensor output;
static std::shared_ptr<torch::jit::GenericInstructionList> model;

void allocate_input_buffer(int c, int h, int w) {
  input = at::zeros({1, c, h, w}, at::dtype(at::kFloat));
}

float* input_buffer() {
  return input.data<float>();
}

float* output_buffer() {
  return output.data<float>();
}

bool is_model_loaded() {
  return !!model;
}

void load_model(std::istream& input) {
  model = torch::jit::loadInstructionList(input);
}

void run_model() {
  auto exec = std::make_shared<torch::jit::InstructionExecutor>(model);
  std::vector<torch::jit::IValue> inputs{input};
  output = exec->run(inputs).toTensor();
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  std::ifstream input(argv[1]);
  load_model(input);
  std::cout << (is_model_loaded() ? "OK" : "Failed") << std::endl;
  allocate_input_buffer(1, 28, 28);
  for (int i = 0; i < 1 * 28 * 28; i++)
    input_buffer()[i] = 1.;
  run_model();
  for (int i = 0; i < 5; i++)
    std::cout << output_buffer()[i] << " ";
  std::cout << std::endl;
}
