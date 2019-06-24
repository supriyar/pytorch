#include <torch/csrc/lite_interpreter/import_instructions.h>
#include <torch/csrc/jit/generic_instruction.h>
#include <torch/csrc/lite_interpreter/instruction_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/script.h> // One-stop header.
#include <fstream>
#include "predictor.h"
#include <chrono>
using namespace std::chrono;

static at::Tensor input;
static at::Tensor output;
static std::shared_ptr<torch::jit::GenericInstructionList> model;

void allocate_input_buffer(int c, int h, int w) {
  input = at::zeros({1, h, w, c}, at::dtype(at::kFloat));
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

  std::ifstream File;
  File.open("./mnist_in0.txt");
  std::cout << " Loading input " << std::endl;
  for(int i = 0; i < 28*28; ++i)
  {
      float x;
      File >> x;
      input_buffer()[i] = x;
      //std::cout << " " << input_buffer()[i];
  }

  File.close();
  //for (int i = 0; i < 1 * 28 * 28; i++)
  //  input_buffer()[i] = 1.;
  auto start = high_resolution_clock::now();
  std::cout << "Starting time " << std::endl;
  run_model();
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Ending time " << std::endl;
  std::cout << "Time taken for QNNPACK " << duration.count() << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << output_buffer()[i] << " ";
  std::cout << std::endl;
}
