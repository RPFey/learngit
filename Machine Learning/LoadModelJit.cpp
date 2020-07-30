#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <time.h>
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[]) {

  cv::Mat img = cv::imread(argv[2]);
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(224, 224));
  cv::namedWindow("origin", cv::WINDOW_NORMAL);
  cv::imshow("origin", resize_img);

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  std::clock_t startTime,endTime;

  cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
  cv::Mat matFloat;
  resize_img.convertTo(matFloat, CV_32F);
  auto size = matFloat.size();
  auto nChannels = matFloat.channels();
  auto tensor = torch::from_blob(matFloat.data, {1, size.height, size.width, nChannels});

  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor.permute({0, 3, 1, 2}));

  startTime = std::clock();
  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  endTime = clock();
  std::cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
  std::cout << output.sizes() << '\n';
  at::Tensor clas = output.argmax(1);
  std::cout << clas << '\n';

  cv::waitKey(0);
  cv::destroyAllWindows();
  return 0;
}
