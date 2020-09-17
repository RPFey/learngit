[TOC]

# TensorRT

## Installation

首先是按照官网上用 `.deb` 文件安装。教程见[此处](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian)。注意 `CUDA` `CUDNN` 版本号， python 包可选。安装教程也要注意是否与 `TensorRT` 相对应。

之后用 samples 连接编译不过。主要是不知道 `.deb` 将头文件和库放在哪里了，而且 sample 中是按照相对路径连接的，所以又下载了 `.tar` 文件。设置动态连接库的环境变量后

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
```

运行成功！

> NVIDIA 官网有 cudnn code samples, 也就是 `*-doc-*.deb` 包，用 `dpkg` 安装之后，在 `/usr/local/src` 下找到 code samples, 用其中的 MNIST 例程检查 cudnn 是否安装成功。
> `CUDA` 安装好像自带了一个版本的 TensorRT，注意在 Make.config 中交换一下连接文件搜索路径的顺序。

## TensorRT Construction

### Instantiate a TensorRT object in C++

* create a global object of type `ILogger`

需要创建 `ILogger` 的对象或派生对象。官方文档给出示例：

```c++
class Logger : public ILogger
 {
     void log(Severity severity, const char* msg) override
     {
         // suppress info-level messages
         if (severity != Severity::kINFO)
             std::cout << msg << std::endl;
     }
 } gLogger;
```

在 `sampleONNXMNIST` 中，位于 `commons/logging.h` 中 `class Logger`

* create `nvinfer1::IBuilder` 

> sample 中 SampleUniquePtr 实际是用了 std::unique_ptr 在 SampleOnnxMNIST 声明中。

```c++
auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
```

* create `INetworkDefinition`

Dynamic shape support requires that the `kEXPLICIT_BATCH` flag is set. With dynamic shapes, any of the input dimensions can vary at run-time, and there are no implicit dimensions in the network specification. We shift the bits to set the corresponding flag.

```c++
const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
```

* Create Network Parser

```c++
auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
```

* Build the network

for onnx model

```c++
parser->parseFromFile(onnx_filename, ILogger::Severity::kWARNING);
for (int i = 0; i < parser.getNbErrors(); ++i)
{
    std::cout << parser->getError(i)->desc() << std::endl;
}
```

#### 自定义网络结构

```c++
// 用 Ibuilder 创建网络
INetworkDefinition* network = builder->createNetworkV2(0U);

// 1. 定义输入
ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});

// 2. 一个典型的卷积，BN，激活层
Weights emptywts{DataType::kFLOAT, nullptr, 0};
// 最后的参数对应的是 kernel 和 bias 的权重, 后面有对 Weights 类的说明
IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 32, DimsHW{3, 3}, KernelWeights, emptywts);
// 设置 stride 和 padding
conv1->setStrideNd(DimsHW{s, s});
conv1->setPaddingNd(DimsHW{p, p});

// BN 实际用的是 `addScale`, 对输入做了线性变换
// 注意输入是从上一层获取 Output，实际是一个 ITensor
IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-5);
// Activation 用的是 `addActivation`，参数用类中的方法设置。
```

* Build the engine

```c++
builder->setMaxBatchSize(maxBatchSize); // batch inference
IBuilderConfig* config = builder->createBuilderConfig();
config->setMaxWorkspaceSize(1 << 20);
ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

//check the input and output dimensions
network->getNbInputs() // get the number of input tensors
network->getInput(0) // get the Input tensor class ITensor
```

`IBuilderConfig` 可以选择不同的优化算法。

### Inference

前推需要把数据拷贝到 cuda 上，然后执行网络，再拷贝回 Host 上，与 CUDA 编程类似。

例程中，采用 `.executeV2()` 运行，传入的是输入输出对应的指针。在 mEngine 中输入输出向量都有相对应的 BindingIndex 和 Name 。这些由函数 `getBindingIndex`, `getBindingName` 互相转化。`.executeV2()` 传入参数是 `void**` , 在示例中是 `vector<void*>` 头元素的地址。`vector` 中按照 mEngine `Tensor` 的索引顺序存储相对应的向量（或者数据）的首地址。

## C++ API

### ITensor

### Dims

Dims 是一个基类，由此派生出了一系列和维数有关的类。一般在定义张量大小的时候传入。

### Weights

```c++
//! by \p values field should be preserved until the build is complete.

class Weights
{
public:
    DataType type;      //!< The type of the weights.
    const void* values; //!< The weight values, in a contiguous array.
    int64_t count;      //!< The number of weights in the array.
};
```
