#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cstdlib>



#define INPUT_DIMENSION 100

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "           \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n";   \
            exit(err);                                                        \
        }                                                                     \
    } while (0)

// ----------------------------------------------------------------------
// DeviceTensor：封装设备端内存及张量形状（这里以二维单通道张量和一维向量为例）
// ----------------------------------------------------------------------
class DeviceTensor {
public:
    float* d_data;
    int width, height, channels; // 若为向量，可令 height=channels=1

    DeviceTensor(int w, int h = 1, int c = 1)
        : width(w), height(h), channels(c) {
        int total = width * height * channels;
        CUDA_CHECK(cudaMalloc(&d_data, total * sizeof(float)));
    }

    ~DeviceTensor(){
        cudaFree(d_data);
    }

    int size() const {
        return width * height * channels;
    }
};

// ----------------------------------------------------------------------
// CUDA 核函数
// ----------------------------------------------------------------------

// 2D 卷积核：假设单通道、核大小为 3x3，stride=1，无 padding
__global__ void conv2dKernel(const float* input, int in_width, int in_height,
                             float* output, int out_width, int out_height,
                             const float* kernel, float bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_width * out_height;
    if (idx < total) {
        int out_x = idx % out_width;
        int out_y = idx / out_width;
        float sum = 0.0f;
        // 遍历 3x3 kernel
        for (int ky = 0; ky < 3; ky++) {
            for (int kx = 0; kx < 3; kx++) {
                int in_x = out_x + kx;
                int in_y = out_y + ky;
                sum += input[in_y * in_width + in_x] * kernel[ky * 3 + kx];
            }
        }
        output[idx] = sum + bias;
    }
}

// ReLU 核函数：逐元素激活
__global__ void reluKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = input[idx];
        output[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

// 复制核函数：将 input 复制到 output
__global__ void copyKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

// FullyConnected 核函数：每个线程计算一个输出神经元
__global__ void fullyConnectedKernel(const float* input, int in_size,
                                     const float* weights, const float* biases,
                                     float* output, int out_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
        float sum = 0.0f;
        for (int j = 0; j < in_size; j++) {
            sum += input[j] * weights[idx * in_size + j];
        }
        output[idx] = sum + biases[idx];
    }
}

// ----------------------------------------------------------------------
// Operator 抽象基类
// ----------------------------------------------------------------------
class Operator {
public:
    // forward 接口：输入为 DeviceTensor*，返回新分配的 DeviceTensor*（调用者负责释放）
    virtual DeviceTensor* forward(DeviceTensor* input) = 0;
    virtual ~Operator() {}
};

// ----------------------------------------------------------------------
// Convolution2DOperator：实现 3x3 卷积，改变张量尺寸
// ----------------------------------------------------------------------
class Convolution2DOperator : public Operator {
public:
    // kernel: 长度为9的数组
    Convolution2DOperator(const std::vector<float>& kernel, float bias)
        : bias_(bias) {
        // 将 kernel 数组复制到设备常量内存（这里简化为动态分配）
        CUDA_CHECK(cudaMalloc(&d_kernel_, 9 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_kernel_, kernel.data(), 9 * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~Convolution2DOperator() {
        cudaFree(d_kernel_);
    }

    // 假设输入为单通道二维 tensor，输出尺寸为 (in_width-2) x (in_height-2)
    virtual DeviceTensor* forward(DeviceTensor* input) override {
        int out_width = input->width - 2;
        int out_height = input->height - 2;
        DeviceTensor* output = new DeviceTensor(out_width, out_height, 1);
        int total = out_width * out_height;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        conv2dKernel<<<gridSize, blockSize>>>(input->d_data, input->width, input->height,
                                              output->d_data, out_width, out_height,
                                              d_kernel_, bias_);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        return output;
    }
private:
    float bias_;
    float* d_kernel_;
};

// ----------------------------------------------------------------------
// ReLUOperator：逐元素激活，形状不变
// ----------------------------------------------------------------------
class ReLUOperator : public Operator {
public:
    virtual DeviceTensor* forward(DeviceTensor* input) override {
        DeviceTensor* output = new DeviceTensor(input->width, input->height, input->channels);
        int total = input->size();
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        reluKernel<<<gridSize, blockSize>>>(input->d_data, output->d_data, total);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        return output;
    }
};

// ----------------------------------------------------------------------
// FlattenOperator：将二维 tensor 转换为一维向量（新 tensor）
// ----------------------------------------------------------------------
class FlattenOperator : public Operator {
public:
    virtual DeviceTensor* forward(DeviceTensor* input) override {
        int total = input->size();
        DeviceTensor* output = new DeviceTensor(total, 1, 1);
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        copyKernel<<<gridSize, blockSize>>>(input->d_data, output->d_data, total);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        return output;
    }
};

// ----------------------------------------------------------------------
// FullyConnectedOperator：将一维向量映射到指定维度
// ----------------------------------------------------------------------
class FullyConnectedOperator : public Operator {
public:
    FullyConnectedOperator(int input_size, int output_size)
        : in_size_(input_size), out_size_(output_size) {
        // 分配并初始化权重与偏置（这里随机初始化）
        weights_.resize(input_size * output_size);
        biases_.resize(output_size);
        for (auto &w : weights_) {
            w = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        for (auto &b : biases_) {
            b = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        CUDA_CHECK(cudaMalloc(&d_weights_, weights_.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_biases_, biases_.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_weights_, weights_.data(), weights_.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_biases_, biases_.data(), biases_.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~FullyConnectedOperator() {
        cudaFree(d_weights_);
        cudaFree(d_biases_);
    }

    virtual DeviceTensor* forward(DeviceTensor* input) override {
        // 输入为一维向量，大小为 in_size_
        DeviceTensor* output = new DeviceTensor(out_size_, 1, 1);
        int blockSize = 256;
        int gridSize = (out_size_ + blockSize - 1) / blockSize;
        fullyConnectedKernel<<<gridSize, blockSize>>>(input->d_data, in_size_,
                                                      d_weights_, d_biases_,
                                                      output->d_data, out_size_);
        CUDA_CHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        return output;
    }
private:
    int in_size_, out_size_;
    std::vector<float> weights_;
    std::vector<float> biases_;
    float* d_weights_;
    float* d_biases_;
};

// ----------------------------------------------------------------------
// Sequential：串联多个算子
// ----------------------------------------------------------------------
class Sequential {
public:
    Sequential(const std::vector<Operator*>& ops)
        : ops_(ops) {}

    // 从输入 tensor 开始依次调用每个算子的 forward
    DeviceTensor* forward(DeviceTensor* input) {
        DeviceTensor* cur = input;
        for (size_t i = 0; i < ops_.size(); i++) {
            DeviceTensor* next = ops_[i]->forward(cur);
            // 释放上一步的 tensor（如果不是原始输入）
            if (i > 0) {
                delete cur;
            }
            cur = next;
        }
        return cur;
    }
private:
    std::vector<Operator*> ops_;
};

// ----------------------------------------------------------------------
// 主函数：构造网络并执行
// ----------------------------------------------------------------------
int main() {
    // 假设输入为单通道 8x8 图像
    int in_width = INPUT_DIMENSION, in_height = INPUT_DIMENSION;
    DeviceTensor* d_input = new DeviceTensor(in_width, in_height, 1);
    int total = d_input->size();
    std::vector<float> h_input(total, 1.8f);
    // 为观察效果修改部分值
    h_input[0] = -1.0f;
    h_input[10] = 2.0f;
    h_input[20] = -0.5f;
    CUDA_CHECK(cudaMemcpy(d_input->d_data, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    // 构造各个算子：
    // 1. 卷积层：3x3 卷积核（随机示例），输出尺寸 N -2 * N - 2
    std::vector<float> conv_kernel = {
         0.2f, -0.1f, 1.2f,
        -0.1f,  0.3f, 0.1f,
         0.0f,  0.15f, -0.2f
    };
    Convolution2DOperator conv2d(conv_kernel, 0.5f);
    // 2. ReLU 激活层
    ReLUOperator relu1;
    // 3. Flatten 层：将 6x6 转换为 36 维向量
    FlattenOperator flatten;
    // 4. 全连接层：将 36 维向量映射到 10 维输出
    FullyConnectedOperator fc((INPUT_DIMENSION - 2)^2, 10);
    // 5. 再次 ReLU 激活
    ReLUOperator relu2;

    // 将各算子串联
    std::vector<Operator*> ops = { &conv2d, &relu1, &flatten, &fc, &relu2 };
    Sequential seq(ops);

    // 在 GPU 上连续执行所有算子
    DeviceTensor* d_output = seq.forward(d_input);
    // d_output 形状为 10x1x1

    // 将最终结果复制回主机
    int out_size = d_output->size();
    std::vector<float> h_output(out_size);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output->d_data, out_size * sizeof(float), cudaMemcpyDeviceToHost));

    // 输出结果
    std::cout << "Final output:" << std::endl;
    for (int i = 0; i < out_size; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // 清理内存
    delete d_input;
    delete d_output;

    return 0;
}
