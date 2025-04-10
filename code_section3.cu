#include <iostream>
#include <cstdlib>
#include <cudnn.h>
#include <cuda_runtime.h>  // 添加 CUDA 运行时头文件

#define CHECK_CUDNN(expression)                                \
  {                                                            \
    cudnnStatus_t status = (expression);                       \
    if (status != CUDNN_STATUS_SUCCESS) {                      \
      std::cerr << "Error on line " << __LINE__ << ": "        \
                << cudnnGetErrorString(status) << std::endl;   \
      std::exit(EXIT_FAILURE);                                 \
    }                                                          \
  }

#define CHECK_CUDA(expression)                                 \
  {                                                            \
    cudaError_t status = (expression);                         \
    if (status != cudaSuccess) {                               \
      std::cerr << "CUDA Error on line " << __LINE__ << ": "   \
                << cudaGetErrorString(status) << std::endl;    \
      std::exit(EXIT_FAILURE);                                 \
    }                                                          \
  }

int main() {
    // 1. 创建 cuDNN 句柄
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // 2. 创建输入张量描述符：格式 NCHW
    cudnnTensorDescriptor_t input_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, 
                     CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 32, 32));

    // 3. 创建滤波器描述符
    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, 
                     CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 16, 3, 5, 5));

    // 4. 创建卷积描述符
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                     2, 2,  // padding
                     1, 1,  // stride
                     1, 1,  // dilation
                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 5. 创建输出张量描述符
    int n, c, h, w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc,
                     input_desc, filter_desc, &n, &c, &h, &w));
    cudnnTensorDescriptor_t output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
                     CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

    // 6. 选择卷积前向算法
    cudnnConvolutionFwdAlgo_t algo;
    #ifdef CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                     input_desc, filter_desc, conv_desc, output_desc,
                     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));
    #else
        // 旧版本 cuDNN 使用默认算法
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    #endif
    // 7. 查询工作空间大小
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                     input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_bytes));
    std::cout << "工作空间大小: " << workspace_bytes << " bytes" << std::endl;

    // 8. 分配内存
    float *d_input, *d_filter, *d_output, *d_workspace;
    size_t input_bytes = 1 * 3 * 32 * 32 * sizeof(float);
    size_t filter_bytes = 16 * 3 * 5 * 5 * sizeof(float);
    size_t output_bytes = n * c * h * w * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_filter, filter_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    if (workspace_bytes > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));
    } else {
        d_workspace = nullptr;
    }

    // (可选) 初始化数据，可以使用 cudaMemset 或 cudaMemcpy 拷贝主机数据到 GPU
    CHECK_CUDA(cudaMemset(d_input, 1, input_bytes));
    CHECK_CUDA(cudaMemset(d_filter, 2.2, filter_bytes));
    CHECK_CUDA(cudaMemset(d_output, 3.3, output_bytes));

    // 9. 执行卷积前向计算
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn,
                     &alpha,
                     input_desc, d_input,
                     filter_desc, d_filter,
                     conv_desc,
                     algo,
                     d_workspace, workspace_bytes,
                     &beta,
                     output_desc, d_output));

    std::cout << "卷积前向计算成功！" << std::endl;

    // 9.5 从设备拷贝输出数据到主机并打印前几个值
    float* h_output = new float[output_bytes / sizeof(float)];
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    // 打印前10个输出元素
    std::cout << "卷积输出结果 (前200个元素): ";
    for (int i = 0; i < 200 && i < (output_bytes / sizeof(float)); ++i) {
    std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;


    // 10. 清理资源
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    if (d_workspace) CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    return 0;
}
