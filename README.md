# CUDA_Buildup



 Storing and annotating the enlightening code (moment) in the process of **Parallel computing**  learning to backtrack the chain of thought in the future.

[Personal notes](https://www.yuque.com/g/hangtaili/dgelan/rsbgr0610q24gur1/collaborator/join?token=s6HcOiIqymvvLq8k&source=doc_collaborator# 《并行计算》)



## Code section 1: 

A simple network model

**组成部分**：

Operator（算子） : RELU, Convolution 2D, Flatten, fully Connect 

> Not only implement in parallel form, but also higher computation speed with the help of the NVIDIA CUDA （GPU）     （ CUDA-structured code 用CUDA的API实现)

one abstract class Operator (pure virtual function)

simple inheritor: like RELU (只需要实现forward函数，做简单的变化，类中不需要新建成员变量来存储weight 和 bias)

comlicated inheritor：like Conv （构造函数初始化不同大小的卷积核，存储weight和bias，未来还需要Backpropagation）



Class DeviceTensor: following the C++ RAII thought, DeviceTensor is used to manage the lifecycle of the GPU allocated memory （实现中但凡需要申请显存都用这个类来实例化）



CUDA kernel：How to abstract need to go back to the principles of Math



Class Sequential:  using Vector to form a simple model which is composed of serial operators.

(形成最终模型的结构 + forward函数来处理输入X，得到输出)



## Code section 2:

A simple example of how to use API of the Cublas and CublasLt for matrix multiplication.



Key point:

1. Before estimation the performance of GPU, it needs warm-up.

2. In CUDA's execution model, an event is queued as a command in a stream

   ```
   CHECK_CUDA(cudaEventRecord(start));
   kernel_func();
   CHECK_CUDA(cudaEventRecord(stop));
   CHECK_CUDA(cudaEventSynchronize(stop));
       
   CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop))
   gpu先执行start event，再执行kernel，再执行stop， 而sync只是阻塞主线程等待stop event的运行结束

3. why the cublastLt is more complex than cublas?

   > cublas的API需要和blas保持兼容
   >
   > cublasLt**为了更高性能、更灵活的优化路径**，NVIDIA 把计算过程的每一步都“显式化”，你可以精细控制它的：
   >
   > - 计算方式（比如 Tensor Core vs 普通 FMA）；
   > - 数据布局（row-major, col-major, tensor NHWC/NCHW 等）；
   > - 内存复用（reorder 缓存）；
   > - 并行策略（tile size/block size/stream 等）；
   > - 混合精度（比如 FP32 accumulate on FP16 input）；
   >
   > **这些在 cublas 老接口中是写死的，而在 cublasLt 中是开放出来的。

   

4. The flow of  cublasLt execution : **Handle-Create-Layout-Desc-Layout-Execute-Destroy**

5. The result of experiment：

   ![pic](./pic/pic1.png)
