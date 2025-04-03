# CUDA_Buildup



 Storing and annotating the enlightening code (moment) in the process of **Parallel computing**  learning to backtrack the chain of thought in the future.





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



