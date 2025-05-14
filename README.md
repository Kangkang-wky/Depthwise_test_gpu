# Depthwise_test_gpu
Implementation of Optimized Depthwise Convolution Kernels and PyTorch Extensions on Deep Computing Unit

## Introduction
Depthwise Separable Convolution decomposes the standard convolution operation into “Depthwise Convolution” and “Pointwise Convolution” to reduce computational overhead. This method has been applied to various deep learning neural network models, such as the MobileNet series and EfficientNet series. This project focuses on optimizing the depthwise separable convolution. The optimized convolution kernel functions are encapsulated as PyTorch extensions, allowing them to be called within PyTorch.

## Directories
- Depthwise
  - Kernel: kernels and tests for depthwise convolution
  - Extension
    - DCU_Depthwise_Extension: pytorch extensions for depthwise convolution
    - Test_DCU_Depthwise_Extension: tests for depthwise extensions
