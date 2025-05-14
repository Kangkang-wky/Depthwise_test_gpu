#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <stdlib.h>
#include <iomanip>
#include <time.h>
#include <random>
#include <vector>
#include <fstream>
#include <unistd.h>


#include "cudnn.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "warmup.h"
#include "Filter3x3_Input7x7_Stride1.h"
#include "Filter3x3_Input14x14_Stride1.h"
#include "Filter3x3_Input14x14_Stride2.h"
#include "Filter3x3_Input28x28_Stride1.h"
#include "Filter3x3_Input28x28_Stride2.h"
#include "Filter3x3_Input56x56_Stride1.h"
#include "Filter3x3_Input56x56_Stride2.h"
#include "Filter3x3_Input112x112_Stride1.h"
#include "Filter3x3_Input112x112_Stride2.h"
#include "Filter5x5_Input7x7_Stride1.h"
#include "Filter5x5_Input14x14_Stride1.h"
#include "Filter5x5_Input14x14_Stride2.h"
#include "Filter5x5_Input28x28_Stride1.h"
#include "Filter5x5_Input56x56_Stride2.h"

using namespace std;

/*
CUDA and CUDNN Error Handling

CHECK_CUDA(err) - to check if an CUDA API call returned some error.
*/

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t cuda_error = (func);                                           \
    if (cuda_error != cudaSuccess)                                             \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__,                           \
             cudaGetErrorString(cuda_error));                                  \
  }

// cudnn runtime error
#define CHECK_CUDNN(func)                                                      \
  {                                                                            \
    cudnnStatus_t cudnn_status = (func);                                       \
    if (cudnn_status != CUDNN_STATUS_SUCCESS)                                  \
      printf("%s %d CUDNN: %s\n", __FILE__, __LINE__,                          \
             cudnnGetErrorString(cudnn_status));                               \
  }

// cublas runtime error
#define CHECK_CUBLAS(func)                                                     \
  {                                                                            \
    cublasStatus_t cublas_status = (func);                                     \
    if (cublas_status != CUBLAS_STATUS_SUCCESS)                                \
      printf("%s %d CUBLAS: %s\n", __FILE__, __LINE__,                         \
             cublasGetErrorString(cublas_status));                             \
  }

// check cuda runtime is success error ?  
inline bool isCudaSuccess(cudaError_t status) {
  cudaError_t error = status;
  if (error != cudaSuccess) {
    std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)
              << std::endl;
    return false;
  }
  return true;
}


// benchmark 基准测试部分
const char *cudnnAlgName[] = {
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
    "CUDNN_CONVOLUTION_FWD_ALGO_COUNT"};

/*
compareOutput():
	Compare the result calculated by our kernel and that by the CUDNN library.
	Use CUDNN library as a reference.
Input:
	n            - batch number
	c            - channel number
	h            - height
	w            - width
	kernelOutput - output data of our kernel
	cudnnOutput  - output data of the CUDNN
	delta        - a small value. Allowed numerical differece between each element
Output:
	-1           - our kernel is wrong
	0            - out kernel is correct
*/
int compareOutput(int n, int c, int h, int w, const float* kernelOutput, const float* cudnnOutput, float delta) {
	int i, j, k, l;

	// Loop over each element, and compare the value.
	// If the difference is small, then accept, or, reject and return.
	for (i = 0; i < n; i++) {
		for (j = 0; j < c; j++) {
			for (k = 0; k < h; k++) {
				for (l = 0; l < w; l++) {
					if (abs(kernelOutput[i * c * h * w + j * h * w + k * w + l] - cudnnOutput[i * c * h * w + j * h * w + k * w + l]) > delta) {
						printf("%f, %f\n", kernelOutput[i * c * h * w + j * h * w + k * w + l], cudnnOutput[i * c * h * w + j * h * w + k * w + l]);
						printf("Wrong! Output Batch Idx: %d, Channel Idx: %d, Row Idx: %d, Col Idx: %d\n", i, j, k, l);
						return -1;
					}
				}
			}
		}
	}
	return 0;
}

/*
To test depthwise convolution kernels.
*/
int main(int argc, char* argv[]) {
	// Input dimension
	int inputBatchNumber = 0;
	int inputChannel = 0;
	int inputHeight = 0;
	int inputWidth = 0;

	// Filter dimension
	int filterLayerNumber = 0;
	int filterChannel = 0;
	int filterHeight = 0;
	int filterWidth = 0;

	// Output dimension
	int outputBatchNumber = 0;
	int outputChannel = 0;
	int outputHeight = 0;
	int outputWidth = 0;

	// padding on height and width
	int paddingHeight = 0;
	int paddingWidth = 0;

	// stride
	int stride = 1;

	float alpha = 1.0;
	float beta = 0.0;

	// Initialize all required parameters
	// Input dimensions
	inputBatchNumber = atoi(argv[1]);
	inputChannel = atoi(argv[2]);
	inputHeight = atoi(argv[3]);
	inputWidth = inputHeight;           // Assume that inputs are square

	// Filter dimensions
	filterLayerNumber = inputChannel;
	filterChannel = 1;
	filterHeight = atoi(argv[4]);
	filterWidth = filterHeight;         // Assume that filters are square

	// Padding size
	if (filterWidth == 3) {
		paddingHeight = 1;
		paddingWidth = 1;
	}
	else if (filterWidth == 5) {
		paddingHeight = 2;
		paddingWidth = 2;
	}

	// Stride
	stride = atoi(argv[5]);

	// Output dimensions
	outputBatchNumber = inputBatchNumber;
	outputChannel = inputChannel;
	outputHeight = (inputHeight + paddingHeight * 2 - filterHeight) / stride + 1;
	outputWidth = (inputWidth + paddingWidth * 2 - filterWidth) / stride + 1;

	// Data size
	int inputSize = inputBatchNumber * inputChannel * inputHeight * inputWidth;
	int filterSize = filterLayerNumber * filterChannel * filterHeight * filterWidth;
	int outputSize = outputBatchNumber * outputChannel * outputHeight * outputWidth;

	// allocate host memory and device memory for input data, and copy it from host to device.
	float* hostInput = (float*)malloc(inputSize * sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < inputSize; i++) {
		hostInput[i] = (float)(float(rand()) / float((RAND_MAX)) * 5.0);
	}
	float* deviceInput;
	CHECK_CUDA(cudaMalloc((void**)&deviceInput, inputSize * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(deviceInput, hostInput, inputSize * sizeof(float), cudaMemcpyHostToDevice));

	// allocate host memory and device memory for filter data, and copy it from host to device.
	float* hostFilter = (float*)malloc(filterSize * sizeof(float));
	srand(time(NULL));
	for (int i = 0; i < filterSize; i++) {
		hostFilter[i] = (float)(float(rand()) / float((RAND_MAX)) * 5.0);
	}
	float* deviceFilter;
	CHECK_CUDA(cudaMalloc((void**)&deviceFilter, filterSize * sizeof(float)));
	CHECK_CUDA(cudaMemcpy(deviceFilter, hostFilter, filterSize * sizeof(float), cudaMemcpyHostToDevice));

	// allocate host memory and device memory for kernel output data
	float* hostKernelOutput = (float*)malloc(outputSize * sizeof(float));
	float* deviceKernelOutput;
	CHECK_CUDA(cudaMalloc((void**)&deviceKernelOutput, outputSize * sizeof(float)));

	// allocate host memory and device memory for Cudnn output data
	float* hostCudnnOutput = (float*)malloc(outputSize * sizeof(float));
	float* deviceCudnnOutput;
	CHECK_CUDA(cudaMalloc((void**)&deviceCudnnOutput, outputSize * sizeof(float)));

	// Use CUDA event to measure running time
	float elapsedTime = 0.0;
	float kernelTime = 0.0;
	float cudnnTime = 0.0;
	cudaEvent_t start, stop;
	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	// GPU warm up for benchmarking
	warmup<<<1024, 128>>>();

	// Kernel Invocation
	if (stride == 1) {
		if (filterHeight == 3) {
			if (inputHeight == 7) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				cudaEventRecord(start);
				Filter3x3_Input7x7_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 16);
				dim3 blockSize(14 * 16, 1);
				cudaEventRecord(start);
				Filter3x3_Input14x14_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(28 * 8, 1);
				cudaEventRecord(start);
				Filter3x3_Input28x28_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel);
				dim3 blockSize(4 * 56, 1);
				cudaEventRecord(start);
				Filter3x3_Input56x56_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 112) {
				dim3 gridSize(outputBatchNumber, outputChannel * 4);
				dim3 blockSize(2 * 112, 1);
				cudaEventRecord(start);
				Filter3x3_Input112x112_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
		}
		else if (filterHeight == 5) {
			if (inputHeight == 7) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				cudaEventRecord(start);
				Filter5x5_Input7x7_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 16);
				dim3 blockSize(14 * 16, 1);
				cudaEventRecord(start);
				Filter5x5_Input14x14_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(28 * 8, 1);
				cudaEventRecord(start);
				Filter5x5_Input28x28_Stride1<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
		}
	}
	else if (stride == 2) {
		if (filterHeight == 3) {
			if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				cudaEventRecord(start);
				Filter3x3_Input14x14_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 28) {
				dim3 gridSize(outputBatchNumber, outputChannel / 8);
				dim3 blockSize(14 * 8, 1);
				cudaEventRecord(start);
				Filter3x3_Input28x28_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel / 2);
				dim3 blockSize(28 * 2, 1);
				cudaEventRecord(start);
				Filter3x3_Input56x56_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 112) {
				dim3 gridSize(outputBatchNumber, outputChannel * 2);
				dim3 blockSize(56 * 4, 1);
				cudaEventRecord(start);
				Filter3x3_Input112x112_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
		}
		else if (filterHeight == 5) {
			if (inputHeight == 14) {
				dim3 gridSize(outputBatchNumber, outputChannel / 32);
				dim3 blockSize(7 * 32, 1);
				cudaEventRecord(start);
				Filter5x5_Input14x14_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
			else if (inputHeight == 56) {
				dim3 gridSize(outputBatchNumber, outputChannel / 2);
				dim3 blockSize(28 * 2, 1);
				cudaEventRecord(start);
				Filter5x5_Input56x56_Stride2<<<gridSize, blockSize>>> (
					deviceInput, deviceFilter, deviceKernelOutput,
					inputBatchNumber, inputChannel, inputHeight, inputWidth,
					filterLayerNumber, filterHeight, filterWidth,
					outputBatchNumber, outputChannel, outputHeight, outputWidth,
					paddingWidth, stride,
					alpha, beta);
				cudaEventRecord(stop);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsedTime, start, stop);
				kernelTime = elapsedTime;
			}
		}
	}
	
	// Copy kernel output from device to host
	CHECK_CUDA(cudaMemcpy(hostKernelOutput, deviceKernelOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
	
    // Create cudnn
    cudnnHandle_t convCudnn;
    CHECK_CUDNN(cudnnCreate(&convCudnn));
    
    // input descriptor
    cudnnTensorDescriptor_t convInputDescriptor;
	CHECK_CUDNN(cudnnCreateTensorDescriptor(&convInputDescriptor));
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(convInputDescriptor,
											/*format=*/CUDNN_TENSOR_NHWC,
											/*dataType=*/CUDNN_DATA_FLOAT,
											/*batch_size=*/inputBatchNumber,
											/*in_channels=*/inputChannel,
											/*image_height=*/inputHeight,
											/*image_width=*/inputWidth));

    // filter descriptor
    cudnnFilterDescriptor_t convKernelDescriptor;
	CHECK_CUDNN(cudnnCreateFilterDescriptor(&convKernelDescriptor));
	CHECK_CUDNN(cudnnSetFilter4dDescriptor(convKernelDescriptor,
											/*dataType=*/CUDNN_DATA_FLOAT,
											/*format=*/CUDNN_TENSOR_NHWC,
											/*out_channels=*/filterLayerNumber,
											/*in_channels=*/filterChannel,
											/*kernel_height=*/filterHeight,
											/*kernel_width=*/filterWidth));

    // output descriptor
    cudnnTensorDescriptor_t convOutputDescriptor;
	CHECK_CUDNN(cudnnCreateTensorDescriptor(&convOutputDescriptor));
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(convOutputDescriptor,
											/*format=*/CUDNN_TENSOR_NHWC,
											/*dataType=*/CUDNN_DATA_FLOAT,
											/*batch_size=*/outputBatchNumber,
											/*out_channels=*/outputChannel,
											/*image_height=*/outputHeight,
											/*image_width=*/outputWidth));

    // convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
	// 初始化 conv 描述符
	CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
												/*pad_height=*/pad,
												/*pad_width=*/pad,
												/*vertical_stride=*/stride,
												/*horizontal_stride=*/stride,
												/*dilation_height=*/1,
												/*dilation_width=*/1,
												/*mode=*/CUDNN_CROSS_CORRELATION,
												/*dataType=*/CUDNN_DATA_FLOAT));

	// 分组卷积设置, 分组卷积为 inputChannel
	CHECK_CUDNN(cudnnSetConvolutionGroupCount(convDesc, inputChannel));


	// set algorithm
	int algo_type = 1;
	cudnnConvolutionFwdAlgo_t algo = cudnnConvolutionFwdAlgo_t(algo_type);

 	// create workspace
    size_t workspaceSize = 0;
    void* workspaceData = nullptr;
	// 为 conv 申请空间大小
	CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
		convCudnn, convInputDescriptor, convKernelDescriptor, convDesc,
		convOutputDescriptor, algo, &workspaceSize));
	// 为 conv 实际分配空间
	CHECK_CUDA(cudaMalloc(&workspaceData, workspaceSize));


    // Use Cudnn to check kernel result and measure running time
    cudaEventRecord(start);
	CHECK_CUDNN(cudnnConvolutionForward(
		convCudnn, &alpha, convInputDescriptor, deviceInput, convKernelDescriptor,
		deviceFilter, convDesc, algo, workspaceData, workspaceSize, &beta,
		convOutputDescriptor, output))
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudnnTime = elapsedTime;

    // Copy CUDNN result from device to host
    CHECK_CUDA(cudaMemcpy(hostCudnnOutput, deviceCudnnOutput, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Compare Kernel result and Cudnn result
    if (compareOutput(outputBatchNumber, outputChannel, outputHeight, outputWidth, hostKernelOutput, hostCudnnOutput, 1) == 0) {
		printf("Kernel Calculation Correct.\n");
		printf("Cudnn time : %f ms.\n", cudnnTime);
		printf("Kernel time : %f ms.\n", kernelTime);
    }

	free(hostInput);
	free(hostFilter);
	free(hostKernelOutput);
	free(hostCudnnOutput);

	cudaFree(deviceInput);
	cudaFree(deviceFilter);
	cudaFree(deviceKernelOutput);
	cudaFree(deviceCudnnOutput);

	CHECK_CUDA(cudnnDestroy(convCudnn));
    CHECK_CUDA(cudnnDestroyTensorDescriptor(convInputDescriptor));
    CHECK_CUDA(cudnnDestroyTensorDescriptor(convOutputDescriptor));
    CHECK_CUDA(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDA(cudnnDestroyFilterDescriptor(convKernelDescriptor));
    CHECK_CUDA(cudaFree(workspaceData));

	CHECK_CUDA(cudaDeviceReset());
	return 0;
}