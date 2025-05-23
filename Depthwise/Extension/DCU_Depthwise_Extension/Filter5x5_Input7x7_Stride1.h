/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 7 x 7, stride 1, padding 2

The number of channel must be multiple of 32.
Used in the MobileNet V2 and EfficientNet B0, in case of
	1) 7 x 7 x 1152 -> 7 x 7 x 1152, stride = 1, fitler = 5
*/
template <typename scalar_t>
__global__ void Filter5x5_Input7x7_Stride1(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	// every 32 channels is a group.
	__shared__ float filterData[32 * 25];	// filter is 5 x 5 = 25
	__shared__ float inputData[32 * 7 * 11]; // original input is 7 x 7, padded to be 11 x 11. ignore up and bottom padding, so 7 x 11

	float inTemp0, inTemp1, inTemp2, inTemp3, inTemp4;
	float sum0, sum1, sum2, sum3, sum4;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 32;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	filterData[threadIdx.x + 32 * 7] = filter[filterLoadSrcIdx + 32 * 7];
	filterData[threadIdx.x + 32 * 7 * 2] = filter[filterLoadSrcIdx + 32 * 7 * 2];
	// load rest of the filter value. 25 * 32 in total
	if (threadIdx.x < 25 * 32 - 3 * 32 * 7) {
		filterData[32 * 7 * 3 + threadIdx.x] = filter[32 * 7 * 3 + filterLoadSrcIdx];
	}

	// set left and right padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 1] = 0;
	inputData[leftPaddingIdx + 9] = 0; // right side padding
	inputData[leftPaddingIdx + 10] = 0; // right side padding
	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 4 + threadIdx.x + 2;	// each thread find its own load destination.

	inputData[inputLoadDstIdx] = input[inputLoadSrcIdx];
	inputData[inputLoadDstIdx + 32 * 11 * 1] = input[inputLoadSrcIdx + 32 * 7 * 1];
	inputData[inputLoadDstIdx + 32 * 11 * 2] = input[inputLoadSrcIdx + 32 * 7 * 2];
	inputData[inputLoadDstIdx + 32 * 11 * 3] = input[inputLoadSrcIdx + 32 * 7 * 3];
	inputData[inputLoadDstIdx + 32 * 11 * 4] = input[inputLoadSrcIdx + 32 * 7 * 4];
	inputData[inputLoadDstIdx + 32 * 11 * 5] = input[inputLoadSrcIdx + 32 * 7 * 5];
	inputData[inputLoadDstIdx + 32 * 11 * 6] = input[inputLoadSrcIdx + 32 * 7 * 6];
	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth +
		threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / inputWidth) * paddedWidth * inputHeight + threadIdx.x % inputWidth;
	int filterAccessBase = (threadIdx.x / inputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 3 times:
	// 		1. filter's 3rd row (when filter is sliding through the 1st row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 3rd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 5] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;

	// 2nd row
	// convolve with filter 4 times:
	//		1. filter's 4th row (when filter is sliding through the 1st row of input)
	// 		2. filter's 3rd row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 2nd row (when filter is sliding through the 3rd row of input) 
	//		4. filter's 1st row (when filter is sliding through the 4th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;
	sum3 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 4] * inTemp4;

	// 3rd row
	// convolve with filter 5 times:
	//		1. filter's 5th row (when filter is sliding through the 1st row of input)
	// 		2. filter's 4th row (when filter is sliding through the 2nd row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 3rd row of input) 
	//		4. filter's 2nd row (when filter is sliding through the 4th row of input) 
	//		5. filter's 1st row (when filter is sliding through the 5th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 5] * inTemp0;
	sum4 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 6] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 7] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 8] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 9] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 4th row
	// convolve with filter 5 times:
	//		1. filter's 5th row (when filter is sliding through the 2nd row of input)
	// 		2. filter's 4th row (when filter is sliding through the 3rd row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 4th row of input) 
	//		4. filter's 2nd row (when filter is sliding through the 5th row of input) 
	//		5. filter's 1st row (when filter is sliding through the 6th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 10] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 5] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 11] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 6] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 12] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 7] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 13] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 8] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 14] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 9] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 5th row
	// convolve with filter 5 times:
	//		1. filter's 5th row (when filter is sliding through the 3rd row of input)
	// 		2. filter's 4th row (when filter is sliding through the 4th row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 5th row of input) 
	// 		4. filter's 2nd row (when filter is sliding through the 6th row of input) 
	//		5. filter's 1st row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
	sum3 = sum3 + filterData[filterAccessBase + 15] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 10] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
	sum3 = sum3 + filterData[filterAccessBase + 16] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 11] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
	sum3 = sum3 + filterData[filterAccessBase + 17] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 12] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
	sum3 = sum3 + filterData[filterAccessBase + 18] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 13] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
	sum3 = sum3 + filterData[filterAccessBase + 19] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 14] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 6th row
	// convolve with filter 4 times:
	//		1. filter's 5th row (when filter is sliding through the 4th row of input)
	//		2. filter's 4th row (when filter is sliding through the 5th row of input)
	// 		3. filter's 3rd row (when filter is sliding through the 6th row of input) 
	//		4. filter's 2nd row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 20] * inTemp0;
	sum4 = sum4 + filterData[filterAccessBase + 15] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 21] * inTemp1;
	sum4 = sum4 + filterData[filterAccessBase + 16] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 22] * inTemp2;
	sum4 = sum4 + filterData[filterAccessBase + 17] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 23] * inTemp3;
	sum4 = sum4 + filterData[filterAccessBase + 18] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum3 = sum3 + filterData[filterAccessBase + 24] * inTemp4;
	sum4 = sum4 + filterData[filterAccessBase + 19] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;

	output[outputIdx] = sum3 * alpha + beta;
	outputIdx += outputWidth;

	// 7th row
	// convolve with filter 3 times:
	// 		1. filter's 5th row (when filter is sliding through the 5th row of input) 
	//		2. filter's 4th row (when filter is sliding through the 6th row of input) 
	//		3. filter's 3rd row (when filter is sliding through the 7th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 20] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 21] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 22] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 23] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum4 = sum4 + filterData[filterAccessBase + 24] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;

	output[outputIdx] = sum4 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	output[outputIdx] = sum1 * alpha + beta;
}
