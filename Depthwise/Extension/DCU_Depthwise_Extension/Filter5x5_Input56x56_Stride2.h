/*
Depthwise Convolution Kernel.

Case: filter 5 x 5, input 56 x 56, stride 2, padding 2

Used in the MobileNet V2 and EfficientNet B0, in case of
	1)	56 x 56 x 144 -> 28 x 28 x 144, stride = 2, filter = 5
*/
template <typename scalar_t>
__global__ void Filter5x5_Input56x56_Stride2(const scalar_t* __restrict__ input, const scalar_t* __restrict__ filter, scalar_t* __restrict__ output,
	int inputBatchNumber, int inputChannel, int inputHeight, int inputWidth,
	int filterLayerNumber, int filterHeight, int filterWidth,
	int outputBatchNumber, int outputChannel, int outputHeight, int outputWidth,
	int padding, int stride,
	float alpha, float beta) {

	__shared__ float filterData[2 * 25];	// filter is 5 x 5 = 25
	__shared__ float inputData[2 * 56 * 60]; // original input is 56 x 56, padded to be 60 x 60. ignore up and bottom padding, so 56 x 60

	float inTemp0, inTemp1, inTemp2, inTemp3, inTemp4;
	float sum0, sum1, sum2;  // to accumulate the row sum result. rolling recycle.

	int channelGroupSize = 2;
	int paddedWidth = inputWidth + 2 * padding;

	// load filter
	int filterLoadSrcIdx = blockIdx.y * channelGroupSize * filterWidth * filterHeight + threadIdx.x;
	if (threadIdx.x < 2 * 25) {
		filterData[threadIdx.x] = filter[filterLoadSrcIdx];
	}

	// set padding
	int leftPaddingIdx = threadIdx.x * paddedWidth;
	inputData[leftPaddingIdx] = 0;
	inputData[leftPaddingIdx + 1] = 0;

	inputData[leftPaddingIdx + paddedWidth - 2] = 0;
	inputData[leftPaddingIdx + paddedWidth - 1] = 0;

	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + 1] = 0;

	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + paddedWidth - 2] = 0;
	inputData[leftPaddingIdx + (channelGroupSize / 2) * inputHeight * paddedWidth + paddedWidth - 1] = 0;

	__syncthreads();

	// load input
	// for all threads in the same block, use blockIdx.x to find correct batch index, use blockIdx.y to find correct input channel.
	int inputLoadIdxBase = blockIdx.x * inputChannel * inputHeight * inputWidth + blockIdx.y * channelGroupSize * inputHeight * inputWidth;
	int inputLoadSrcIdx = inputLoadIdxBase + threadIdx.x;	// each thread find its own load source.
	int inputLoadDstIdx = (threadIdx.x / inputWidth) * 4 + threadIdx.x + 2;	// each thread find its own load destination.

	#pragma unroll
	for (int i = 0; i < 112; i++) {
		inputData[inputLoadDstIdx + 60 * i] = input[inputLoadSrcIdx + 56 * i];
	}

	__syncthreads();

	// convolution
	int outputIdx = blockIdx.x * outputChannel * outputHeight * outputWidth +
		blockIdx.y * channelGroupSize * outputHeight * outputWidth +
		(threadIdx.x / outputWidth) * outputHeight * outputWidth + threadIdx.x % outputWidth;

	int inputAccessBase = (threadIdx.x / outputWidth) * paddedWidth * inputHeight + threadIdx.x % outputWidth * 2;
	int filterAccessBase = (threadIdx.x / outputWidth) * filterHeight * filterWidth;
	int inputAccessOffset = 0;

	// 1st row
	// convolve with filter 2 times:
	// 		1. filter's 3rd row (when filter is sliding through the 1st row of input)
	//		2. filter's 1st row (when filter is sliding through the 3rd row of input)
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = filterData[filterAccessBase + 10] * inTemp0;
	sum1 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;

	// 2nd row
	// convolve with filter 2 times:
	//		1. filter's 4th row (when filter is sliding through the 1st row of input)
	//		2. filter's 2nd row (when filter is sliding through the 3rd row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;

	#pragma unroll
	for (int i = 0; i < 8; i++) {
		// 3rd row, 45
		// convolve with filter 3 times:
		//		1. filter's 5th row (when filter is sliding through the 1st row of input)
		//		2. filter's 3rd row (when filter is sliding through the 3rd row of input) 
		//		3. filter's 1st row (when filter is sliding through the 5th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
		sum2 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum0 * alpha + beta;
		outputIdx += outputWidth;

		// 4th row
		// convolve with filter 2 times:
		// 		1. filter's 4th row (when filter is sliding through the 3rd row of input) 
		//		2. filter's 2nd row (when filter is sliding through the 5th row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;

		// 5th row
		// convolve with filter 3 times:
		//		1. filter's 5th row (when filter is sliding through the 3rd row of input)
		//		2. filter's 3rd row (when filter is sliding through the 5th row of input) 
		//		3. filter's 1st row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
		sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
		sum0 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
		sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
		sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
		sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
		sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;

		output[outputIdx] = sum1 * alpha + beta;
		outputIdx += outputWidth;

		// 6th row
		// convolve with filter 2 times:
		// 		1. filter's 4th row (when filter is sliding through the 5th row of input) 
		// 		2. filter's 2nd row (when filter is sliding through the 7th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;

		// 7th row
		// convolve with filter 3 times:
		//		1. filter's 5th row (when filter is sliding through the 5th row of input)
		//		2. filter's 3rd row (when filter is sliding through the 7th row of input)
		//		3. filter's 1st row (when filter is sliding through the 9th row of input) 
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
		sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;
		sum1 = filterData[filterAccessBase + 0] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
		sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 1] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
		sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 2] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
		sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 3] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
		sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 4] * inTemp4;
		output[outputIdx] = sum2 * alpha + beta;
		outputIdx += outputWidth;

		// 8th row, 50th row
		// convolve with filter 2 times:
		//		1. filter's 4th row (when filter is sliding through the 7th row of input)
		//		2. filter's 2nd row (when filter is sliding through the 9th row of input)
		inputAccessOffset += paddedWidth;
		inTemp0 = inputData[inputAccessBase + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;
		sum1 = sum1 + filterData[filterAccessBase + 5] * inTemp0;

		inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;
		sum1 = sum1 + filterData[filterAccessBase + 6] * inTemp1;

		inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;
		sum1 = sum1 + filterData[filterAccessBase + 7] * inTemp2;

		inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;
		sum1 = sum1 + filterData[filterAccessBase + 8] * inTemp3;

		inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
		sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
		sum1 = sum1 + filterData[filterAccessBase + 9] * inTemp4;
	}


	// 51st row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 49th row of input)
	//		2. filter's 3rd row (when filter is sliding through the 51th row of input) 
	//		3. filter's 1st row (when filter is sliding through the 53rd row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 20] * inTemp0;
	sum1 = sum1 + filterData[filterAccessBase + 10] * inTemp0;
	sum2 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 21] * inTemp1;
	sum1 = sum1 + filterData[filterAccessBase + 11] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 22] * inTemp2;
	sum1 = sum1 + filterData[filterAccessBase + 12] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 23] * inTemp3;
	sum1 = sum1 + filterData[filterAccessBase + 13] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 24] * inTemp4;
	sum1 = sum1 + filterData[filterAccessBase + 14] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 4] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
	outputIdx += outputWidth;

	// 52nd row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 51st row of input) 
	//		2. filter's 2nd row (when filter is sliding through the 53rd row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 15] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 16] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 17] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 18] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 19] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 9] * inTemp4;

	// 53rd row
	// convolve with filter 3 times:
	//		1. filter's 5th row (when filter is sliding through the 51st row of input)
	//		2. filter's 3rd row (when filter is sliding through the 53rd row of input) 
	//		3. filter's 1st row (when filter is sliding through the 55th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 20] * inTemp0;
	sum2 = sum2 + filterData[filterAccessBase + 10] * inTemp0;
	sum0 = filterData[filterAccessBase + 0] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 21] * inTemp1;
	sum2 = sum2 + filterData[filterAccessBase + 11] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 1] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 22] * inTemp2;
	sum2 = sum2 + filterData[filterAccessBase + 12] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 2] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 23] * inTemp3;
	sum2 = sum2 + filterData[filterAccessBase + 13] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 3] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum1 = sum1 + filterData[filterAccessBase + 24] * inTemp4;
	sum2 = sum2 + filterData[filterAccessBase + 14] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 4] * inTemp4;

	output[outputIdx] = sum1 * alpha + beta;
	outputIdx += outputWidth;

	// 54th row
	// convolve with filter 2 times:
	// 		1. filter's 4th row (when filter is sliding through the 53rd row of input) 
	// 		2. filter's 2nd row (when filter is sliding through the 55th row of input) 
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 15] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 5] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 16] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 6] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 17] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 7] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 18] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 8] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 19] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 9] * inTemp4;

	// 55th row
	// convolve with filter 2 times:
	//		1. filter's 5th row (when filter is sliding through the 53th row of input)
	//		2. filter's 3rd row (when filter is sliding through the 55th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 20] * inTemp0;
	sum0 = sum0 + filterData[filterAccessBase + 10] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 21] * inTemp1;
	sum0 = sum0 + filterData[filterAccessBase + 11] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 22] * inTemp2;
	sum0 = sum0 + filterData[filterAccessBase + 12] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 23] * inTemp3;
	sum0 = sum0 + filterData[filterAccessBase + 13] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum2 = sum2 + filterData[filterAccessBase + 24] * inTemp4;
	sum0 = sum0 + filterData[filterAccessBase + 14] * inTemp4;
	output[outputIdx] = sum2 * alpha + beta;
	outputIdx += outputWidth;

	// 56th row
	// convolve with filter 1 times:
	//		1. filter's 4th row (when filter is sliding through the 55th row of input)
	inputAccessOffset += paddedWidth;
	inTemp0 = inputData[inputAccessBase + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 15] * inTemp0;

	inTemp1 = inputData[inputAccessBase + 1 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 16] * inTemp1;

	inTemp2 = inputData[inputAccessBase + 2 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 17] * inTemp2;

	inTemp3 = inputData[inputAccessBase + 3 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 18] * inTemp3;

	inTemp4 = inputData[inputAccessBase + 4 + inputAccessOffset];
	sum0 = sum0 + filterData[filterAccessBase + 19] * inTemp4;
	output[outputIdx] = sum0 * alpha + beta;
}
