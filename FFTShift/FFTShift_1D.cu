// /usr/local/cuda-8.0/bin/nvcc -O3 -gencode arch=compute_50,code=sm_50 -m64  FFTShift.cu -lcufft_static -lculibos --relocatable-device-code=true

// --- This code compares four different ways to perform the FFTshift:
//
// --- 1)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>

#include <cufft.h>
#include <cufftXt.h>

#include "Utilities.cuh"
#include "TimingGPU.cuh"

//#define DEBUG

#define BLOCKSIZE 256

/*****************************************/
/* FFTSHIFT 1D IN-PLACE MEMORY MOVEMENTS */
/*****************************************/
__global__ void fftshift_1D_inplace_memory_movements(float2 * __restrict__ d_inout, const unsigned int N)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N/2)
    {
		float2 temp = d_inout[tid];
        d_inout[tid] = d_inout[tid + (N / 2)];
        d_inout[tid + (N / 2)] = temp;
    }
}

/*********************************************/
/* FFTSHIFT 1D OUT-OF-PLACE MEMORY MOVEMENTS */
/*********************************************/
__global__ void fftshift_1D_outofplace_memory_movements(const float2 * __restrict__ d_in, float2 * __restrict__ d_out, const unsigned int N)
{
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N/2)
    {
        d_out[tid] = d_in[tid + (N / 2)];
        d_out[tid + (N / 2)] = d_in[tid];
    }
}

/**********************************************/
/* FFTSHIFT 1D INPLACE CHESSBOARD - VERSION 1 */
/**********************************************/
__device__ float2 fftshift_1D_chessboard_callback_v1(void *d_in, size_t offset, void *callerInfo, void *sharedPtr) {

	float a		= (float)(1-2*((int)offset%2));

	float2	out = ((float2*)d_in)[offset];
	out.x = out.x * a;
	out.y = out.y * a;
	return out;
}

__device__ cufftCallbackLoadC fftshift_1D_chessboard_callback_v1_Ptr = fftshift_1D_chessboard_callback_v1;

/**********************************************/
/* FFTSHIFT 1D INPLACE CHESSBOARD - VERSION 2 */
/**********************************************/
__device__ float2 fftshift_1D_chessboard_callback_v2(void *d_in, size_t offset, void *callerInfo, void *sharedPtr) {

	float a = pow(-1.,(double)(offset&1));

	float2	out = ((float2*)d_in)[offset];
	out.x = out.x * a;
	out.y = out.y * a;
	return out;
}

__device__ cufftCallbackLoadC fftshift_1D_chessboard_callback_v2_Ptr = fftshift_1D_chessboard_callback_v2;

/**********************************************/
/* FFTSHIFT 1D INPLACE CHESSBOARD - VERSION 3 */
/**********************************************/
__device__ float2 fftshift_1D_chessboard_callback_v3(void *d_in, size_t offset, void *callerInfo, void *sharedPtr) {

	float2	out = ((float2*)d_in)[offset];

	if ((int)offset&1) {

		out.x = -out.x;
		out.y = -out.y;

	}
	return out;
}

__device__ cufftCallbackLoadC fftshift_1D_chessboard_callback_v3_Ptr = fftshift_1D_chessboard_callback_v3;

/********/
/* MAIN */
/********/
int main()
{
    const int N = 524288;
//    const int N = 16;

    TimingGPU timerGPU;

    // --- Host side input array
	float2 *h_vect = (float2 *)malloc(N * sizeof(float2));
	for (int i = 0; i < N; i++) {
		h_vect[i].x = (float)rand() / (float)RAND_MAX;
		h_vect[i].y = (float)rand() / (float)RAND_MAX;
	}

	// --- Host side output arrays
	float2 *h_out1 = (float2 *)malloc(N * sizeof(float2));
	float2 *h_out2 = (float2 *)malloc(N * sizeof(float2));
	float2 *h_out3 = (float2 *)malloc(N * sizeof(float2));
	float2 *h_out4 = (float2 *)malloc(N * sizeof(float2));
	float2 *h_out5 = (float2 *)malloc(N * sizeof(float2));

	// --- Device side input arrays
	float2 *d_vect1; gpuErrchk(cudaMalloc(&d_vect1, N * sizeof(float2)));
	float2 *d_vect2; gpuErrchk(cudaMalloc(&d_vect2, N * sizeof(float2)));
	float2 *d_vect3; gpuErrchk(cudaMalloc(&d_vect3, N * sizeof(float2)));
	float2 *d_vect4; gpuErrchk(cudaMalloc(&d_vect4, N * sizeof(float2)));
	float2 *d_vect5; gpuErrchk(cudaMalloc(&d_vect5, N * sizeof(float2)));
    gpuErrchk(cudaMemcpy(d_vect1, h_vect, N * sizeof(float2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vect2, h_vect, N * sizeof(float2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vect3, h_vect, N * sizeof(float2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vect4, h_vect, N * sizeof(float2), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_vect5, h_vect, N * sizeof(float2), cudaMemcpyHostToDevice));

	// --- Device side output arrays
	float2 *d_out1; gpuErrchk(cudaMalloc(&d_out1, N * sizeof(float2)));
	float2 *d_out2; gpuErrchk(cudaMalloc(&d_out2, N * sizeof(float2)));
	float2 *d_out3; gpuErrchk(cudaMalloc(&d_out3, N * sizeof(float2)));
	float2 *d_out4; gpuErrchk(cudaMalloc(&d_out4, N * sizeof(float2)));
	float2 *d_out5; gpuErrchk(cudaMalloc(&d_out5, N * sizeof(float2)));

	/***************************************************************/
	/* VERSION 1: cuFFT + IN-PLACE MEMORY MOVEMENTS BASED FFTSHIFT */
	/***************************************************************/
	cufftHandle planinverse; cufftSafeCall(cufftPlan1d(&planinverse, N, CUFFT_C2C, 1));
	timerGPU.StartCounter();
	cufftSafeCall(cufftExecC2C(planinverse, d_vect1, d_vect1, CUFFT_INVERSE));
	fftshift_1D_inplace_memory_movements<<<iDivUp(N/2, BLOCKSIZE), BLOCKSIZE>>>(d_vect1, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	printf("In-place memory movements elapsed time:  %3.3f ms \n", timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_out1, d_vect1, N * sizeof(float2), cudaMemcpyDeviceToHost));

	/*******************************************************************/
	/* VERSION 2: cuFFT + OUT-OF-PLACE MEMORY MOVEMENTS BASED FFTSHIFT */
	/*******************************************************************/
	cufftHandle planinverse_v1; cufftSafeCall(cufftPlan1d(&planinverse_v1, N, CUFFT_C2C, 1));
	timerGPU.StartCounter();
	cufftSafeCall(cufftExecC2C(planinverse_v1, d_vect2, d_vect2, CUFFT_INVERSE));
	fftshift_1D_outofplace_memory_movements<<<iDivUp(N/2, BLOCKSIZE), BLOCKSIZE>>>(d_vect2, d_out2, N);
#ifdef DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	printf("Out-of-place memory movements elapsed time:  %3.3f ms \n", timerGPU.GetCounter());
	gpuErrchk(cudaMemcpy(h_out2, d_out2, N * sizeof(float2), cudaMemcpyDeviceToHost));

	// --- Checking the results
	for (int i=0; i<N; i++) if ((h_out1[i].x != h_out2[i].x)||(h_out1[i].y != h_out2[i].y)) { printf("Out-of-place memory movements test failed!\n"); return 0; }

	printf("Out-of-place memory movements test passed!\n");

	/***************************************************/
	/* VERSION 3: CHESSBOARD MULTIPLICATION V1 + cuFFT */
	/***************************************************/
	cufftCallbackLoadC hfftshift_1D_chessboard_callback_v1_Ptr;

	gpuErrchk(cudaMemcpyFromSymbol(&hfftshift_1D_chessboard_callback_v1_Ptr, fftshift_1D_chessboard_callback_v1_Ptr, sizeof(hfftshift_1D_chessboard_callback_v1_Ptr)));
	cufftHandle planinverse_v2; cufftSafeCall(cufftCreate(&planinverse_v2));
	size_t work_size_v2; cufftSafeCall(cufftMakePlan1d(planinverse_v2, N, CUFFT_C2C, 1, &work_size_v2));
	cufftSafeCall(cufftXtSetCallback(planinverse_v2, (void **)&hfftshift_1D_chessboard_callback_v1_Ptr, CUFFT_CB_LD_COMPLEX, NULL));
	timerGPU.StartCounter();
	cufftSafeCall(cufftExecC2C(planinverse_v2, d_vect3, d_out3, CUFFT_INVERSE));
	printf("Chessboard v1 elapsed time:  %3.3f ms \n", timerGPU.GetCounter());

	gpuErrchk(cudaMemcpy(h_out3, d_out3, N*sizeof(float2), cudaMemcpyDeviceToHost));

	// --- Checking the results
	for (int i=0; i<N; i++) if ((h_out1[i].x != h_out3[i].x)||(h_out1[i].y != h_out3[i].y)) { printf("Chessboard v1 test failed!\n"); return 0; }

	printf("Chessboard v1 test passed!\n");

	/****************************************/
	/* CHESSBOARD MULTIPLICATION V2 + cuFFT */
	/****************************************/
	cufftCallbackLoadC hfftshift_1D_chessboard_callback_v2_Ptr;

	gpuErrchk(cudaMemcpyFromSymbol(&hfftshift_1D_chessboard_callback_v2_Ptr, fftshift_1D_chessboard_callback_v2_Ptr, sizeof(hfftshift_1D_chessboard_callback_v2_Ptr)));
	cufftHandle planinverse_v3; cufftSafeCall(cufftCreate(&planinverse_v3));
	size_t work_size_v3; cufftSafeCall(cufftMakePlan1d(planinverse_v3, N, CUFFT_C2C, 1, &work_size_v3));
	cufftSafeCall(cufftXtSetCallback(planinverse_v3, (void **)&hfftshift_1D_chessboard_callback_v2_Ptr, CUFFT_CB_LD_COMPLEX, 0));
	timerGPU.StartCounter();
	cufftSafeCall(cufftExecC2C(planinverse_v3, d_vect4, d_out4, CUFFT_INVERSE));
	printf("Chessboard v2 elapsed time:  %3.3f ms \n", timerGPU.GetCounter());

	gpuErrchk(cudaMemcpy(h_out4, d_out4, N*sizeof(float2), cudaMemcpyDeviceToHost));

	// --- Checking the results
	for (int i=0; i<N; i++) if ((h_out1[i].x != h_out4[i].x)||(h_out1[i].y != h_out4[i].y)) { printf("Chessboard v2 test failed!\n"); return 0; }

	printf("Chessboard v2 test passed!\n");

	/****************************************/
	/* CHESSBOARD MULTIPLICATION V3 + cuFFT */
	/****************************************/
	cufftCallbackLoadC hfftshift_1D_chessboard_callback_v3_Ptr;

	gpuErrchk(cudaMemcpyFromSymbol(&hfftshift_1D_chessboard_callback_v3_Ptr, fftshift_1D_chessboard_callback_v3_Ptr, sizeof(hfftshift_1D_chessboard_callback_v3_Ptr)));
	cufftHandle planinverse_v4; cufftSafeCall(cufftCreate(&planinverse_v4));
	size_t work_size_v4; cufftSafeCall(cufftMakePlan1d(planinverse_v4, N, CUFFT_C2C, 1, &work_size_v4));
	cufftSafeCall(cufftXtSetCallback(planinverse_v4, (void **)&hfftshift_1D_chessboard_callback_v3_Ptr, CUFFT_CB_LD_COMPLEX, 0));
	timerGPU.StartCounter();
	cufftSafeCall(cufftExecC2C(planinverse_v4, d_vect5, d_out5, CUFFT_INVERSE));
	printf("Chessboard v3 elapsed time:  %3.3f ms \n", timerGPU.GetCounter());

	gpuErrchk(cudaMemcpy(h_out5, d_out5, N*sizeof(float2), cudaMemcpyDeviceToHost));

	// --- Checking the results
	for (int i=0; i<N; i++) if ((h_out1[i].x != h_out5[i].x)||(h_out1[i].y != h_out5[i].y)) { printf("Chessboard v3 test failed!\n"); return 0; }

	printf("Chessboard v3 test passed!\n");

	return 0;
}
