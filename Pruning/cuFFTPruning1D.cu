#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <math.h>
#include <fstream>

#include <cuda.h>
#include <cufft.h>
#include <cublas.h>

#include <omp.h>

#include <fftw3.h>

#include "TimingCPU.h"
#include "TimingGPU.cuh"
#include "Utilities.cuh"

#define PI_d			3.141592653589793

#define BLOCKSIZEX		8
#define BLOCKSIZEY		8

#define BLOCKSIZE		512

/******************/
/* STEP #1 ON CPU */
/******************/
void step1CPU(fftw_complex * __restrict h_xpruning, const fftw_complex * __restrict h_x, const int N, const int K) {

//	double factor = -2. * PI_d / (K * N);
//	int n;
//	omp_set_nested(1);
//#pragma omp parallel for private(n) num_threads(4)
//	for (int k = 0; k < K; k++) {
//		double arg1 = factor * k;
//#pragma omp parallel for num_threads(4)
//		for (n = 0; n < N; n++) {
//			double arg = arg1 * n;
//			double cosarg = cos(arg);
//			double sinarg = sin(arg);
//			h_xpruning[k * N + n][0] = h_x[n][0] * cosarg - h_x[n][1] * sinarg;
//			h_xpruning[k * N + n][1] = h_x[n][0] * sinarg + h_x[n][1] * cosarg;
//		}
//	}

	//double factor = -2. * PI_d / (K * N);
	//int k;
	//omp_set_nested(1);
	//#pragma omp parallel for private(k) num_threads(4)
	//for (int n = 0; n < N; n++) {
	//	double arg1 = factor * n;
	//	#pragma omp parallel for num_threads(4)
	//	for (k = 0; k < K; k++) {
	//		double arg = arg1 * k;
	//		double cosarg = cos(arg);
	//		double sinarg = sin(arg);
	//		h_xpruning[k * N + n][0] = h_x[n][0] * cosarg - h_x[n][1] * sinarg;
	//		h_xpruning[k * N + n][1] = h_x[n][0] * sinarg + h_x[n][1] * cosarg;
	//	}
	//}

	//double factor = -2. * PI_d / (K * N);
	//for (int k = 0; k < K; k++) {
	//	double arg1 = factor * k;
	//	for (int n = 0; n < N; n++) {
	//		double arg = arg1 * n;
	//		double cosarg = cos(arg);
	//		double sinarg = sin(arg);
	//		h_xpruning[k * N + n][0] = h_x[n][0] * cosarg - h_x[n][1] * sinarg;
	//		h_xpruning[k * N + n][1] = h_x[n][0] * sinarg + h_x[n][1] * cosarg;
	//	}
	//}

	//double factor = -2. * PI_d / (K * N);
	//for (int n = 0; n < N; n++) {
	//	double arg1 = factor * n;
	//	for (int k = 0; k < K; k++) {
	//		double arg = arg1 * k;
	//		double cosarg = cos(arg);
	//		double sinarg = sin(arg);
	//		h_xpruning[k * N + n][0] = h_x[n][0] * cosarg - h_x[n][1] * sinarg;
	//		h_xpruning[k * N + n][1] = h_x[n][0] * sinarg + h_x[n][1] * cosarg;
	//	}
	//}

	double factor = -2. * PI_d / (K * N);
	#pragma omp parallel for num_threads(8)
	for (int n = 0; n < K * N; n++) {
		int row = n / N;
		int col = n % N;
		double arg = factor * row * col;
		double cosarg = cos(arg);
		double sinarg = sin(arg);
		h_xpruning[n][0] = h_x[col][0] * cosarg - h_x[col][1] * sinarg;
		h_xpruning[n][1] = h_x[col][0] * sinarg + h_x[col][1] * cosarg;
	}
}

/******************/
/* STEP #3 ON CPU */
/******************/
void step3CPU(fftw_complex * __restrict h_xhatpruning, const fftw_complex * __restrict h_xhatpruning_temp, const int N, const int K) {

	//int k;
	//omp_set_nested(1);
	//#pragma omp parallel for private(k) num_threads(4)
	//for (int p = 0; p < N; p++) {
	//	#pragma omp parallel for num_threads(4)
	//	for (k = 0; k < K; k++) {
	//		h_xhatpruning[p * K + k][0] = h_xhatpruning_temp[p + k * N][0];
	//		h_xhatpruning[p * K + k][1] = h_xhatpruning_temp[p + k * N][1];
	//	}
	//}	
	
	//int p;
	//omp_set_nested(1);
	//#pragma omp parallel for private(p) num_threads(4)
	//for (int k = 0; k < K; k++) {
	//	#pragma omp parallel for num_threads(4)
	//	for (p = 0; p < N; p++) {
	//		h_xhatpruning[p * K + k][0] = h_xhatpruning_temp[p + k * N][0];
	//		h_xhatpruning[p * K + k][1] = h_xhatpruning_temp[p + k * N][1];
	//	}
	//}

	//for (int p = 0; p < N; p++) {
	//	for (int k = 0; k < K; k++) {
	//		h_xhatpruning[p * K + k][0] = h_xhatpruning_temp[p + k * N][0];
	//		h_xhatpruning[p * K + k][1] = h_xhatpruning_temp[p + k * N][1];
	//	}
	//}
	
	//for (int k = 0; k < K; k++) {
	//	for (int p = 0; p < N; p++) {
	//		h_xhatpruning[p * K + k][0] = h_xhatpruning_temp[p + k * N][0];
	//		h_xhatpruning[p * K + k][1] = h_xhatpruning_temp[p + k * N][1];
	//	}
	//}

	#pragma omp parallel for num_threads(8)
	for (int p = 0; p < K * N; p++) {
		int col = p % N;
		int row = p / K;
		h_xhatpruning[col * K + row][0] = h_xhatpruning_temp[col + row * N][0];
		h_xhatpruning[col * K + row][1] = h_xhatpruning_temp[col + row * N][1];
	}

	//for (int p = 0; p < N; p += 2) {
	//	for (int k = 0; k < K; k++) {
	//		for (int p0 = 0; p0 < 2; p0++) {
	//			h_xhatpruning[(p + p0) * K + k][0] = h_xhatpruning_temp[(p + p0) + k * N][0];
	//			h_xhatpruning[(p + p0) * K + k][1] = h_xhatpruning_temp[(p + p0) + k * N][1];
	//		}
	//	}
	//}

}

/******************/
/* STEP #1 ON GPU */
/******************/
//__global__ void step1GPUkernel(double2 * __restrict__ d_xpruning, const double2 * __restrict__ d_x, const int N, const int K) {
//
//	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
//
//	if (tidx >= K * N) return;
//
//	double factor = -2. * PI_d / (K * N);
//	int row = tidx / N;
//	int col = tidx % N;
//	double arg = factor * row * col;
//	double cosarg = cos(arg);
//	double sinarg = sin(arg);
//	d_xpruning[tidx].x = d_x[col].x * cosarg - d_x[col].y * sinarg;
//	d_xpruning[tidx].y = d_x[col].x * sinarg + d_x[col].y * cosarg;
//}
//
//void step1GPU(double2 * __restrict__ d_xpruning, const double2 * __restrict__ d_x, const int N, const int K) {
//
//	step1GPUkernel << <iDivUp(N * K, BLOCKSIZE), BLOCKSIZE>> >(d_xpruning, d_x, N, K);
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//}

//__global__ void step1GPUkernel(double2 * __restrict__ d_xpruning, const double2 * __restrict__ d_x, const int N, const int K) {
//
//	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
//	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;
//
//	if ((tidx >= N) || (tidy >= K)) return;
//
//	double factor = -2. * PI_d / (K * N);
//	double arg1 = factor * tidy;
//	double arg = arg1 * tidx;
//	double cosarg = cos(arg);
//	double sinarg = sin(arg);
//	d_xpruning[tidy * N + tidx].x = d_x[tidx].x * cosarg - d_x[tidx].y * sinarg;
//	d_xpruning[tidy * N + tidx].y = d_x[tidx].x * sinarg + d_x[tidx].y * cosarg;
//
//}
//
//void step1GPU(double2 * __restrict__ d_xpruning, const double2 * __restrict__ d_x, const int N, const int K) {
//
//	dim3 GridSize(iDivUp(N, BLOCKSIZEX), iDivUp(K, BLOCKSIZEY));
//	dim3 BlockSize(BLOCKSIZEX, BLOCKSIZEY);
//
//	step1GPUkernel << <GridSize, BlockSize>> >(d_xpruning, d_x, N, K);
//	gpuErrchk(cudaPeekAtLastError());
//	gpuErrchk(cudaDeviceSynchronize());
//}

__global__ void step1GPUkernel(double2 * __restrict__ d_xpruning, const double2 * __restrict__ d_x, const int N, const int K) {

	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= K) || (tidy >= N)) return;

	double factor = -2. * PI_d / (K * N);
	double arg1 = factor * tidx;
	double arg = arg1 * tidy;
	double cosarg = cos(arg);
	double sinarg = sin(arg);
	d_xpruning[tidx * N + tidy].x = d_x[tidy].x * cosarg - d_x[tidy].y * sinarg;
	d_xpruning[tidx * N + tidy].y = d_x[tidy].x * sinarg + d_x[tidy].y * cosarg;

}

void step1GPU(double2 * __restrict__ d_xpruning, const double2 * __restrict__ d_x, const int N, const int K) {

	dim3 GridSize(iDivUp(K, BLOCKSIZEX), iDivUp(N, BLOCKSIZEY));
	dim3 BlockSize(BLOCKSIZEX, BLOCKSIZEY);

	step1GPUkernel << <GridSize, BlockSize >> >(d_xpruning, d_x, N, K);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

/******************/
/* STEP #3 ON GPU */
/******************/
__global__ void step3GPUkernel(double2 * __restrict__ d_xhatpruning, const double2 * __restrict__ d_xhatpruning_temp, const int N, const int K) {

	const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	const int tidy = threadIdx.y + blockIdx.y * blockDim.y;

	if ((tidx >= N) || (tidy >= K)) return;

	d_xhatpruning[tidx * K + tidy].x = d_xhatpruning_temp[tidx + tidy * N].x;
	d_xhatpruning[tidx * K + tidy].y = d_xhatpruning_temp[tidx + tidy * N].y;

	//printf("%f %f\n", d_xhatpruning_temp[tidx + tidy * N].x, d_xhatpruning_temp[tidx + tidy * N].y);
	//printf("%i %f %f\n", tidx * K + tidy, d_xhatpruning[tidx * K + tidy].x, d_xhatpruning[tidx * K + tidy].y);
}

void step3GPU(double2 * __restrict d_xhatpruning, const double2 * __restrict d_xhatpruning_temp, const int N, const int K) {

	dim3 GridSize(iDivUp(N, BLOCKSIZEX), iDivUp(K, BLOCKSIZEY));
	dim3 BlockSize(BLOCKSIZEX, BLOCKSIZEY);

	step3GPUkernel << <GridSize, BlockSize >> >(d_xhatpruning, d_xhatpruning_temp, N, K);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
}

/********/
/* MAIN */
/********/
void main() {

	int N = 10;
	int K = 100000;

	// --- CPU memory allocations
	fftw_complex *h_x = (fftw_complex *)malloc(N     * sizeof(fftw_complex));
	fftw_complex *h_xzp = (fftw_complex *)calloc(N * K, sizeof(fftw_complex));
	fftw_complex *h_xpruning = (fftw_complex *)malloc(N * K * sizeof(fftw_complex));
	fftw_complex *h_xhatpruning = (fftw_complex *)malloc(N * K * sizeof(fftw_complex));
	fftw_complex *h_xhatpruning_temp = (fftw_complex *)malloc(N * K * sizeof(fftw_complex));
	fftw_complex *h_xhat = (fftw_complex *)malloc(N * K * sizeof(fftw_complex));
	double2		 *h_xhatGPU = (double2 *)malloc(N * K * sizeof(double2));

	// --- GPU memory allocations
	double2 *d_x;					gpuErrchk(cudaMalloc((void**)&d_x,					N     * sizeof(double2)));
	double2 *d_xzp;					gpuErrchk(cudaMalloc((void**)&d_xzp,				N * K * sizeof(double2)));
	gpuErrchk(cudaMemset(d_xzp, 0, N * K * sizeof(double2)));
	double2 *d_xpruning;			gpuErrchk(cudaMalloc((void**)&d_xpruning,			N * K * sizeof(double2)));
	double2 *d_xhatpruning;			gpuErrchk(cudaMalloc((void**)&d_xhatpruning,		N * K * sizeof(double2)));
	double2 *d_xhatpruning_temp;	gpuErrchk(cudaMalloc((void**)&d_xhatpruning_temp,	N * K * sizeof(double2)));
	double2 *d_xhat;				gpuErrchk(cudaMalloc((void**)&d_xhat,				N * K * sizeof(double2)));

	// --- Random number generation of the data sequence on the CPU - moving the data from CPU to GPU
	srand(time(NULL));
	for (int k = 0; k < N; k++) {
		h_x[k][0] = (double)rand() / (double)RAND_MAX;
		h_x[k][1] = (double)rand() / (double)RAND_MAX;
	}
	gpuErrchk(cudaMemcpy(d_x, h_x, N * sizeof(double2), cudaMemcpyHostToDevice));

	//double *h_xreal = (double *)malloc(N * sizeof(double));
	//std::ifstream infilereal, infileimag;
	//infilereal.open("datareal.txt");
	//infileimag.open("dataimag.txt");
	//for (int i = 0; i < N; i++) {
	//	infilereal >> h_x[i][0];
	//	infileimag >> h_x[i][1];
	//}
	//infilereal.close();
	//infileimag.close();

	// --- Zero padding the input sequence
	memcpy(h_xzp, h_x, N * sizeof(fftw_complex));
	gpuErrchk(cudaMemcpy(d_xzp, h_x, N * sizeof(double2), cudaMemcpyHostToDevice));

	// --- FFTW and cuFFT plans
	fftw_plan h_plan_zp		 = fftw_plan_dft_1d(N * K, h_xzp, h_xhat, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan h_plan_pruning = fftw_plan_many_dft(1, &N, K, h_xpruning, NULL, 1, N, h_xhatpruning_temp, NULL, 1, N, FFTW_FORWARD, FFTW_ESTIMATE);
	
	cufftHandle d_plan_zp;			cufftSafeCall(cufftPlan1d(&d_plan_zp, N * K, CUFFT_Z2Z, 1));
	cufftHandle d_plan_pruning;		cufftSafeCall(cufftPlanMany(&d_plan_pruning, 1, &N, NULL, 1, N, NULL, 1, N, CUFFT_Z2Z, K));

	double totalTimeCPU = 0., totalTimeGPU = 0.;
	double partialTimeCPU, partialTimeGPU;
	
	/****************************/
	/* STANDARD APPROACH ON CPU */
	/****************************/
	printf("Number of processors available = %i\n", omp_get_num_procs());
	printf("Number of threads              = %i\n", omp_get_max_threads());

	TimingCPU timerCPU;
	timerCPU.StartCounter();
	fftw_execute(h_plan_zp);
	printf("\nStadard on CPU: \t \t %f\n", timerCPU.GetCounter());

	/******************/
	/* STEP #1 ON CPU */
	/******************/
	timerCPU.StartCounter();
	step1CPU(h_xpruning, h_x, N, K);
	partialTimeCPU = timerCPU.GetCounter();
	totalTimeCPU = totalTimeCPU + partialTimeCPU;
	printf("\nOptimized first step CPU: \t %f\n", totalTimeCPU);

	/******************/
	/* STEP #2 ON CPU */
	/******************/
	timerCPU.StartCounter();
	fftw_execute(h_plan_pruning);
	partialTimeCPU = timerCPU.GetCounter();
	totalTimeCPU = totalTimeCPU + partialTimeCPU;
	printf("Optimized second step CPU: \t %f\n", timerCPU.GetCounter());

	/******************/
	/* STEP #3 ON CPU */
	/******************/
	timerCPU.StartCounter();
	step3CPU(h_xhatpruning, h_xhatpruning_temp, N, K);
	partialTimeCPU = timerCPU.GetCounter();
	totalTimeCPU = totalTimeCPU + partialTimeCPU;
	printf("Optimized third step CPU: \t %f\n", partialTimeCPU);

	printf("Total time CPU: \t \t %f\n", totalTimeCPU);

	/****************************/
	/* STANDARD APPROACH ON GPU */
	/****************************/
	TimingGPU timerGPU;
	timerGPU.StartCounter();
	cufftSafeCall(cufftExecZ2Z(d_plan_zp, (cufftDoubleComplex *)d_xzp, (cufftDoubleComplex *)d_xhat, CUFFT_FORWARD));
	printf("\nStadard on GPU: \t \t %f\n", timerGPU.GetCounter());

	/******************/
	/* STEP #1 ON GPU */
	/******************/
	timerGPU.StartCounter();
	step1GPU(d_xpruning, d_x, N, K);
	partialTimeGPU = timerGPU.GetCounter();
	totalTimeGPU = totalTimeGPU + partialTimeGPU;
	printf("\nOptimized first step CPU: \t %f\n", totalTimeGPU);

	//gpuErrchk(cudaMemcpy(h_xhatGPU, d_xpruning, N * K * sizeof(double2), cudaMemcpyDeviceToHost));
	//double rmserror = 0., norm = 0.;
	//for (int k = 0; k < K * N; k++) {
	//	rmserror = rmserror + (h_xhatGPU[k].x - h_xpruning[k][0]) * (h_xhatGPU[k].x - h_xpruning[k][0]) + (h_xhatGPU[k].y - h_xpruning[k][1]) * (h_xhatGPU[k].y - h_xpruning[k][1]);
	//	norm = norm + h_xpruning[k][0] * h_xpruning[k][0] + h_xpruning[k][1] * h_xpruning[k][1];
	//	//printf("%f %f %f %f\n", h_xhatGPU[k].x, h_xhatGPU[k].y, h_xhat[k][0], h_xhat[k][1]);
	//}
	//printf("rmserror between first step CPU - GPU %f\n", 100. * sqrt(rmserror / norm));

	/******************/
	/* STEP #2 ON GPU */
	/******************/
	timerGPU.StartCounter();
	cufftSafeCall(cufftExecZ2Z(d_plan_pruning, (cufftDoubleComplex *)d_xpruning, (cufftDoubleComplex *)d_xhatpruning_temp, CUFFT_FORWARD));
	partialTimeGPU = timerGPU.GetCounter();
	totalTimeGPU = totalTimeGPU + partialTimeGPU;
	printf("Optimized second step GPU: \t %f\n", partialTimeGPU);

	//gpuErrchk(cudaMemcpy(h_xhatGPU, d_xhatpruning_temp, N * K * sizeof(double2), cudaMemcpyDeviceToHost));
	//double rmserror = 0., norm = 0.;
	//for (int k = 0; k < K * N; k++) {
	//	rmserror = rmserror + (h_xhatGPU[k].x - h_xhatpruning_temp[k][0]) * (h_xhatGPU[k].x - h_xhatpruning_temp[k][0]) + (h_xhatGPU[k].y - h_xhatpruning_temp[k][1]) * (h_xhatGPU[k].y - h_xhatpruning_temp[k][1]);
	//	norm = norm + h_xhatpruning_temp[k][0] * h_xhatpruning_temp[k][0] + h_xhatpruning_temp[k][1] * h_xhatpruning_temp[k][1];
	//	//printf("%f %f %f %f\n", h_xhatGPU[k].x, h_xhatGPU[k].y, h_xhatpruning_temp[k][0], h_xhatpruning_temp[k][1]);
	//}
	//printf("rmserror between first step CPU - GPU %f\n", 100. * sqrt(rmserror / norm));

	/******************/
	/* STEP #3 ON GPU */
	/******************/
	cublasHandle_t handle; cublasSafeCall(cublasCreate(&handle));
	//cuDoubleComplex alpha;
	double2 alpha;		alpha.x = 1.;	alpha.y = 0.;
	double2 beta;		beta.x  = 0.;	beta.y  = 0.;

	timerGPU.StartCounter();
	//cublasSafeCall(cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, K, N, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)d_xhatpruning_temp, N, (cuDoubleComplex *)&beta, (cuDoubleComplex *)d_xhatpruning_temp, N, (cuDoubleComplex *)d_xhatpruning, K));
	step3GPU(d_xhatpruning, d_xhatpruning_temp, N, K);
	partialTimeGPU = timerGPU.GetCounter();
	totalTimeGPU = totalTimeGPU + partialTimeGPU;
	printf("Optimized third step GPU: \t %f\n", partialTimeGPU);

	printf("Total time GPU: \t \t %f\n", totalTimeGPU);

	//gpuErrchk(cudaMemcpy(h_xhatGPU, d_xhatpruning, N * K * sizeof(double2), cudaMemcpyDeviceToHost));
	//double rmserror = 0., norm = 0.;
	//for (int k = 0; k < N; k++) {
	//	rmserror = rmserror + (h_xhatGPU[k].x - h_xhatpruning[k][0]) * (h_xhatGPU[k].x - h_xhatpruning[k][0]) + (h_xhatGPU[k].y - h_xhatpruning[k][1]) * (h_xhatGPU[k].y - h_xhatpruning[k][1]);
	//	norm = norm + h_xhatpruning[k][0] * h_xhatpruning[k][0] + h_xhatpruning[k][1] * h_xhatpruning[k][1];
	//	printf("%f %f %f %f\n", h_xhatGPU[k].x, h_xhatGPU[k].y, h_xhatpruning[k][0], h_xhatpruning[k][1]);
	//}
	//printf("rmserror between first step CPU - GPU %f\n", 100. * sqrt(rmserror / norm));

	double rmserror = 0.;
	double norm = 0.;
	for (int n = 0; n < N; n++) {
		rmserror = rmserror + (h_xhatpruning[n][0] - h_xhat[n][0]) * (h_xhatpruning[n][0] - h_xhat[n][0]) + (h_xhatpruning[n][1] - h_xhat[n][1]) * (h_xhatpruning[n][1] - h_xhat[n][1]);
		norm = norm + h_xhat[n][0] * h_xhat[n][0] + h_xhat[n][1] * h_xhat[n][1];
		//printf("%f %f %f %f\n", h_xhat[n][0], h_xhat[n][0], h_xhatpruning[n][0], h_xhatpruning[n][1]);
	}
	printf("\nrmserror %f\n", 100. * sqrt(rmserror / norm));

	fftw_destroy_plan(h_plan_zp);

}
