
#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <cuda.h>

#include <stdio.h>


// cuda filter to convert the RGBA or GBRA format picture to NV12

#ifdef BT601
#define Ycoeff ((float4)(0.299f, 0.587f, 0.114f, 0.f))
#define Ucoeff ((float4)(-0.14713f, -0.28886f, 0.436f, 0.f))
#define Vcoeff ((float4)(0.615f, -0.51499f, -0.10001f, 0.f))

// BGR
#define YcoeffB ((float4)(0.114f, 0.587f, 0.299f, 0.f))
#define UcoeffB ((float4)(0.436f, -0.28886f, -0.14713f, 0.f))
#define VcoeffB ((float4)(-0.10001f, -0.51499, 0.615f, 0.f))

#else
#ifdef BT709

#define Ycoeff ((float4)(0.2126f, 0.7152f, 0.0722f, 0.f))
#define Ucoeff ((float4)(-0.09991f, -0.33609f, -.436f, 0.f))
#define Vcoeff ((float4)(0.615f, -0.55861f, -0.05639f, 0.f))

// BGR
#define YcoeffB ((float4)(0.0722f, 0.7152f, 0.2126f, 0.f))
#define UcoeffB ((float4)(0.436f, -0.33609f, -0.09991f, 0.f))
#define VcoeffB ((float4)(-0.05639f, -0.55861f, 0.615f, 0.f))

#else

#define Ycoeff ((float4)(0.257f, 0.504f, 0.098f, 0.f))
#define Ucoeff ((float4)(-0.148f, -0.291f, 0.439f, 0.f))
#define Vcoeff ((flaot4)(0.439f, -0.368f, -0.071f, 0.f))

#define YcoeffB ((float4)(0.098f, 0.504f, 0.257f, 0.f))
#define UcoeffB ((float4)(0.439f, -0.291f, -0.148f, 0))
#define VcoeffB ((float4)(-0.071f, -0.368f, 0.439f, 0))

#endif // BT709
#endif // BT601
// the cuda kernel to convert rgba data to nv12
__global__ void cuda_kernel_rgba_to_nv12(unsigned char * inputSrc, unsigned char * output, int alignedWidth, int width, int height, int pitch){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int yIndex = 0;
	int uvIndex = width * height;
	int offset = 0;

	unsigned int pixel;

	// in the case where, due to quantization into frids, we have more threads than pixels, skip the threads which dont's correspond to valid pixels
	if( x >= width || y >= height) return;
	// get the pointer to the pixel at (x, y)
	
	pixel = *(unsigned int *)((inputSrc + y * alignedWidth) + 4 * x);
	offset = y * pitch + x;

	int Y = 0, U = 0, V = 0; // the yuv values
	int a = 0, r = 0, g = 0, b = 0;

#if 1
	r = (pixel & 0xff000000) >> 24; // a is not used obviously
	g = (pixel & 0xff0000) >> 16;
	b = (pixel & 0xff00) >> 8;
	a = (pixel & 0xff) >> 0;

#else
	r = (pixel & 0xff000000) >> 24;
	g = (pixel & 0xff0000) >> 16;
	b = (pixel & 0xff00) >> 8;
#endif

	// rgb to yuv
	Y = (( 66 * r + 129 * g + 25 * b + 128 ) >> 8 ) + 16;
	V = (( -38 * r + 74 * g + 112 * b + 128 ) >> 8) + 128;
	U = (( 112 * r - 94 * g - 18 * b + 128 ) >> 8) + 128;

	// write to output array
#if 0
	output[y * alignedWidth + x] = Y;
	output[alignedWidth * height + (y >> 1) * alignedWidth + ( x >> 1) * 2] = U;
	output[alignedWidth * height + ( y >> 1) * alignedWidth + ( x >> 1) * 2 + 1] = V;
#else 
	output[y * pitch + x] = Y;
	output[pitch * height + (y >> 1) * pitch + (x >> 1) * 2] = U;
	output[pitch * height + (y >> 1) * pitch + (x >> 1) * 2 + 1] = V;
#endif
}

// the cuda kernel to convert gbra data to nv12
__global__ void cuda_kernel_gbra_to_nv12(unsigned char * inputSrc, unsigned char * output, int srcAlignedWidth, int width, int height, int pitch){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int yIndex = 0;
	int uvIndex = width * height;
	int offset = 0;

	unsigned int pixel;

	// in the case where, due to quantization into frids, we have more threads than pixels, skip the threads which dont's correspond to valid pixels
	if( x >= width || y >= height) return;
	// get the pointer to the pixel at (x, y)
	
	pixel = *(unsigned int *)((inputSrc + y * srcAlignedWidth) + 4 * x);
	offset = y * pitch + x;

	int Y = 0, U = 0, V = 0; // the yuv values
	int a = 0, r = 0, g = 0, b = 0;

#if 1
	g = (pixel & 0xff000000) >> 24; // a is not used obviously
	b = (pixel & 0xff0000) >> 16;
	r = (pixel & 0xff00) >> 8;
	a = (pixel & 0xff) >> 0;

#else
	r = (pixel & 0xff000000) >> 24;
	g = (pixel & 0xff0000) >> 16;
	b = (pixel & 0xff00) >> 8;
#endif

	// rgb to yuv
	Y = (( 66 * r + 129 * g + 25 * b + 128 ) >> 8 ) + 16;
	V = (( -38 * r + 74 * g + 112 * b + 128 ) >> 8) + 128;
	U = (( 112 * r - 94 * g - 18 * b + 128 ) >> 8) + 128;

	// write to output array
#if 0
	output[y * alignedWidth + x] = Y;
	output[alignedWidth * height + (y >> 1) * alignedWidth + ( x >> 1) * 2] = U;
	output[alignedWidth * height + ( y >> 1) * alignedWidth + ( x >> 1) * 2 + 1] = V;
#else 
	output[y * pitch + x] = Y;
	output[pitch * height + (y >> 1) * pitch + (x >> 1) * 2] = U;
	output[pitch * height + (y >> 1) * pitch + (x >> 1) * 2 + 1] = V;
#endif
}

// the cuda kernel to convert argb data to nv12
__global__ void cuda_kernel_argb_to_nv12(unsigned char * inputSrc, unsigned char * output, int srcAlignedWidth, int width, int height, int pitch){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int yIndex = 0;
	int uvIndex = width * height;
	int offset = 0;

	unsigned int pixel;

	// in the case where, due to quantization into frids, we have more threads than pixels, skip the threads which dont's correspond to valid pixels
	if( x >= width || y >= height) return;
	// get the pointer to the pixel at (x, y)
	
	pixel = *(unsigned int *)((inputSrc + y * srcAlignedWidth) + 4 * x);
	offset = y * pitch + x;

	int Y = 0, U = 0, V = 0; // the yuv values
	int a = 0, r = 0, g = 0, b = 0;

#if 1
	a = (pixel & 0xff000000) >> 24; // a is not used obviously
	r = (pixel & 0xff0000) >> 16;
	g = (pixel & 0xff00) >> 8;
	b = (pixel & 0xff) >> 0;

#else
	r = (pixel & 0xff000000) >> 24;
	g = (pixel & 0xff0000) >> 16;
	b = (pixel & 0xff00) >> 8;
#endif

	// rgb to yuv
	Y = (( 66 * r + 129 * g + 25 * b + 128 ) >> 8 ) + 16;
	V = (( -38 * r + 74 * g + 112 * b + 128 ) >> 8) + 128;
	U = (( 112 * r - 94 * g - 18 * b + 128 ) >> 8) + 128;

	// write to output array
#if 0
	output[y * alignedWidth + x] = Y;
	output[alignedWidth * height + (y >> 1) * alignedWidth + ( x >> 1) * 2] = U;
	output[alignedWidth * height + ( y >> 1) * alignedWidth + ( x >> 1) * 2 + 1] = V;
#else 
	output[y * pitch + x] = Y;
	output[pitch * height + (y >> 1) * pitch + (x >> 1) * 2] = U;
	output[pitch * height + (y >> 1) * pitch + (x >> 1) * 2 + 1] = V;
#endif
}


// the cuda kernel to convert rgb data to nv12
__global__ void cuda_kernel_rgb_to_nv12(unsigned char * inputSrc, unsigned char * output, int srcAlignedWidth, int width, int height, int pitch){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int yIndex = 0;
	int uvIndex = width * height;
	int offset = 0;

	unsigned int pixel;

	// in the case where, due to quantization into frids, we have more threads than pixels, skip the threads which dont's correspond to valid pixels
	if( x >= width || y >= height) return;
	// get the pointer to the pixel at (x, y)
	
	pixel = *(unsigned int *)((inputSrc + y * srcAlignedWidth) + 3 * x);
	offset = y * pitch + x;

	int Y = 0, U = 0, V = 0; // the yuv values
	int a = 0, r = 0, g = 0, b = 0;

#if 1
	r = (pixel & 0xff000000) >> 24; // a is not used obviously
	g = (pixel & 0xff0000) >> 16;
	b = (pixel & 0xff00) >> 8;
	//b = (pixel & 0xff) >> 0;

#else
	r = (pixel & 0xff000000) >> 24;
	g = (pixel & 0xff0000) >> 16;
	b = (pixel & 0xff00) >> 8;
#endif

	// rgb to yuv
	Y = (( 66 * r + 129 * g + 25 * b + 128 ) >> 8 ) + 16;
	V = (( -38 * r + 74 * g + 112 * b + 128 ) >> 8) + 128;
	U = (( 112 * r - 94 * g - 18 * b + 128 ) >> 8) + 128;

	// write to output array
#if 0
	output[y * srcAlignedWidth + x] = Y;
	output[srcAlignedWidth * height + (y >> 1) * srcAlignedWidth + ( x >> 1) * 2] = U;
	output[srcAlignedWidth * height + ( y >> 1) * srcAlignedWidth + ( x >> 1) * 2 + 1] = V;
#else 
	output[y * pitch + x] = Y;
	output[pitch * height + (y >> 1) * pitch + ( x >> 1) * 2] = U;
	output[pitch * height + ( y >> 1) * pitch+ ( x >> 1) * 2 + 1] = V;
#endif
}

extern "C"
	void cuda_rgba_to_nv12(void * input, CUdeviceptr dstPtr, int alignedWidth, int height, int width, int pitch){
		cudaError_t error = cudaSuccess;
		dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x -1)/Db.x, (height+Db.y -1)/Db.y);

		cuda_kernel_rgba_to_nv12<<<Dg, Db>>>((unsigned char * )input, (unsigned char *)dstPtr, alignedWidth, width, height, pitch);

		error = cudaGetLastError();
		if(error != cudaSuccess){
			printf("cuda_kernel_rgba_to_nv12() failed to launch error = %d\n", error);

		}
}

extern "C"
	void cuda_gbra_to_nv12(void * input, CUdeviceptr dstPtr, int alignedWidth, int height, int width, int pitch){
		cudaError_t error = cudaSuccess;
		dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x -1)/Db.x, (height+Db.y -1)/Db.y);

		cuda_kernel_gbra_to_nv12<<<Dg, Db>>>((unsigned char * )input, (unsigned char *)dstPtr, alignedWidth, width, height, pitch);

		error = cudaGetLastError();
		if(error != cudaSuccess){
			printf("cuda_kernel_rgba_to_nv12() failed to launch error = %d\n", error);

		}
}

extern "C"
	void cuda_argb_to_nv12(void * input, CUdeviceptr dstPtr, int alignedWidth, int height, int width, int pitch){
		cudaError_t error = cudaSuccess;
		dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x -1)/Db.x, (height+Db.y -1)/Db.y);

		cuda_kernel_argb_to_nv12<<<Dg, Db>>>((unsigned char * )input, (unsigned char *)dstPtr, alignedWidth, width, height, pitch);

		error = cudaGetLastError();
		if(error != cudaSuccess){
			printf("cuda_kernel_rgba_to_nv12() failed to launch error = %d\n", error);

		}
}

extern "C"
	void cuda_rgb_to_nv12(void * input, CUdeviceptr dstPtr, int alignedWidth, int height, int width, int pitch){
		cudaError_t error = cudaSuccess;
		dim3 Db = dim3(16, 16);  // block dimensions are fixed to be 256 threads
		dim3 Dg = dim3((width + Db.x -1)/Db.x, (height+Db.y -1)/Db.y);

		cuda_kernel_argb_to_nv12<<<Dg, Db>>>((unsigned char * )input, (unsigned char *)dstPtr, alignedWidth, width, height, pitch);

		error = cudaGetLastError();
		if(error != cudaSuccess){
			printf("cuda_kernel_rgba_to_nv12() failed to launch error = %d\n", error);

		}
}
