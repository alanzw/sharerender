#ifndef __CUDAFILTER_H__
#define __CUDAFILTER_H__
// this is for the cuda filter
// use the cuda runtime api

#include <d3d9.h>
#include <d3d10.h>
#include <d3d11.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda.h>
#include <cuda_d3d9_interop.h>
#include "..\Utility\DXDataTypes.h"
#include "..\VideoUtility\inforecoder.h"
//#include "..\VideoUtility\inforecoder.h"
#include "GpuMat.hpp"


//#define USE_NORMAL

#define SAVE_FILTER_IMAGE

extern InfoRecorder * infoRecorder;

class CudaFilter{
	Surface2D * sourceSurface;

	//GpuMat * srcMat;
	//IDirect3DDevice9 * device;
#if 0
	void * pDevice;
#else
	IDXGISwapChain * swapChain;
	DXDevice * dxDevice;   // must set, the same pointer with render
#endif
	DX_VERSION dxVersion;  // the version of the directx

	//D3DFORMAT surfaceFormat;
	cudaGraphicsResource_t  graphicResource;
	D3DFORMAT surfaceFormat;

	GpuMat *srcMat;
	int width, height;
	int pitch;
public:
	//CudaFilter(IDirect3DDevice9* device);
	CudaFilter(void * device, DX_VERSION);
	~CudaFilter();

	bool initSurface(int width, int height);
	//void setSurface(IDirect3DSurface9 * surface, int widht, int height);
	void setSurface(DXSurface * surface, int width, int height);

	void runKernels(CUdeviceptr dstPtr, Surface2D * source = NULL);
	void runKernels(GpuMat & dst);

	//bool updateSurface(IDirect3DSurface9 * surface);
	bool updateSurface(DXSurface * surface);

	HRESULT registerD3DResourceWithCUDA(void * source = NULL);
	HRESULT unregisterD3DResourceWithCUDA();
	HRESULT initD3D(HWND hwnd, DX_VERSION ver);
	HRESULT initCuda(DX_VERSION ver);
	HRESULT releaseCuda();

};

#endif