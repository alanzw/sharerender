#ifndef __CUDAFILTER_H__
#define __CUDAFILTER_H__
// this is for the cuda filter

#include <d3d9.h>
#include <d3d10.h>
#include <d3d11.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <cuda.h>
#include "../LibCore/DXDataTypes.h"
#define SAVE_FILTER_IMAGE

//#define ONLY_D3D9


class CudaFilter{
	Surface2D * sourceSurface;
	//IDirect3DDevice9 * device;

	void * pDevice;
	DX_VERSION dxVersion;  // the version of the directx

	//D3DFORMAT surfaceFormat;

public:
	//CudaFilter(IDirect3DDevice9* device);
	CudaFilter(void * device, DX_VERSION);
	~CudaFilter();

	bool initSurface(int width, int height);
	//void setSurface(IDirect3DSurface9 * surface, int widht, int height);
	void setSurface(DXSurface * surface, int width, int height);

	void runKernels(CUdeviceptr dstPtr, Surface2D * source = NULL);

	//bool updateSurface(IDirect3DSurface9 * surface);
	bool updateSurface(DXSurface * surface);

	HRESULT registerD3DResourceWithCUDA();
	HRESULT unregisterD3DResourceWithCUDA();
	HRESULT initD3D(HWND hwnd);
	HRESULT initCuda(DX_VERSION ver);
	HRESULT releaseCuda();

};

#endif