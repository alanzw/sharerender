#include <cuda_d3d9_interop.h>
//#include <rendercheck_d3d9.h>
#include "CudaFilter.h"
#include <d3dx9.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>

#include <cutil_inline.h>

#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"

// this is for the cuda filter

extern "C"
{
	void cuda_rgba_to_nv12(void * input, CUdeviceptr output, int alignedWidht, int height, int width, int pitch);
	void cuda_gbra_to_nv12(void * input, CUdeviceptr output, int alignedWidth, int height, int width, int pitch);
	void cuda_argb_to_nv12(void * intpu, CUdeviceptr output, int alignedWidth, int height, int width, int pitch);
	void cuda_rgb_to_nv12(void * input, CUdeviceptr output, int alignedWidht, int height, int width, int pitch);
}


// constructor and destructor
#ifdef ONLY_D3D9
CudaFilter::CudaFilter(IDirect3DDevice9 * device){
	this->device = device;
	sourceSurface = NULL;
	surfaceFormat = D3DFORMAT::D3DFMT_A8R8G8B8;   // default use argb format
}
#endif

CudaFilter::CudaFilter(void *device, DX_VERSION ver){
	pDevice = device;
	dxVersion = ver;
}

CudaFilter::~CudaFilter(){

}

bool CudaFilter::initSurface(int width, int height){
	if(sourceSurface){
		infoRecorder->logError("[CudaFilter]: already has a surface.\n");
		return false;
	}

	if(this->pDevice == NULL){
		infoRecorder->logError("[CudaFilter]: NULL device, error.\n");
		return false;
	}
	sourceSurface = new Surface2D();
	sourceSurface->width = width;
	sourceSurface->height = height;
	sourceSurface->dxVersion = dxVersion;

	if(dxVersion == DX9){
		HRESULT hr = ((IDirect3DDevice9 *)pDevice)->CreateOffscreenPlainSurface(width, height, D3DFORMAT::D3DFMT_A8R8G8B8, D3DPOOL::D3DPOOL_DEFAULT, &sourceSurface->pSurface, NULL);
		if(FAILED(hr)){
			infoRecorder->logError("[CudaFilter]: create the filter surface failed.\n");
			return false;
		}
		else{
			this->registerD3DResourceWithCUDA();
		}
	}else if(dxVersion == DX10){

	}
	else if(dxVersion == DX10_1){

	}else if(dxVersion == DX11){

	}
	return true;
}

bool CudaFilter::updateSurface(DXSurface * surface){

}

#ifdef ONLY_D3D9
bool CudaFilter::updateSurface(IDirect3DSurface9 * surface){
	if(surface == NULL){
		infoRecorder->logError("[CudaFilter]: update surface get NULL surface.\n");
		return false;
	}

	HRESULT hr = device->StretchRect(surface, NULL, this->sourceSurface->pSurface, NULL, D3DTEXF_LINEAR);
	if(FAILED(hr)){
		infoRecorder->logError("[CudaFilter]: copy the surface data failed.\n");
		return false;
	}
	return true;
}
#endif
void CudaFilter::setSurface(DXSurface * s, int width, int height){

}
#ifdef ONLY_D3D9
void CudaFilter::setSurface(IDirect3DSurface9 * surface, int width, int height){
	if(sourceSurface){
		// already?
		infoRecorder->logError("[CudaFilter]: already has a surface.\n");

	}
	else{
		sourceSurface = new Surface2D();
	}
	sourceSurface->pSurface = surface;
	sourceSurface->width = width;
	sourceSurface->height = height;
}
#endif

HRESULT CudaFilter::initCuda(DX_VERSION ver){
	
	if(ver == DX9){
		if(pDevice == NULL){
			infoRecorder->logError("[CudaFilter]: NULL d3d device.\n");
		}
		cudaD3D9SetDirect3DDevice(device);
		cutilCheckMsg("[CudaFilter]: cudaD3D9SetDirect3DDevice failed.");
	}else if(ver == DX10){

	}else if(ver == DX11){

	}
	return S_OK;
}

HRESULT CudaFilter::releaseCuda(){
	cudaThreadExit();
	cutilCheckMsg("[CudaFilter]: cudaThreadExit failed.");
	return S_OK;
}

HRESULT CudaFilter::registerD3DResourceWithCUDA(){
	// register the Direct3D resources that we'll use 
	// we'll read from surface, so don't set any special map flags for it
	if(dxVersion == DX9){
		cudaD3D9RegisterResource(sourceSurface->pSurface, cudaD3D9RegisterFlagsNone);
	}else if(dxVersion == DX10){

	}else if(dxVersion == DX11){

	}
	return S_OK;
}

HRESULT CudaFilter::unregisterD3DResourceWithCUDA(){
	if(dxVersion == DX9){
		cudaD3D9UnregisterResource(sourceSurface->pSurface);
		cutilCheckMsg("cudaD3D9UnregisterResource (sourceSurface) failed.");
	}else if(dxVersion == DX10 || dxVersion == DX10_1){
		//cudaD3D10UnregisterResource(
	}else if(dxVersion == DX10_1){

	}
	return S_OK;
}

// run the cuda part of the computation

// the dstPtr is the device memory ptr for storing the converted data
void CudaFilter::runKernels(CUdeviceptr dstPtr, Surface2D * source){
	void * pData = NULL;
	size_t pitch = 0;

	Surface2D * ps = NULL;
	if(source == NULL){
		ps = this->sourceSurface;
	}else{

		ps = source;
	}

	if(dxVersion == DX9){
		// first map the resources
		IDirect3DResource9 * ppResources[1] = {
			ps->pSurface
		};

		cudaD3D9MapResources(1, ppResources);
		cutilCheckMsg("cudaD3D9MapResources(1) failed.");

		cutilSafeCallNoSync(cudaD3D9ResourceGetMappedPointer(&pData, sourceSurface->pSurface, 0, 0));
		cutilSafeCallNoSync(cudaD3D9ResourceGetMappedPitch(&pitch, NULL, sourceSurface->pSurface, 0, 0));

		// run the convert logic
		if(surfaceFormat == D3DFORMAT::D3DFMT_A8R8G8B8){
			cuda_argb_to_nv12(pData, dstPtr, sourceSurface->pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}else if(surfaceFormat == D3DFORMAT::D3DFMT_R8G8B8){
			cuda_rgb_to_nv12(pData, dstPtr, sourceSurface->pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}
		//else if(surfaceFormat == D3DFORMAT::D3DFMT_


		// end the convert logic

		// last, unmap the resource.
		cudaD3D9UnmapResources(1, ppResources);
		cutilCheckMsg("cudaD3D9UnmapResources(1) failed.");
	}else if(dxVersion == DX10 || dxVersion == DX10_1){

	}else if(dxVersion == DX11){

	}
}