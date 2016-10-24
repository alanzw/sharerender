

#include <d3dx9.h>
#include <d3dx9tex.h>

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d9_interop.h>
#include <cuda_d3d10_interop.h>
#include <cuda_d3d11_interop.h>

#include "check_d3d9.h"

#include "CudaFilter.h"
#include "GpuMat.hpp"
#include "glob.hpp"

#include "bmpformat.h"

//#define D3D9_SPEC

#define __DEBUG__

// this is for the cuda filter

extern "C"
{
	void cuda_rgba_to_nv12(void * input, CUdeviceptr output, int alignedWidht, int height, int width, int pitch);
	void cuda_gbra_to_nv12(void * input, CUdeviceptr output, int alignedWidth, int height, int width, int pitch);
	void cuda_argb_to_nv12(void * intpu, CUdeviceptr output, int alignedWidth, int height, int width, int pitch);
	void cuda_rgb_to_nv12(void * input, CUdeviceptr output, int alignedWidht, int height, int width, int pitch);

	void RGB_to_YV12(const GpuMat &src, GpuMat &dst);
	bool YV12MatToARGB(GpuMat &src, GpuMat &dst);

	bool YV12ToARGB(unsigned char* pYV12,unsigned char* pARGB,int width,int height);

	void RGBA_to_NV12(const GpuMat & src, GpuMat & dst);
}


IDirect3DSurface9 * sysSurface = NULL;

// constructor and destructor
#ifdef ONLY_D3D9
CudaFilter::CudaFilter(IDirect3DDevice9 * device){
	this->device = device;
	sourceSurface = NULL;
	surfaceFormat = D3DFORMAT::D3DFMT_A8R8G8B8;   // default use argb format
}
#endif

CudaFilter::CudaFilter(void *device, DX_VERSION ver){
	srcMat = NULL;
	dxDevice = new DXDevice();
	dxDevice->d10Device1 = (ID3D10Device1 *)device;
	dxVersion = ver;
	sourceSurface = NULL; /// new Surface2D();
	//sourceSurface
}

CudaFilter::~CudaFilter(){
	if(sourceSurface){
		delete sourceSurface;
		sourceSurface = NULL;
	}

}

bool CudaFilter::initSurface(int width, int height){
	if(sourceSurface){
		infoRecorder->logError("[CudaFilter]: already has a surface.\n");
		return false;
	}

	if(this->dxDevice->d9Device == NULL){
		infoRecorder->logError("[CudaFilter]: NULL device, error.\n");
		return false;
	}

	infoRecorder->logError("[CudaFilter]: init surface width:%d, height:%d.\n", width, height);
	sourceSurface = new Surface2D();
	sourceSurface->width = width;
	sourceSurface->height = height;
	sourceSurface->dxVersion = dxVersion;
	sourceSurface->dxSurface = new DXSurface();
	sourceSurface->dxSurface->d10Surface = NULL;
	surfaceFormat = D3DFMT_A8R8G8B8; 

	// create the source gpu mat
	if(srcMat == NULL){
		srcMat = new GpuMat(width, height, CV_8UC3);
		if(srcMat == NULL){
			return false;
		}
	}

#if 0
	if(dxVersion == DX9){
		// create surface
		HRESULT hr = dxDevice->d9Device->CreateOffscreenPlainSurface(width, height, D3DFMT_A8R8G8B8, D3DPOOL::D3DPOOL_DEFAULT, &(sourceSurface->dxSurface->d9Surface), NULL);
		if(FAILED(hr)){
			// error
			return false;
		}

	}else if(dxVersion == DX10){

	}else if(dxVersion == DX10_1){

	}else if(dxVersion == DX11){

	}
#endif

	return true;
}

// copy the surface data to mapped source surface
bool CudaFilter::updateSurface(DXSurface * surface){
	if(sourceSurface == NULL){
		return false;
	}
	if(dxVersion == DX9){
#if 1
		if(sourceSurface->dxSurface->d10Surface == NULL){
			sourceSurface->pSurface = NULL;
			//infoRecorder->logError("[CudaFilter]: to create the source surface.Surface2D: 0x%p, width:%d, height:%d.\n", sourceSurface, sourceSurface->width, sourceSurface->height);
			D3DSURFACE_DESC desc;
			surface->d9Surface->GetDesc(&desc);

			IDirect3DSurface9 * temp = NULL;

			HRESULT hrr = ((IDirect3DDevice9 *)dxDevice->d9Device)->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, D3DPOOL::D3DPOOL_SYSTEMMEM , &(temp), NULL);
			if(SUCCEEDED(hrr)){
				hrr = ((IDirect3DDevice9 *)dxDevice->d9Device)->GetRenderTargetData(surface->d9Surface, temp);
				if(FAILED(hrr)){
					infoRecorder->logError("[CudaFilter]: GetRenderTargetData failed.\n");
				}
			}
			D3DLOCKED_RECT rect;
			hrr = temp->LockRect(&rect, NULL, D3DLOCK_DISCARD);

			CHECK_D3D_ERROR(hrr);

			if(FAILED(hrr)){
				infoRecorder->logError("[CudaFilter]: lock rect failed, pitch:%d.\n", rect.Pitch);
			}else{
				infoRecorder->logError("[CudaFilter]: lock rect success, pitch:%d.\n", rect.Pitch);
			}
			temp->UnlockRect();
			temp->Release();

			this->width = desc.Width;
			this->height = desc.Height;
			this->pitch = rect.Pitch;
			
			infoRecorder->logError("[CudaFilter]: create a source surface, widht:%d, heiht:%d, pitch:%d.\n", desc.Width, desc.Height, pitch);
			// create the surface
			//HRESULT hr = ((IDirect3DDevice9 *)dxDevice->d9Device)->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, desc.Pool , &(sourceSurface->pSurface), NULL);

			HRESULT hr = ((IDirect3DDevice9 *)dxDevice->d9Device)->CreateRenderTarget(desc.Width, desc.Height, desc.Format, D3DMULTISAMPLE_TYPE::D3DMULTISAMPLE_NONE, 0, FALSE, &(sourceSurface->pSurface), NULL);
			
			CHECK_D3D_ERROR(hr);

			if(FAILED(hr)){
				//Log::slog("[CudaFilter]: create the filter surface failed.\n");
				infoRecorder->logError("[CudaFilter]: create source surface failed.\n");
				return false;
			}
			else{
				sourceSurface->dxSurface->d9Surface = sourceSurface->pSurface;
				//sourceSurface->dxSurface->d9Surface = surface->d9Surface;
				if(this->registerD3DResourceWithCUDA() == S_FALSE){
					// register failed.
					return false;
				}
			}
#if 1
			hr = dxDevice->d9Device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, D3DPOOL::D3DPOOL_SYSTEMMEM, &sysSurface, NULL);
			if(FAILED(hr)){
				assert(sysSurface == NULL);
			}

			CHECK_D3D_ERROR(hr);

#endif
		}

		//dxDevice->d9Device->UpdateSurface(surface->d9Surface, NULL, sourceSurface->dxSurface->d9Surface, NULL);
		// it said that, I must use getRenderTargetData to copy the surface to off-scrren surface, cause, stretch did not support if the destination surface is an off-screen plain surface but the source is not. right now, the 'surface' is a renderTarget.
#if 1
		HRESULT hrr = dxDevice->d9Device->StretchRect(surface->d9Surface, NULL, sourceSurface->dxSurface->d9Surface, NULL, D3DTEXTUREFILTERTYPE::D3DTEXF_NONE);

		//CHECK_D3D_ERROR(hrr);
#if 0
		char name[100] = {0};
		static int in = 0;
		sprintf(name, "from-sframe-%d.jpg", in);
		hrr = dxDevice->d9Device->GetRenderTargetData(surface->d9Surface, sysSurface);
		CHECK_D3D_ERROR(hrr);

		D3DXSaveSurfaceToFile(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_JPG, sysSurface, NULL, NULL);
		memset(name, 0, 100);
		sprintf(name, "frame-source-surface-%d.jpg", in);
		hrr = dxDevice->d9Device->GetRenderTargetData(sourceSurface->dxSurface->d9Surface, sysSurface);

		CHECK_D3D_ERROR(hrr);

		D3DXSaveSurfaceToFile(name, D3DXIFF_JPG, sysSurface, NULL, NULL);

		sprintf(name, "filter-surface-%d.jpg", in++);
		D3DXSaveSurfaceToFile(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_JPG, sourceSurface->dxSurface->d9Surface, NULL, NULL);

#endif
#endif
		//CHECK_D3D_ERROR(hrr);
		if(FAILED(hrr)){
			infoRecorder->logError("[CudaFilter]: copy dx9 surface to source surface failed.\n");
		}

		return true;
#else
		
#endif
	}else if(dxVersion == DX10){
		if(sourceSurface == NULL){
			// create the surface
			D3D10_TEXTURE2D_DESC desc;
			surface->d10Surface->GetDesc(&desc);
			desc.BindFlags = 0;
			desc.CPUAccessFlags = D3D10_CPU_ACCESS_FLAG::D3D10_CPU_ACCESS_READ;
			desc.Usage = D3D10_USAGE_STAGING;
			HRESULT hr = dxDevice->d10Device->CreateTexture2D(&desc, NULL, &sourceSurface->dxSurface->d10Surface);
			if(FAILED(hr)){

			}else{
				registerD3DResourceWithCUDA();
			}

			
		}
		dxDevice->d10Device->CopyResource(sourceSurface->dxSurface->d10Surface, surface->d10Surface);
		//12345alandxDevice->d10Device->UpdateSubresource();
	}else if(DX10_1 == dxVersion){
		if(sourceSurface == NULL){
			// create the surface
			D3D10_TEXTURE2D_DESC desc;
			surface->d10Surface->GetDesc(&desc);
			desc.BindFlags = 0;
			desc.CPUAccessFlags = D3D10_CPU_ACCESS_FLAG::D3D10_CPU_ACCESS_READ;
			desc.Usage = D3D10_USAGE_STAGING;
			HRESULT hr = dxDevice->d10Device1->CreateTexture2D(&desc, NULL, &sourceSurface->dxSurface->d10Surface);
			if(FAILED(hr)){

			}else{
				registerD3DResourceWithCUDA();
			}
		}
		dxDevice->d10Device1->CopyResource(sourceSurface->dxSurface->d10Surface, surface->d10Surface);
	}
	else if(dxVersion == DX11){
		if(sourceSurface == NULL){
			// create the surface
			D3D11_TEXTURE2D_DESC desc;
			surface->d11Surface->GetDesc(&desc);
			desc.BindFlags = 0;
			desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			desc.Usage = D3D11_USAGE_STAGING;

			HRESULT hr = dxDevice->d11Device->CreateTexture2D(&desc, NULL, &sourceSurface->dxSurface->d11Surface);
			if(FAILED(hr)){

			}else{
				registerD3DResourceWithCUDA();
			}
		}
		// need to create the device context, so we need IDXGISwapChain  
		//dxDevice->d11Device->
		
	}
}


void CudaFilter::setSurface(DXSurface * s, int width, int height){

}


HRESULT CudaFilter::initCuda(DX_VERSION ver){
	
	if(dxDevice->d10Device == NULL){
		infoRecorder->logError("[CudaEncoder]: get NULL device for init.\n");
		return S_FALSE;
	}
	if(ver == DX9){
#ifndef D3D9_SPEC
		if(cudaD3D9SetDirect3DDevice(dxDevice->d9Device)!=cudaSuccess){
			return S_FALSE;
		}
#else
		infoRecorder->logError("[CudaFilter]: set D3D9Device:0x%p.\n", dxDevice->d9Device);
#if 0
		if(cudaD3D9SetDirect3DDevice(dxDevice->d9Device) != cudaSuccess){
			infoRecorder->logError("[CudaFilter]: cudaD3D9SetDirect3DDevice failed.\n");
			return S_FALSE;
		}
#endif
#endif
	}
	else if(ver == DX10 || ver == DX10_1){
		cudaD3D10SetDirect3DDevice(dxDevice->d10Device);
	}
	else if(ver == DX11){
		cudaD3D11SetDirect3DDevice(dxDevice->d11Device);
	}
	return S_OK;
}

HRESULT CudaFilter::releaseCuda(){
	cudaThreadExit();
	//cutilCheckMsg("[CudaFilter]: cudaThreadExit failed.");
	return S_OK;
}

HRESULT CudaFilter::registerD3DResourceWithCUDA(void * surface){
	// register the Direct3D resources that we'll use 
	// we'll read from surface, so don't set any special map flags for it
	//DebugBreak();
	infoRecorder->logError("[CudaFilter]: registerD3DResourceWithCUDA.\n");
	cudaError err;
	if(dxVersion == DX9){
		//cudaD3D9RegisterResource(sourceSurface->pSurface, cudaD3D9RegisterFlagsNone);
#ifndef D3D9_SPEC
		if(surface){
			sourceSurface->dxSurface->d9Surface = (IDirect3DSurface9 *)surface;
			char name[100] = {0};
			static int index = 0;
			sprintf(name, "before-register-%d.jpg", index++);
			D3DXSaveSurfaceToFile(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_JPG, (IDirect3DSurface9 *)surface, NULL, NULL);
		}
		if(cudaGraphicsD3D9RegisterResource(&graphicResource, sourceSurface->dxSurface->d9Surface, cudaGraphicsRegisterFlagsNone) != cudaSuccess){
			infoRecorder->logError("[CudaFilter]: cudaGraphicsD3D9RegisterResource failed.\n");
			return S_FALSE;
		}
		infoRecorder->logError("[CudaFilter]: cudaGraphicsD3D9RegisterResource, surface:0x%p, returned graphicResource:0x%p.\n", sourceSurface->dxSurface->d9Surface, graphicResource);


		if(sourceSurface->inited == false){
			// malloc the memory
			infoRecorder->logError("[CudaFilter]: init the surface, malloc linear memory.\n");
			//err = 
#if 1

#ifndef USE_NORMAL

			srcMat = new GpuMat(sourceSurface->height, sourceSurface->width, 4);
			//srcMat->channels = 4;
			infoRecorder->logError("[CudaFilter]: GpuMat created, cols:%d, rows:%d, step:%d.\n", srcMat->cols, srcMat->rows, srcMat->step);
#else
			cudaError err =cudaMallocPitch(&(sourceSurface->cudaLinearMemory), (size_t *)&(sourceSurface->pitch), sourceSurface->width * 4, sourceSurface->height);
			if(err != cudaSuccess){
				infoRecorder->logError("[CudaFilter]: cudaMallocPitch failed with: %d.\n", err);
			}
			else{
				infoRecorder->logError("[CudaFilter]: cudaMallocPitch, width:%d, height:%d, pitch:%d.\n", sourceSurface->width * 4, sourceSurface->height, sourceSurface->pitch);
			}
#endif
#else
			CUresult re =cuMemAllocPitch((CUdeviceptr *)&(sourceSurface->cudaLinearMemory), (size_t *)&(sourceSurface->pitch), sourceSurface->width, sourceSurface->height, 16);
			if(re != CUDA_SUCCESS){
				infoRecorder->logError("[CuadFilter]: malloc cuda pitch failed.\n");
			}
#endif
			sourceSurface->inited = true;
		}
#else
		if(surface){
			sourceSurface->dxSurface->d9Surface = (IDirect3DSurface9* )surface;
		}
		cudaError_t err;
		if((err = cudaD3D9RegisterResource(sourceSurface->dxSurface->d9Surface, cudaD3D9RegisterFlagsNone)) != cudaSuccess){
			infoRecorder->logError("[CudaFilter]: cudaD3DRegisterResource failed with :%d, surface:0x%p\n", err, sourceSurface->dxSurface->d9Surface);
			switch(err){
			case  cudaErrorInvalidDevice:
				infoRecorder->logError("[CudaFilter]: invalidDevice for Register Resource, means Direct3D interoperability is not initialized on this context. d3d device:0x%p, surface:0x%p\n", this->dxDevice->d9Device, sourceSurface->dxSurface->d9Surface);
				break;
			case  cudaErrorInvalidResourceHandle:
				infoRecorder->logError("[CudaFilter]: incorrect surface type to register.\n");
				break;

			case  cudaErrorUnknown:
				infoRecorder->logError("[CudaFilter]: the surface cannot be registered.\n");
				break;
			}
			return S_FALSE;
		}
#endif
	}else if(dxVersion == DX10){
		
		cudaGraphicsD3D10RegisterResource((cudaGraphicsResource **)&graphicResource, sourceSurface->dxSurface->d10Surface, cudaGraphicsRegisterFlagsNone);
		//cudaD3D10RegisterResource(sourceSurface->dxSurface->d10Surface, cudaD3D10RegisterFlagsNone);
	}else if(dxVersion == DX11){
		cudaGraphicsD3D11RegisterResource((cudaGraphicsResource **)&graphicResource, sourceSurface->dxSurface->d11Surface, cudaGraphicsRegisterFlagsNone);
	}
	return S_OK;
}

HRESULT CudaFilter::unregisterD3DResourceWithCUDA(){
#ifndef D3D9_SPEC
	cudaGraphicsUnregisterResource(graphicResource);
#else
	cudaD3D9UnregisterResource(sourceSurface->pSurface);
#endif
		//cutilCheckMsg("cudaD3D9UnregisterResource (sourceSurface) failed.");
	
	return S_OK;
}

// init the d3d for cuda filter, create device for d3d9, create IDXGISwapChain for D3D10 and D3D11
HRESULT CudaFilter::initD3D(HWND hwnd, DX_VERSION ver){
	// guess, must the original device or swapchain pointer

	return D3D_OK;
}

// the GpuMat is the device meory to store the converted data
void CudaFilter::runKernels(GpuMat & dst){
	void * pData = NULL;
	cudaError_t ret;
	size_t size = 0;
	size_t pitch = 0;
	char text[100] = {0};
	//Surface2D * ps = sourceSurface;
#ifdef __DEBUG__
	static int index = 0;
	index++;
	char surfaceName[100] = {0};
	sprintf(surfaceName, "surface-%d.jpg", index);
	// store the surface 
	//D3DXSaveSurfaceToFile(surfaceName, D3DXIFF_JPG, sourceSurface->dxSurface->d9Surface, NULL, NULL);

#endif
	cudaGraphicsResource * ppResource[1] = {
		graphicResource
	};

	if((ret = cudaGraphicsMapResources(1, ppResource)) != cudaSuccess){
		infoRecorder->logError("[CudaFilter]: map resource failed with %d.\n", ret);
	}

#if 1
	cudaArray * cuArray;
	cudaError err = cudaGraphicsSubResourceGetMappedArray(&cuArray, graphicResource, 0, 0);
	if(err != cudaSuccess){
		infoRecorder->logError("[CudaFilter]: cudaGraphicsSubResourceGetmappedArray failed with :%d\n.\n", ret);
	}

	infoRecorder->logError("[CuaFilter]: cudaMemcpy2DFromArray, pitch:%d, width:%d, height:%d, surface pithc:%d.\n", sourceSurface->pitch, sourceSurface->width * 4, sourceSurface->height, this->pitch);

	err = cudaMemcpy2DFromArray(srcMat->data,  srcMat->step, cuArray, 0, 0, this->pitch, sourceSurface->height, cudaMemcpyDeviceToDevice);

	switch(err){
	case cudaSuccess:
		infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFromArray  success.\n");
		break;
	case cudaErrorInvalidValue:
		infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidValue.\n");
		break;
	case cudaErrorInvalidDevicePointer:
		infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidDevicePointer.\n");
		break;
	case cudaErrorInvalidPitchValue:
		infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidPitchValue.\n");
		break;
	case cudaErrorInvalidMemcpyDirection:
		infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidMemcpyDirection.\n");
		break;
	}
#endif
	// here is the problem !!!!!!
	infoRecorder->logError("[CudaFilter]: before unmap the resources. graphics:0x%p\n", graphicResource);

	cudaGraphicsUnmapResources(1, ppResource);

	infoRecorder->logError("[CudaFilter]: before call cuda process, source surface: width:%d, height:%d, pitch:%d.\n", sourceSurface->width, sourceSurface->height, sourceSurface->pitch);
	// create the source mat

#if 0
	// run the convert logic
	if(surfaceFormat == D3DFORMAT::D3DFMT_A8R8G8B8){
		cuda_argb_to_nv12(pData, dstPtr, pitch, sourceSurface->height, sourceSurface->width, pitch);
	}else if(surfaceFormat == D3DFORMAT::D3DFMT_R8G8B8){
		cuda_rgb_to_nv12(pData, dstPtr, pitch, sourceSurface->height, sourceSurface->width, pitch);
	}else{
		cuda_argb_to_nv12(pData, dstPtr, pitch, sourceSurface->height, sourceSurface->width, pitch);
	}
#else

	// call RGB_to_YV12, convert RGB to YV12 and store to dst mat
#if 0
	RGB_to_YV12(*srcMat, dst);
#else
	RGBA_to_NV12(*srcMat, dst);

#endif


#if 0
	static int ind = 0;
	char name[100] = {0};
	sprintf(name, "dst-nv12-%d.bmp", ind++);
	// save NV12 file
	BYTE * yv = new BYTE[srcMat->rows * srcMat->cols * 3 / 2];

	cudaMemcpy2D(yv, srcMat->cols, dst.data, dst.step, dst.cols, dst.rows, cudaMemcpyDeviceToHost);

	BYTE * rgb = new BYTE[sizeof(BYTE) * dst.cols * srcMat->rows * 3];

	NV12ToBGR24_Native((unsigned char *)yv, rgb, srcMat->cols, srcMat->rows);
	long s2 = 0;
	BYTE * c = ConvertRGBToBMPBuffer(rgb, srcMat->cols, srcMat->rows, &s2);
	SaveBMP(c, srcMat->cols, srcMat->rows, s2, name);

	//YV12MatToARGB(dst, *pDst);
	//pDst->saveBMP(name);
	//delete pDst;
	delete[] yv;
	delete[] rgb;
	delete[] c;
#endif

#if 0
	static int ind = 0;
	char name[100] = {0};

	int padding = 0;
	int scanlinebytes = srcMat->cols * 3;
	while ( ( scanlinebytes + padding ) % 4 != 0 )     // DWORD = 4 bytes
		padding++;
	// get the padded scanline width
	int psw = scanlinebytes + padding;
	
	char * data = (char *)malloc(sizeof(char) * psw * srcMat->cols);
	int rgbStep = 3;
	int argbStep = 4;

	memset(name, 0, 100);
	

	D3DLOCKED_RECT rect;
	//sysSurface = NULL;
	HRESULT hr;

	hr = this->dxDevice->d9Device->GetRenderTargetData(sourceSurface->dxSurface->d9Surface, sysSurface);

	CHECK_D3D_ERROR(hr);

	// save from surface, that all right, the data is right
	sprintf(name, "surface-%d.jpg", ind);
	D3DXSaveSurfaceToFile(name, D3DXIFF_JPG, sysSurface, NULL, NULL);


	D3DSURFACE_DESC desc;
	hr = sysSurface->GetDesc(&desc);
	if(FAILED(hr)){
		CHECK_D3D_ERROR(hr);
	}
	else{
		if(desc.Format == D3DFORMAT::D3DFMT_A8R8G8B8){
			infoRecorder->logError("[CudaFilter]: get sysSurface format A8R8G8B8.\n");
		}
		else if(desc.Format == D3DFORMAT::D3DFMT_R8G8B8){
			infoRecorder->logError("[CudaFilter]: get sysSurface format R8G8B8.\n");
		}
		else if(desc.Format == D3DFORMAT::D3DFMT_X8R8G8B8){
			infoRecorder->logError("[CudaFilter]: get sysSurface format D3DFMT_X8R8G8B8.\n");
		}else if(desc.Format == D3DFORMAT::D3DFMT_A8B8G8R8){
			infoRecorder->logError("[CudaFilter]: get sysSurface format D3DFMT_A8B8G8R8.\n");
		}else if(desc.Format == D3DFORMAT::D3DFMT_X8B8G8R8){
			infoRecorder->logError("[CudaFilter]: get sysSurface format D3DFMT_X8B8G8R8.\n");
		}
	}

	hr = sysSurface->LockRect(&rect, NULL, 0);
	if(FAILED(hr)){
		CHECK_D3D_ERROR(hr);
	}else{

		char * src = (char *)rect.pBits;
		for(int i = 0; i < srcMat->rows; i++){
			for(int j = 0; j < srcMat->cols; j++){
				// B
				data[ i * psw + j * rgbStep + 0] = src[i * rect.Pitch + j * argbStep + 2];
				// G
				data[ i * psw + j * rgbStep + 1] = src[i * rect.Pitch + j * argbStep + 1];
				// R
				data[ i * psw + j * rgbStep + 2] = src[i * rect.Pitch + j * argbStep + 0];
			}
		}
	}
	sysSurface->UnlockRect();
	//sysSurface->Release();
	sprintf(name, "surface-from-mem-%d.bmp", ind);
	SaveBMP((BYTE *)data, srcMat->cols, srcMat->rows, psw * srcMat->rows, name);
	free(data);


	// the source data is right.
	sprintf(name, "src-%d.bmp", ind);
	//if(!srcMat->saveBMP(name)){
		
	//}

	// to save the converted image to bmp
	memset(name, 0, 100);
	sprintf(name, "dst-%d.bmp", ind++);

	// convert from yv12 to rgb
#if 0
	static unsigned char * pARGB = NULL;
	if(pARGB == NULL){
		pARGB = (unsigned char *)malloc(sizeof(unsigned char) * srcMat->rows * srcMat->step);
	}

	YV12ToARGB(dst.data, pARGB, srcMat->cols, srcMat->rows);
	//savebmp(pARGB, name, srcMat->cols, srcMat->rows, 4);

	unsigned char * pRGB = (unsigned char *)malloc(sizeof(unsigned char) * dst.rows * dst.cols * 4);   // alined to 4 bytes

	for(int i= 0; i < dst.rows; i++){
		for(int j = 0; j < dst.cols * 4; j++){
			pRGB[ i*  dst.cols + j] = pARGB[i * dst.cols + j];
		}
	}

	SaveBMP(pRGB, dst.cols, dst.rows, 4 * dst.cols * dst.rows, name);

	free(pRGB);
	//SaveBMP(pARGB, srcMat->cols, srcMat->rows, srcMat->step * srcMat->rows, name);
	//dst.saveBMP(name);
#else
	// create the result GpuMat
	//GpuMat * pDst = new GpuMat(srcMat->rows, srcMat->cols, srcMat->type());


	// read from encoeded yuv data
	BYTE * yv = new BYTE[srcMat->rows * srcMat->cols * 3 / 2];

	cudaMemcpy2D(yv, srcMat->cols, dst.data, dst.step, dst.cols, dst.rows, cudaMemcpyDeviceToHost);

	BYTE * rgb = new BYTE[sizeof(BYTE) * dst.cols * srcMat->rows * 3];

	YV12ToBGR24_Native((unsigned char *)yv, rgb, srcMat->cols, srcMat->rows);
	long s2 = 0;
	BYTE * c = ConvertRGBToBMPBuffer(rgb, srcMat->cols, srcMat->rows, &s2);
	SaveBMP(c, srcMat->cols, srcMat->rows, s2, name);

	//YV12MatToARGB(dst, *pDst);
	//pDst->saveBMP(name);
	//delete pDst;
	delete[] yv;
	delete[] rgb;
	delete[] c;
#endif

#endif

#endif
}

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

#if 1
	size_t size = 0;
	// use the cuda runtime api
	
		// first map the resources
	// error here
	char text[100] = {0};
	cudaError_t ret = cudaSuccess;

#ifndef D3D9_SPEC
		if((ret = cudaGraphicsMapResources(1, &graphicResource)) != cudaSuccess){
			
			infoRecorder->logError("[CudaFilter]: cudaGraphicsMapResources failed with :%d\n", ret);
		}

#if 0
		if((ret = cudaGraphicsResourceGetMappedPointer(&pData, &size, graphicResource))!= cudaSuccess){
			// error
			sprintf(text, "get mapped resource pointer failed with %p", ret);
			MessageBox(NULL, text, "ERROR", MB_OK);
		}
#else
		cudaArray * cuArray;
		
		cudaError err = cudaGraphicsSubResourceGetMappedArray(&cuArray, graphicResource, 0, 0);
		if(err != cudaSuccess){
			infoRecorder->logError("[CudaFilter]: cudaGraphicsSubResourceGetmappedArray failed with :%d\n.\n", ret);
		}
#endif

		infoRecorder->logError("[CuaFilter]: cudaMemcpy2DFromArray, pitch:%d, width:%d, height:%d, surface pithc:%d.\n", sourceSurface->pitch, sourceSurface->width * 4, sourceSurface->height, this->pitch);

#ifndef USE_NORMAL
		err = cudaMemcpy2DFromArray(srcMat->data, srcMat->cols * srcMat->step, cuArray, 0, 0, sourceSurface->width * 4, sourceSurface->height, cudaMemcpyDeviceToDevice);
#else

		err = cudaMemcpy2DFromArray(sourceSurface->cudaLinearMemory,sourceSurface->pitch, cuArray,0, 0, sourceSurface->width * 4, sourceSurface->height, cudaMemcpyDeviceToDevice);
#endif

		switch(err){
		case cudaSuccess:
			infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFromArray  success.\n", err);

			break;
		case cudaErrorInvalidValue:
			infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidValue.\n");
			break;
		case cudaErrorInvalidDevicePointer:
			infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidDevicePointer.\n");
			break;
		case cudaErrorInvalidPitchValue:
			infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidPitchValue.\n");
			break;
		case cudaErrorInvalidMemcpyDirection:
			infoRecorder->logError("[CudaFilter]: cudaMemcpy2DFrameArray got cudaErrorInvalidMemcpyDirection.\n");
			break;
		}

		cudaGraphicsUnmapResources(1, &graphicResource);

		//this->unregisterD3DResourceWithCUDA();
		//cudaD3D9ResourceGetMappedPointer(&pData, sourceSurface->pSurface, 0, 0);
		//cudaD3D9ResourceGetMappedPitch(&pitch, NULL, sourceSurface->pSurface, 0, 0);
#if 0
		// run the convert logic
		if(surfaceFormat == D3DFORMAT::D3DFMT_A8R8G8B8){
			cuda_argb_to_nv12(pData, dstPtr, sourceSurface->pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}else if(surfaceFormat == D3DFORMAT::D3DFMT_R8G8B8){
			cuda_rgb_to_nv12(pData, dstPtr, sourceSurface->pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}
#else
		// run the convert logic
		infoRecorder->logError("[CudaFilter]: before call cuda process, source surface: width:%d, height:%d, pitch:%d.\n", sourceSurface->width, sourceSurface->height, sourceSurface->pitch);

#ifdef USE_NORMAL
		if(surfaceFormat == D3DFORMAT::D3DFMT_A8R8G8B8){
			cuda_argb_to_nv12(sourceSurface->cudaLinearMemory, dstPtr, pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}else if(surfaceFormat == D3DFORMAT::D3DFMT_R8G8B8){
			cuda_rgb_to_nv12(sourceSurface->cudaLinearMemory, dstPtr, pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}
#else


#endif
		infoRecorder->logError("[CudaFilter]: end call cuda process.\n");
#endif

		// copy

		// end the convert logic

		// last, unmap the resource.
		
		//cudaD3D9UnmapResources(1, ppResources);
		//cutilCheckMsg("cudaD3D9UnmapResources(1) failed.");
#else
#if 0
	if((ret = cudaD3D9RegisterResource(sourceSurface->dxSurface->d9Surface, cudaD3D9RegisterFlagsNone)) != cudaSuccess){
		sprintf(text, "map resource failed with %p", ret);
		MessageBox(NULL, text, "ERROR", MB_OK);	
	}
#endif

	if((ret = cudaD3D9MapResources(1, (IDirect3DResource9 **)&(sourceSurface->dxSurface->d9Surface))) != cudaSuccess){
		infoRecorder->logError("[CudaFilter]: cudaD3D9MapResources failed with:%d\n", ret);
		//sprintf(text, "map resource failed with %p", ret);
		//MessageBox(NULL, text, "ERROR", MB_OK);	
	}
	// get the pointer and pitch
	cudaD3D9ResourceGetMappedPointer(&pData, sourceSurface->dxSurface->d9Surface, 0, 0);
	cudaD3D9ResourceGetMappedSize(&size, sourceSurface->dxSurface->d9Surface, 0, 0);
	cudaD3D9ResourceGetMappedPitch(&pitch, NULL, sourceSurface->dxSurface->d9Surface, 0, 0);

	infoRecorder->logError("[CudaFilter]: get mapped pointer: 0x%p, size:%d, pitch:%d.\n", pData, size, pitch);

#ifdef __DEBUG__
	// copyt the rgb image to host, and store

#endif

	// run the convert logic
		if(surfaceFormat == D3DFORMAT::D3DFMT_A8R8G8B8){
			infoRecorder->logError("[CudaFilter]: surface format is ARGB.\n");
			cuda_argb_to_nv12(pData, dstPtr, pitch, sourceSurface->height, sourceSurface->width, pitch);
		}else if(surfaceFormat == D3DFORMAT::D3DFMT_R8G8B8){
			infoRecorder->logError("[CudaFilter]: surface format is RGB.\n");
			cuda_rgb_to_nv12(pData, dstPtr, pitch, sourceSurface->height, sourceSurface->width, pitch);
		}else{
			infoRecorder->logError("[CudaFilter]: surface format is else.\n");
			cuda_argb_to_nv12(pData, dstPtr, pitch, sourceSurface->height, sourceSurface->width, pitch);
		}


		cudaD3D9UnmapResources(1, (IDirect3DResource9 **)&(sourceSurface->dxSurface->d9Surface));

#endif

		
	
#else

	if(dxVersion == DX9){
		// first map the resources
		IDirect3DResource9 * ppResources[1] = {
			ps->pSurface
		};

		//cudaGraphicsMapResources(1, graphicResource);
		cudaD3D9MapResources(1, ppResources);
		//cutilCheckMsg("cudaD3D9MapResources(1) failed.");
		//cutilSafeCallNoSync()
		cudaD3D9ResourceGetMappedPointer(&pData, sourceSurface->pSurface, 0, 0);
		cudaD3D9ResourceGetMappedPitch(&pitch, NULL, sourceSurface->pSurface, 0, 0);

		// run the convert logic
		if(surfaceFormat == D3DFORMAT::D3DFMT_A8R8G8B8){
			cuda_argb_to_nv12(pData, dstPtr, sourceSurface->pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}else if(surfaceFormat == D3DFORMAT::D3DFMT_R8G8B8){
			cuda_rgb_to_nv12(pData, dstPtr, sourceSurface->pitch, sourceSurface->height, sourceSurface->width, sourceSurface->pitch);
		}
		
		// end the convert logic

		// last, unmap the resource.
		cudaD3D9UnmapResources(1, ppResources);
		//cutilCheckMsg("cudaD3D9UnmapResources(1) failed.");
	}else if(dxVersion == DX10 || dxVersion == DX10_1){
		cudaD3D10Ma
	}else if(dxVersion == DX11){

	}
#endif
}