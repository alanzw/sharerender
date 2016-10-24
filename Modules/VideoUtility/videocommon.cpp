#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "videocommon.h"
#include "../LibCore/MemOp.h"
#include "../LibCore/inforecorder.h"
#include "../LibCore/TimeTool.h"
#include <cuda_d3d9_interop.h>
#include <cudaD3D9.h>

namespace cg{
	int align_malloc(int size, void **ptr, int *alignment){
		if((*ptr = malloc(size + 16)) == NULL)
			return -1;
#ifdef __X86_64__
		*alignment = 16 - (((long long)*ptr)&0x0f);
#else
		*alignment = 16 - (((unsigned) * ptr) & 0x0f);
#endif
		return 0;
	}

	ImageFrame::ImageFrame(){
		imgBufInternal = NULL;
		imgBuf = NULL;
		realWidth = 0;
		realHeight = 0;
		realStride = 0;
		realSize = 0;
		maxStride = 0;
		imgBufSize = 0;
		alignment = 0;
		pixelFormat = AV_PIX_FMT_NONE;

	}
	bool ImageFrame::init(int maxW, int maxH, int maxS){
		int i ;
		for(i = 0; i< MAX_STRIDE; i++){
			lineSize[i] = maxS;
		}
		this->maxStride = maxS;
		imgBufSize = maxH * maxS;
		if(align_malloc(imgBufSize, (void **)&imgBufInternal, &alignment) < 0){
			// error
			return false;
		}
		imgBuf = imgBufInternal + alignment;
		bzero(imgBuf, imgBufSize);
		return true;
	}
	ImageFrame::~ImageFrame(){
		if(imgBuf != NULL){
			free(imgBuf);
			imgBuf = NULL;
		}
		return;
	}


	void ImageFrame::DupFrame(ImageFrame * src, ImageFrame * dst){
		int j = 0;

		for(j = 0; j< MAX_STRIDE; j++){
			dst->lineSize[j] = src->lineSize[j];
		}
		dst->realHeight = src->realHeight;
		dst->realWidth = src->realWidth;
		dst->realStride = src->realStride;
		dst->realSize = src->realSize;
		bcopy(src->imgBuf, dst->imgBuf, dst->imgBufSize);

	}
	// print the image information
	void ImageFrame::print(){
		cg::core::infoRecorder->logError("[ImageFrame]: frame %p info, (%d x %d), pitch:%d.\n", this, realWidth, realHeight, realStride);
	}

	//////// surfaceFrame //////////
	// print the surface information
	void SurfaceFrame::print(){
		cg::core::infoRecorder->logError("[SurfaceFrame]: frame %p info, (%d x %d), pitch:%d, registered:%s, cu res:%p.\n", this, width, height, pitch, registered ? "true": "false", cuResource);
	}

	SurfaceFrame::SurfaceFrame(){
		dxVersion = cg::core::DXNONE;
		dxSurface = new cg::core::DXSurface();
		width = 0;
		height = 0;
		registered = false; 
		cuResource = NULL; 
	}

	SurfaceFrame::~SurfaceFrame(){
		if(registered){
			unregisterToCUDA();
		}
		if(dxSurface){
			delete dxSurface;
			dxSurface = NULL;
		}
		dxVersion = cg::core::DXNONE;
	}

	// interop with CUDA
	bool SurfaceFrame::registerToCUDA(){
		CUresult err = CUDA_SUCCESS;
		if(registered)
			return registered;

		if((err = cuGraphicsD3D9RegisterResource(&cuResource, dxSurface->d9Surface, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST)) !=  CUDA_SUCCESS){
			// failed to register
			cg::core::infoRecorder->logError("[SurfaceFrame]: register the surface %p to res %p failed with:%d.\n", dxSurface->d9Surface, cuResource, err);
			return false;
		}
		registered = true;
		return true;
	}

	bool SurfaceFrame::unregisterToCUDA(){
		CUresult err = CUDA_SUCCESS;
		if((err = cuGraphicsUnregisterResource(cuResource))!= CUDA_SUCCESS){
			cg::core::infoRecorder->logError("[SurfaceFrame]: unregister res %p failed with %d.\n", cuResource, err);
			return false;
		}
		return true;
	}


	void SourceFrame::print(){
		if(type == IMAGE)
			ImageFrame::print();
		else
			SurfaceFrame::print();
	}


	int ve_malloc(int size, void **ptr, int * alignment){
		if((*ptr =malloc(size + 18)) == NULL){
			return -1;
		}
		*alignment = 16 - (((unsigned) *ptr) & 0x0f);
		return 0;
	}

	long ccg_gettid(){
		return GetCurrentThreadId();
	}

	struct ccgRect * ccg_fillrect(struct ccgRect * rect, int left, int top, int right, int bottom){
		if (rect == NULL){
			return NULL;
		}
#define SWAP(a, b) do {int tmp = a; a = b; b = tmp; }while(0);
		if (left > right)
			SWAP(left, right);
		if (top > bottom)
			SWAP(top, bottom);
#undef SWAP
		rect->left = left;
		rect->top = top;
		rect->right = right;
		rect->bottom = bottom;

		rect->width = rect->right - rect->left + 1;
		rect->height = rect->bottom - rect->top + 1;
		rect->linesize = rect->width * RGBA_SIZE;
		rect->size = rect->width * rect->height * RGBA_SIZE;

		if (rect->width <= 0 || rect->height <= 0){
			return NULL;
		}
		return rect;
	}


}