#define WINDOWS_LEAN_AND_MEAN
#pragma push_macro("_WINSOCKAPI_")
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#include <string>
#include "NvEncoderCudaInterop.h"
#include "../common/nvUtils.h"

// only need for test
#include "../common/nvFileIO.h"
#include <cuda_d3d9_interop.h>
#include "../../LibCore/InfoRecorder.h"
#include "../../VideoUtility/pipeline.h"

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined(__aarch64__)
#define PTX_FILE "preproc64_cuda.ptx"
#else
#define PTX_FILE "preproc32_cuda.ptx"
#endif

#define __cu(a) do { CUresult ret; if(( ret = (a)) != CUDA_SUCCESS){ fprintf(stderr, "%s has returned CUDA error%d\n", #a, ret); cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: %s has returned CUDA error %d\n", #a, ret); return NV_ENC_ERR_GENERIC; }} while(0)


#define BITSTREAM_BUFFER_SIZE 2 * 1024 * 1024
namespace cg{
	namespace nvenc{

		CNvEncoderCudaInteropImpl::CNvEncoderCudaInteropImpl(IDirect3DDevice9 * device, int _width, int _height, /*SOURCE_TYPE useSourceType_,*/ pipeline * imgPipe_, VideoWriter *writer_)
			:Encoder(0, _height, _width,/* useSourceType_,*/ NVENC_ENCODER, imgPipe_, writer_){
				m_pNvHWEncoder = new CNvHWEncoder();

				m_cuContext = NULL;
				m_cuModule = NULL;
				m_cuInterleaveUVFunction = NULL;

				m_uEncodeBufferCount = 0;
				memset(&m_stEncoderInput, 0, sizeof(m_stEncoderInput));
				memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));

				memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));
				memset(m_ChromaDevPtr, 0, sizeof(m_ChromaDevPtr));

				inited = true;
		}

		CNvEncoderCudaInteropImpl::~CNvEncoderCudaInteropImpl(){
			if(m_pNvHWEncoder){
				delete m_pNvHWEncoder;
				m_pNvHWEncoder = NULL;
			}
		}

		// init the CUDA
		NVENCSTATUS CNvEncoderCudaInteropImpl::InitCuda(uint32_t deviceID){

			CUresult cuResult = CUDA_SUCCESS;
			CUdevice cuDevice = 0;
			CUcontext cuContextCurr;
			int deviceCount = 0;
			int SMminor = 0, SMmajor = 0;

			// CUDA interfaces
			__cu(cuInit(0));

			__cu(cuDeviceGetCount(&deviceCount));
			if(deviceCount == 0){
				return NV_ENC_ERR_NO_ENCODE_DEVICE;
			}
			if(deviceID > (unsigned int)deviceCount - 1){
				PRINTERR("Invalid Device Id = %d\n", deviceID);
				return NV_ENC_ERR_INVALID_ENCODERDEVICE;
			}
			// now we get the actual device
			__cu(cuDeviceGet(&cuDevice, deviceID));
			__cu(cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID));
			if(((SMmajor << 4 )+ SMminor) < 0x30){
				PRINTERR("GPU %d does not have NVENC capabilities exiting.\n", deviceID);
				cg::core::infoRecorder->logError("GPU %d does not have NVENC capabilities exiting.\n", deviceID);
				return NV_ENC_ERR_NO_ENCODE_DEVICE;
			}
			// Create CUDA context and pop the current one
			__cu(cuCtxCreate(&m_cuContext, 0, cuDevice));

			// in  this branch we use compilation with parameters
			const unsigned int jitNumOptions = 3;
			CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
			void ** jitOptVals = new void *[jitNumOptions];

			// set up size of compilation log buffer
			jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
			int jitLogBufferSize = 1024;
			jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

			// set up pointer to the compilation log buffer
			jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
			char *jitLogBuffer = new char[jitLogBufferSize];
			jitOptVals[1] = jitLogBuffer;

			// set up pointer to set the Maximum # of registers for a particular kernel
			jitOptions[2] = CU_JIT_MAX_REGISTERS;
			int jitRegCount = 32;
			jitOptVals[2] = (void *)(size_t)jitRegCount;


			std::string ptx_source;
			FILE * fp = fopen(PTX_FILE, "rb");
			if(!fp){
				PRINTERR("unable to read ptx file %s\n", PTX_FILE);
				cg::core::infoRecorder->logError("unable to read ptx file %s\n", PTX_FILE);
				return NV_ENC_ERR_INVALID_PARAM;
			}
			fseek(fp, 0, SEEK_END);
			int file_size = ftell(fp);
			char * buf = new char[file_size + 1];
			fseek(fp, 0, SEEK_SET);
			fread(buf, sizeof(char), file_size, fp);
			fclose(fp);
			buf[file_size] = '\0';
			ptx_source = buf;
			delete[] buf;

			cuResult = cuModuleLoadDataEx(&m_cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);
			if (cuResult != CUDA_SUCCESS)
			{
				return NV_ENC_ERR_OUT_OF_MEMORY;
			}

			delete[] jitOptions;
			delete[] jitOptVals;
			delete[] jitLogBuffer;

			// load the function from module file
			__cu(cuModuleGetFunction(&m_cuInterleaveUVFunction, m_cuModule, "InterleaveUV"));
			__cu(cuModuleGetFunction(&m_cuBGRAToNV12Function, m_cuModule, "RGBAToNV12_2"));
			__cu(cuModuleGetFunction(&m_cuBGRToNV12Function, m_cuModule, "RGBToNV12_2"));

			__cu(cuCtxPopCurrent(&cuContextCurr));
			return NV_ENC_SUCCESS;
		}

		// allocate IO buffers for CUDA
		NVENCSTATUS CNvEncoderCudaInteropImpl::AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight){
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_uEncodeBufferCount);

			CCudaAutoLock cuLock(m_cuContext);

			__cu(cuMemAlloc(&m_ChromaDevPtr[0], uInputWidth*uInputHeight / 4));
			__cu(cuMemAlloc(&m_ChromaDevPtr[1], uInputWidth*uInputHeight / 4));

			__cu(cuMemAllocHost((void **)&m_yuv[0], uInputWidth*uInputHeight));
			__cu(cuMemAllocHost((void **)&m_yuv[1], uInputWidth*uInputHeight / 4));
			__cu(cuMemAllocHost((void **)&m_yuv[2], uInputWidth*uInputHeight / 4));

			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++)
			{
				// allocate the ARGB buffer for input
				__cu(cuMemAllocPitch(&m_stEncodeBuffer[i].stInputBfr.pNV12TempdevPtr, (size_t *)&m_stEncodeBuffer[i].stInputBfr.uNV12TempStride, uInputWidth * 4, uInputHeight, 16));
				__cu(cuMemAllocPitch(&m_stEncodeBuffer[i].stInputBfr.pNV12devPtr, (size_t *)&m_stEncodeBuffer[i].stInputBfr.uNV12Stride, uInputWidth, uInputHeight * 3 / 2, 16));

				nvStatus = m_pNvHWEncoder->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, (void*)m_stEncodeBuffer[i].stInputBfr.pNV12devPtr, 
					uInputWidth, uInputHeight, m_stEncodeBuffer[i].stInputBfr.uNV12Stride, &m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;

				m_stEncodeBuffer[i].stInputBfr.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
				m_stEncodeBuffer[i].stInputBfr.dwWidth = uInputWidth;
				m_stEncodeBuffer[i].stInputBfr.dwHeight = uInputHeight;

				nvStatus = m_pNvHWEncoder->NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;
				m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

#if defined(NV_WINDOWS)
				nvStatus = m_pNvHWEncoder->NvEncRegisterAsyncEvent(&m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;
				m_stEncodeBuffer[i].stOutputBfr.bWaitOnEvent = true;
#else
				m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
#endif
			}

			m_stEOSOutputBfr.bEOSFlag = TRUE;
#if defined(NV_WINDOWS)
			nvStatus = m_pNvHWEncoder->NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
			if (nvStatus != NV_ENC_SUCCESS)
				return nvStatus;
#else
			m_stEOSOutputBfr.hOutputEvent = NULL;
#endif
			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS CNvEncoderCudaInteropImpl::ReleaseIOBuffers(){
			CCudaAutoLock cuLock(m_cuContext);
			for(int i = 0; i < 3; i++){
				if(m_yuv[i]){
					cuMemFreeHost(m_yuv[i]);
					m_yuv[i] = NULL;
				}
			}
			for(uint32_t i = 0; i < m_uEncodeBufferCount; i++){
				cuMemFree(m_stEncodeBuffer[i].stInputBfr.pNV12devPtr);
				m_pNvHWEncoder->NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = NULL;

#if defined(NV_WINDOWS)
				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				nvCloseFile(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
#endif
			}

			if(m_stEOSOutputBfr.hOutputEvent){
#if defined(NV_WINDOWS)
				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
				nvCloseFile(m_stEOSOutputBfr.hOutputEvent);
				m_stEOSOutputBfr.hOutputEvent = NULL;
#endif
			}
			return NV_ENC_SUCCESS;

		}

		NVENCSTATUS CNvEncoderCudaInteropImpl::FlushEncoder(){
			NVENCSTATUS nvStatus = m_pNvHWEncoder->NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
			if(nvStatus != NV_ENC_SUCCESS)
			{
				assert(0);
				return nvStatus;
			}

			EncodeBuffer *pEncodeBuffer = m_EncodeBufferQueue.GetPending();
			while(pEncodeBuffer)
			{
				m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, writer);
				pEncodeBuffer = m_EncodeBufferQueue.GetPending();
				// UnMap the input buffer after frame is done
				if (pEncodeBuffer && pEncodeBuffer->stInputBfr.hInputSurface)
				{
					nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
					pEncodeBuffer->stInputBfr.hInputSurface = NULL;
				}
			}
#if defined(NV_WINDOWS)
			if (WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0)
			{
				assert(0);
				nvStatus = NV_ENC_ERR_GENERIC;
			}
#endif
			return nvStatus;
		}

		NVENCSTATUS CNvEncoderCudaInteropImpl::Deinitialize(){
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			ReleaseIOBuffers();

			nvStatus = m_pNvHWEncoder->NvEncDestroyEncoder();
			if (nvStatus != NV_ENC_SUCCESS)
			{
				assert(0);
			}

			__cu(cuCtxDestroy(m_cuContext));

			return NV_ENC_SUCCESS;
		}

		// copy the surface and convert the format to NV12
		NVENCSTATUS CNvEncoderCudaInteropImpl::CopySurfaceAndConvert(EncodeBuffer * pEncodeBuffer, CUgraphicsResource cuResource, int src_width, int src_height, int src_stride){

			// check the size
			if(src_width != encoderWidth || src_height != encoderHeight){
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImp]: size not match this size (%d, %d) != src size(%d, %d).\n", encoderWidth, encoderHeight, src_width, src_height);
				return NV_ENC_ERR_GENERIC;
			}

			CCudaAutoLock cuLock(m_cuContext);
			CUgraphicsResource ppResorces[1] = { cuResource };
			CUresult err = cuGraphicsMapResources(1, ppResorces, NULL);
			if(err != CUDA_SUCCESS){
				// failed
				return NV_ENC_ERR_GENERIC;
			}
			CUarray cuArray;
			err = cuGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);
			if(err != CUDA_SUCCESS){
				//failed to get the array of the surface
				return NV_ENC_ERR_GENERIC;
			}

			CUDA_ARRAY_DESCRIPTOR desc;
			memset((void *)&desc, 0, sizeof(CUDA_ARRAY_DESCRIPTOR));
			err =  cuArrayGetDescriptor(&desc, cuArray);
			if(err != CUDA_SUCCESS){
				return NV_ENC_ERR_GENERIC;
			}

			cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImp]: CUDA array, (%d %d), channels:%d.\n", desc.Width, desc.Height, desc.NumChannels);

			// copy data from D3D resource to CUDA device ptr
			CUDA_MEMCPY2D copy2D;
			memset(&copy2D, 0, sizeof(CUDA_MEMCPY2D));

			copy2D.srcXInBytes = 0;
			copy2D.srcY = 0;
			copy2D.srcMemoryType = CU_MEMORYTYPE_ARRAY;
			copy2D.srcHost = 0;
			copy2D.srcDevice = 0;
			copy2D.srcArray = cuArray;

			copy2D.srcPitch = desc.Width * desc.NumChannels;

			copy2D.dstXInBytes = 0;
			copy2D.dstY = 0;
			copy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			copy2D.dstHost = 0;
			copy2D.dstDevice = pEncodeBuffer->stInputBfr.pNV12TempdevPtr;   // actually, the buffer is ARGB format
			copy2D.dstArray = 0;
			copy2D.dstPitch = pEncodeBuffer->stInputBfr.uNV12TempStride;

			copy2D.WidthInBytes = encoderWidth * desc.NumChannels;
			copy2D.Height = encoderHeight;

			__cu(cuMemcpy2D(&copy2D));

			// launch the kernel to convert format
#define BLOCK_X 32
#define BLOCK_Y 16
			dim3 block(BLOCK_X, BLOCK_Y, 1);
			dim3 grid((encoderWidth + BLOCK_X - 1) / BLOCK_X * 2, (encoderHeight + BLOCK_Y - 1) / BLOCK_Y * 2, 1);
#undef BLOCK_Y
#undef BLOCK_X

			CUdeviceptr dNV12Chroma = (CUdeviceptr)((unsigned char*)pEncodeBuffer->stInputBfr.pNV12devPtr + pEncodeBuffer->stInputBfr.uNV12Stride*encoderHeight);
			//void *args[6] = { &m_ChromaDevPtr[0], &m_ChromaDevPtr[1], &dNV12Chroma, &chromaWidth, &chromaHeight, &chromaWidth, &chromaWidth, &pEncodeBuffer->stInputBfr.uNV12Stride};
			void *args[6] = { 
				&(pEncodeBuffer->stInputBfr.pNV12TempdevPtr),   // argb ptr
				&(pEncodeBuffer->stInputBfr.pNV12devPtr),		// nv12 ptr
				&(pEncodeBuffer->stInputBfr.uNV12TempStride),   // argb stride
				&(pEncodeBuffer->stInputBfr.uNV12Stride),		// nv12 stride
				&encoderWidth,											// width
				&encoderHeight											// height
			};										

			if(desc.NumChannels == 3){
				// RGB
				__cu(cuLaunchKernel(m_cuBGRToNV12Function, grid.x, grid.y, grid.z,
					block.x, block.y, block.z, 0, NULL, args, NULL));
			}
			else if(desc.NumChannels == 4){
				// ARGB
				__cu(cuLaunchKernel(m_cuBGRAToNV12Function, grid.x, grid.y, grid.z, 
					block.x, block.y, block.z, 0, NULL, args, NULL));
			}	

			CUresult cuResult = cuStreamQuery(NULL);

			// unmap the resource
			__cu(cuGraphicsUnmapResources(1, ppResorces, 0));

			// if error, return
			if (!((cuResult == CUDA_SUCCESS) || (cuResult == CUDA_ERROR_NOT_READY)))
			{
				return NV_ENC_ERR_GENERIC;
			}

			return NV_ENC_SUCCESS;
		}

		// copy the nv12 cpu array to gpu
		NVENCSTATUS CNvEncoderCudaInteropImpl::CopyNV12HtoD(EncodeBuffer * pEncodeBuffer, char * src, int src_width, int src_height, int src_stride){
			CCudaAutoLock cuLock(m_cuContext);
			CUDA_MEMCPY2D copyParam;
			memset(&copyParam, 0, sizeof(copyParam));

			// copy
			copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			copyParam.dstDevice = pEncodeBuffer->stInputBfr.pNV12devPtr;
			copyParam.dstPitch = pEncodeBuffer->stInputBfr.uNV12Stride;
			copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
			copyParam.srcHost = src;   // the src contains the whole NV12 data
			copyParam.srcPitch = src_stride;
			copyParam.WidthInBytes = src_width;
			copyParam.Height = src_height * 3 / 2;    // the height is image height, so the NV12 height is 3/2 of the image height
			__cu(cuMemcpy2D(&copyParam));

			return NV_ENC_SUCCESS;
		}

		NVENCSTATUS CNvEncoderCudaInteropImpl::ConvertYUVToNV12(EncodeBuffer * pEncodeBuffer, unsigned char * yuv[3], int width, int height){
			CCudaAutoLock cuLock(m_cuContext);
			// copy luma
			CUDA_MEMCPY2D copyParam;
			memset(&copyParam, 0, sizeof(copyParam));
			copyParam.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			copyParam.dstDevice = pEncodeBuffer->stInputBfr.pNV12devPtr;
			copyParam.dstPitch = pEncodeBuffer->stInputBfr.uNV12Stride;
			copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
			copyParam.srcHost = yuv[0];
			copyParam.srcPitch = width;
			copyParam.WidthInBytes = width;
			copyParam.Height = height;
			__cu(cuMemcpy2D(&copyParam));

			// copy chroma

			__cu(cuMemcpyHtoD(m_ChromaDevPtr[0], yuv[1], width*height / 4));
			__cu(cuMemcpyHtoD(m_ChromaDevPtr[1], yuv[2], width*height / 4));

#define BLOCK_X 32
#define BLOCK_Y 16
			int chromaHeight = height / 2;
			int chromaWidth = width / 2;
			dim3 block(BLOCK_X, BLOCK_Y, 1);
			dim3 grid((chromaWidth + BLOCK_X - 1) / BLOCK_X, (chromaHeight + BLOCK_Y - 1) / BLOCK_Y, 1);
#undef BLOCK_Y
#undef BLOCK_X

			CUdeviceptr dNV12Chroma = (CUdeviceptr)((unsigned char*)pEncodeBuffer->stInputBfr.pNV12devPtr + pEncodeBuffer->stInputBfr.uNV12Stride*height);
			void *args[8] = { &m_ChromaDevPtr[0], &m_ChromaDevPtr[1], &dNV12Chroma, &chromaWidth, &chromaHeight, &chromaWidth, &chromaWidth, &pEncodeBuffer->stInputBfr.uNV12Stride};

			__cu(cuLaunchKernel(m_cuInterleaveUVFunction, grid.x, grid.y, grid.z,
				block.x, block.y, block.z,
				0,
				NULL, args, NULL));
			CUresult cuResult = cuStreamQuery(NULL);
			if (!((cuResult == CUDA_SUCCESS) || (cuResult == CUDA_ERROR_NOT_READY)))
			{
				return NV_ENC_ERR_GENERIC;
			}
			return NV_ENC_SUCCESS;
		}

		// tool functions

		// the thread functions
		void CNvEncoderCudaInteropImpl::onQuit(){
			FlushEncoder();
						
			// de-initialize 
			Deinitialize();
			unregisterEvent();

			Encoder::onQuit();
		}
		// the function that execute each loop
		BOOL CNvEncoderCudaInteropImpl::run(){
			// load frame,
			struct pooldata * data = NULL;
			EncodeBuffer *pEncodeBuffer = NULL;
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			long long pts = -1LL;
			
			//cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: to load surface.\n");	
			
			if(!(data = loadFrame())){
				cg::core::infoRecorder->logError("[CNvEncoderCudaInpteropImpl]: load frame from pipeline failed.\n");
				return TRUE;
			}
			// convert format
			SourceFrame * frame = (SourceFrame *)data->ptr;
			pts = writer->updataPts(frame->imgPts);
			//frame->print();

#if 0
			// get the device buffer
			pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			if(!pEncodeBuffer)
			{
				pEncodeBuffer = m_EncodeBufferQueue.GetPending();
				// TODO, update the code to support RTSP

				m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, writer);
				// UnMap the input buffer after frame done
				if (pEncodeBuffer->stInputBfr.hInputSurface)
				{
					nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
					pEncodeBuffer->stInputBfr.hInputSurface = NULL;
				}
				pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			}
#else
			pEncodeBuffer = m_EncodeBufferQueue.GetPending();
			// TODO, update the code to support RTSP
			if(!pEncodeBuffer){

			}else{
				pTimer->Start();
				m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, writer);
				packTime = pTimer->Stop();
				if(this->refIntraMigrationTimer){
					UINT intramigration = this->refIntraMigrationTimer->Stop();
					cg::core::infoRecorder->logError("[Global]: intra-migration time: %f (ms), in NVENC encoder.\n", 1000.0 * intramigration / this->refIntraMigrationTimer->getFreq());
					this->refIntraMigrationTimer = NULL;
				}
				// UnMap the input buffer after frame done
				if (pEncodeBuffer->stInputBfr.hInputSurface)
				{
					nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
					pEncodeBuffer->stInputBfr.hInputSurface = NULL;
				}
			}
			pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			if(!pEncodeBuffer)
			{
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: NO available buffer to use.\n");
			}
#endif
			pTimer->Start();
			if(frame->type == IMAGE){
				unsigned char * src = frame->imgBuf;
				CCudaAutoLock cuLock(m_cuContext);
				if(frame->pixelFormat == PIX_FMT_NV12)
					CopyNV12HtoD(pEncodeBuffer, (char *)src, frame->realWidth, frame->realHeight, frame->realStride);
				else if(frame->pixelFormat == PIX_FMT_ARGB){
					// raw image format
					cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: ARGB raw image format. not supported now.\n");
					return FALSE;
				}
				else if(frame->pixelFormat == PIX_FMT_BGRA){
					// raw image format
					cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: BGRA raw image format. not supported now.\n");
					return FALSE;
				}
				else{
					cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: input image format not specified.\n");
					return FALSE;    // error
				}
				return TRUE;
			}
			else if(frame->type == SURFACE){
				IDirect3DSurface9 * surface = frame->dxSurface->d9Surface;

				if(!surface){
					cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: get NULL surface to encode,\n");
					return TRUE;
				}
				CCudaAutoLock cuLock(m_cuContext);
				if(!frame->registerToCUDA()){
					// register failed
					cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: surface frame register failed.\n");
				}
				// now, map the surface and convert the format, copy the data to NV12Ptr
				if((nvStatus = CopySurfaceAndConvert(pEncodeBuffer, frame->cuResource, frame->width, frame->height, frame->pitch)) != NV_ENC_SUCCESS){
					// failed to convert format use surface and CUDA
					cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: failed to copy surface and convert the format using CUDA.\n");
					return TRUE;
				}
			}else{
				// invalid input
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: get invalid input.\n");
				return FALSE;
			}

			releaseData(data);

			// encode
			// map the registered resource to hInputSurface, because NVENC use the hInputSurface handle to encode
			nvStatus = m_pNvHWEncoder->NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				PRINTERR("Failed to Map input buffer %p\n", pEncodeBuffer->stInputBfr.hInputSurface);
				return FALSE;
			}

			nvStatus = m_pNvHWEncoder->NvEncEncodeFrame(pEncodeBuffer, NULL, encodeConfig.width, encodeConfig.height);
			if(nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: encode frame failed with:%d.\n", nvStatus);
			}
			encodeTime = pTimer->Stop();
			cg::core::infoRecorder->addEncodeTime(getEncodeTime());

			return TRUE;
		}

		void InitDefaultCuda(EncodeConfig & config){

		}
		// the function that execute right after the thread started
		BOOL CNvEncoderCudaInteropImpl::onThreadStart(){
			cg::core::infoRecorder->logTrace("[CNvEncoderCudaInteropImpl]: on thread start.\n");
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			memset(&encodeConfig, 0, sizeof(EncodeConfig));

			encodeConfig.endFrameIdx = INT_MAX;
			encodeConfig.bitrate = 5000000;
			encodeConfig.rcMode = NV_ENC_PARAMS_RC_CONSTQP;
			encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH;
			encodeConfig.deviceType = NV_ENC_CUDA;
			encodeConfig.codec = NV_ENC_H264;
			encodeConfig.fps = 30;
			encodeConfig.qp = 28;
			encodeConfig.presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
			encodeConfig.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

			// fill the size
			encodeConfig.width = (encoderWidth == 0 ? 800 : encoderWidth); // as default
			encodeConfig.height = (encoderHeight == 0 ? 600 : encoderHeight); // as default
			
			if(encodeConfig.width == 0 || encodeConfig.height == 0){
				// error 
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: onThreadStart(), encoder config is invalid.\n");
				return FALSE;
			}
				
			// Init CUDA part
			nvStatus = InitCuda(encodeConfig.deviceID);
			if(nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: onThreadStart(), init cuda failed with:%d.\n", nvStatus);
				return FALSE;
			}

			nvStatus = m_pNvHWEncoder->Initialize((void *)m_cuContext, NV_ENC_DEVICE_TYPE_CUDA);
			if(nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: onThreadStart(), initialize HWEncoder failed with:%d.\n", nvStatus);
				return FALSE;
			}

			encodeConfig.presetGUID = m_pNvHWEncoder->GetPresetGUID(encodeConfig.encoderPreset, encodeConfig.codec);
			// show config
			PrintEncodeConfig(encodeConfig, true);
			nvStatus = m_pNvHWEncoder->CreateEncoder(&encodeConfig);
			if(nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: onThreadStart(), create encoder failed with:%d.\n", nvStatus);
				return FALSE;
			}

			m_uEncodeBufferCount = encodeConfig.numB + 4;

			nvStatus = AllocateIOBuffers(encodeConfig.width, encodeConfig.height);
			if(nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: onThreadStart(), AllocateIOBuffers failed with:%d.\n", nvStatus);
				return FALSE;
			}
			// register the pipeline surface to CUDA

			registerEvent();
			// initialize done.
			return TRUE;
		}

		void CNvEncoderCudaInteropImpl::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
			// do nothing now
		}

		// register the surface frame to CUDA
		NVENCSTATUS CNvEncoderCudaInteropImpl::reisgerResourceToCUDA(SurfaceFrame * frame){
			if(!frame->registerToCUDA()){
				// register failed
				return NV_ENC_ERR_GENERIC;
			}
			return NV_ENC_SUCCESS;
		}

		void PrintEncodeConfig(const EncodeConfig & config,bool useOutput, bool useInput /* = false */){
			if(useInput)
				cg::core::infoRecorder->logError("Encoding input				: \"%s\"\n", config.inputFileName);
			if(useOutput)
				cg::core::infoRecorder->logError("		output				: \"%s\"\n", config.outputFileName);
			cg::core::infoRecorder->logError("		codec				: \"%s\"\n", config.codec == NV_ENC_HEVC ? "HEVC" : "H264");
			cg::core::infoRecorder->logError("		size				: %d x %d\n", config.width, config.height);
			cg::core::infoRecorder->logError("		bitrate				: %d bits/sec\n", config.bitrate);
			cg::core::infoRecorder->logError("		vbvMaxBitrate		: %d bits/sec\n", config.vbvMaxBitrate);
			cg::core::infoRecorder->logError("		vbvsize				: %d bits\n", config.vbvSize);
			cg::core::infoRecorder->logError("		fps					: %d frames/sec\n", config.fps);
			cg::core::infoRecorder->logError("		rcMode				: %s\n", config.rcMode == NV_ENC_PARAMS_RC_CONSTQP? "CONSTQP": 
				config.rcMode == NV_ENC_PARAMS_RC_VBR ? "VBR" :
				config.rcMode == NV_ENC_PARAMS_RC_CBR ? "CBR" :
				config.rcMode == NV_ENC_PARAMS_RC_VBR_MINQP ? "VBR MINQP" :
				config.rcMode == NV_ENC_PARAMS_RC_2_PASS_QUALITY ? "TWO_PASS_QUALITY" :
				config.rcMode == NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP ? "TWO_PASS_FRAMESIZE_CAP" :
				config.rcMode == NV_ENC_PARAMS_RC_2_PASS_VBR ? "TWO_PASS_VBR" : "UNKNOWN");
			if(config.gopLength == NVENC_INFINITE_GOPLENGTH)
				cg::core::infoRecorder->logError("		goplength			: INFINITE GOP \n");
			else
				cg::core::infoRecorder->logError("		goplength			: %d \n", config.gopLength);
			cg::core::infoRecorder->logError("		B frames			: %d \n", config.numB);
			cg::core::infoRecorder->logError("		QP					: %d \n", config.qp);
			cg::core::infoRecorder->logError("		preset				: %s \n", (config.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HQ_GUID) ? "LOW_LATENCY_HQ" :
				(config.presetGUID == NV_ENC_PRESET_LOW_LATENCY_HP_GUID) ? "LOW_LATENCY_HP" :
				(config.presetGUID == NV_ENC_PRESET_HQ_GUID)? "HQ_PRESET":
				(config.presetGUID == NV_ENC_PRESET_HP_GUID)? "HP_PRESET":
				(config.presetGUID == NV_ENC_PRESET_LOSSLESS_HP_GUID)? "LOSSLESS_HP" : "LOW_LATENCY_DEFAULT");
			cg::core::infoRecorder->logError("\n");
		}
	}
}