//this file is for video encode, as the NVIDIA example


/*
* Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

// This preprocessor definition is required for the use case where we want to use
// GPU device memory input with the CUDA H.264 Encoder.  This define is needed because
// 64-bit CUdevicePtrs are not supported in R260 drivers with NVCUVENC.  With driver versions
// after R265, ?UdevicePtrs are supported with NVCUVENC.  CUDA kernels that want to interop
// with the CUDA H.264 Encoder, this define must also be present for drivers <= R260.
//#define CUDA_FORCE_API_VERSION 3010    // as annotations, use the preprocessor definition for using GPU device memory input
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <cstring>
#include <cassert>

#include "CudaEncoder.h"
#include "pipeline.h"
#include "../LibCore/cthread.h"

#include "../LibCore/TimeTool.h"
#include "../LibCore/glob.hpp"
#include "../libcore/BmpFormat.h"

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "nvcuvenc.lib")
#pragma comment(lib, "nvcuvid.lib")
#pragma comment(lib, "d3dx9.lib")


#define __cu(a) do { CUresult ret; if(( ret = (a)) != CUDA_SUCCESS){ fprintf(stderr, "%s has returned CUDA error%d\n", #a, ret); cg::core::infoRecorder->logError("[CudaEncoder]: %s has returned CUDA error %d\n", #a, ret); }} while(0)

namespace cg{
	namespace nvcuvenc{
		HANDLE CudaEncoder::mutex;

		// TODO: be careful with callbacks, we need to check the dependency
		
		// encoder callback for net

		EncoderCallbackNet::EncoderCallbackNet(VideoWriter * _writer): writer(_writer), isKeyFrame_(false), buf_(NULL), refCudaEncodingTimer(NULL), refIntraMigrationTimer(NULL), encodeTime(0){
			buf_ = new unsigned char[BitStreamBufferSize];
			assert(buf_ != NULL);
		}

		EncoderCallbackNet::EncoderCallbackNet(): refCudaEncodingTimer(NULL), refIntraMigrationTimer(NULL), encodeTime(0){
			buf_ = new unsigned char[BitStreamBufferSize];
			assert(buf_ != NULL);
			//m_fOutput = NULL;
		}

		EncoderCallbackNet::~EncoderCallbackNet(){
			if(buf_){
				delete buf_;
				buf_ = NULL;
			}
		}

		////////// handler for cuda encoder  //////////
		// get the buffer to write the result
		unsigned char * EncoderCallbackNet::acquireBitStream(int * bufferSize){
			*bufferSize = static_cast<int>(BitStreamBufferSize - 2);
			cg::core::infoRecorder->logTrace("[EncoderCallback]: encode buffer: %p.\n", buf_ + 2);
			return (buf_ + 2);
		}
		// the result is in 'data' and it can be write to file
		void EncoderCallbackNet::releaseBitStream(unsigned char * data, int size){
			// write to file ?

			// get the frame data
			if(refCudaEncodingTimer){
				encodeTime = refCudaEncodingTimer->Stop();
				cg::core::infoRecorder->addEncodeTime(getEncodeTime()); 
			}

			cg::core::infoRecorder->logTrace("[EncoderCallback]: get encoded buffer: %p.\n", data);
			int64_t pts = writer->getUpdatedPts();
			AVPacket pkt;
			av_init_packet(&pkt);
			pkt.data = data;
			pkt.size = size;
			writer->sendPacket(0, &pkt, pts);

			if(refIntraMigrationTimer){
				UINT intramigration = refIntraMigrationTimer->Stop();
				cg::core::infoRecorder->logError("[Global]: intra-migration: %f (ms), in cuda encoder.\n", 1000.0 * intramigration / refIntraMigrationTimer->getFreq());
				refIntraMigrationTimer = NULL;
			}
		}

		void EncoderCallbackNet::onBeginFrame(int frameNumber, PicType picType){
			// nothing to do now
		}

		void EncoderCallbackNet::onEndFrame(int frameNumber, PicType picType){
			// nothing to do now
		}

		//////////////////// self define allocator for GpuMat
		class CGAllocator: public cv::cuda::GpuMat::Allocator{
		public:
			bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize);
			void free(cv::cuda::GpuMat* mat);
		};

		bool CGAllocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize){
			if(rows > 1 && cols > 1){
				__cu(cuMemAllocPitch((CUdeviceptr *)&mat->data, (size_t *)&mat->step, elemSize * cols, rows, 16));
			}else{
				// single row or single column must be continuous
				__cu(cuMemAlloc((CUdeviceptr *)&mat->data, elemSize * cols * rows));
				mat->step = elemSize * cols;
			}
			mat->refcount = (int *)cv::fastMalloc(sizeof(int));
			return true;
		}
		void CGAllocator::free(cv::cuda::GpuMat* mat){
			__cu(cuMemFree((CUdeviceptr)mat->datastart));
			cv::fastFree(mat->refcount);
		}


		//////////////////// cuda encoder impl  /////////////////
		CudaEncoder::CudaEncoder(int width, int height,/* SOURCE_TYPE _srcType, */pipeline * _imgPipe, VideoWriter * _writer, IDirect3DDevice9 * _device): Encoder(0, height, width,/* _srcType,*/ CUDA_ENCODER, _imgPipe, _writer), size_(width, height), d9Device(_device), argbMat(NULL), writer_(_writer), pCallbackNet(NULL){
			if(mutex == NULL){
				mutex = CreateMutex(NULL, FALSE, NULL);
			}
			if(d_writer.empty()){
				pCallbackNet = new EncoderCallbackNet(writer);
				pCallbackNet->setRefCudaEncodingTimer(pTimer);
				cv::Ptr<EncoderCallbackNet> callbackNet(pCallbackNet);
				// create the video writer
				WaitForSingleObject(mutex, INFINITE);
				if(!initCuda(0)){
					cg::core::infoRecorder->logError("[CudaEncoder]: init the cuda failed.\n");
				}

				CCudaAutoLock cuLock(m_cuContext);
				// if use raw surface as input
				d_writer = cv::cudacodec::createVideoWriter(callbackNet, cv::Size(encoderWidth, encoderHeight), 40, cv::cudacodec::SF_BGR);
				
				ReleaseMutex(mutex);
			}
			
			if(argbMat == NULL){
				CCudaAutoLock cuLock(m_cuContext);
				argbMat = new cv::cuda::GpuMat(height, width, CV_8UC4, new CGAllocator());
			}
			
		}
		
		void CudaEncoder::setRefIntraMigrationTimer(cg::core::PTimer * refTimer){
			refIntraMigrationTimer = refTimer;
			//refIntraMigrationTimerForCuda = refTimer;
			pCallbackNet->setRefIntraMigrationTimer(refTimer);

		}
		
		bool CudaEncoder::initEncoder(){
			inited = true;
			return inited;
		}

		// init the cuda part
		bool CudaEncoder::initCuda(uint32_t deviceID){
			//
			cg::core::infoRecorder->logTrace("[CudaEncoder]: init the cuda, get the context.\n");
#if 0
			CUresult cuError = CUDA_SUCCESS;
			CUcontext cuCtxCurr;
			CUdevice cuDevice;
			__cu(cuD3D9CtxCreate(&m_cuContext, &m_cuDevice, CU_CTX_SCHED_AUTO, d9Device));
#else
			CUresult cuResult = CUDA_SUCCESS;
			CUdevice cuDevice = 0;
			CUcontext cuContextCurr;
			int deviceCount = 0;
			int SMminor = 0, SMmajor = 0;

			// CUDA interfaces
			CUDA_ERROR_INVALID_DEVICE;
			__cu(cuInit(0));

			__cu(cuDeviceGetCount(&deviceCount));
			if(deviceCount == 0){
				return false;
			}
			if(deviceID > (unsigned int)deviceCount - 1){
				cg::core::infoRecorder->logError("Invalid Device Id = %d\n", deviceID);
				return false;
			}
			// now we get the actual device
			__cu(cuDeviceGet(&cuDevice, deviceID));
			//__cu(cuCtxCreate(&m_cuContext, CU_CTX_SCHED_AUTO, cuDevice));
			__cu(cuD3D9CtxCreate(&m_cuContext, &m_cuDevice, CU_CTX_SCHED_AUTO, d9Device));
			__cu(cuCtxPopCurrent(&cuContextCurr));
#endif

			cg::core::infoRecorder->logTrace("[CudaEncoder]: get context:%p.\n", m_cuContext);
			return true;
		}


		int CudaEncoder::loadFrameToGpuMat(SourceFrame & frame, cv::cuda::GpuMat & mat){
			if(frame.type == IMAGE){
				// load the image
				CUresult cuErr = CUDA_SUCCESS;
				CUDA_MEMCPY2D copy2D;
				memset(&copy2D, 0, sizeof(CUDA_MEMCPY2D));

				copy2D.srcXInBytes = 0;
				copy2D.srcY = 0;
				copy2D.srcMemoryType = CU_MEMORYTYPE_HOST;
				copy2D.srcHost = frame.imgBuf;
				copy2D.srcDevice = 0;
				copy2D.srcArray = 0;
				copy2D.srcPitch = frame.realStride;

				copy2D.dstXInBytes = 0;
				copy2D.dstY = 0;
				copy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
				copy2D.dstHost = 0;
				copy2D.dstDevice = (CUdeviceptr)mat.data;   // actually, the buffer is ARGB format
				copy2D.dstArray = 0;
				copy2D.dstPitch = mat.step;

				copy2D.WidthInBytes = frame.realStride;
				copy2D.Height = encoderHeight;

				cuErr = cuMemcpy2D(&copy2D);
				return cuErr;
			}
			else if(frame.type == SURFACE){
				// load the surface
				CUresult cuErr = CUDA_SUCCESS;
				CUgraphicsResource ppResources[1] = {frame.cuResource};
				cuErr = cuGraphicsMapResources(1, ppResources, NULL);
				if(cuErr != CUDA_SUCCESS){
					return cuErr;
				}
				do{
					// copy data
					CUarray cuArray;
					cuErr = cuGraphicsSubResourceGetMappedArray(&cuArray, frame.cuResource, 0, 0);
					if(cuErr != CUDA_SUCCESS){
						break;
					}

					CUDA_ARRAY_DESCRIPTOR desc;
					memset((void *)&desc, 0, sizeof(CUDA_ARRAY_DESCRIPTOR));
					cuErr = cuArrayGetDescriptor(&desc, cuArray);
					if(cuErr != CUDA_SUCCESS)
						break;

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
					copy2D.dstDevice = (CUdeviceptr)mat.data;   // actually, the buffer is ARGB format
					copy2D.dstArray = 0;
					copy2D.dstPitch = mat.step;

					copy2D.WidthInBytes = encoderWidth * desc.NumChannels;
					copy2D.Height = encoderHeight;

					cuErr = cuMemcpy2D(&copy2D);
					if(cuErr != CUDA_SUCCESS){
						break;
					}

				}while(0);

				// unmap the resource
				cuErr = cuGraphicsUnmapResources(1, ppResources, 0);
				return cuErr;
			}
			else{
				// error, the frame type is not specificed
				cg::core::infoRecorder->logError("[CudaEncoder]: the frame type is not specficed.\n");
				return -1;
			}
		}
		

#if 1
		// load image frame to GpouMat, the image frame contains raw argb data
		int CudaEncoder::loadImageToGpuMat(SourceFrame & frame, cv::cuda::GpuMat & mat){
			CUresult cuErr = CUDA_SUCCESS;
			CUDA_MEMCPY2D copy2D;
			memset(&copy2D, 0, sizeof(CUDA_MEMCPY2D));

			copy2D.srcXInBytes = 0;
			copy2D.srcY = 0;
			copy2D.srcMemoryType = CU_MEMORYTYPE_HOST;
			copy2D.srcHost = frame.imgBuf;
			copy2D.srcDevice = 0;
			copy2D.srcArray = 0;
			copy2D.srcPitch = frame.realStride;

			copy2D.dstXInBytes = 0;
			copy2D.dstY = 0;
			copy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			copy2D.dstHost = 0;
			copy2D.dstDevice = (CUdeviceptr)mat.data;   // actually, the buffer is ARGB format
			copy2D.dstArray = 0;
			copy2D.dstPitch = mat.step;

			copy2D.WidthInBytes = frame.realStride;
			copy2D.Height = encoderHeight;

			cuErr = cuMemcpy2D(&copy2D);
			return cuErr;
		}


		// load surface data to GpuMat, the surface frame contains IDirect3DSurface9
		int CudaEncoder::loadSurfaceToGpuMat(SourceFrame & frame, cv::cuda::GpuMat & mat){
			//DebugBreak();
			CCudaAutoLock cuLock(m_cuContext);
			cg::core::infoRecorder->logTrace("[CudaEncoder]: load surface to gpu mat, fame resource:%p.\n", frame.cuResource);
			CUresult cuErr = CUDA_SUCCESS;
			CUgraphicsResource ppResources[1] = {frame.cuResource};
			__cu(cuErr = cuGraphicsMapResources(1, ppResources, NULL));
			if(cuErr != CUDA_SUCCESS){
				
				return cuErr;
			}
			do{
				// copy data
				CUarray cuArray;
				__cu(cuErr = cuGraphicsSubResourceGetMappedArray(&cuArray, frame.cuResource, 0, 0));
				if(cuErr != CUDA_SUCCESS){
					break;
				}

				CUDA_ARRAY_DESCRIPTOR desc;
				memset((void *)&desc, 0, sizeof(CUDA_ARRAY_DESCRIPTOR));
				cuErr = cuArrayGetDescriptor(&desc, cuArray);
				if(cuErr != CUDA_SUCCESS)
					break;

				CUDA_MEMCPY2D copy2D;
				memset(&copy2D, 0, sizeof(CUDA_MEMCPY2D));

				copy2D.srcXInBytes = 0;
				copy2D.srcY = 0;
				copy2D.srcMemoryType = CU_MEMORYTYPE_ARRAY;
				copy2D.srcHost = 0;
				copy2D.srcDevice = 0;
				copy2D.srcArray = cuArray;

				copy2D.srcPitch = 0; // desc.Width * desc.NumChannels;

				copy2D.dstXInBytes = 0;
				copy2D.dstY = 0;
				copy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
				copy2D.dstHost = 0;
				copy2D.dstDevice = (CUdeviceptr)mat.data;   // actually, the buffer is ARGB format
				copy2D.dstArray = 0;
				copy2D.dstPitch = mat.step;

				copy2D.WidthInBytes = encoderWidth * desc.NumChannels;
				copy2D.Height = encoderHeight;

				__cu(cuErr = cuMemcpy2D(&copy2D));
				if(cuErr != CUDA_SUCCESS){
					break;
				}

			}while(0);
			// unmap the resource
			__cu(cuErr = cuGraphicsUnmapResources(1, ppResources, 0));
			return cuErr;
		}
#endif
		BOOL CudaEncoder::run(){
			struct pooldata * data = NULL;
			
			if(!(data = loadFrame())){
				cg::core::infoRecorder->logTrace("[CudaEncoder]: load frame from pipe failed.\n");
				return TRUE;
			}

			pTimer->Start();

			SourceFrame * frame = (SourceFrame *)data->ptr;
			writer_->updataPts(frame->imgPts);
			if(frame->type == IMAGE){
				cg::core::infoRecorder->logTrace("[CudaEncoder]: load image.\n");
				// load the cpu array data to GPU and encode
				if(!loadImageToGpuMat(*frame, *argbMat)){
					// load failed
					releaseData(data);
					return TRUE;
				}

			}else if(frame->type == SURFACE){
				// use ARGB surface as input
				cg::core::infoRecorder->logTrace("[CudaEncoder]: load surface.\n");
				// load the cpu array data to GPU and encode
				
				CCudaAutoLock cuLock(m_cuContext);
				if(!frame->registerToCUDA()){
					// register failed
					cg::core::infoRecorder->logError("[CudaEncoder]: surface frame register failed.\n");
					releaseData(data);
					return FALSE;
				}
				// now, map the surface
				if(loadSurfaceToGpuMat(*frame, *argbMat) != CUDA_SUCCESS){
					// load failed
					cg::core::infoRecorder->logError("[CudaEncoder]: load surface to gpu failed.\n");
					releaseData(data);
					return TRUE;
				}
			}else{
				cg::core::infoRecorder->logError("[CudaEncoder]: use source type or pipe is error.\n");
			}
			releaseData(data);
			// now, ARGB data is in argbMat, then, encode

			d_writer->write(*argbMat);

			return TRUE;
		}

		BOOL CudaEncoder::onThreadStart(){
			// check the dependency

			// register the event
			registerEvent();
			// here check the CV_Encoder
			if(d_writer.empty()){
				return FALSE;
				// create a callback 
				pCallbackNet = new EncoderCallbackNet(writer);
				// set the cuda encoding timer to the callback
				pCallbackNet->setRefCudaEncodingTimer(pTimer);
				cv::Ptr<EncoderCallbackNet> callbackNet(pCallbackNet);
				// check the callbacks dependency
				// create the video writer
				WaitForSingleObject(mutex, INFINITE);
				initCuda(0);
				// if use raw surface as input
				d_writer = cv::cudacodec::createVideoWriter(callbackNet, cv::Size(encoderWidth, encoderHeight), 40, cv::cudacodec::SF_BGR);
				ReleaseMutex(mutex);
			}
			if(argbMat == NULL){
				CCudaAutoLock cuLock(m_cuContext);
				argbMat = new cv::cuda::GpuMat(encoderHeight, encoderWidth, CV_8UC4, new CGAllocator());
			}
			return TRUE;
		}

		void CudaEncoder::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
			// now do noting
		}

		void CudaEncoder::onQuit(){
			// flush encoder
			// release the resources
			unregisterEvent();

		}
	}
}
