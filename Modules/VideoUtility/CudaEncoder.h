#ifndef NV_VIDEO_ENCODER
#define NV_VIDEO_ENCODER

#include <stdio.h>
#include <stdlib.h>
#include <d3d9.h>

#include "opencv2/core.hpp"
#include "opencv2/core/directx.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/highgui.hpp"

#include <cuda.h>
#include <cudaD3D9.h>
#include <math.h> // log10
#include <nvcuvid.h> // this is for the Cuvideoctxlock
#include <NVEncoderAPI.h>
#include <NVEncodeDataTypes.h>
#include "types.h"
#include <map>

#include "helper_timer.h"
#include <device_launch_parameters.h>

#include "../VideoUtility/encoder.h"
//#include "../CudaFilter/CudaFilter.h"
#include "../LibCore/InfoRecorder.h"
#include "videocommon.h"


#ifdef _DEBUG
#pragma comment(lib, "opencv_core300d.lib")
#pragma comment(lib, "opencv_cudacodec300d.lib")
#pragma comment(lib, "opencv_cudev300d.lib")
#pragma comment(lib, "opencv_video300d.lib")
#pragma comment(lib, "opencv_videoio300d.lib")
#else

#pragma comment(lib, "opencv_core300.lib")
#pragma comment(lib, "opencv_cudacodec300.lib")
#pragma comment(lib, "opencv_cudev300.lib")
#pragma comment(lib, "opencv_video300.lib")
#pragma comment(lib, "opencv_videoio300.lib")
#endif


#define USE_PTR


#ifndef UINT64_C
#define UINT64_C(val) val##ui64
#endif
//#define FILE_INPUT
namespace cg{

	void computeFPS(); //compute the fps

	enum INPUT_TYPE{
		INPUT_NONE,
		CPUIMAGE,
		GPUIMAGE
	};

	namespace nvcuvenc{
		const int BitStreamBufferSize = 1024 * 1024 * 2;
		class CCudaAutoLock{
		private:
			CUcontext								m_pCtx;
		public:
			CCudaAutoLock(CUcontext pCtx): m_pCtx(pCtx){ cuCtxPushCurrent(m_pCtx); }
			~CCudaAutoLock(){ CUcontext cuLast = NULL; cuCtxPopCurrent(&cuLast); }
		};

		class EncoderCallbackNet: public cv::cudacodec::EncoderCallBack{
		public:
			EncoderCallbackNet();	
			//EncoderCallbackNet(const string & outputFileName);
			EncoderCallbackNet(VideoWriter * _writer);
			virtual ~EncoderCallbackNet();

			unsigned char *							acquireBitStream(int * bufferSize);
			void									releaseBitStream(unsigned char * data, int size);
			void									onBeginFrame(int frameNumber, PicType picType);
			void									onEndFrame(int frameNumber, PicType picType);

			inline void setRefIntraMigrationTimer(cg::core::PTimer * pTimer){ refIntraMigrationTimer = pTimer; }
			inline void setRefCudaEncodingTimer(cg::core::PTimer * pTimer){ refCudaEncodingTimer = pTimer; }
			inline float getEncodeTime(){ return (float)1000.0 * encodeTime / refCudaEncodingTimer->getFreq(); }

		private:
			bool									isKeyFrame_;
			uchar *									buf_;     // store the output data
			int										bufSize;
			VideoWriter*							writer;

			cg::core::PTimer *						refCudaEncodingTimer;  // the timer for cuda encoding counting
			cg::core::PTimer *						refIntraMigrationTimer;
			UINT									encodeTime;
		};

		class CudaEncoder: public Encoder{

			static HANDLE mutex; // the mutex is necessary to init the OpenCV library

			cv::Ptr<cv::cudacodec::VideoWriter>		d_writer;
			cv::cuda::GpuMat						*argbMat;
			std::string								outputFileName;   // the name for output file	
			cv::Size								size_;
			CUcontext								m_cuContext;
			CUdevice								m_cuDevice;
			IDirect3DDevice9 *						d9Device;
			VideoWriter *							writer_;

			EncoderCallbackNet *					pCallbackNet;

		protected:

			int										loadFrameToGpuMat(SourceFrame & frame, cv::cuda::GpuMat & mat);
			int										loadSurfaceToGpuMat(SourceFrame & frame, cv::cuda::GpuMat & mat);
			int										loadImageToGpuMat(SourceFrame & frame, cv::cuda::GpuMat & mat);

		public:
			CudaEncoder(int widht, int height, /*SOURCE_TYPE _srcType, */pipeline* _sourcePipe,  VideoWriter* _writer, IDirect3DDevice9 * deivce);

			bool									initEncoder();
			bool									initCuda(uint32_t deviceID);
			virtual BOOL							run();
			virtual BOOL							onThreadStart();
			virtual void							onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
			virtual void							onQuit();

			//inline void								setSourceType(SOURCE_TYPE type){ useSourceType = type; }
			virtual void							setRefIntraMigrationTimer(cg::core::PTimer * refTimer);
		};

	}
}

#endif // NV_VIDEO_ENCODER