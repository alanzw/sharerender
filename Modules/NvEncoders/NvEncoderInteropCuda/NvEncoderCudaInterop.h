#ifndef __NVENCODERCUDAINTEROP_H__
#define __NVENCODERCUDAINTEROP_H__
// inter op with cuda
#include <vector_types.h>   
#include "../common/CNvEncoder.h"
#include "../../LibCore/CThread.h"
#include "../../VideoUtility/encoder.h"
#include "../../VideoUtility/videocommon.h"
#include "../../VideoUtility/pipeline.h"

namespace cg{
	namespace nvenc{

		// for Interop with CUDA
		class CCudaAutoLock{
		private:
			CUcontext m_pCtx;
		public:
			CCudaAutoLock(CUcontext pCtx): m_pCtx(pCtx){ cuCtxPushCurrent(m_pCtx); }
			~CCudaAutoLock(){ CUcontext cuLast = NULL; cuCtxPopCurrent(&cuLast); }
		};

		class CNvEncoderCudaInteropImpl: public CNvEncoder, public Encoder{
		public:

			CNvEncoderCudaInteropImpl(IDirect3DDevice9 * device, int width, int height, /*SOURCE_TYPE useSourceType_, */pipeline * imgPipe_, VideoWriter *writer_);
			virtual ~CNvEncoderCudaInteropImpl();

		protected:
			CUcontext				m_cuContext;
			CUmodule				m_cuModule;
			CUfunction				m_cuInterleaveUVFunction;
			CUfunction				m_cuBGRToNV12Function;
			CUfunction				m_cuBGRAToNV12Function;
			CUdeviceptr				m_ChromaDevPtr[2];
			EncodeConfig			m_stEncoderInput;

			uint8_t *				m_yuv[3];

		protected:
			NVENCSTATUS				Deinitialize();
			NVENCSTATUS				InitCuda(uint32_t deviceID);
			NVENCSTATUS				AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight);
			NVENCSTATUS				ReleaseIOBuffers();
			NVENCSTATUS				FlushEncoder();
			NVENCSTATUS				ConvertYUVToNV12(EncodeBuffer * pEncodeBuffer, unsigned char * yuv[3], int widht, int height);
			NVENCSTATUS				CopyNV12HtoD(EncodeBuffer * pEncodeBuffer, char * src, int src_width, int src_height, int src_stride);
			NVENCSTATUS				CopySurfaceAndConvert(EncodeBuffer * pEncodeBuffer, CUgraphicsResource cuResource, int src_width, int src_height, int src_stride);
			NVENCSTATUS				reisgerResourceToCUDA(SurfaceFrame * frame);

		public:

			virtual BOOL			onThreadStart();
			virtual void			onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
			virtual BOOL			run();
			virtual void			onQuit();

		private:
			void InitDefaultCuda(EncodeConfig & config);


			EncodeConfig			encodeConfig;   
			CUdeviceptr				srcBuffer;    /// the buffer for surface input
			CUgraphicsResource		cuGraphicResource;
		};
	}
}
#endif