#ifndef __NVENCODERD3DINTEROP_H__
#define __NVENCODERD3DINTEROP_H__
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <d3d9.h>
#include <InitGuid.h>
#include <Dxva2api.h>

#include "../common/nvCPUOPSys.h"
#include "../common/CNvEncoder.h"
#include "../../LibCore/CThread.h"
#include "../../VideoUtility/encoder.h"
#include "../../VideoUtility/videocommon.h"
#include "../../VideoUtility/VideoWriter.h"
#include "../../VideoUtility/pipeline.h"
namespace cg{
	namespace nvenc{
		class CNvEncoderD3DInteropImpl: public CNvEncoder, public cg::core::CThread{
		public:
			CNvEncoderD3DInteropImpl();
			CNvEncoderD3DInteropImpl(IDirect3DDevice9 * device, int height, int width, pipeline * pipe_);
			virtual ~CNvEncoderD3DInteropImpl();

		protected:
			IDirect3D9Ex                                        *m_pD3DEx;
			IDirect3DDevice9                                    *m_pD3D9Device;
			IDirectXVideoProcessorService                       *m_pDXVA2VideoProcessServices;
			IDirectXVideoProcessor                              *m_pDXVA2VideoProcessor;
			DXVA2_ValueRange                                     m_Brightness;
			DXVA2_ValueRange                                     m_Contrast;
			DXVA2_ValueRange                                     m_Hue;
			DXVA2_ValueRange                                     m_Saturation;

			EncodeBuffer                                         m_stEncodeBuffer[MAX_ENCODE_QUEUE];
			CNvQueue<EncodeBuffer>                               m_EncodeBufferQueue;
			cg::VideoWriter										*writer;
		protected:
			NVENCSTATUS			Deinitialize();
			NVENCSTATUS			InitD3D9(unsigned int deviceID = 0, IDirect3DDevice9 * device = NULL);
			NVENCSTATUS			AllocateIOBuffers(unsigned int dwInputWidth, unsigned int dwInputHeight);
			NVENCSTATUS			ReleaseIOBuffers();
			NVENCSTATUS			FlushEncoder();
			NVENCSTATUS			ConverRGBToNV12(IDirect3DSurface9 *pSrcRGB, IDirect3DSurface9 * pDstNV12, uint32_t width, uint32_t height);


			pooldata *			loadFrame();
			void				releaseData(pooldata * data);

		public:
			virtual BOOL		onThreadStart();
			virtual void		onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
			virtual void		onQuit();
			virtual BOOL		run();
		private:
			// only use surface as input
			EncodeConfig		encodeConfig;
			pipeline *			surfaceSource;
			HANDLE				cond, condMutex;
			int					width, height;
		};
	}
}


#endif   // __NVENCODERD3DINTEROP_H__