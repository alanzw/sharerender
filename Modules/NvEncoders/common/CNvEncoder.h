#ifndef __CNVENCODER_H__
#define __CNVENCODER_H__
#define WINDOWS_LEAN_AND_MEAN
#include "NvHWEncoder.h"

// defines

#define MAX_ENCODE_QUEUE 32
#define SEDT_VER(configStruct, type) { configStruct.version = type##_VER; }

///////////////////////

// the base for NvEncoder, NvEncoderD3Dinterop and NvEncoderCudaInterop
// all in namespace NV
namespace cg{
	namespace nvenc{
		template<class T>
		class CNvQueue{
			T ** m_pBuffer;
			unsigned int			m_uSize;
			unsigned int			m_uPendingCount;
			unsigned int			m_uAvailableIdx;
			unsigned int			m_uPendingndex;
		public:
			CNvQueue(): m_pBuffer(NULL), m_uSize(0), m_uPendingCount(0), m_uAvailableIdx(0), m_uPendingndex(0){}

			~CNvQueue(){ delete [] m_pBuffer; }

			bool Initialize(T *pItems, unsigned int uSize){
				m_uSize = uSize;
				m_uPendingCount = 0;
				m_uAvailableIdx = 0;
				m_uPendingndex	= 0;
				m_pBuffer = new T*[m_uSize];
				for(unsigned int i = 0; i < m_uSize; i++){
					m_pBuffer[i] = &pItems[i];
				}
				return true;
			}
			T * GetAvailable(){
				T * pItem = NULL;
				if(m_uPendingCount == m_uSize) return NULL;
				pItem = m_pBuffer[m_uAvailableIdx];
				m_uAvailableIdx = (m_uAvailableIdx+1)% m_uSize;
				m_uPendingCount += 1;
				return pItem;
			}
			T * GetPending(){
				if(m_uPendingCount == 0) return NULL;

				T * pItem = m_pBuffer[m_uPendingndex];
				m_uPendingndex = (m_uPendingndex + 1) %m_uSize;
				m_uPendingCount -= 1;
				return pItem;
			}

		};

		typedef struct _EncoderConfig{
			uint8_t *				yuv[3];
			uint32_t				stride[3];
			uint32_t				width;
			uint32_t				height;
		}EncodeFrameConfig;

		typedef struct _EncodeFrameConfig
		{
			IDirect3DTexture9 *		pRGBTexture;
			uint32_t				width;
			uint32_t				height;
		}D3DEncodeFrameConfig;

		typedef enum{
			NV_ENC_DX9	= 0,
			NV_ENC_DX11 = 1,
			NV_ENC_CUDA = 2,
			NV_ENC_DX10 = 3,
		}NvEncodeDeviceType;	


		class CNvEncoder{
		public:
			virtual ~CNvEncoder();
			// protected attributes
		protected:
			CNvHWEncoder *			m_pNvHWEncoder;
			uint32_t				m_uEncodeBufferCount;
			EncodeBuffer			m_stEncodeBuffer[MAX_ENCODE_QUEUE];
			CNvQueue<EncodeBuffer>	m_EncodeBufferQueue;
			EncodeOutputBuffer		m_stEOSOutputBfr;
			// functions
		protected:
			virtual NVENCSTATUS		AllocateIOBuffers(uint32_t uInputWidth, uint32_t uInputHeight) = 0;
			virtual NVENCSTATUS		ReleaseIOBuffers() = 0;
			virtual NVENCSTATUS		FlushEncoder() = 0;
		};
		// global 

		void PrintEncodeConfig(const EncodeConfig & config,bool useOutput, bool useInput = false);
	}
}

#endif