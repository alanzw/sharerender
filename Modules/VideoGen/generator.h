#ifndef __GENERATOR_H__
#define __GENERATOR_H__

#include <WinSock2.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "../LibCore/cthread.h"
#include "../LibDistrubutor/Context.h"
#include "../VideoUtility/cudaEncoder.h"
#include "../VideoUtility/FilterRGB2YUV.h"
#include "../VideoUtility/wrapper.h"
#include "../X264Encoder/X264Encoder.h"

#define NVENC
#define USE_NVENC_CUDA
//#define USE_NVENC_D3D
//#define USE_NVENC_NOR

// define the user message for VideoGenerator

#define WM_USR_ENCODER  (WM_USER + 100)


// this definition will disable the video generation
//#define NO_VIDEO_GEN

#define USE_TEST

#ifdef NVENC
#if defined (USE_NVENC_CUDA)
#include "../NvEncoders/NvEncoderInteropCuda/NvEncoderCudaInterop.h"
#elif defined(USE_NVENC_D3D)
#include "../NvEncoders/NvEncoderInteropD3D/NvEncoderD3DInterop.h"
#elif defined( USE_NVENC_NOR)
#include "..\NVENCEncoder\nvencencoder.h"
#include "../VideoUtility/rtspconf.h"
#endif
#endif  //NVENC

using namespace cg::core;
using namespace cg::rtsp;

// confirmed, DXGISwapChain is the device to D3D9
namespace cg{
	// global variables
	class VideoGen : public CThread{
		int id;   // the id of the VideoGen

		static int GenCounter;   // the counter for VideoGen
		static HANDLE genNotifier; // notify the server whenever a new generator is created.
		static map<IDENTIFIER, VideoGen *> genMap;  // the server port to find the videoGen
		static bool globalInited;
		
		bool isD3D;

		// change encoding device
		bool isChangedDevice;
		ENCODER_TYPE changeToEncoder;

		string object;   // the name of rtsp service

		// encoder type
		ENCODER_TYPE encoderType;   // registered encoder type
		ENCODER_TYPE useEncoderType;  // current use encoder type
		SOURCE_TYPE useSourceType;

		cg::nvcuvenc::CudaEncoder * encoder;
		X264Encoder * x264Encoder;
		Filter * x264Filter;   // only used for X264Encoder

#ifdef NVENC
#if defined(USE_NVENC_CUDA)
		cg::nvenc::CNvEncoderCudaInteropImpl * nvEncoder;	// the encoder with CUDA
#elif defined(USE_NVENC_D3D)
		cg::nvenc::CNvEncoderD3DInteropImpl * nvEncoder;	// the NVENC encoder with D3D
#elif defined(USE_NVENC_NOR)
		cg::CNvEncoderH264 * nvEncoder;   // the VNENC encoder original, not working well
#endif
#endif // NVENC
		HWND windowHandle;
		Wrapper * wrapper;
		VideoWriter * writer;   // the writer, write to file or/and to network
		cg::rtsp::RTSPContext * ctx; // the RTSPcontext for the video Generator, created when init

		cg::core::DX_VERSION dxVer;   // the version for directx
		bool useNvenc;   // whether the hardware support NVENC

		// pipelines
		pipeline *sourcePipe;  // pipeline for wrapper, 

		IDXGISwapChain * swapChain;   // for dx10 and dx11
		IDirect3DDevice9 * d9Device; // for dx9
		cg::core::DXDevice * dxDevice;

		HANDLE presentEvent;   // set from out side, triggered when present()
		HANDLE changeDeviceEvent;  // event for change event, whenever want to change encode device

		int height, width;

		LARGE_INTEGER captureTv, initialTv, freq;
		int frameInterval;
		struct timeval  tv;
		
		// video result option
		FILE* videoOutput;
		char* videoOutputName;
		bool enableWriteToNet;
		bool enableWriteToFile;

		bool inited;
		bool x264Inited, cudaInited, nvecnInited;  // init flags for each encoders
		bool enableGen;					// if true, start to generate video


		PTimer * intraMigrationTimer;   // the timer for intra migration, set to corresponding encoder when the encoder is actived

		int onEncodeDeviceChange();		// called when encode device is changed
		int setupSurfaceSource();		// setup the surface source
		int setupImageSource();			// setup the image sourced
		int setupWrapper();
		int initVideoGen(core::DX_VERSION ver, void * device);  // for d3d games
		int initVideoGen();				// for non d3d game, the fps is from RTSP config
		
		int initCudaEncoder(void * device);					// for cuda encoder
		int initCudaEncoder(void * dvice, void * context);  // only for D11
		int initX264Encoder();								// for X264Encoder
		int initNVENCEncoder(void * device);				// for NVENC encoder
		bool activeEncoder(ENCODER_TYPE type);				// active given encoder, if not started, start encoding

	public:
		static void Initialize(){
			if(globalInited)
				return;
			GenCounter =0;
			genNotifier = CreateEvent(NULL, FALSE, FALSE, NULL);
			globalInited = true;
		}
		static HANDLE GetNotifier(){ return genNotifier; }
		static VideoGen * findVideoGen(IDENTIFIER _id);
		static void addMap(IDENTIFIER _id, VideoGen * gen);
		inline void setResolution(int _width, int _height){ width = _width, height = _height;}
		inline void setObject(string obj){ object = obj; }
		void setSourceType(SOURCE_TYPE srcType);

		// the flag useNVENC is for the hardware encoder, the x264 encoder is always created
		VideoGen(HWND hwnd, bool useNVENC = false);
		VideoGen(HWND hwnd, void * device, core::DX_VERSION version, bool useNVENC = false);

		// for test
		VideoGen(HWND hwnd, bool useNVENC, bool writeToFile, bool writeNet);
		VideoGen(HWND hwnd, void * device, core::DX_VERSION version, bool useNVENC, bool writeToFile, bool writeNet);	

		VideoGen(HWND hwnd, bool useNVENC, ENCODER_TYPE _encodeType, bool writeToFile, bool wirteNet);
		VideoGen(HWND hwnd, void * device, core::DX_VERSION version, bool useNVENC, ENCODER_TYPE encodeType, bool writeFile, bool writeNet);

		~VideoGen();

		inline void setOutputFileName(char * name){ videoOutputName = _strdup(name); }

		cg::rtsp::RTSPContext* getContext(){ return ctx; }
		bool		 initRTSPContext(SOCKET s);

		inline bool isInited(){ return inited; }
		inline bool isEnableGen(){ return enableGen; }
		inline void setEnableGen(bool val){ enableGen = val; }

		inline int getHeight(){ return height;}
		inline int getWidth(){ return width; }

		inline void setPresentHandle(HANDLE h ){ presentEvent  = h;}
		inline HANDLE getPresentEvent(){ return presentEvent; }


		inline bool isUseNVENC(){ return useNvenc; }
		// thread function
		virtual BOOL stop();
		virtual BOOL run();
		virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
		virtual BOOL onThreadStart();
		virtual void onQuit();

		// intra-migration function
		int changeEncodeDevice(ENCODER_TYPE dstEncoderType);
		void setVideoTag(unsigned int tag){ if(writer)writer->setTags(tag); }
		void setVideoSpecialTag(unsigned char tag);
		void setValueTag(unsigned char valueTag);
		inline void setEncoderType(ENCODER_TYPE type){ 
			encoderType = type; 
			if(encoderType == ADAPTIVE_CUDA){
				useEncoderType = CUDA_ENCODER;
			}else if(encoderType == ADAPTIVE_NVENC){
				useEncoderType = NVENC_ENCODER;

			}else{
				useEncoderType = type;
			}
		}

	};

	extern VideoGen * gGenerator;

	// tool function to write message to screen
	BOOL _stdcall DrawMyText(LPDIRECT3DDEVICE9 pDxdevice,TCHAR* strText ,int nbuf);

}
#endif