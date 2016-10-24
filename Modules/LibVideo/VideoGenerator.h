#ifndef __VIDEOGENERATOR_H__
#define __VIDEOGENERATOR_H__
// this is for the video generator

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#if 1
#include <map>
#include "Commonwin32.h"
#include "Pipeline.h"
#include "Config.h"
#include "RtspConf.h"
#include "RtspContext.h"
#include "Pipeline.h"
#endif

#include "Encoder.h"
#include "X264Encoder.h"
#include "CudaEncoder.h"
#include "FilterRGB2YUV.h"
#include "Wrapper.h"
#include "VSource.h"

#include "VideoCommon.h"


#define USE_CTHREAD

class VideoGenerator: public CThread{
	// for generator itself
	// we will create two types encoder in order to switch the encoder when running.
	bool supportD3D;  // must be set

	IDirect3DDevice9 * d3dDevice;    // must be set if using d3d
	RTSPConf * rtspConf;
	ccgRect * windowRect;    // must be set

	X264Encoder * x264Encoder;
	CudaEncoder * cudaEncoder;
	// filter is easy to switch
	CRITICAL_SECTION section, *pSection;

	CRITICAL_SECTION encoderSection, * pEncoderSection;
	CRITICAL_SECTION wrapperSection, * pWrapperSection;

	// for user
	ENCODER_TYPE useEncoderType;
	Encoder * encoder;

	// only for cpu encoder
	Filter * filter;

	pipeline * imageSource[SOURCES];   // the source of the image
	pipeline * surfaceSource[SOURCES];  // the source of surface

	pipeline * imageDst[SOURCES];    // the image destination for x264 encoder
	pipeline * surfaceDst[SOURCES];   // the destination for  cuda encoder

	// wrapper
	DX_VERSION dxVersion;

	DXWrapper * dxWrapper;				// for games that use D3D, D9, D10 or D11
	WindowWrapper * windowWrapper;     // for pure 2d game
	SOURCE_TYPE useSourceType; 
	
	HANDLE windowEvent; // all initializtion work will wait window event.
	HANDLE presentEvent;   // the present registered with a generator
	HWND captureWindow;
	SOCKET outsideSocket;    // the socket from outside

	int winHeight, winWidth;   // the height and width for the window to capture
	int outputHeight, outputWidth;   // the output size for encoder

	bool running;
	bool inited;
	CRITICAL_SECTION runningSection, * pRunningSection;

	LARGE_INTEGER captureTv, initialTv, freq;
	int frameInterval;

	bool isRunning();
	
#ifndef USE_CTHREAD
	HANDLE vidoeThreadHandle;
	DWORD videoThreadId;

	static DWORD WINAPI VideoGenertorThreadProc(LPVOID param);
#endif

	int loopFor2DGame();   // get game image preodic
	int loopForD3DGame();   // may not used

	// for D3D game only 
	bool doCapture(IDirect3DDevice9 * device, RTSPConf * rtspConf, LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frame_interval);

	int setAndSetupWindow(HWND hwnd);
	int setupImagePipeline(struct ccgImage * image);  // setup the source image pipeline 
	int setupFilterPipeline();  // setup the output image pipeline

	// only for D3D game, use cuda pipeline
	int setupCudaPipeline();
	inline bool isD3DGame(){ return supportD3D; }
public:
	VideoGenerator();
	~VideoGenerator();

	inline void setDXVersion(DX_VERSION v){ dxVersion = v; }
	inline bool isInited(){return inited; }
	inline void setInited(bool val){ inited = val; }

	int initVideoGenerator();   // setup the encoder, filter and the source
	// called by 2D server, need window handle, window height and window width
	int initVideoGenerator(HWND hwnd, int height, int width);
	int initVideoGenerator(HWND hwnd, int srcHeight, int srcWidth, int dstHeight, int dstWidth);    // for scale the output

	// called by D3D server, need window handle , D3D device, height and width.
	int initVideoGenerator(HWND hwnd, IDirect3DDevice9 * device, int height, int width, int x, int y);
	int initVideoGenerator(HWND hwnd, DX_VERSION dxVersion, void * device, int height, int width, int x, int y, int outH, int outW);

	int startModules();   // start all the modules in generator
#ifndef USE_CTHREAD
	void startThread();
	int stop();
#else

	virtual BOOL stop();
	virtual void run();
	virtual void onThreadStart(LPVOID param);
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void onQuit();

#endif

	ENCODER_TYPE getCurEncoderType();
	SOURCE_TYPE getCurSourceType();


	inline void setWindowHwnd(HWND h){ this->captureWindow = h; }
	inline void triggerWindowEvent(){ SetEvent(this->windowEvent); }
	
	// synchronized function to control the video generator's behavior
	// only for d3d games
	int changeEncoderDevice(ENCODER_TYPE type);
	//int stop();
	void lockEncoder();
	void lockWrapper();
	void lockFilter();

	void unlockEncoder();
	void unlockWrapper();
	void unlockFilter();
};

#endif