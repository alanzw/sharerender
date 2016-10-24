#ifndef __GENERATOR_H__
#define __GENERATOR_H__
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "CThread.h"
#include "CudaEncoder.h"
#include "Wrapper.h"


#define NVENC

#ifdef NVENC
#include "NVENCEncoder.h"
#endif

//extern InfoRecorder * infoRecorder;

extern int do_hook();

// confirmed, DXGISwapChain is the device to D3D9

// global variables
class VideoGen : public CThread{
	CudaEncoder * encoder;

#ifdef NVENC
	CNvEncoderH264 * nvEncoder;   // the VNENC encoder
#endif

	DXWrapper * dxWrapper;
	DX_VERSION dxVer;   // the version for directx
	pipeline * imagePipe;  // pipeline for wrapper, 
	pipeline * cudaPipe;  // use surface
	//pipeline * pipeDst;  // pipeline for encoder

	IDXGISwapChain * swapChain;   // for dx10 and dx11
	IDirect3DDevice9 * d9Device; // for dx9
	DXDevice * dxDevice;

	HANDLE presentEvent;   // set from out side

	char * videoFileName;
	HWND windowHandle;
	int height, width;

	bool useNvenc;

	LARGE_INTEGER captureTv, initialTv, freq;
	int frameInterval;
public:
	VideoGen(bool useNVENC = false);
	VideoGen(HWND hwnd, bool useNVENC = false);
	~VideoGen();

	int initVideoGen(DX_VERSION ver, void * device);
	int setupSurfaceSource();
	int initCudaEncoder(void * device);
	int initCudaEncoder(void * dvice, void * context);  // only for D11
	int initNVENCEncoder(void * device);

	inline void setPresentHandle(HANDLE h ){ presentEvent  = h;}
	inline HANDLE getPresentEvent(){ return presentEvent; }
	inline void setVideoName(char * name){ videoFileName = _strdup(name); }
	inline bool isUseNVENC(){ return useNvenc; }
	// thread function
	virtual BOOL stop();
	virtual void run();
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void onThreadStart();
	virtual void onQuit();
};

#endif