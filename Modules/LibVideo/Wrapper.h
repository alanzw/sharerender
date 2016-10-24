#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#include "VideoCommon.h"
#include <d3d9.h>
#include <time.h>
#include <d3dx9.h>
#include <d3d10.h>
#include <d3d11.h>
#include <d3d10_1.h>
#include <d3dx9tex.h>
#include <DxErr.h>
#include "Pipeline.h"


class Wrapper{
	int id;
	CRITICAL_SECTION section;

public:
	HWND captureHwnd;   // the window handle
	RECT screenRect;    // the rect to capture

	pipeline* pipe;

	BITMAPINFO bmpInfo;
    int frameSize;

	int screenWidth, screenHeight;
	SOURCE_TYPE sourceType;   // the soruce type of the wrapper, to capture surface or image
	ENCODER_TYPE encoderType;   // define the destination type
	HANDLE imageNotifier;
	bool running;

	int fps;
	int frame_interval;
	LARGE_INTEGER initialTv, captureTv, freq;
	int capture_initialized;

	Wrapper();
	Wrapper(HWND hwnd);   
	Wrapper(SOURCE_TYPE sourceType, ENCODER_TYPE encoderType);
	virtual ~Wrapper();

	inline void registerImageSource(pipeline * p){ this->pipe = p; }
	inline pipeline * getImagePipe(){ return this->pipe; }

	inline int getId(){ return id; }
	inline void setId(int _id){ id = _id; }
	inline void setHWND(HWND hwnd){ captureHwnd = hwnd; }

	inline void setSourceType(SOURCE_TYPE type){ sourceType = type; }
	inline SOURCE_TYPE getSourceType(){ return sourceType; }

	inline HANDLE getImageNotifier(){ return imageNotifier; }

	inline bool isRunning(){ bool ret = false; EnterCriticalSection(&section); ret = running; LeaveCriticalSection(&section); return ret; }

	inline void setRunning(bool set){ EnterCriticalSection(&section); running  = set; LeaveCriticalSection(&section); }

	virtual int capture(char * buf, int bufLen, struct ccgRect * gRect) = 0;
	virtual int captureInit(int fps){ this->fps = fps; return this->fps; }
	virtual bool isWrapSurface();

	virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval) = 0;

	void makeBitmapInfo(int w, int h, int bitsPerPixel);
    void fillBitmapInfo(int w, int h, int bitsPerPixel);
};



////define the d3d9 wrapper
class D3DWrapper : public Wrapper{

    IDirect3DDevice9 * d3d_device;    // the source device
    IDirect3DSurface9 * surface;
	IDirect3DSurface9 * sysOffscreenSurface, * deviceOffscreenSurface; 

    static int count; // record the class construction times
    static IDirect3D9 * d3d;

	pipeline* cudaPipe;
	HANDLE cudaNotifier;
	// new add
	//pipeline * pipe;
	
public:
	
	inline void registerCudaSource(pipeline * p){ this->cudaPipe = p; cudaPipe->client_register(GetCurrentThreadId(), cudaNotifier); }
	inline pipeline * getCudaPipe(){ return this->cudaPipe; }
	inline HANDLE getCudaNotifier(){ return cudaNotifier; }
	inline 

    D3DWrapper & getWrapper();
	// register device, use cuda encoder
	D3DWrapper(IDirect3DDevice9 * device = NULL) :Wrapper(SOURCE_TYPE::SURFACE, ENCODER_TYPE::CUDA_ENCODER), d3d_device(device){
		cudaPipe = NULL;
		cudaNotifier = CreateEvent(NULL, FALSE, FALSE, NULL);
		surface = NULL;
		sysOffscreenSurface = NULL;
		deviceOffscreenSurface = NULL;
	}
	// default use surface source, but the encoder can be cuda or x264
	D3DWrapper(ENCODER_TYPE encoderType, SOURCE_TYPE srcType = SURFACE):Wrapper(srcType ,encoderType){
		cudaPipe = NULL;
		cudaNotifier = CreateEvent(NULL, FALSE, FALSE, NULL);
		surface = NULL;
		sysOffscreenSurface = NULL;
		deviceOffscreenSurface = NULL;
		d3d_device = NULL;
	}

	virtual ~D3DWrapper();
    //D3DWrapper(IDirect3DDevice9 * device)

    int init(struct ccgImage * image, HWND capHwnd = NULL);   // init the wrapper with image info and window handle
	int init(struct ccgImage * image, int width, int height);
	int deInit();


	inline bool setDevice(IDirect3DDevice9 * device){ d3d_device = device; return true; }
    void makeBitmapInfo(BITMAPINFO * pInfo, int w, int h, int bitsPerPixel);
    void fillBitmapInfo(BITMAPINFO * pInfo, int w, int h, int bitsPerPixel);

    virtual int capture(char * buf, int bufLen, struct ccgRect * gRect);
	virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);

	//virtual int captureInit(int fps);
	IDirect3DSurface9 * capture(ENCODER_TYPE encoderType, int gameWidth, int gameHeight);
	// capture the rendered surface 

	IDirect3DSurface9 * capture(ENCODER_TYPE encoderType, IDirect3DSurface9 ** ppDst);

	bool capture(ENCODER_TYPE encoderType, SurfaceFrame * sframe);

};

class WindowWrapper: public Wrapper{

	HDC captureHdc;
	HDC captureCompatibaleDC;

	HBITMAP captureCompatibleBitmap;
	LPVOID pBits;

	DWORD threadId;
	HANDLE threadHandle;

public:

	//WindowWrapper();
	WindowWrapper(HWND hwnd = NULL);
	

	int init(struct ccgImage * image, HWND capHwnd = NULL);   // init the wrapper with image info and window handle
	int deInit();
	virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);
	virtual int capture(char * buf, int bufLen, struct ccgRect * gRect);
	virtual int captureInit(int fps){ this->fps = fps; return fps;}

	virtual bool isWrapSurface();

#if 0
	bool startThread();
	static DWORD WINAPI WrapperThreadProc(LPVOID param);
#endif
};

// provide the surface
class D3D10Wrapper: public Wrapper{
	IDXGISwapChain * swapChain;
	DXGI_SWAP_CHAIN_DESC desc;
	DXGI_FORMAT dxgiFormat;
	ID3D10Device * device;

	ID3D10RenderTargetView * pRTV;
	ID3D10Resource * srcResource;
	ID3D10Texture2D * srcBuffer, * dstBuffer;

	D3D10_MAPPED_TEXTURE2D mappedScreen;

	pipeline * cudaPipe;
	HANDLE cudaNotifier;


public:
	inline void registerCudaSource(pipeline * p){ this->cudaPipe = p; }
	inline pipeline * getCudaPipe(){ return cudaPipe; }
	inline HANDLE getCudaNotifier(){ return cudaNotifier; }

	D3D10Wrapper(IDXGISwapChain * chain);
	virtual ~D3D10Wrapper();

	virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);
	virtual int capture(char * buf, int bufLen, struct ccgRect * gRect);
};

class D3D10Wrapper1: public Wrapper{
	IDXGISwapChain * swapChain;
	DXGI_SWAP_CHAIN_DESC desc;
	DXGI_FORMAT dxgiFormat;
	ID3D10Device * device;

	ID3D10RenderTargetView * pRTV;
	ID3D10Resource * srcResource;
	ID3D10Texture2D * srcBuffer, *dstBuffer;
	D3D10_MAPPED_TEXTURE2D mappedScreen;

	pipeline * cudaPipe;
	HANDLE cudaNotifier;
public:

	inline void registerCudaSource(pipeline * p){ this->cudaPipe = p; }
	inline pipeline * getCudaPipe(){ return cudaPipe; }
	inline HANDLE getCudaNotifier(){ return cudaNotifier; }

	D3D10Wrapper1(IDXGISwapChain * chain);
	virtual ~D3D10Wrapper1();


	virtual int capture(char * buf, int bufLen, struct ccgRect * gRect);
	virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);

};

class D3D11Wrapper: public Wrapper{
	IDXGISwapChain * swapChain;
	DXGI_SWAP_CHAIN_DESC sdesc;
	DXGI_FORMAT dxgiFormat;

	ID3D11Device * device;
	ID3D11DeviceContext * deviceContext;
	ID3D11RenderTargetView * pRTV;
	ID3D11Resource * srcResource;

	ID3D11Texture2D * srcBuffer, * dstBuffer;
	D3D11_TEXTURE2D_DESC desc;
	D3D11_MAPPED_SUBRESOURCE mappedScreen;

	pipeline * cudaPipe;
	HANDLE cudaNotifier;
public:

#if 0
	inline void registerImageSource(pipeline * p){ this->pipe = p; }
	inline void registerSurfaceSource(pipeline * p){ this->cudaPipe = p; }

#else
	inline void registerCudaSource(pipeline * p){ cudaPipe = p; }
	inline pipeline * getCudaPipe(){ return cudaPipe; }
	inline HANDLE getCudaNotifier(){ return cudaNotifier; }
#endif
	D3D11Wrapper(IDXGISwapChain * chain);
	virtual ~D3D11Wrapper();

	virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);
	virtual int capture(char * buf, int bufLen, struct ccgRect * gRect);

};


union DXWrapper{
	D3DWrapper * d9Wrapper;
	D3D10Wrapper * d10Wrapper;
	D3D10Wrapper1 * d10Wrapper1;
	D3D11Wrapper * d11Wrapper;
};

#endif