#ifndef __WRAPPER_H__
#define __WRAPPER_H__

#include "videocommon.h"
#include <d3d9.h>
#include <time.h>
#include <d3dx9.h>
#include <d3d10.h>
#include <d3d11.h>
#include <d3d10_1.h>
#include <d3dx9tex.h>
#include <DxErr.h>
#include "pipeline.h"
#include "../LibCore/TimeTool.h"


namespace cg{

	enum ERR_WRAPPER{
		ERR_D3D_ERROR = -5,
		ERR_NULL_PIPE,
		ERR_SIZE_NOT_MATCH,
		ERR_NULL_DEVICE,
		ERR_FAILED,
		WRAPPER_OK
	};

	class Wrapper{
		CRITICAL_SECTION section;

	protected:

		HDC captureHdc;
		HDC captureCompatibaleDC;

		HBITMAP captureCompatibleBitmap;
		LPVOID pBits;

		HWND captureHwnd;   // the window handle
		RECT screenRect;    // the rect to capture
		RECT windowRect;

		short windowWidth, windowHeight;    // window width and height

		pipeline * sourcePipe;   // the pipeline to store the captured data
		BITMAPINFO bmpInfo;
		SOURCE_TYPE sourceType;   // the source type of the wrapper, to capture surface or image


		cg::core::PTimer * pTimer;
		UINT captureTime;

		void makeBitmapInfo(int w, int h, int bitsPerPixel);
		void fillBitmapInfo(int w, int h, int bitsPerPixel);

		bool initilize();

	public:
		inline float getCaptureTime(){ return (float)1000.0 * captureTime / pTimer->getFreq(); }
		Wrapper(HWND hwnd, int _winWidth, int _winHeight, pipeline *_src_pipe);
		virtual ~Wrapper();

		bool isWrapSurface();
		// must be called before capture
		virtual bool changeSourceType(SOURCE_TYPE dstType) = 0;
		virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval) = 0;
	};

	////define the d3d9 wrapper /////
	class D3DWrapper : public Wrapper{
		IDirect3DDevice9 * d3d_device;		// the source device

		IDirect3DSurface9 * surface;		// the surface
		IDirect3DSurface9 * sysOffscreenSurface, * noAARenderTarget; 

		static int count;					// record the class construction times
		static IDirect3D9 * d3d;

		

	public:

		// use source type and window height and width create the D3DWrapper
		D3DWrapper(HWND h, IDirect3DDevice9 * device, int winWidth, int winHeight, pipeline *_src_pipe):Wrapper(h, winWidth, winHeight, _src_pipe), d3d_device(device), surface(NULL), sysOffscreenSurface(NULL), noAARenderTarget(NULL){

			//pTimer = new cg::core::PTimer();

		}

		virtual ~D3DWrapper();
		virtual bool changeSourceType(SOURCE_TYPE dstType);
		virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);

	private:
		// return the surface, if source type is IMAGE, the surface is located in host memory, if source type is SURFACE, the surface is located in device memory
		// capture the surface for CUDA encoder or NVENC encoder, send a surface to fill
		bool capture(IDirect3DSurface9 * surface);
		// interface for d3d capture
	};

	class WindowWrapper : public Wrapper{

		BITMAP bitMap;
		BYTE * rgbaBuffer;
		int frameSize;

		int WindowWrapper::capture(char * buf, int bufLen, ccgRect* gRect);

	public:
		// only support image source
		WindowWrapper(HWND hwnd, int winWidth, int winHeigh, pipeline *_imgPipe): Wrapper(hwnd, winWidth, winHeigh, _imgPipe), rgbaBuffer(NULL){
			changeSourceType(IMAGE);
		}
		virtual bool changeSourceType(SOURCE_TYPE dstType);

		int init(struct ccgImage * image, HWND capHwnd = NULL);   // init the wrapper with image info and window handle
		int deInit();
		virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);
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

	public:

		D3D10Wrapper(HWND h, IDXGISwapChain * chain, int winWidth, int winHeihgt, pipeline *_imgPipe): Wrapper(h, winWidth, winHeihgt, _imgPipe), device(NULL), pRTV(NULL), srcResource(NULL), srcBuffer(NULL), dstBuffer(NULL){}

		virtual ~D3D10Wrapper();
		virtual bool changeSourceType(SOURCE_TYPE dstType);
		virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);
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

	public:

		D3D10Wrapper1(HWND h, IDXGISwapChain * chain, int winWidth, int winHeihgt, pipeline *_imgPipe):Wrapper(h, winWidth, winHeihgt, _imgPipe), device(NULL), pRTV(NULL), srcBuffer(NULL), srcResource(NULL), dstBuffer(NULL){}

		virtual ~D3D10Wrapper1();
		virtual bool changeSourceType(SOURCE_TYPE dstType);
		virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);
	private:
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

	public:

		D3D11Wrapper(HWND h, IDXGISwapChain * chain, int winWidth, int winHeight, pipeline *_imgPipe):Wrapper(h, winWidth, winHeight, _imgPipe), device(NULL), pRTV(NULL), srcBuffer(NULL), srcResource(NULL), dstBuffer(NULL), deviceContext(NULL){}

		virtual ~D3D11Wrapper();
		virtual bool changeSourceType(SOURCE_TYPE dstType);
		virtual bool capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval);
	};

}

#endif