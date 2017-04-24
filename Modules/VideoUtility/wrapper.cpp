#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "wrapper.h"
#include <d3dx9.h>

#include "../LibCore/bmpformat.h"
#include "../LibCore/InfoRecorder.h"

#define USE_COPY_RT    // use the COPY render target to get antialiased screen
#define USE_RENDER_TARGET    // enable when use logic server to gen video

//#define SAVE_WARPPER_IMAGE

namespace cg{

	// used by image source, only
	Wrapper::Wrapper(HWND hwnd, int _winWidth, int _winHeight, pipeline *_src_pipe): sourceType(SOURCE_NONE),captureHwnd(hwnd), windowHeight(windowHeight), windowWidth(windowWidth), sourcePipe(_src_pipe), pBits(NULL), pTimer(NULL), captureTime(0){
		pTimer = new cg::core::PTimer();
	}

	Wrapper::~Wrapper(){
		
		if(sourcePipe){
			delete sourcePipe;
			sourcePipe = NULL;
		}
	}

	bool Wrapper::isWrapSurface(){
		bool ret = true;
		EnterCriticalSection(&this->section);
		if(sourceType == SURFACE){
			ret = true;
		}
		else{
			ret = false;
		}
		LeaveCriticalSection(&this->section);
		return ret;
	}

	void Wrapper::makeBitmapInfo(int w, int h, int bitsPerPixel){
		ZeroMemory(&bmpInfo, sizeof(BITMAPINFO));
		bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		bmpInfo.bmiHeader.biBitCount = bitsPerPixel;
		bmpInfo.bmiHeader.biCompression = BI_RGB;
		bmpInfo.bmiHeader.biWidth = w;
		bmpInfo.bmiHeader.biHeight = h;
		bmpInfo.bmiHeader.biPlanes = 1; // must be 1
		bmpInfo.bmiHeader.biSizeImage = bmpInfo.bmiHeader.biHeight
			* bmpInfo.bmiHeader.biWidth
			* bmpInfo.bmiHeader.biBitCount / 8;
		return;
	}

	void Wrapper::fillBitmapInfo(int w, int h, int bitsPerPixel){
		BITMAPINFO * pInfo = &bmpInfo;
		ZeroMemory(pInfo, sizeof(BITMAPINFO));
		pInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		pInfo->bmiHeader.biBitCount = bitsPerPixel;
		pInfo->bmiHeader.biCompression = BI_RGB;
		pInfo->bmiHeader.biWidth = w;
		pInfo->bmiHeader.biHeight = h;
		pInfo->bmiHeader.biPlanes = 1; // must be 1
		pInfo->bmiHeader.biSizeImage = pInfo->bmiHeader.biHeight
			* pInfo->bmiHeader.biWidth
			* pInfo->bmiHeader.biBitCount / 8;
		return;
	}

	// init the wrapper

	// change the source type, check whether support the type
	bool WindowWrapper::changeSourceType(SOURCE_TYPE dstType){
		if(dstType == SURFACE)
			return false;
		sourceType = dstType;
		return true;
	}
	bool D3DWrapper::changeSourceType(SOURCE_TYPE dstType){
		sourceType = dstType;
		return true;
	}
	bool D3D10Wrapper::changeSourceType(SOURCE_TYPE dstType){
		sourceType = dstType;
		return true;
	}
	bool D3D10Wrapper1::changeSourceType(SOURCE_TYPE dstType){
		sourceType = dstType;
		return true;
	}
	bool D3D11Wrapper::changeSourceType(SOURCE_TYPE dstType){
		sourceType = dstType;
		return true;
	}

	D3DWrapper::~D3DWrapper(){
		if (surface)
		{
			surface->Release();
			surface = NULL;
		}
		if (d3d_device){
			d3d_device->Release();
			d3d_device = NULL;
		}
	}

	// capture the surface data to given surface
	bool D3DWrapper::capture(IDirect3DSurface9* pSurface){
		HRESULT hr = D3D_OK;
		
		if(pSurface == NULL){
			cg::core::infoRecorder->logError("[D3DWrapper]: capture, bnut get NULL surface to store.\n");
			// the surface is NULL, error
			return false;
		}
		cg::core::infoRecorder->logError("[D3DWrapper]: capture rener target to surface.\n");

		IDirect3DSurface9 * rt  = NULL;
		if(FAILED(hr= d3d_device->GetRenderTarget(0, &rt))){
			cg::core::infoRecorder->logError("[D3DWrapper]: get render target failed with:%d.\n", hr);
			return false;
		}


#ifdef SAVE_WARPPER_IMAGE
		static int index = 0;
		char name[1024] = {0};
		sprintf(name, "capture-%d.bmp", index++);
		D3DXSaveSurfaceToFileA(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_BMP, rt, NULL, NULL);

#endif

		if(FAILED(hr = d3d_device->StretchRect(rt, NULL, pSurface, NULL, D3DTEXF_NONE))){
			cg::core::infoRecorder->logError("[D3DWrapper]: copy render target failed with:%d.\n", hr);
			return false;
		}

		rt->Release();
		return true;
	}

	// the d3d9 wrapper may only need the window handle, get the height and width, create a new D3D9Device
	// the capture function for D3D9
	bool D3DWrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){

		D3DLOCKED_RECT lockedRect;
		D3DSURFACE_DESC desc;
		struct pooldata * data = NULL;
		SourceFrame * sframe = NULL; // for surface

		// check the source pipeline
		if(!sourcePipe){
			cg::core::infoRecorder->logError("[D3DWrapper]: the source pipeline is not specified.\n");
			return false;
		}

		data = sourcePipe->allocate_data();
		if(data == NULL){
			// pool is full
			return false;
		}
		// check the device
		if(d3d_device == NULL){
			cg::core::infoRecorder->logError("[D3DWrapper]: get NULL d3d device.\n");
			return false;
		}

		pTimer->Start();

		sframe = (SourceFrame *)data->ptr;
		sframe->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) / (1+frameInterval);

		if(this->isWrapSurface()){
			cg::core::infoRecorder->logTrace("[D3DWrapper]: capture surface.\n");
			// get the surface
			if(frameInterval == 0){
				frameInterval = 2;
			}

			sframe->type = SURFACE;
			if(this->capture(sframe->getD3D9Surface())){
				sourcePipe->store_data(data);

				captureTime = pTimer->Stop();
				cg::core::infoRecorder->addCaptureTime(getCaptureTime());

				sourcePipe->notify_all();
				return true;
			}else{
				// failed.
				core::infoRecorder->logError("[D3DWrapper]: capture NULL surface for D3DWrapper.\n");
				return false;
			}
		}
		else{
			core::infoRecorder->logTrace("[D3DWrapper]: capture image.\n");
			// get the image, store to image pipeline

			sframe->type = IMAGE;
			//sframe->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) / frameInterval;

			unsigned char * buf = sframe->getImgBuf();
			char * src = NULL;
			HRESULT hr = D3D_OK;
			// get the image to sysOffscreenSurface
#ifndef USE_COPY_RT
			IDirect3DSurface9 * renderTarget = NULL;
			d3d_device->GetRenderTarget(0, &renderTarget);

			if(sysOffscreenSurface == NULL){
				// if not created, create one

				renderTarget->GetDesc(&desc);
				if(desc.Width != windowWidth || desc.Height != windowHeight ){
					cg::core::infoRecorder->logTrace("[D3DWrapper]: render target size not match window size, target size (%d x %d), window size (%d x %d), render target format:%d(desire:%d).\n", desc.Width, desc.Height, windowWidth, windowHeight, desc.Format, D3DFMT_A8R8G8B8);
				}
				if(desc.Format == D3DFMT_A8R8G8B8 || desc.Format == D3DFMT_X8R8G8B8){					
					cg::core::infoRecorder->logTrace("[D3DWrapper]: RGB render target format is:%d, supported(ARGB: %d or XRGB: %d\n", desc.Format, D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8);
				}
				else if(desc.Format == D3DFMT_A8B8G8R8 || desc.Format == D3DFMT_X8B8G8R8){
					cg::core::infoRecorder->logTrace("[D3DWrapper]: BGR render target format is:%d, supported(ABGR: %d or XBGR: %d\n", desc.Format, D3DFMT_A8B8G8R8, D3DFMT_X8B8G8R8);
				}
				else{
					cg::core::infoRecorder->logError("[D3DWrapper]: unsupported render target format is:%d, supported(BGR or RGB)\n", desc.Format);

					return false;
				}


				if(FAILED(d3d_device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &sysOffscreenSurface, NULL))){
					cg::core::infoRecorder->logError("[D3DWrapper]: create system offscreeen surface failed.\n");
					sourcePipe->release_data(data);
					return false;
				}
			}

			renderTarget->Release();

			hr = d3d_device->GetFrontBufferData(0, sysOffscreenSurface);
			if(FAILED(hr)){
				cg::core::infoRecorder->logError("[D3DWrapper]: get front buffer data failed.\n");		
			}

			if(FAILED(hr)){
					
				if(hr == D3DERR_DRIVERINTERNALERROR){
					cg::core::infoRecorder->logError("[D3DWrapper]: d3d error: D3DERR_DRIVERINTERNALERROR\n");

				}
				else if(hr == D3DERR_DEVICELOST){
					cg::core::infoRecorder->logError("[D3DWrapper]: d3d error: D3DERR_DEVICELOST\n");
				}
				else if(hr == D3DERR_INVALIDCALL){
					cg::core::infoRecorder->logError("[D3DWrapper]: d3d error: D3DERR_INVALIDCALL\n");
				}
				return false;
			}

#else // USE_RENDER_TARGET
			IDirect3DSurface9 * renderTarget = NULL;
			hr = d3d_device->GetRenderTarget(0, &renderTarget);

#if 0
			static int index = 0;
			char name[1024] = {0};
			sprintf(name, "ShadowRun/capture-%d.bmp", index++);
			D3DXSaveSurfaceToFileA(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_BMP, sysOffscreenSurface, NULL, NULL);

#endif

			// create the surface in system memory
			if(sysOffscreenSurface == NULL){
				// if not created, create one
				renderTarget->GetDesc(&desc);
				if(desc.Width != windowWidth || desc.Height != windowHeight ){
					cg::core::infoRecorder->logTrace("[D3DWrapper]: render target size not match window size, target size (%d x %d), window size (%d x %d), render target format:%d(desire:%d).\n", desc.Width, desc.Height, windowWidth, windowHeight, desc.Format, D3DFMT_A8R8G8B8);
				}

				if(desc.Format == D3DFMT_A8R8G8B8 || desc.Format == D3DFMT_X8R8G8B8){					cg::core::infoRecorder->logTrace("[D3DWrapper]: RGB render target format is:%d, supported(ARGB: %d or XRGB: %d\n", desc.Format, D3DFMT_A8R8G8B8, D3DFMT_X8R8G8B8);
				}
				else if(desc.Format == D3DFMT_A8B8G8R8 || desc.Format == D3DFMT_X8B8G8R8){
					cg::core::infoRecorder->logTrace("[D3DWrapper]: BGR render target format is:%d, supported(ABGR: %d or XBGR: %d\n", desc.Format, D3DFMT_A8B8G8R8, D3DFMT_X8B8G8R8);
				}
				else{
					cg::core::infoRecorder->logError("[D3DWrapper]: unsupported render target format is:%d, supported(BGR or RGB)\n", desc.Format);

					return false;
				}

				if(FAILED(d3d_device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format/*D3DFMT_A8R8G8B8*/, D3DPOOL_SYSTEMMEM, &sysOffscreenSurface, NULL))){
					cg::core::infoRecorder->logError("[D3DWrapper]: create system offscreeen surface failed.\n");
					sourcePipe->release_data(data);
					return false;
				}
			}

			// check and create render target
			if(noAARenderTarget == NULL){
				//
				renderTarget->GetDesc(&desc);
				hr = d3d_device->CreateRenderTarget(desc.Width, desc.Height, desc.Format /*D3DFMT_A8R8G8B8*/, D3DMULTISAMPLE_NONE, 0, FALSE, &noAARenderTarget, NULL);
				if(FAILED(hr) || !noAARenderTarget){
					cg::core::infoRecorder->logError("[D3DWrapper]: Create Render Target Failed with:%x.\n", hr);
					return false;
				}
			}

			// stretch render target 

			hr = d3d_device->StretchRect(renderTarget, NULL, noAARenderTarget, NULL, D3DTEXF_NONE);
			if(FAILED(hr)){
				cg::core::infoRecorder->logError("[D3DWrapper]: Stretch Render Target Failed with:%x.\n", hr);
				return false;
			}

			hr = d3d_device->GetRenderTargetData(noAARenderTarget, sysOffscreenSurface);
			if(FAILED(hr)){
				cg::core::infoRecorder->logError("[D3DWrapper]: Get Render Target Data failed with: %x.\n", hr);
				return false;
			}
			// release the render target
			renderTarget->Release();

#ifdef SAVE_WARPPER_IMAGE
			static int index = 0;
			char name[1024] = {0};
			sprintf(name, "ShadowRun/capture-%d.bmp", index++);
			D3DXSaveSurfaceToFileA(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_BMP, sysOffscreenSurface, NULL, NULL);

#endif

#endif // USE_RENDER_TARGET

			///// copy data from surface.

			if(FAILED(sysOffscreenSurface->GetDesc(&desc))){
				cg::core::infoRecorder->logError("[D3DWrapper]: sys off screen surface get desc failed.\n");
				sourcePipe->release_data(data);
				return false;
			}

			hr = sysOffscreenSurface->LockRect(&lockedRect, NULL, D3DLOCK_READONLY);
			if(FAILED(hr)){
				cg::core::infoRecorder->logError("[D3DWrapper]: lock system memory surface failed with %s.\n", hr == D3DERR_INVALIDCALL ? "D3DERR_INVALIDCALL" : (hr == D3DERR_WASSTILLDRAWING ? "D3DERR_WASSTILLDRWING" : "UNKNOWN"));
				sourcePipe->release_data(data);
				return false;
			}

			sframe->realWidth = desc.Width;
			sframe->realHeight = desc.Height;
			sframe->realStride = desc.Width << 2;
			sframe->realSize = sframe->realHeight * sframe->realStride;
			sframe->lineSize[0] = sframe->realStride;

			// copy data
			src = (char *)lockedRect.pBits;

			for(UINT i = 0; i < desc.Height; i++){
				CopyMemory(buf, src, sframe->realStride);
				src+= lockedRect.Pitch;
				buf+=sframe->realStride;
			}

			sysOffscreenSurface->UnlockRect();
			captureTime = pTimer->Stop();
			cg::core::infoRecorder->addCaptureTime(getCaptureTime());
			
			sourcePipe->store_data(data);
			sourcePipe->notify_all();
			return true;
		}
	}

	// for window wrapper, this wrapper captures GDI window, low performance
	int WindowWrapper::init(struct ccgImage *image, HWND _captureHwnd){
		// use GDI
		captureHwnd = _captureHwnd;
		ZeroMemory(&screenRect, sizeof(RECT));
		fillBitmapInfo(
			GetSystemMetrics(SM_CXSCREEN),
			GetSystemMetrics(SM_CYSCREEN),
			BITSPERPIXEL);

		frameSize = bmpInfo.bmiHeader.biSizeImage;
		//
		bmpInfo.bmiHeader.biSize = sizeof(bmpInfo.bmiHeader);
		if(image){
			image->width = bmpInfo.bmiHeader.biWidth;
			image->height = bmpInfo.bmiHeader.biHeight;
			image->bytes_per_line = (BITSPERPIXEL >> 3) * image->width;
		}

		// add capture window code here
		captureHdc = GetWindowDC(captureHwnd);
		captureCompatibaleDC = CreateCompatibleDC(captureHdc);

		GetWindowRect(captureHwnd, &windowRect);
		//captureCompatibleBitmap = CreateDIBSection(captureHdc, &bmpInfo, DIB_RGB_COLORS, &pBits, NULL, 0);
		captureCompatibleBitmap = CreateCompatibleBitmap(captureHdc, windowRect.right - windowRect.left, windowRect.bottom - windowRect.top);

		if (captureCompatibaleDC == NULL || captureCompatibleBitmap == NULL){
			cg::core::infoRecorder->logError("[WindowWrapper]: unable to create compatitable DC/Bitmap.\n");
			goto initErrorQuit;
			return -1;
		}
		HBITMAP oldBitMap = (HBITMAP)SelectObject(captureCompatibaleDC, captureCompatibleBitmap);
		PrintWindow(captureHwnd, captureCompatibaleDC, 0);

		//GetWindowRect(_captureHwnd, &windowRect);
		GetObject(captureCompatibleBitmap,sizeof(BITMAP), &bitMap);

		// update the bitmap info
		bmpInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		bmpInfo.bmiHeader.biWidth = bitMap.bmWidth;
		bmpInfo.bmiHeader.biHeight = bitMap.bmHeight;
		bmpInfo.bmiHeader.biPlanes = 1;
		bmpInfo.bmiHeader.biBitCount = 24;    ///?
		bmpInfo.bmiHeader.biCompression = BI_RGB;

		cg::core::infoRecorder->logError("[WindowWrapper]: get bitmap (%d x %d).\n", bitMap.bmWidth, bitMap.bmHeight);

		//bitMap.
		DWORD dwSize = ((bitMap.bmWidth * 24 + 31) / 32) * 4 * bitMap.bmHeight;
		DWORD rgbaSize = ((bitMap.bmWidth * 32 + 31) / 32) * 4 * bitMap.bmHeight;

		if(NULL == pBits)
			pBits = new BYTE[dwSize + sizeof(BITMAPINFOHEADER)];
		if(NULL == rgbaBuffer)
			rgbaBuffer = new BYTE[rgbaSize];

		LPBITMAPINFOHEADER lpbi = (LPBITMAPINFOHEADER)pBits;
		(*lpbi).biSize = sizeof(BITMAPINFOHEADER); 
		(*lpbi).biWidth = bitMap.bmWidth;
		(*lpbi).biHeight = bitMap.bmHeight;
		(*lpbi).biPlanes = 1;
		(*lpbi).biBitCount = 24;
		(*lpbi).biCompression = BI_RGB;

		GetDIBits(captureCompatibaleDC, captureCompatibleBitmap, 0, bitMap.bmHeight, (BYTE *)lpbi +sizeof(BITMAPINFOHEADER), (BITMAPINFO *)lpbi, DIB_RGB_COLORS);

		SelectObject(captureCompatibaleDC, oldBitMap);
		//DeleteObject(captureCompatibleBitmap);
		//DeleteObject(captureCompatibaleDC);
		//ReleaseDC(captureHwnd, captureHdc);
		return 0;

		//
initErrorQuit:
		deInit();
		return -1;
	}

	int WindowWrapper::deInit(){
		if (captureCompatibleBitmap != NULL){
			DeleteObject(captureCompatibleBitmap);
			captureCompatibleBitmap = NULL;
			pBits = NULL;
		}
		if (captureCompatibaleDC != NULL){
			DeleteDC(captureCompatibaleDC);
			captureCompatibaleDC = NULL;
		}
		if (captureHdc != NULL){
			ReleaseDC(captureHwnd, captureHdc);
			captureHdc = NULL;
		}
		return 0;
	}



	bool WindowWrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){
		core::infoRecorder->logError("[WindowWrapper]: this capture should never be used.\n");
		struct pooldata * data = NULL;
		SourceFrame * frame =  NULL;

		if(!sourcePipe){
			cg::core::infoRecorder->logError("[WindowWrapper]: get NULL pipeline.\n");
			return false;
		}

		pTimer->Start();
		data = sourcePipe->allocate_data();
		frame = (SourceFrame*)data->ptr;

		frame->type = IMAGE;
		//frameInterval ++;
		// get the pipeline 
		frame->pixelFormat = PIX_FMT_RGBA;
		frame->realWidth = windowRect.right - windowRect.left;
		frame->realHeight = windowRect.bottom - windowRect.top;
		frame->realStride = frame->realWidth << 2;
		frame->realSize = frame->realWidth * frame->realStride;
		frame->lineSize[0] = frame->realStride;

		// form the rect
		ccgRect rect;

		rect.left = windowRect.left;
		rect.right =  windowRect.right;
		rect.top = windowRect.top;
		rect.bottom = windowRect.bottom;
		rect.width = windowRect.right - windowRect.left;
		rect.height = windowRect.bottom - windowRect.top;
		rect.size = rect.width * rect.height;
		rect.linesize = (BITSPERPIXEL >> 3) * rect.width;
		// do the capture
		this->capture((char *)frame->imgBuf, frame->imgBufSize, &rect);
		core::infoRecorder->logError("[WindowWrapper]: image buffer size:%d.\n", frame->imgBufSize);
		// draw coursor

		frame->imgPts = core::pcdiff_us(captureTv, initialTv, freq) / frameInterval;

		captureTime = pTimer->Stop();

		sourcePipe->store_data(data);
		sourcePipe->notify_all();
		return true;
	}

	static void SaveHwndToBmpFile(HWND hWnd, LPCTSTR lpszPath)
	{
		HDC hDC = ::GetWindowDC(hWnd);
		//ASSERT(hDC);

		HDC hMemDC = ::CreateCompatibleDC(hDC);
		//ASSERT(hMemDC);

		RECT rc;
		::GetWindowRect(hWnd, &rc);

		HBITMAP hBitmap = ::CreateCompatibleBitmap(hDC, rc.right - rc.left, rc.bottom - rc.top);
		//ASSERT(hBitmap);

		HBITMAP hOldBmp = (HBITMAP)::SelectObject(hMemDC, hBitmap);
		::PrintWindow(hWnd, hMemDC, 0);

		BITMAP bitmap = {0};
		::GetObject(hBitmap, sizeof(BITMAP), &bitmap);
		BITMAPINFOHEADER bi = {0};
		BITMAPFILEHEADER bf = {0};

		CONST int nBitCount = 24;
		bi.biSize = sizeof(BITMAPINFOHEADER);
		bi.biWidth = bitmap.bmWidth;
		bi.biHeight = bitmap.bmHeight;
		bi.biPlanes = 1;
		bi.biBitCount = nBitCount;
		bi.biCompression = BI_RGB;
		DWORD dwSize = ((bitmap.bmWidth * nBitCount + 31) / 32) * 4 * bitmap.bmHeight;

		HANDLE hDib = GlobalAlloc(GHND, dwSize + sizeof(BITMAPINFOHEADER));
		LPBITMAPINFOHEADER lpbi = (LPBITMAPINFOHEADER)GlobalLock(hDib);
		*lpbi = bi;

		::GetDIBits(hMemDC, hBitmap, 0, bitmap.bmHeight, (BYTE*)lpbi + sizeof(BITMAPINFOHEADER), (BITMAPINFO*)lpbi, DIB_RGB_COLORS);

		SaveBMP((BYTE *)lpbi + sizeof(BITMAPINFOHEADER), 800, 600, 2400 * 600, "extern-save.bmp");

		GlobalUnlock(hDib);
		GlobalFree(hDib);

		::SelectObject(hMemDC, hOldBmp);
		::DeleteObject(hBitmap);	
		::DeleteObject(hMemDC);
		::ReleaseDC(hWnd, hDC);
	}

	static void captureToArray(HWND hWnd, BYTE * buffer){
		HDC hDC = ::GetWindowDC(hWnd);
		HDC hMemDC = ::CreateCompatibleDC(hDC);

		RECT rc;
		::GetWindowRect(hWnd, &rc);

		HBITMAP hBitmap = ::CreateCompatibleBitmap(hDC, rc.right - rc.left, rc.bottom - rc.top);
		//ASSERT(hBitmap);

		HBITMAP hOldBmp = (HBITMAP)::SelectObject(hMemDC, hBitmap);
		::PrintWindow(hWnd, hMemDC, 0);

		BITMAP bitmap = {0};
		::GetObject(hBitmap, sizeof(BITMAP), &bitmap);
		BITMAPINFOHEADER bi = {0};
		BITMAPFILEHEADER bf = {0};

		CONST int nBitCount = 24;
		bi.biSize = sizeof(BITMAPINFOHEADER);
		bi.biWidth = bitmap.bmWidth;
		bi.biHeight = bitmap.bmHeight;
		bi.biPlanes = 1;
		bi.biBitCount = nBitCount;
		bi.biCompression = BI_RGB;
		DWORD dwSize = ((bitmap.bmWidth * nBitCount + 31) / 32) * 4 * bitmap.bmHeight;

		//HANDLE hDib = GlobalAlloc(GHND, dwSize + sizeof(BITMAPINFOHEADER));
		LPBITMAPINFOHEADER lpbi = (LPBITMAPINFOHEADER)buffer;
		*lpbi = bi;
		//lpbi = buffer;

		::GetDIBits(hMemDC, hBitmap, 0, bitmap.bmHeight, (BYTE*)lpbi + sizeof(BITMAPINFOHEADER), (BITMAPINFO*)lpbi, DIB_RGB_COLORS);

		//SaveBMP((BYTE *)lpbi + sizeof(BITMAPINFOHEADER), 800, 600, 2400 * 600, "extern-save.bmp");

		::SelectObject(hMemDC, hOldBmp);
		::DeleteObject(hBitmap);	
		::DeleteObject(hMemDC);
		::ReleaseDC(hWnd, hDC);
	}
#if 1
	int WindowWrapper::capture(char * buf, int bufLen, ccgRect* gRect){

		core::infoRecorder->logError("[WindowWrapper]: capture, buf len:%d\n", bufLen);
		int linesize, height;
		char * source, *dst;
		if (gRect == NULL && bufLen < frameSize){
			return -1;
		}
		if (gRect != NULL && bufLen < gRect->size)
			return -1;
		if (captureCompatibaleDC == NULL || pBits == NULL)
			return -1;


#if 0
		captureToArray(captureHwnd, (BYTE *)pBits);
#else

		HBITMAP hOldBitMap = (HBITMAP)SelectObject(captureCompatibaleDC, captureCompatibleBitmap);
		PrintWindow(captureHwnd, captureCompatibaleDC, 0);
		GetDIBits(captureCompatibaleDC, captureCompatibleBitmap, 0, bitMap.bmHeight, (BYTE *)pBits + sizeof(BITMAPINFOHEADER), (BITMAPINFO *)pBits, DIB_RGB_COLORS);
#endif

		if(!ConvertBMPToRGBBuffer((BYTE *)pBits + sizeof(BITMAPINFOHEADER), rgbaBuffer, bitMap.bmWidth, bitMap.bmHeight)){
			core::infoRecorder->logError("[WindowWrapper]: convert bmp BMP buffer to RGBA failed.\n");

		}
#if 1
		//// XXX: image are stored upside-down
		linesize = (BITSPERPIXEL >> 3) *(gRect->width);
		if (gRect == NULL){
			//error
			source = ((char *)rgbaBuffer) - (linesize * screenRect.bottom);
			dst = buf;
			height = screenRect.bottom + 1;
			while (height-- > 0){
				CopyMemory(dst, source, linesize);
				dst += linesize;
				source -= linesize;
			}
		}
		else{
			source = (char *)rgbaBuffer;
			//source += linesize *(gRect->bottom - gRect->top - 1);
			//source += RGBA_SIZE * gRect->left;
			dst = buf;
			height = gRect->height;
			while (height-- > 0){
				memcpy(dst, source, gRect->linesize);
				//CopyMemory(dst, source, gRect->linesize);
				dst += gRect->linesize;
				source += gRect->linesize;
			}
		}

		SelectObject(captureCompatibaleDC, hOldBitMap);
#endif
		return gRect == NULL ? frameSize : gRect->size;
	}

#endif
	//// D3DWrappers//////

	D3D10Wrapper::~D3D10Wrapper(){

	}
	D3D10Wrapper1::~D3D10Wrapper1(){

	}

	D3D11Wrapper::~D3D11Wrapper(){

	}


	bool D3D10Wrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){
		HRESULT hr;

		struct pooldata * data = NULL;
		cg::SourceFrame * frame = NULL;

		if(!(data = sourcePipe->allocate_data())){
			// pool is FULL
			return false;
		}
		frame = (cg::SourceFrame *)data->ptr;

		if(this->swapChain == NULL){
			//error
			return false;
		}
		else{
			hr = swapChain->GetDevice(IID_ID3D10Device, (void **)&this->device);
			if(SUCCEEDED(hr)){

			}
		}

		device->OMGetRenderTargets(1, &pRTV, NULL);
		pRTV->GetResource(&srcResource);

		D3D10_TEXTURE2D_DESC tDesc;

		srcBuffer = (ID3D10Texture2D *)srcResource;
		srcBuffer->GetDesc(&tDesc);
		tDesc.BindFlags = 0;
		tDesc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
		tDesc.Usage = D3D10_USAGE_STAGING;

		hr = device->CreateTexture2D(&tDesc, NULL, &dstBuffer);
		if(FAILED(hr)){
			OutputDebugString("Failed to create texture2d.\n");
		}
		device->CopyResource(dstBuffer, srcBuffer);

		// if use surface, store the dst buffer, or read the image data
		if(this->isWrapSurface()){
			// store the surface to pipeline
			// use the ID3D10Texture2D
			
			frame->dxSurface->d10Surface = dstBuffer;
			frame->width = tDesc.Width;
			frame->height= tDesc.Height;

			frame->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) /frameInterval;

			sourcePipe->store_data(data);
			sourcePipe->notify_all();

			//dstBuffer->Unmap(0);
			device->Release();
			srcResource->Release();
			srcBuffer->Release();
			pRTV->Release();;
		}
		else{
			// wrapper the image
			hr = dstBuffer->Map(0, D3D10_MAP_READ, 0, &mappedScreen);
			if(FAILED(hr)){
				OutputDebugString("Failed to map from dst buffer.\n");
			}

			// copy image
			do{
				unsigned char * src = NULL, * dst = NULL;
				
				//frame->pixelFormat = PIX_FMT_BGRA;
				frame->realWidth = tDesc.Width;
				frame->realHeight = tDesc.Height;
				frame->realStride = tDesc.Width<<2;
				frame->realSize = frame->realWidth * frame->realStride;
				frame->lineSize[0] = frame->realStride;

				src = (unsigned char *)mappedScreen.pData;
				dst = (unsigned char *)frame->imgBuf;
				for(int i = 0; i< frame->realHeight; i++){
					CopyMemory(dst, src, frame->realStride);
					src+= mappedScreen.RowPitch;
					dst+=frame->realStride;
				}
				frame->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) / frameInterval;
			}while(0);
			// copy to all channels

			// store the data
			sourcePipe->store_data(data);
			sourcePipe->notify_all();

			dstBuffer->Unmap(0);
			device->Release();
			srcResource->Release();
			srcBuffer->Release();
			pRTV->Release();
			dstBuffer->Release();
		}
		return true;
	}

	bool D3D10Wrapper1::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){
		HRESULT hr;
		struct pooldata * data = NULL;
		cg::SourceFrame * frame = NULL;

		if(!(data = sourcePipe->allocate_data())){
			// pool is FULL
			return false;
		}
		frame = (cg::SourceFrame *)data->ptr;

		if(this->swapChain == NULL){
			//error
			return false;
		}
		else{
			hr = swapChain->GetDevice(IID_ID3D10Device1, (void **)&this->device);
			if(SUCCEEDED(hr)){

			}
			else
				return false;
		}

		device->OMGetRenderTargets(1, &pRTV, NULL);
		pRTV->GetResource(&srcResource);

		D3D10_TEXTURE2D_DESC tDesc;

		srcBuffer = (ID3D10Texture2D *)srcResource;
		srcBuffer->GetDesc(&tDesc);
		tDesc.BindFlags = 0;
		tDesc.CPUAccessFlags = D3D10_CPU_ACCESS_READ;
		tDesc.Usage = D3D10_USAGE_STAGING;

		hr = device->CreateTexture2D(&tDesc, NULL, &dstBuffer);
		if(FAILED(hr)){
			OutputDebugString("Failed to create texture2d.\n");
		}
		device->CopyResource(dstBuffer, srcBuffer);

		// if use surface, store the dst buffer, or read the image data
		if(this->isWrapSurface()){
			// store the surface to pipeline
			// use the ID3D10Texture2D
			frame->type = SURFACE;
			frame->dxSurface->d10Surface = dstBuffer;
			frame->width = tDesc.Width;
			frame->height= tDesc.Height;

			frame->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) /frameInterval;

			sourcePipe->store_data(data);
			sourcePipe->notify_all();

			//dstBuffer->Unmap(0);
			device->Release();
			srcResource->Release();
			srcBuffer->Release();
			pRTV->Release();
		}
		else{
			// wrapper the image
			hr = dstBuffer->Map(0, D3D10_MAP_READ, 0, &mappedScreen);
			if(FAILED(hr)){
				OutputDebugString("Failed to map from dst buffer.\n");
			}

			frame->type = IMAGE;
			// copy image
			do{
				unsigned char * src = NULL, * dst = NULL;
				//frame->pixelFormat = PIX_FMT_BGRA;
				frame->realWidth = tDesc.Width;
				frame->realHeight = tDesc.Height;
				frame->realStride = tDesc.Width<<2;
				frame->realSize = frame->realWidth * frame->realStride;
				frame->lineSize[0] = frame->realStride;

				src = (unsigned char *)mappedScreen.pData;
				dst = (unsigned char *)frame->imgBuf;
				for(int i = 0; i< frame->realHeight; i++){
					CopyMemory(dst, src, frame->realStride);
					src+= mappedScreen.RowPitch;
					dst+=frame->realStride;
				}
				frame->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) / frameInterval;
			}while(0);
			// copy to all channels

			// store the data
			sourcePipe->store_data(data);
			sourcePipe->notify_all();

			dstBuffer->Unmap(0);
			device->Release();
			srcResource->Release();
			srcBuffer->Release();
			pRTV->Release();
			dstBuffer->Release();
		}
		return true;
	}

	// capture function that used for DX11
	bool D3D11Wrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){
		HRESULT hr;
		bool ret = false;
		struct pooldata * data = NULL;
		cg::SourceFrame * frame = NULL;
		void * pVoid = NULL;
		ID3D11Device * pDevice = NULL;

		if(!(data = sourcePipe->allocate_data())){
			// pool is FULL
			return false;
		}
		frame = (cg::SourceFrame *)data->ptr;
		ID3D11DeviceContext * pDeviceContext = NULL;
		hr = this->swapChain->GetDevice(IID_ID3D11DeviceContext, (void **)&pVoid);
		pDevice = (ID3D11Device *)pVoid;
		hr = this->swapChain->GetDevice(IID_ID3D11DeviceContext, (void **)&pVoid);

		pDeviceContext = (ID3D11DeviceContext *)pVoid;

		ID3D11RenderTargetView *pRTV = NULL;
		ID3D11Resource * pSrcResource = NULL;

		pDeviceContext->OMGetRenderTargets(1, &pRTV, NULL);
		pRTV->GetResource(&pSrcResource);

		ID3D11Texture2D * pSrcBuffer = (ID3D11Texture2D *)pSrcResource;
		ID3D11Texture2D * pDstBuffer = NULL;

		D3D11_TEXTURE2D_DESC desc;
		pSrcBuffer->GetDesc(&desc);
		desc.BindFlags = 0;
		desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;   //here, we do not consider the cuda accessing ?
		desc.Usage = D3D11_USAGE_STAGING;

		hr = pDevice->CreateTexture2D(&desc, NULL, &pDstBuffer);
		if(FAILED(hr)){
			OutputDebugString("[D3D11Wrapper]:Failed to create buffer.");
			cg::core::infoRecorder->logError("[D3D11Wrapper]: Failed to create buffer.\n");
			return false;
		}

		pDeviceContext->CopyResource(pDstBuffer, pSrcBuffer);

		D3D11_MAPPED_SUBRESOURCE mappedScreen;
		hr = pDeviceContext->Map(pDstBuffer, 0, D3D11_MAP_READ, 0, &mappedScreen);
		if(FAILED(hr)){
			OutputDebugString("Failed to map from DeviceContext");
			cg::core::infoRecorder->logError("[D3D11Wrapper]: failed to create buffer.\n");
		}
		// copy image
		if(this->isWrapSurface()){
			// store the surface to pipeline
			frame->type = SURFACE;
			frame->dxSurface->d11Surface = dstBuffer;
			frame->width = desc.Width;
			frame->height = desc.Height;
			frame->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) /frameInterval;

			sourcePipe->store_data(data);
			sourcePipe->notify_all();

			//pDeviceContext->Unmap(pDstBuffer, 0);
			pDevice->Release();
			pDeviceContext->Release();
			pSrcResource->Release();
			pSrcBuffer->Release();
			pRTV->Release();
			//pDstBuffer->Release();

			ret = true;
		}
		else{
			do{
				unsigned char * src = NULL, *dst = NULL;
				//frame->pixelFormat = PIX_FMT_BGRA;
				frame->realWidth = desc.Width;
				frame->realHeight = desc.Height;
				frame->realStride = desc.Width << 2;
				frame->realSize = frame->realWidth * frame->realStride;
				frame->lineSize[0] = frame->realStride;
				src = (unsigned char *)mappedScreen.pData;
				dst = (unsigned char *)frame->imgBuf;

				for( int i = 0; i< frame->realHeight; i++){
					CopyMemory(dst, src, frame->realStride);
					src += mappedScreen.RowPitch;
					dst += frame->realStride;
				}
				frame->imgPts = cg::core::pcdiff_us(captureTv, initialTv, freq) /frameInterval;
			}while(0);

			sourcePipe->store_data(data);
			sourcePipe->notify_all();

			pDeviceContext->Unmap(pDstBuffer, 0);
			pDevice->Release();
			pDeviceContext->Release();
			pSrcResource->Release();
			pSrcBuffer->Release();
			pRTV->Release();
			pDstBuffer->Release();
			//pDstBuffer->Release();

			ret = true;
		}
		return ret;
	}
}