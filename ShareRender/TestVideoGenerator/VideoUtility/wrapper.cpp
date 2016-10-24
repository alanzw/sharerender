#include "videocommon.h"
#include "wrapper.h"
#include <d3dx9.h>

#include "log.h"
#include "TimeTool.h"

//#include <Windows.h>

static IDirect3D9 * g_pD3D;

Wrapper::Wrapper(){
	captureHwnd = NULL;
	pipe = NULL;
	id = 0;
	sourceType = SOURCE_TYPE::SOURCE_NONE;
	encoderType = ENCODER_TYPE::ENCODER_NONE;
	imageNotifier = CreateEvent(NULL, FALSE, FALSE, NULL);
	InitializeCriticalSection(&section);
	running = false;
	fps = 30;
	screenWidth = 0;
	screenHeight = 0;
}

// used by image source, noly
Wrapper::Wrapper(HWND hwnd):captureHwnd(hwnd), id(0), pipe(NULL){
	sourceType = SOURCE_TYPE::IMAGE;
	encoderType = ENCODER_TYPE::X264_ENCODER;
	imageNotifier = CreateEvent(NULL, FALSE, FALSE, NULL);
	InitializeCriticalSection(&section);
}

Wrapper::Wrapper(SOURCE_TYPE sourceType, ENCODER_TYPE encoderType){
	captureHwnd = NULL;
	pipe = NULL;
	id = 0;
	this->sourceType =sourceType;
	this->encoderType = encoderType;
	imageNotifier = CreateEvent(NULL, FALSE, FALSE, NULL);
	InitializeCriticalSection(&section);
}

Wrapper::~Wrapper(){
	if(pipe){
		delete pipe;
		pipe = NULL;
	}
	if(imageNotifier){
		CloseHandle(imageNotifier);
		imageNotifier = NULL;
	}
	running = false;
}

bool Wrapper::isWrapSurface(){
	bool ret = true;
	EnterCriticalSection(&this->section);
	if(this->sourceType == SURFACE){
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


D3DWrapper::~D3DWrapper(){
	if(cudaPipe){
		delete cudaPipe;
		cudaPipe = NULL;
	}
	if(cudaNotifier){
		CloseHandle(cudaNotifier);
		cudaNotifier = NULL;
	}
}

int D3DWrapper::init(struct ccgImage* image, int width, int height){
	fillBitmapInfo(&bmpInfo, width, height, BITSPERPIXEL);
	frameSize = bmpInfo.bmiHeader.biSizeImage;
	image->width = bmpInfo.bmiHeader.biWidth;
	image->height = bmpInfo.bmiHeader.biHeight;
	image->bytes_per_line = (BITSPERPIXEL >> 3) * image->width;

	return 0;
}

int D3DWrapper::init(struct ccgImage *image, HWND captureHwnd){
	D3DDISPLAYMODE		ddm;
	D3DPRESENT_PARAMETERS	d3dpp;
	//
	//captureHwnd = GetDesktopWindow();
	//
	ZeroMemory(&screenRect, sizeof(RECT));
	fillBitmapInfo(
		&bmpInfo,
		GetSystemMetrics(SM_CXSCREEN),
		GetSystemMetrics(SM_CYSCREEN),
		BITSPERPIXEL);
	frameSize = bmpInfo.bmiHeader.biSizeImage;
	//
	image->width = bmpInfo.bmiHeader.biWidth;
	image->height = bmpInfo.bmiHeader.biHeight;
	image->bytes_per_line = (BITSPERPIXEL >> 3) * image->width;
	screenWidth = image->width;
	screenHeight = image->height;

	if (!g_pD3D){
		if ((g_pD3D = Direct3DCreate9(D3D_SDK_VERSION)) == NULL)
		{
			Log::slog("Unable to Create Direct3D\n");
			return -1;
		}
	}

	if (FAILED(g_pD3D->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &ddm)))
	{
		Log::slog("Unable to Get Adapter Display Mode\n");
		goto initErrorQuit;
	}

	ZeroMemory(&d3dpp, sizeof(D3DPRESENT_PARAMETERS));

	d3dpp.Windowed = D3D_WINDOW_MODE;
	d3dpp.Flags = D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
	d3dpp.BackBufferFormat = ddm.Format;
	d3dpp.BackBufferHeight = screenRect.bottom = ddm.Height;
	d3dpp.BackBufferWidth = screenRect.right = ddm.Width;
	d3dpp.MultiSampleType = D3DMULTISAMPLE_NONE;
	d3dpp.SwapEffect = D3DSWAPEFFECT_COPY; //DISCARD;
	d3dpp.hDeviceWindow = captureHwnd /*NULL*/ /*hWnd*/;
	d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_DEFAULT;
	d3dpp.FullScreen_RefreshRateInHz = D3DPRESENT_RATE_DEFAULT;

	makeBitmapInfo(&bmpInfo, ddm.Width, ddm.Height, BITSPERPIXEL);
	frameSize = bmpInfo.bmiHeader.biSizeImage;

	if (FAILED(g_pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, captureHwnd /*NULL*/ /*hWnd*/,
		D3DCREATE_SOFTWARE_VERTEXPROCESSING, &d3dpp, &d3d_device)))
	{
		Log::slog("Unable to Create Device\n");
		goto initErrorQuit;
	}
	if (FAILED(d3d_device->CreateOffscreenPlainSurface(ddm.Width, ddm.Height,
		D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM/*D3DPOOL_SCRATCH*/, &surface, NULL)))
	{
		Log::slog("Unable to Create Surface\n");
		goto initErrorQuit;
	}

	return 0;
	//
initErrorQuit:
	deInit();
	return -1;
}

int D3DWrapper::deInit(){
	if (surface)
	{
		surface->Release();
		surface = NULL;
	}
	if (d3d_device){
		d3d_device->Release();
		d3d_device = NULL;
	}
	return 0;
}

void D3DWrapper::makeBitmapInfo(BITMAPINFO * pInfo, int w, int h, int bitsPerPixel){
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


void D3DWrapper::fillBitmapInfo(BITMAPINFO * pInfo, int w, int h, int bitsPerPixel){
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

// the d3d9 wrapper may only need the window handle, get the heigh and widht, create a new D3D9Device
// the capture function for D3D9
bool D3DWrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){

	D3DLOCKED_RECT lockedRect;
	D3DSURFACE_DESC desc;
	struct pooldata * data = NULL;
	ImageFrame * iframe = NULL;   // for image
	SurfaceFrame * sframe = NULL; // for surface
	VFrame * vframe = NULL;
	//DebugBreak();

	if(this->isWrapSurface()){
		//DebugBreak();
		infoRecorder->logError("[D3DWrapper]: cpature surface.\n");
		// get the surface, store to cuda pipeline
		// cheeck the device
		if(this->d3d_device == NULL){
			Log::slog("[D3DWrapper]: get NULL d3d device.\n");
			return false;
		}
		if(this->cudaPipe == NULL){
			Log::slog("[D3DWrapper]: get NULL piepline.\n");
			return false;
		}
		if(frameInterval == 0){
			frameInterval = 2;
		}
		//DebugBreak();
		data = cudaPipe->allocate_data();
		sframe = (SurfaceFrame *)data->ptr;
		vframe = (VFrame *)data->ptr;
		vframe->type = SURFACE;
		sframe->type = SURFACE;
		sframe->imgPts = pcdiff_us(captureTv, initialTv, freq) / (1+frameInterval);
		
		if(this->capture(CUDA_ENCODER, sframe)){
			cudaPipe->store_data(data);
			cudaPipe->notify_all();
			infoRecorder->logError("[D3DWrapper]: store surface %p, type addr:%p, vframe type addr:%p, DXSurface:0x%p.\n", data->ptr, &(sframe->type), &(vframe->type), sframe->dxSurface->d10Surface);


			///infoRecorder->logError("[D3DWrapper]: 
		}else{
			// failed.
			Log::slog("[D3DWrapper]: capture NULL surface for D3DWrapper.\n");
			return false;
		}
	}
	else{
		infoRecorder->logError("[D3DWrapper]: cpature surface.\n");
		// get the image, store to image pipeline
		if(!d3d_device || !pipe){
			Log::slog("[D3DWrapper]: get NULL pipeline.\n");
		}
		data = pipe->allocate_data();
		iframe = (ImageFrame*)data->ptr;
		
		iframe->type = IMAGE;
		iframe->imgPts = pcdiff_us(captureTv, initialTv, freq) / frameInterval;
		IDirect3DSurface9 * result = NULL;
		this->capture(X264_ENCODER, &result);
		if(result == NULL){
			Log::slog("[D3DWrapper]: capture NULL surface for image.\n");
			return false;
		}
		result->GetDesc(&desc);
		HRESULT hr = result->LockRect(&lockedRect, NULL, NULL);
		if(FAILED(hr)){
			Log::slog("[D3DWrapper]: lock rect failed.\n");
			return false;
		}
		// copy image
		do{
			int imageSize = 0;
			unsigned char * src = NULL, *dst = NULL;
			//data = pipe->allocate_data();
			//iframe->pixelFormat= PIX_FMT_BGRA;
			iframe->realWidth = desc.Width;
			iframe->realHeight = desc.Height;
			iframe->realStride = desc.Width << 2;
			iframe->realSize = iframe->realHeight * iframe->realStride;
			iframe->lineSize[0] = iframe->realStride;

			src = (unsigned char *)lockedRect.pBits;
			dst = (unsigned char *)iframe->imgBuf;

			for(int i = 0; i<iframe->realHeight; i++){
				CopyMemory(dst, src, iframe->realStride);
				src+= lockedRect.Pitch;
				dst+= iframe->realStride;
				imageSize += lockedRect.Pitch;
			}
			Log::log("[D3DWrapper]: get image data:%d, height:%d, pitch:%d.\n", imageSize, iframe->realHeight, lockedRect.Pitch);
			

		}while(0);
		pipe->store_data(data);
		pipe->notify_all();
		result->UnlockRect();
		result->Release();
	}
}

int D3DWrapper::capture(char * buf, int bufLen, struct ccgRect * gRect){
#if 0
	D3DLOCKED_RECT lockedRect;
	if (gRect == NULL && bufLen < frameSize)
		return -1;
	if (gRect != NULL && bufLen < gRect->size)
		return -1;
	if (d3d_device == NULL || surface == NULL)
		return -1;
	// get front buffer
	d3d_device->GetFrontBufferData(0, surface);
	// lock
	if (FAILED(surface->LockRect(&lockedRect, &(screenRect),
		D3DLOCK_NO_DIRTY_UPDATE | D3DLOCK_NOSYSLOCK | D3DLOCK_READONLY)))
	{
		Log::slog("Unable to Lock Front Buffer Surface\n");
		return -1;
	}
	// copy
	if (gRect == NULL) {
		CopyMemory(buf, lockedRect.pBits, lockedRect.Pitch * screenRect.bottom);
	}
	else {
		int i;
		char *src, *dst;
		src = (char *)lockedRect.pBits;
		src += lockedRect.Pitch * gRect->top;
		src += RGBA_SIZE * gRect->left;
		dst = (char*)buf;
		//
		for (i = 0; i < gRect->height; i++) {
			CopyMemory(dst, src, gRect->linesize);
			src += lockedRect.Pitch;
			dst += gRect->linesize;
		}
	}
	// Unlock
	surface->UnlockRect();

	return gRect == NULL ? frameSize : gRect->size;
#endif
	return 0;
}
#if 0
bool D3DWrapper::startThread(){
	return true;
}
// the thread proc for the wrapper
DWORD WINAPI D3DWrapper::WrapperThreadProc(LPVOID param){
	return 0;
}
#endif
IDirect3DSurface9 * D3DWrapper::capture(ENCODER_TYPE encoderType, IDirect3DSurface9 ** ppDst){
	HRESULT hr;
	D3DSURFACE_DESC desc;
	IDirect3DSurface9 * renderSurface = NULL, *oldRenderSurface = NULL, * resolvedSurface = *ppDst;
	D3DLOCKED_RECT lockedRect;
	int i = 0;
	struct pooldata * data = NULL;
	ImageFrame * frame = NULL;

	renderSurface = oldRenderSurface = NULL;
	
#if 0
	NoneFrame * no = new NoneFrame();
	VFrame * vf = (VFrame *)no;

	infoRecorder->logError("none type addr:%p, VFrame type:%p.\n", &(no->type), &(vf->type));
#endif

	hr = d3d_device->GetRenderTarget(0, &renderSurface);
	if (FAILED(hr)){
		if (hr == D3DERR_INVALIDCALL){
			Log::log("[capture]: GetRenderTarget failed (INVALIDCALL).\n");
		}
		else if (hr == D3DERR_NOTFOUND){
			Log::log("[capture]: GetRenderTarget failed (D3DERR_NOTFOUND).\n");
		}
		else{
			Log::log("[caputre]: GetRenderTarget failed. (Other).\n");
		}
	}
	if (renderSurface == NULL){
		Log::log("[capture]: renderSurface is NULL.\n");
		return false;
	}

	renderSurface->GetDesc(&desc);

	// check if the surfaceo of local game enable multisampling 
	// multisampling enable will avoid locking in the surface
	// if yes, create a no-multisampling surface and copy frame into it

	// create new render target to transfor
#if 0
	if (desc.MultiSampleType != D3DMULTISAMPLE_NONE){
		if (resolvedSurface == NULL){
			hr = d3d_device->CreateRenderTarget(desc.Width, desc.Height, desc.Format, D3DMULTISAMPLE_NONE,
				0, FALSE, &resolvedSurface, NULL);
			if (FAILED(hr)){
				Log::log("[capture]: CreateRenderTarget (resolvedSurface) failed.\n");
				infoRecorder->logError("[D3DWrapper]: CreateRenderTarget (resolvedSurface) failed.\n");
				renderSurface->Release();

				return NULL;
			}
		}
		hr = d3d_device->StretchRect(renderSurface, NULL, resolvedSurface, NULL, D3DTEXF_NONE);
		if (FAILED(hr)){
			Log::log("[capture]: StretchRect failed.\n");
			renderSurface->Release();
			return NULL;
		}

		oldRenderSurface = renderSurface;
		renderSurface = resolvedSurface;
		// the stretched surface is in gpu memory, so it can be used as offscreen surface
		deviceOffscreenSurface = resolvedSurface;
		infoRecorder->logError("[D3DWrapper]: get the render target. deviceOffScreenSurface:0x%p.\n", deviceOffscreenSurface);
	}
	else{
		Log::log("[capture]: render target is not Multisampled.\n");
		infoRecorder->logError("[D3DWrapper]: render target is not Multisampled.\n");
		deviceOffscreenSurface = renderSurface;
	}
#else
	// create a new RenderTarget no matter whether the render target is MULTI_SAMPLED or not
	if (resolvedSurface == NULL){
		hr = d3d_device->CreateRenderTarget(desc.Width, desc.Height, desc.Format, D3DMULTISAMPLE_NONE,
			0, FALSE, &resolvedSurface, NULL);
		if (FAILED(hr)){
			Log::log("[capture]: CreateRenderTarget (resolvedSurface) failed.\n");
			infoRecorder->logError("[D3DWrapper]: CreateRenderTarget (resolvedSurface) failed.\n");
			renderSurface->Release();

			return NULL;
		}
		*ppDst = resolvedSurface;
	}
	hr = d3d_device->StretchRect(renderSurface, NULL, resolvedSurface, NULL, D3DTEXF_NONE);
	//Gernhr = d3d_device->GetRenderTargetData
	if (FAILED(hr)){
		infoRecorder->logError("[capture]: StretchRect failed.\n");
		renderSurface->Release();
		return NULL;
	}

	oldRenderSurface = renderSurface;
	renderSurface = resolvedSurface;
	// the stretched surface is in gpu memory, so it can be used as offscreen surface
	deviceOffscreenSurface = resolvedSurface;
	infoRecorder->logError("[D3DWrapper]: get the render target. deviceOffScreenSurface:0x%p.\n", deviceOffscreenSurface);

#endif

	// surface is only for cuda encoder
	// create offline surface in system memory or device memory
#if 1
	if(encoderType == ENCODER_TYPE::X264_ENCODER){
		if(sysOffscreenSurface == NULL){
			hr = d3d_device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &sysOffscreenSurface, NULL);
			if (FAILED(hr)){
				infoRecorder->logError("[capture]: Create offscreen surface failed.\n");
				renderSurface->Release();
				return NULL;
			}
		}

		// copy the render-target data from device memory to system memory
		hr = d3d_device->GetRenderTargetData(renderSurface, sysOffscreenSurface);
		if (FAILED(hr)){
			infoRecorder->logError("[capture]: GetRenderTargetData failed.\n");
			if (hr == D3DERR_DRIVERINTERNALERROR){
				infoRecorder->logError("[capture]: GetRenderTargetData failed code: D3DERR_DIRVERINTERNALERROR.\n");
			}
			else if (hr == D3DERR_DEVICELOST){
				infoRecorder->logError("[capture]: GetRenderTargetData failed code: D3DERR_DEVICELOST.\n");
			}
			else if (hr == D3DERR_INVALIDCALL){
				infoRecorder->logError("[capture]: GetRenderTargetData failed code: D3DERR_INVALIDCALL.\n");
			}
			else{
				infoRecorder->logError("[capture]: Get render target data failed with code:%d\n", hr);
			}
			if (oldRenderSurface)
				oldRenderSurface->Release();
			else
				renderSurface->Release();
			return NULL;
		}

		if (oldRenderSurface)
			oldRenderSurface->Release();
		else
			renderSurface->Release(); 

		return sysOffscreenSurface;

	}else if(this->encoderType == ENCODER_TYPE::CUDA_ENCODER){
#endif
		if(deviceOffscreenSurface == NULL){
			infoRecorder->logError("[D3DWrapper]: device offscreen surface is NULL? error ?\n");
			hr = d3d_device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, D3DPOOL_DEFAULT, &deviceOffscreenSurface, NULL);
			if (FAILED(hr)){
				infoRecorder->logError("[capture]: Create offscreen surface failed.\n");
				renderSurface->Release();
				return NULL;
			}

		}
		//DebugBreak();
#if 1
		char name[100] = {0};
		static int index= 0;
		sprintf(name, "capture-%d.jpg", index++);
		HRESULT hrr =D3DXSaveSurfaceToFile(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_JPG, deviceOffscreenSurface, NULL, NULL);

		if(FAILED(hrr)){
			infoRecorder->logError("[D3DWrapper]: save surface to file failed.\n");
		}
#endif
		//
#if 1

		// render surface is done here.
		if (oldRenderSurface)
			oldRenderSurface->Release();
		else{
			//renderSurface->Release();
		}
		// just return the gpu memeory surface
		return deviceOffscreenSurface;
	}
#endif

	// render surface is done here.
	if (oldRenderSurface)
		oldRenderSurface->Release();
	else
		renderSurface->Release();

	return NULL;
}

bool D3DWrapper::capture(ENCODER_TYPE encoderType, SurfaceFrame * sframe){
	HRESULT hr;
	D3DSURFACE_DESC desc;
	IDirect3DSurface9 * renderSurface = NULL, *oldRenderSurface = NULL, * resolvedSurface = sframe->dxSurface->d9Surface;
	D3DLOCKED_RECT lockedRect;
	int i = 0;
	struct pooldata * data = NULL;
	ImageFrame * frame = NULL;

	renderSurface = oldRenderSurface = NULL;

	hr = d3d_device->GetRenderTarget(0, &renderSurface);
	if (FAILED(hr)){
		if (hr == D3DERR_INVALIDCALL){
			Log::log("[capture]: GetRenderTarget failed (INVALIDCALL).\n");
		}
		else if (hr == D3DERR_NOTFOUND){
			Log::log("[capture]: GetRenderTarget failed (D3DERR_NOTFOUND).\n");
		}
		else{
			Log::log("[caputre]: GetRenderTarget failed. (Other).\n");
		}
	}
	if (renderSurface == NULL){
		Log::log("[capture]: renderSurface is NULL.\n");
		return false;
	}

	renderSurface->GetDesc(&desc);

	// check if the surfaceo of local game enable multisampling 
	// multisampling enable will avoid locking in the surface
	// if yes, create a no-multisampling surface and copy frame into it

	// create new render target to transfor

	// create a new RenderTarget no matter whether the render target is MULTI_SAMPLED or not
	if (resolvedSurface == NULL){
		infoRecorder->logError("[capture]: to create device surface for pipeline, total should be 8!\n");
		hr = d3d_device->CreateRenderTarget(desc.Width, desc.Height, desc.Format, D3DMULTISAMPLE_NONE,
			0, FALSE, &resolvedSurface, NULL);
		if (FAILED(hr)){
			Log::log("[capture]: CreateRenderTarget (resolvedSurface) failed.\n");
			infoRecorder->logError("[D3DWrapper]: CreateRenderTarget (resolvedSurface) failed.\n");
			renderSurface->Release();

			return false;
		}
		sframe->dxSurface->d9Surface = resolvedSurface;
	}

	hr = d3d_device->StretchRect(renderSurface, NULL, resolvedSurface, NULL, D3DTEXF_NONE);
	//Gernhr = d3d_device->GetRenderTargetData
	if (FAILED(hr)){
		infoRecorder->logError("[capture]: StretchRect failed.\n");
		renderSurface->Release();
		return false;
	}

	oldRenderSurface = renderSurface;
	renderSurface = resolvedSurface;
	// the stretched surface is in gpu memory, so it can be used as offscreen surface
	deviceOffscreenSurface = resolvedSurface;
	infoRecorder->logError("[D3DWrapper]: get the render target. deviceOffScreenSurface:0x%p.\n", deviceOffscreenSurface);

	// surface is only for cuda encoder
	// create offline surface in system memory or device memory
#if 1
	if(encoderType == ENCODER_TYPE::X264_ENCODER){
		if(sysOffscreenSurface == NULL){
			hr = d3d_device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, D3DPOOL_SYSTEMMEM, &sysOffscreenSurface, NULL);
			if (FAILED(hr)){
				infoRecorder->logError("[capture]: Create offscreen surface failed.\n");
				renderSurface->Release();
				return false;
			}
		}

		// copy the render-target data from device memory to system memory
		hr = d3d_device->GetRenderTargetData(renderSurface, sysOffscreenSurface);
		if (FAILED(hr)){
			infoRecorder->logError("[capture]: GetRenderTargetData failed.\n");
			if (hr == D3DERR_DRIVERINTERNALERROR){
				infoRecorder->logError("[capture]: GetRenderTargetData failed code: D3DERR_DIRVERINTERNALERROR.\n");
			}
			else if (hr == D3DERR_DEVICELOST){
				infoRecorder->logError("[capture]: GetRenderTargetData failed code: D3DERR_DEVICELOST.\n");
			}
			else if (hr == D3DERR_INVALIDCALL){
				infoRecorder->logError("[capture]: GetRenderTargetData failed code: D3DERR_INVALIDCALL.\n");
			}
			else{
				infoRecorder->logError("[capture]: Get render target data failed with code:%d\n", hr);
			}
			if (oldRenderSurface)
				oldRenderSurface->Release();
			else
				renderSurface->Release();
			return NULL;
		}

		if (oldRenderSurface)
			oldRenderSurface->Release();
		else
			renderSurface->Release(); 

		return true;

	}else if(this->encoderType == ENCODER_TYPE::CUDA_ENCODER){
#endif
		if(deviceOffscreenSurface == NULL){
			infoRecorder->logError("[D3DWrapper]: device offscreen surface is NULL? error ?\n");
			hr = d3d_device->CreateOffscreenPlainSurface(desc.Width, desc.Height, desc.Format, D3DPOOL_DEFAULT, &deviceOffscreenSurface, NULL);
			if (FAILED(hr)){
				infoRecorder->logError("[capture]: Create offscreen surface failed.\n");
				renderSurface->Release();
				return false;
			}

		}
		//DebugBreak();
#if 0
		char name[100] = {0};
		static int index= 0;
		sprintf(name, "capture-%d.jpg", index++);
		HRESULT hrr =D3DXSaveSurfaceToFile(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_JPG, deviceOffscreenSurface, NULL, NULL);

		if(FAILED(hrr)){
			infoRecorder->logError("[D3DWrapper]: save surface to file failed.\n");
			
		}
#endif
		//
#if 1

		// render surface is done here.
		if (oldRenderSurface)
			oldRenderSurface->Release();
		else{
			//renderSurface->Release();
		}
		// just return the gpu memeory surface
		return true;
	}
#endif

	// render surface is done here.
	if (oldRenderSurface)
		oldRenderSurface->Release();
	else
		renderSurface->Release();

	return false;
}



//capture the surface to encode
IDirect3DSurface9 * D3DWrapper::capture(ENCODER_TYPE encoderType, int gameWidth, int gameHeight){

	HRESULT hr;
	D3DSURFACE_DESC desc;
	IDirect3DSurface9 * renderSurface = NULL, *oldRenderSurface = NULL, * resolvedSurface = NULL;
	D3DLOCKED_RECT lockedRect;
	int i = 0;
	struct pooldata * data = NULL;
	ImageFrame * frame = NULL;

#if 0
	if (vsource_initialized == 0){
		return NULL;
	}
#endif

	Log::log("[Channel]: capture the screen.\n");
	renderSurface = oldRenderSurface = NULL;

	hr = d3d_device->GetRenderTarget(0, &renderSurface);
	if (FAILED(hr)){
		if (hr == D3DERR_INVALIDCALL){
			Log::log("[capture]: GetRenderTarget failed (INVALIDCALL).\n");
		}
		else if (hr == D3DERR_NOTFOUND){
			Log::log("[capture]: GetRenderTarget failed (D3DERR_NOTFOUND).\n");
		}
		else{
			Log::log("[caputre]: GetRenderTarget failed. (Other).\n");
		}
	}
	if (renderSurface == NULL){
		Log::log("[capture]: renderSurface is NULL.\n");
		return false;
	}

	renderSurface->GetDesc(&desc);

	if (desc.Width != gameWidth || desc.Height != gameHeight){
		Log::log("[capture]: game width and height are not match !\n");
		return false;
	}

	if (capture_initialized == 0){
		frame_interval = 1000000 / fps; // in the unif of us
		frame_interval++;
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&initialTv);
		capture_initialized = 1;
	}
	else{
		QueryPerformanceCounter(&captureTv);
	}

	// check if the surfaceo of local game enable multisampling 
	// multisampling enable will avoid locking in the surface
	// if yes, create a no-multisampling surface and copy frame into it
	if (desc.MultiSampleType != D3DMULTISAMPLE_NONE){
		if (resolvedSurface == NULL){
			hr = d3d_device->CreateRenderTarget(gameWidth, gameHeight, desc.Format, D3DMULTISAMPLE_NONE,
				0, FALSE, &resolvedSurface, NULL);
			if (FAILED(hr)){
				Log::log("[capture]: CreateRenderTarget (resolvedSurface) failed.\n");
				renderSurface->Release();

				return NULL;
			}
		}
		hr = d3d_device->StretchRect(renderSurface, NULL, resolvedSurface, NULL, D3DTEXF_NONE);
		if (FAILED(hr)){
			Log::log("[capture]: StretchRect failed.\n");
			renderSurface->Release();
			return NULL;
		}

		oldRenderSurface = renderSurface;
		renderSurface = resolvedSurface;
		// the stretched surface is in gpu memory, so it can be used as offscreen surface
		deviceOffscreenSurface = resolvedSurface;
	}
	else{
		Log::log("[capture]: render target is Multisampled.\n");
	}


	// create offline surface in system memory or device memory
	if(this->encoderType == ENCODER_TYPE::X264_ENCODER){
		if(sysOffscreenSurface == NULL){
			hr = d3d_device->CreateOffscreenPlainSurface(gameWidth, gameHeight, desc.Format, D3DPOOL_SYSTEMMEM, &sysOffscreenSurface, NULL);
			if (FAILED(hr)){
				Log::log("[capture]: Create offscreen surface failed.\n");
				renderSurface->Release();
				return NULL;
			}
		}

		// copy the render-target data from device memory to system memory
		hr = d3d_device->GetRenderTargetData(renderSurface, sysOffscreenSurface);
		if (FAILED(hr)){
			Log::log("[capture]: GetRenderTargetData failed.\n");
			if (hr == D3DERR_DRIVERINTERNALERROR){
				Log::log("[capture]: GetRenderTargetData failed code: D3DERR_DIRVERINTERNALERROR.\n");
			}
			else if (hr == D3DERR_DEVICELOST){
				Log::log("[capture]: GetRenderTargetData failed code: D3DERR_DEVICELOST.\n");
			}
			else if (hr == D3DERR_INVALIDCALL){
				Log::log("[capture]: GetRenderTargetData failed code: D3DERR_INVALIDCALL.\n");
			}
			else{
				Log::log("[capture]: Get render target data failed with code:%d\n", hr);
			}
			if (oldRenderSurface)
				oldRenderSurface->Release();
			else
				renderSurface->Release();
			return NULL;
		}

		if (oldRenderSurface)
			oldRenderSurface->Release();
		else
			renderSurface->Release();

		return sysOffscreenSurface;

	}else if(this->encoderType == ENCODER_TYPE::CUDA_ENCODER){
		if(deviceOffscreenSurface == NULL){
			hr = d3d_device->CreateOffscreenPlainSurface(gameWidth, gameHeight, desc.Format, D3DPOOL_DEFAULT, &deviceOffscreenSurface, NULL);
			if (FAILED(hr)){
				Log::log("[capture]: Create offscreen surface failed.\n");
				renderSurface->Release();
				return NULL;
			}
		}
		//


		// render surface is done here.
		if (oldRenderSurface)
			oldRenderSurface->Release();
		else
			renderSurface->Release();

		// just return the gpu memeory surface
		return deviceOffscreenSurface;
	}

	// render surface is done here.
	if (oldRenderSurface)
		oldRenderSurface->Release();
	else
		renderSurface->Release();

	return NULL;
}

// for window wrapper
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
	image->width = bmpInfo.bmiHeader.biWidth;
	image->height = bmpInfo.bmiHeader.biHeight;
	image->bytes_per_line = (BITSPERPIXEL >> 3) * image->width;
	screenWidth = image->width;
	screenHeight = image->height;

	// add capture window code here
	captureHdc = GetDC(captureHwnd);
	captureCompatibaleDC = CreateCompatibleDC(captureCompatibaleDC);
	captureCompatibleBitmap = CreateDIBSection(captureHdc, &bmpInfo, DIB_RGB_COLORS, &pBits, NULL, 0);

	if (captureCompatibaleDC == NULL || captureCompatibleBitmap == NULL){
		Log::slog("[WindowWrapper]: unbale to create compatibale DC/Bitmap.\n");
		return -1;
	}
	SelectObject(captureCompatibaleDC, captureCompatibleBitmap);

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

#if 0

void WindowWrapper::makeBitmapInfo(int w, int h, int bitsPerPixel){
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

void WindowWrapper::fillBitmapInfo(int w, int h, int bitsPerPixel){
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

#endif

bool WindowWrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){
	infoRecorder->logError("[WindowWrapper]: this capture should never be used.\n");
	return false;
}
int WindowWrapper::capture(char * buf, int bufLen, struct ccgRect * gRect){
	int linesize, height;
	char * source, *dst;
	if (gRect == NULL && bufLen < frameSize){
		return -1;
	}
	if (gRect != NULL && bufLen < gRect->size)
		return -1;
	if (captureCompatibaleDC == NULL || pBits == NULL)
		return -1;

	BitBlt(captureCompatibaleDC, 0, 0, screenRect.right + 1, screenRect.bottom + 1, captureHdc, 0, 0, SRCCOPY | CAPTUREBLT);

	//// XXX: image are sotred upside-down
	linesize = (BITSPERPIXEL >> 3) *(screenRect.right + 1);
	if (gRect == NULL){
		source = ((char *)pBits) - (linesize * screenRect.bottom);
		dst = buf;
		height = screenRect.bottom + 1;
		while (height-- > 0){
			CopyMemory(dst, source, linesize);
			dst += linesize;
			source -= linesize;
		}
	}
	else{
		source = (char *)pBits;
		source += linesize *(screenRect.bottom - gRect->top);
		source += RGBA_SIZE * gRect->left;
		dst = buf;
		height = gRect->height;
		while (height-- > 0){
			CopyMemory(dst, source, gRect->linesize);
			dst += gRect->linesize;
			source -= linesize;
		}
	}

	return gRect == NULL ? frameSize : gRect->size;
}

bool WindowWrapper::isWrapSurface(){
	return false;
}

#if 0
bool WindowWrapper::startThread(){
	threadHandle = chBEGINTHREADEX(NULL, 0, WrapperThreadProc, this, FALSE, &threadId);
	return true;
}
// the thread proc for the wrapper
DWORD WINAPI WindowWrapper::WrapperThreadProc(LPVOID param){

	WindowWrapper * wrapper = (WindowWrapper *)param;
	// first check the window handle
	if (!wrapper || !wrapper->captureHwnd){
		Log::slog("[WindowWrapper]: get NULL wrapper or NULL window handle in wrapper thread proc.\n");
		return -1;
	}
	// how to set the cycle
	WaitForSingleObject(wrapper->notifier, INFINITE);

	while (wrapper->running){

	}
	return 0;
}
#endif

WindowWrapper::WindowWrapper(HWND h): Wrapper(h){
	pBits = NULL;
}




//// D3DWrappers//////

D3D10Wrapper::D3D10Wrapper(IDXGISwapChain * chain):Wrapper(SURFACE, CUDA_ENCODER), swapChain(chain){
	device = NULL;
	pRTV = NULL;
	srcResource = NULL;
	srcBuffer = NULL;
	dstBuffer = NULL;

	pipe = NULL;
}

D3D10Wrapper::~D3D10Wrapper(){
	if(cudaPipe){
		delete cudaPipe;
		cudaPipe = NULL;
	}
	if(cudaNotifier){
		CloseHandle(cudaNotifier);
		cudaNotifier = NULL;
	}
}
D3D10Wrapper1::~D3D10Wrapper1(){
	if(cudaPipe){
		delete cudaPipe;
		cudaPipe = NULL;
	}
	if(cudaNotifier){
		CloseHandle(cudaNotifier);
		cudaNotifier = NULL;
	}

}

D3D10Wrapper1::D3D10Wrapper1(IDXGISwapChain * chain):Wrapper(SURFACE, CUDA_ENCODER), swapChain(chain){
	device = NULL;
	pRTV = NULL;
	srcResource = NULL;
	srcBuffer = NULL;
	dstBuffer = NULL;

	pipe = NULL;
}

D3D11Wrapper::~D3D11Wrapper(){
	if(cudaPipe){
		delete cudaPipe;
		cudaPipe = NULL;
	}
	if(cudaNotifier){
		CloseHandle(cudaNotifier);
		cudaNotifier = NULL;
	}
}

D3D11Wrapper::D3D11Wrapper(IDXGISwapChain * chain):Wrapper(SURFACE, CUDA_ENCODER), swapChain(chain){
	device = NULL;
	pRTV = NULL;
	srcResource = NULL;
	srcBuffer = NULL;
	dstBuffer = NULL;
	deviceContext = NULL;

	pipe = NULL;
}

bool D3D10Wrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){
	HRESULT hr;

	struct pooldata * data;
	ImageFrame * frame = NULL;
	
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
		data = cudaPipe->allocate_data();
		SurfaceFrame * surface = (SurfaceFrame *)data->ptr;
		surface->dxSurface->d10Surface = dstBuffer;
		surface->width = tDesc.Width;
		surface->height= tDesc.Height;

		surface->imgPts = pcdiff_us(captureTv, initialTv, freq) /frameInterval;

		cudaPipe->store_data(data);
		cudaPipe->notify_all();

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
			data = this->pipe->allocate_data();
			frame = (ImageFrame *)data->ptr;
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
			frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frameInterval;
		}while(0);
		// copy to all channels

		// store the data
		pipe->store_data(data);
		pipe->notify_all();

		dstBuffer->Unmap(0);
		device->Release();
		srcResource->Release();
		srcBuffer->Release();
		pRTV->Release();
		dstBuffer->Release();
	}
	return true;
}

int D3D10Wrapper::capture(char * buf, int bufLen, struct ccgRect * gRect){
	HRESULT hr;

	struct pooldata * data;
	ImageFrame * frame = NULL;
	

	if(this->swapChain == NULL){
		//error
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
			data = this->pipe->allocate_data();
			frame = (ImageFrame *)data->ptr;
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
			frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / 30;
		}while(0);
		// copy to all channels

		// store the data
		pipe->store_data(data);
		pipe->notify_all();

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

	struct pooldata * data;
	ImageFrame * frame = NULL;
	

	if(this->swapChain == NULL){
		//error
		return false;
	}
	else{
		hr = swapChain->GetDevice(IID_ID3D10Device1, (void **)&this->device);
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
		data = cudaPipe->allocate_data();
		SurfaceFrame * surface = (SurfaceFrame *)data->ptr;
		surface->dxSurface->d10Surface = dstBuffer;
		surface->width = tDesc.Width;
		surface->height= tDesc.Height;

		surface->imgPts = pcdiff_us(captureTv, initialTv, freq) /frameInterval;

		cudaPipe->store_data(data);
		cudaPipe->notify_all();

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

		// copy image
		do{
			unsigned char * src = NULL, * dst = NULL;
			data = this->pipe->allocate_data();
			frame = (ImageFrame *)data->ptr;
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
			frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frameInterval;
		}while(0);
		// copy to all channels

		// store the data
		pipe->store_data(data);
		pipe->notify_all();

		dstBuffer->Unmap(0);
		device->Release();
		srcResource->Release();
		srcBuffer->Release();
		pRTV->Release();
		dstBuffer->Release();
	}
	return true;
}

int D3D10Wrapper1::capture(char * buf, int bufLen, struct ccgRect * gRect){
	HRESULT hr;

	struct pooldata * data;
	ImageFrame * frame = NULL;

	if(this->swapChain == NULL){
		//error
	}
	else{
		hr = swapChain->GetDevice(IID_ID3D10Device1, (void **)&this->device);
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
		data = cudaPipe->allocate_data();
		SurfaceFrame * surface = (SurfaceFrame *)data->ptr;
		surface->dxSurface->d10Surface = dstBuffer;
		surface->width = tDesc.Width;
		surface->height= tDesc.Height;

		//surface->imgPts = pcdiff_us(captureTv, initialTv, freq) /frameInterval;

		cudaPipe->store_data(data);
		cudaPipe->notify_all();

		//dstBuffer->Unmap(0);
		device->Release();
		srcResource->Release();
		srcBuffer->Release();
		pRTV->Release();
		//dstBuffer->Release();
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
			data = this->pipe->allocate_data();
			frame = (ImageFrame *)data->ptr;
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
			//frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frameInterval;
		}while(0);
		// copy to all channels

		// store the data
		pipe->store_data(data);
		pipe->notify_all();

		dstBuffer->Unmap(0);
		device->Release();
		srcResource->Release();
		srcBuffer->Release();
		pRTV->Release();
		dstBuffer->Release();
	}
	return true;
}

int D3D11Wrapper::capture(char * buf, int bufLen, struct ccgRect * gRect){
	// do nothing
	infoRecorder->logError("[D3D11Wrapper]: should never use this capture function.\n");
	return -1;
}

// capture function that used for DX11
bool D3D11Wrapper::capture(LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frameInterval){
	HRESULT hr;
	bool ret = false;

	struct pooldata * data;
	ImageFrame * frame = NULL;
	void * pVoid = NULL;

	ID3D11Device * pDevice = NULL;

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
		infoRecorder->logError("[D3D11Wrapper]: Failed to create buffer.\n");
		return ret;
	}

	pDeviceContext->CopyResource(pDstBuffer, pSrcBuffer);

	D3D11_MAPPED_SUBRESOURCE mappedScreen;
	hr = pDeviceContext->Map(pDstBuffer, 0, D3D11_MAP_READ, 0, &mappedScreen);
	if(FAILED(hr)){
		OutputDebugString("Failed to map from DeviceContext");
		infoRecorder->logError("[D3D11Wrapper]: failed to create buffer.\n");
	}
	// copy image
	if(this->isWrapSurface()){
		// store the surface to pipeline
		data = cudaPipe->allocate_data();
		SurfaceFrame * surface = (SurfaceFrame *)data->ptr;
		
		surface->dxSurface->d11Surface = dstBuffer;
		surface->width = desc.Width;
		surface->height = desc.Height;
		//surface->pixelFormat = desc.Format;
		surface->imgPts = pcdiff_us(captureTv, initialTv, freq) /frameInterval;

		cudaPipe->store_data(data);
		cudaPipe->notify_all();

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
			unsigned char * src, *dst;
			data = pipe->allocate_data();
			frame = (ImageFrame *)data->ptr;
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
			frame->imgPts = pcdiff_us(captureTv, initialTv, freq) /frameInterval;
		}while(0);

		
		pipe->store_data(data);
		pipe->notify_all();

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