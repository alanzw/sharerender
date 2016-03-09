#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "LibRenderChannel.h"
#include "../LibCore/SmallHash.h"
#include "../VideoGen/generator.h"
#include <WinUser.h>

#ifndef WINDOW_CLASS 
#define WINDOW_CLASS "MYWINDOWCLASS"
#endif
#ifndef WINDOW_NAME
#define WINDOW_NAME "game-client"
#endif

#ifdef BACKBUFFER_TEST

#endif

SmallHash<HWND, HWND> serverToClient;

// for the init part of render channel
void RenderChannel::destroyD3DWindow() {
	printf("Destroying D3D window...\n");

	if(gD3d != NULL) gD3d->Release();
	gD3d = NULL;
}

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wp, LPARAM lp) {
	switch(msg) {
	case WM_DESTROY:
		PostQuitMessage(0);
		break;

	case WM_KEYUP:
		printf("keyup in MsgProc.\n");
		if(wp == VK_ESCAPE) PostQuitMessage(0);
		break;
	
	}
	return DefWindowProc(hWnd, msg, wp, lp);
}

extern HWND CreateWindowWithSDL(int w, int h, int x, int y);

HWND RenderChannel::initWindow(int width, int height, DWORD dwExStyle, DWORD dwStyle) {
	cg::core::infoRecorder->logTrace("init_window(), width=%d, height=%d\n", width, height);
	WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, MsgProc, 0, 0, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, WINDOW_CLASS, NULL };
	RegisterClassEx(&wc);

	if(width < 800) width = 800;
	if(height < 600) height = 600;

#ifdef SDL_WINDOW
	hWnd = CreateWindowWithSDL(width, height, 400, 200);
	window_handle = hWnd;
	//hh = hWnd;
#else
	int borderHeight = GetSystemMetrics(SM_CYBORDER);
	int captionHeight = GetSystemMetrics(SM_CYCAPTION);

	if (useServerStyle) {
		//window_handle = CreateWindow(WINDOW_CLASS, WINDOW_NAME, dwStyle, 0, 0, width, height, GetDesktopWindow(), NULL, wc.hInstance, NULL);
		if(gameName[0] == 'C' || gameName[0] == 'c')
			hWnd = CreateWindow(WINDOW_CLASS, WINDOW_NAME, dwExStyle , 400, 200, width, height, GetDesktopWindow(), NULL, wc.hInstance, NULL);
		else
			hWnd = CreateWindowEx(dwExStyle,WINDOW_CLASS,WINDOW_NAME,dwStyle , 400,200,width,height,GetDesktopWindow(),NULL,wc.hInstance,NULL);
	}
	else
		hWnd = CreateWindowEx(dwExStyle, WINDOW_CLASS, WINDOW_NAME, dwStyle, 400, 200, width, height, GetDesktopWindow(), NULL, wc.hInstance, NULL);
#endif

	//SetEvent(this->ch->windowHandleEvent);

	return hWnd;
}


HRESULT RenderChannel::clientInit() {
	cg::core::infoRecorder->logTrace("client_init() called\n");

	if(gD3d == NULL) {
		gD3d = Direct3DCreate9(D3D_SDK_VERSION);
	}

	if(FAILED(gD3d->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &displayMode))) return false;

	/////////////////////////////////////////////
	//read data
	int id = cc->read_int();
	UINT Adapter = cc->read_uint();
	D3DDEVTYPE DeviceType = (D3DDEVTYPE)(cc->read_uint());
	DWORD BehaviorFlags = cc->read_uint();

	cc->read_byte_arr((char*)(&d3dpp), sizeof(d3dpp));
	d3dpp.BackBufferFormat = displayMode.Format;
	/////////////////////////////////////////////

	cg::core::infoRecorder->logTrace("client_init(), init_window start, Presentation Parameter back buffer witth:%d, back buffer height:%d\n", d3dpp.BackBufferWidth, d3dpp.BackBufferHeight);

#if 0
	d3dpp.BackBufferHeight = 600;
	d3dpp.BackBufferWidth = 800;
#else
	// create window for each
	DWORD style = 0;
	DWORD exStyle = 0;

	HWND hh = initWindow(d3dpp.BackBufferWidth, d3dpp.BackBufferHeight, exStyle, style);
#endif

	if(hh == NULL) {
		cg::core::infoRecorder->logError("window_handle is NULL\n");
	}
	else{
		// add the window map
		serverToClient.addMap(d3dpp.hDeviceWindow, hh);
	}

	d3dpp.hDeviceWindow = hh;
	hWnd = hh;

	// create d3d device and register to CUDA
	LPDIRECT3DDEVICE9 base_device = NULL;
	//d3dpp.Flags |= D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;

	HRESULT hr = gD3d->CreateDevice(Adapter, DeviceType, hh, BehaviorFlags | D3DCREATE_MULTITHREADED, &d3dpp, &base_device);
	device_list[id] = base_device;
	curDevice= base_device;


	//pd3dDevice = cur_device;
	//ReleaseMutex(DeviceHandle);
	//windowHwnd = hh;
#if 0
	videoItem->windowHandle = hWnd;
	videoItem->device = curDevice;
	if(this->presentEvent)
		videoItem->presentEvent = presentEvent;
	else{
		videoItem->presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		presentEvent = videoItem->presentEvent;
	}
	cg::core::infoRecorder->logError("[ClientInit]: item:%p, create present event:%p.\n",videoItem, videoItem->presentEvent);
	videoItem->winWidth = d3dpp.BackBufferWidth;
	videoItem->winHeight = d3dpp.BackBufferHeight;

	VideoContext * vctx = VideoContext::GetContext();
	//figure out the id
	cg::core::infoRecorder->logError("[ClientInit]: create video context for task: %p, item:%p.\n", taskId, videoItem);
	vctx->addMap(taskId, videoItem);
#endif

	switch(hr){
	case D3D_OK:
		cg::core::infoRecorder->logError("client_init(), create devie return D3D_OK. id:%d, device:%p.\n", id, base_device);
		break;
	case D3DERR_DEVICELOST:
		cg::core::infoRecorder->logError("client_init(), create device return D3DERR_DEVICELOST.\n");
		break;
	case D3DERR_INVALIDCALL:
		cg::core::infoRecorder->logError("client_init(), create devie return D3DERR_INVALIDCALL\n");
		break;
	case D3DERR_NOTAVAILABLE:
		cg::core::infoRecorder->logError("client_init(), create device return D3DERR_NOTAVAILABLE.\n");
		break;
	case D3DERR_OUTOFVIDEOMEMORY:
		cg::core::infoRecorder->logError("client_init(), create device return D3DERR_OUTOFVIDEOMEMORY.\n");
		break;
	default:
		break;

	}
	if(base_device == NULL) {
		cg::core::infoRecorder->logError("client_init(), device is NULL, id=%d\n", id);
	}
	else {
		cg::core::infoRecorder->logTrace("client_init(), device is good, id=%d\n", id);
	}

	cg::core::infoRecorder->logTrace("server_init(), CreateDevice end\n");

	ShowWindow(hWnd, SW_SHOWNORMAL);
	UpdateWindow(hWnd);
	//atexit(&RenderChannel::destroyD3DWindow);
	enableRTSP = false;
	enableWriteToFile = true;
#ifndef NO_VIDEO_GEN

#if 0

	if(enableRTSPService() || enableGenerateVideo()){
		if(!generator){
			cg::core::infoRecorder->logError("[client_init(), to create generator.\n");
#ifdef USE_TEST
			//DebugBreak();
			generator = new cg::VideoGen(hWnd, (void *)curDevice, DX9, true, enableWriteToFile, enableRTSP);
#else
			generator = new cg::VideoGen(hWnd, (void *)curDevice, DX9);
#endif // USE_TEST

			// add the generator to map
			cg::RTSPConf * rtspc = cg::RTSPConf::GetRTSPConf();
			cg::VideoGen::addMap(taskId, generator);
		}
	}
#else
	

#endif
#endif // NO_VIDEO_GEN

	return hr;
}