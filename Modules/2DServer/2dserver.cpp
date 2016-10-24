#include <process.h>
#include "2DServer.h"

#include "ccg_win32.h"
#include "ccg_config.h"
#include "rtspconf.h"
#include "ctrlconfig.h"
#include "controller.h"

#include "..\Utility\log.h"
#include "ctrl-sdl.h"

#include "detours\detours.h"

#include "hook-function.h"
//#include "inforecoder.h"

#pragma comment(lib, "detours\detours.lib")


///this file includes the dll main

static int StartHookCalled = 0;

InfoRecorder * infoRecorder = NULL;

HWND (WINAPI *CreateWindowNext)( 
	DWORD dwExStyle,
	LPCSTR lpClassName,
	LPCSTR lpWindowName,
	DWORD dwStyle,
	int X,
	int Y,
	int nWidth,
	int nHeight,
	HWND hWndParent,
	HMENU hMenu,
	HINSTANCE hInstance,
	LPVOID lpParam) = CreateWindowExA;

HWND (WINAPI *CreateWindowExWNext)(
	DWORD dwExStyle,
	LPCWSTR lpClassName,
	LPCWSTR lpWindowName,
	DWORD dwStyle,
	int X,
	int Y,
	int nWidth,
	int nHeight,
	HWND hWndParent,
	HMENU hMenu,
	HINSTANCE hInstance,
	LPVOID lpParam) = CreateWindowExW;

void (WINAPI* ExitProcessNext)(UINT uExitCode) = ExitProcess;

// hook the 2D games, just need to hook the create window and make sure that the game did not use d3d

HWND WINAPI CreateWindowCallback(DWORD dwExStyle,LPCSTR lpClassName,LPCSTR lpWindowName, DWORD dwStyle,int x,int y,int nWidth,int nHeight,HWND hWndParent, HMENU hMenu,HINSTANCE hInstance,LPVOID lpParam) {
	infoRecorder->logError("CreateWindowCallback() called, width:%d, height:%d\n", nWidth, nHeight);
	HWND ret = NULL;
	ret =  CreateWindowNext(dwExStyle,lpClassName,lpWindowName,dwStyle,x,y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);

	// if the width and the height is valid, then, create the thread for video stream
	if(nWidth > 10 && nWidth < 10000 && nHeight > 10 && nHeight < 10000){
		// valid rect for the game window
		// source, filter, encoder
		if(generator == NULL){
			generator = new VideoGenerator();

		}else{
			infoRecorder->logError("[CreateWindow]: multiple valid window?");
		}
	}

	return ret;
}

HWND WINAPI CreateWindowExWCallback( DWORD dwExStyle, LPCWSTR lpClassName, LPCWSTR lpWindowName, DWORD dwStyle,
 int X, int Y, int nWidth, int nHeight, HWND hWndParent, HMENU hMenu, HINSTANCE hInstance, LPVOID lpParam) {
	infoRecorder->logError("CreateWindowExWCallback() called, Widht:%d, Height:%d\n", nWidth, nHeight);
	HWND ret = NULL;
	ret = CreateWindowExWNext(dwExStyle,lpClassName,lpWindowName,dwStyle,X,Y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);
	// if the width and the height is valid, then, create the thread for video stream
	if(nWidth > 10 && nWidth < 10000 && nHeight > 10 && nHeight < 10000){
		// valid rect fot the game window
		// source, filter, encoder
		if(generator == NULL){
			generator = new VideoGenerator();

		}else{
			infoRecorder->logError("[CreateWindow]: multiple valid window?");
		}
		generator->setWindowHwnd(ret);
		generator->triggerWindowEvent();
	}

	return ret;
}

//  hook the 3D games

static int hook_d9(){
	HMODULE hMode;
	if ((hMode = GetModuleHandle("d3d9.dll")) == NULL){
		infoRecorder->logError("[hook d9]: d3d9.dll not load yet.\n");
		if ((hMode = LoadLibrary("d3d9.dll")) == NULL){
			infoRecorder->logError("[hook d9]: load d3d9.dll failed.\n");
			return -1;
		}
	}
	pD3d = (TDirect3DCreate9)GetProcAddress(hMode, "Direct3DCreate9");
	if (pD3d == NULL){
		infoRecorder->logError("[hook d9]: GetProcAddress(Direct3DCreate9) failed.\n");
		return -1;
	}
	//
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());
	DetourAttach(&(LPVOID&)pD3d, hook_d3d);
	DetourTransactionCommit();

	return 0;

}
static int hook_dxgi(){
	HMODULE hMod;
	if ((hMod = GetModuleHandle("dxgi.dll")) == NULL){
		infoRecorder->logError("[hook dxgi]: dxgi.dll not load yet.\n");
		return -1;
	}
	pCreateDXGIFactory = (TCreateDXGIFactory)GetProcAddress(hMod, "CreateDXGIFactory");
	if (pCreateDXGIFactory == NULL){
		infoRecorder->logError("[hook dxgi]: GetProcessAddress(CreateDXGIFactory] failed.\n");
		return -1;
	}
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());
	DetourAttach(&(LPVOID &)pCreateDXGIFactory, hook_CreateDXGIFactory);
	DetourTransactionCommit();
	return 0;
}
static int hook_d10_1(){
	HMODULE hMod;
	if ((hMod = GetModuleHandle("d3d10_1.dll")) == NULL){
		infoRecorder->logError("[hook d10_1]: d3d10_1.dll not load yet.\n");
		if ((hMod = LoadLibrary("d3d10_1.dll")) == NULL){
			infoRecorder->logError("[hook d10_1]: load d3d10_1.dll failed.\n");
			return -1;
		}

	}
	pD3D10CreateDeviceAndSwapChain1 = (TD3D10CreateDeviceAndSwapChain1)GetProcAddress(hMod, "D3D10CreateDeviceAndSwapChain1");
	if (pD3D10CreateDeviceAndSwapChain1 == NULL){
		infoRecorder->logError("GetProcAddress(D3D10CreateDeviceAndSwapChain1) failed.\n");
		return -1;
	}
	// 
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());
	DetourAttach(&(LPVOID &)pD3D10CreateDeviceAndSwapChain1, hook_D3D10CreateDeviceAndSwapChain1);
	DetourTransactionCommit();
	return 0;
}
static int hook_d10(){
	HMODULE hMod;
	if ((hMod = GetModuleHandle("d3d10.dll")) == NULL){
		infoRecorder->logError("[hook d10]: d3d10.dll not load yet.\n");
		if ((hMod = LoadLibrary("d3d10.dll")) == NULL){
			infoRecorder->logError("[hook d10]: load d3d10.dll failed.\n");
			return -1;
		}

	}
	pD3D10CreateDeviceAndSwapChain = (TD3D10CreateDeviceAndSwapChain)GetProcAddress(hMod, "D3D10CreateDeviceAndSwapChain");
	if (pD3D10CreateDeviceAndSwapChain == NULL){
		infoRecorder->logError("GetProcAddress(D3D10CreateDeviceAndSwapChain) failed.\n");
		return -1;
	}
	//
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());
	DetourAttach(&(LPVOID&)pD3D10CreateDeviceAndSwapChain, hook_D3D10CreateDeviceAndSwapChain);
	DetourTransactionCommit();
	return 0;
}
static int hook_d11(){
	HMODULE hMod;
	if ((hMod = GetModuleHandle("d3d11.dll")) == NULL){
		infoRecorder->logError("[hook d11]: d3d11.dll not load yet.\n");
		if ((hMod = LoadLibrary("d3dll.dll")) == NULL){
			infoRecorder->logError("[hook d11]: load d3d11.dll failed.\n");
			return -1;
		}
	}
	pD3D11CreateDeviceAndSwapChain = (TD3D11CreateDeviceAndSwapChain)GetProcAddress(hMod, "D3D11CreateDeviceAndSwapChain");
	if (pD3D11CreateDeviceAndSwapChain == NULL){
		infoRecorder->logError("[hook d11]: GetProcAddress(D3D11CreateDeviceAndSwapChain) failed.\n");
		return -1;
	}
	//
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());
	DetourAttach(&(LPVOID &)pD3D11CreateDeviceAndSwapChain, hook_D3D11CreateDeviceAndSwapChain);
	DetourTransactionCommit();
	return 0;
}

int do_hook(){
#if 0
	if (CoInitializeEx(NULL, COINITBASE_MULTITHREADED) != S_OK){

	}
#endif

	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());
	// hook create window

	DetourAttach((PVOID*)&CreateWindowNext, CreateWindowCallback);
	DetourAttach((PVOID*)&CreateWindowExWNext, CreateWindowExWCallback);

	DetourAttach(&(LPVOID&)ExitProcessNext, ExitProcessCallback);
	DetourTransactionCommit();

	if (!hook_d9() || !hook_d10() || !hook_d10_1() || !hook_d11() || !hook_dxgi()){

	}
	else{
		return -1;
	}
}


VideoGenerator * generator = NULL;

Channel * channel  = NULL;
HANDLE videoThread = NULL;
DWORD videoThreadId = 0;
extern DWORD WINAPI EventVideoServer(LPVOID param);   // the video thread procedure.

// the dll main entry for the 2DServer.dll
BOOL APIENTRY DllMain(HMODULE hModeule, DWORD ul_reason_for_call, LPVOID lpReserved){
	static int first = 0;

	switch(ul_reason_for_call){
	case DLL_PROCESS_ATTACH:
		{
			
			//Log::init("2DServer.log");
			//Log::log(GetCommandLine());
			infoRecorder->logError(GetCommandLine());

			// star the hook precedure
			if(StartHookCalled == 0){
				//StartHook();
				do_hook();
				StartHookCalled = 1;
			}

			CtrlConfig * conf = NULL;
			conf = CtrlConfig::GetCtrlConfig(STREAM_SERVER_CONFIG);
			CtrlMessagerServer * ctrlServer = new CtrlMessagerServer();
			do{
				if (conf->ctrlenable){
					if (ctrlServer->initQueue(32768, sizeof(sdlmsg_t)) < 0){
						conf->ctrlenable = 0;
						break;
					}
					//msgfunc * replayer = &(CtrlReplayer::replayCallback);
					//ctrlServer->setReplay(&(CtrlReplayer::replayCallback));
					ctrlServer->init(conf, CTRL_CURRENT_VERSION);

					CtrlReplayer::setMsgServer(ctrlServer);
#if 0
					if (!ctrlServer->start()){
						Log::slog("Cannot create controller thread, controller disable\n");
						conf->ctrlenable = 0;
						break;
					}
#endif
				}
				//enableRender = conf->enableRender;

				// TODO: start the rtsp thread
				// d3d. GDI? check the source type here, hook funciton will give the information, use the d3d method with high priority

				// use this dll means that use video streaming

#if 0
				// use VideoGenerator
				generator = new VideoGenerator();
				generator->initVideoGenerator();
				generator->startModules();
#endif

#if 0
				// create the channel
				channel = new Channel();
				videoThread = chBEGINTHREADEX(NULL, 0, EventVideoServer, channel, 0, &videoThreadId);
#endif
			} while (0);

			first = 0;
		}
		break;
	case DLL_THREAD_ATTACH: break;
	case DLL_THREAD_DETACH: break;
	case DLL_PROCESS_DETACH:
		{

		}
		WM_ACTIVATE;
	}
	return TRUE;
}

#if 0
// the event thread proc for video streaming.
DWORD WINAPI EventVideoServer(LPVOID param){
	VideoStream * stream = VideoStream::GetStream();
	if(stream == NULL){
		// error
	}

	do{
		usleep(100000);
	}while(stream->getResolutionRetrieved() == 0);

	Log::slog("[EventVideoThreadProc]: run the server.\n");

	// getthe window handle 
	WaitForSingleObject(stream->getWindowEvent(), INFINITE);
	if(stream->cropWindow() < 0){
		// error
	}
	else if(stream->getpRect() == NULL){
		Log::slog("[EventVideoThreadProc]: crop disabled.\n");
	}
	else if(stream->getpRect() != NULL){
		Log::slog("[EventVideoTreahdProc]: corp enabled: (%d, %d)-(%d, %d)\n",0,0,0,0);
	}

	// init the modules


	// run the modules



}


int VideoStream::cropWindow(){
	RECT client;
	POINT lt, rb;
	int dw, dh;

	if(rect == NULL || pRect ==  NULL){
		return -1;
	}

	dw = GetSystemMetrics(SM_CXSCREEN);
	dh = GetSystemMetrics(SM_CYSCREEN);

	if(SetWindowPos(hWnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE | SWP_SHOWWINDOW) == 0){
		Log::slog("[VideoStream]: SetWindowPos failed.\n");
		return -1;
	}
	if(GetClientRect(hWnd, & client)== 0){
		Log::slog("[VideoStream]: GetClientRect failed.\n");
		return -1;
	}
	if(SetForegroundWindow(hWnd) == 0){
		Log::slog("[VideoStream]: SetForegroundWindow failed.\n");
	}

	lt.x = client.left;
	lt.y = client.top;
	rb.x = client.right -1;
	rb.y = client.bottom -1;

	if(ClientToScreen(hWnd, &lt) == 0 || ClientToScreen(hWnd, &rb) == 0){
		Log::slog("[VideoStream]: Map from clienmt coordinate to screen coordinate failed.\n");
		return -1;
	}

	rect->left = lt.x;
	rect->top = lt.y;
	rect->right = rb.x;
	rect->bottom - rb.y;

	// size check: mutliples of 2?
	if((rect->right - rect->left + 1) %2 != 0)
		rect->left--;
	if((rect->bottom - rect->top + 1) %2 != 0)
		rect->top--;

	if(rect->left < 0|| rect->top < 0 || rect->right >= dw || rect->bottom >= dh){
		Log::slog("[VideoStream]: invalid window: (%d, %d)-(%d, %d) w=%d, h=%d (Screen dimension = %dx%d).\n", rect->left, rect->top, rect->right, rect->bottom, rect->right - rect->left +1, rect->bottom - rect->top + 1, dw, dh);
		return -1;
	}
	*pRect = rect;
	return 1;
}

#endif