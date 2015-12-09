#include "hook-function.h"
#include <Windows.h>
#include "generator.h"

VideoGen * generator = NULL;

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


void WINAPI ExitProcessCallback(UINT uExitCode){
	Log::slog("[ExitProcess]: called.\n");
	infoRecorder->logError("[ExitProcess] called.\n");
	infoRecorder->flush();
	return ExitProcessNext(uExitCode);
}


// hook the 2D games, just need to hook the create window and make sure that the game did not use d3d

HWND WINAPI CreateWindowCallback(DWORD dwExStyle,LPCSTR lpClassName,LPCSTR lpWindowName, DWORD dwStyle,int x,int y, int nWidth,int nHeight,HWND hWndParent, HMENU hMenu,HINSTANCE hInstance,LPVOID lpParam) {
	infoRecorder->logError("CreateWindowCallback() called, width:%d, height:%d\n", nWidth, nHeight);


	HWND ret = NULL;
	ret =  CreateWindowNext(dwExStyle,lpClassName,lpWindowName,dwStyle,x,y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);

	// if the width and the height is valid, then, create the thread for video stream
	
	if(nWidth > 10 && nWidth < 10000 && nHeight > 10 && nHeight < 10000){
		// valid rect for the game window
		// source, filter, encoder
		infoRecorder->logError("[CreateWindow]: ex. height:%d, width:%d\n", nHeight, nWidth);
#if 1
		if(generator == NULL){
			infoRecorder->logError("[CreateWindow]: create new VideoGenerator.\n");
			infoRecorder->logError("[VideoGen]: create VideoGen with height:%d, width:%d.\n", nHeight, nWidth);
			generator = new VideoGen(ret, true);

		}else{
			infoRecorder->logError("[CreateWindow]: multiple valid window?");
		}
#endif
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
#if 1
			// valid rect fot the game window
			// source, filter, encoder
			if(generator == NULL){
				infoRecorder->logError("[CreateWindow]: create new VideoGen.\n");
				infoRecorder->logError("[VideoGen]: create VideoGen with height:%d, width:%d.\n", nHeight, nWidth);
				generator = new VideoGen(ret,true);

			}else{
				infoRecorder->logError("[CreateWindow]: multiple valid window?");
			}
			//generator->setWindowHwnd(ret);
			//generator->triggerWindowEvent();
#endif
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

		return 0;
	}
	else{
		return -1;
	}
}