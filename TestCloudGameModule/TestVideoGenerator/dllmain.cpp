// the entry of the dll for generate game video
#include <Windows.h>
#include "log.h"
#include "detours\detours.h"
#include "hook-function.h"
#include <Windows.h>
#include "generator.h"

#pragma comment(lib, "detours//detours.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "nvcuvenc.lib")
#pragma comment(lib, "nvcuvid.lib")


bool startHookCalled = false;


// dll main
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved){
	printf("enter the dll.\n");
	Log::init("testlog.log");
	if(infoRecorder == NULL){
		infoRecorder = new InfoRecorder("test");
	}
	switch(ul_reason_for_call){
	case DLL_PROCESS_ATTACH:
		infoRecorder->logError(GetCommandLine());

		if(startHookCalled == false){
			do_hook();
			startHookCalled = true;
		}


		break;
	case DLL_THREAD_ATTACH: break;
	case DLL_THREAD_DETACH: break;
	case DLL_PROCESS_DETACH:
		WM_ACTIVATE;
		break;
	}
	printf("init the dll finished.\n");
	return TRUE;
}