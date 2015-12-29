// this is for render loader, create the render proxy with given dll
#include <iostream>
#include <Windows.h>
//#include "../LibCore/InfoRecorder.h"
#include "detours/detours.h"

using namespace std;

//#pragma comment(lib, "detours/detours.lib")
#pragma comment(lib, "detours.lib")
#pragma comment(lib, "nvapi.lib")

HANDLE StartProcessWithDll(char * exeName, char * dllName, char * exePath, char * arg){
	PROCESS_INFORMATION pi = {0};
	STARTUPINFO si = {0};
	si.cb = sizeof(si);

	HANDLE hnd = GetCurrentProcess();
	LPSECURITY_ATTRIBUTES lp_attributes;
	LPSECURITY_ATTRIBUTES lpThreadAttributes;
	STARTUPINFO startupInfo = { sizeof(startupInfo) };
	memset(&startupInfo, 0, sizeof(STARTUPINFO));
	startupInfo.cb = sizeof(STARTUPINFO);
	startupInfo.dwFlags = 0;
	startupInfo.wShowWindow = SW_HIDE;

	char cmdLine[MAX_PATH];

	DWORD id = GetCurrentProcessId();

	if(arg)
		sprintf(cmdLine, "%s\\%s %s",exePath, exeName, arg);
	else
		sprintf(cmdLine, "%s\\%s",exePath, exeName);

	printf("cmd line is %s, dll name:%s, path:%s\n", cmdLine, dllName, exePath);
	BOOL ret = DetourCreateProcessWithDll(NULL, cmdLine, NULL, NULL, TRUE, CREATE_DEFAULT_ERROR_MODE,
		NULL, exePath, &si, &pi, dllName, NULL);

	if (!ret) {
		char err_str[200];
		sprintf(err_str, "Game Start %s Failed", exeName);
		MessageBox(NULL, err_str, "Error", MB_OK);
	}

	return pi.hProcess;
}


// the main entry for render loader
int main(int argc, char ** argv){
	// setup the default 
	char args[1024] = {0};

	if(argc > 2){
		// argc > 2, Loader [render] [args]
		// parse the cmd line
		for(int i = 1; i < argc; i++){
			sprintf(args, "%s ", argv[i]);
		}
	}
	else{
		// parse default args
		sprintf(args, "%s", "-s D:\\GameTest\\AllResults -f 30 -o Render.video.264 -e 3");
	}
	char curDirectory[MAX_PATH] = {0};
	GetCurrentDirectory(MAX_PATH, curDirectory);
	// craete the render poxy process
	HANDLE hp = StartProcessWithDll("RenderRroxy.exe", "GameVideoGenerator.dll", curDirectory, args);
	WaitForSingleObject(hp, INFINITE);
	return 0;
}
