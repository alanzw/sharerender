#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <stdio.h>
#include "detours/detours.h"
#include <string>

using namespace std;

#pragma comment(lib, "detours/detours.lib")

int main(int argc, char** argv) {

	PROCESS_INFORMATION pi = {0};
	STARTUPINFO si = {0};
	si.cb = sizeof(si);

	HANDLE hnd = GetCurrentProcess();

	printf("Listening\nhandle %d\n", hnd);

	LPSECURITY_ATTRIBUTES lp_attributes;
	LPSECURITY_ATTRIBUTES lpThreadAttributes;
	STARTUPINFO startupInfo = {sizeof(startupInfo)};
	memset(&startupInfo,0,sizeof(STARTUPINFO));
	startupInfo.cb = sizeof(STARTUPINFO);
	startupInfo.dwFlags=0;
	startupInfo.wShowWindow = SW_HIDE;

	PROCESS_INFORMATION processInformation;
	char cmdLine[100];
	string AppName;
	int recv_len = 0;
	char RecvB[100];

	// use this to test, all connections establish is logic server
	if(argc == 1 || argc == 2 || argc == 3) {
		DWORD id = GetCurrentProcessId();

		int dump_mesh = 0;
		if(argc == 3) dump_mesh = 1;

		sprintf(cmdLine,"%s %d %d %d %d",argv[1], dump_mesh, -1,-1, id);
		printf("cmd line is %s\n", cmdLine);

		bool ret = DetourCreateProcessWithDll(NULL,cmdLine, NULL, NULL, TRUE, CREATE_DEFAULT_ERROR_MODE,
			NULL, NULL, &si, &pi, "LogicServer.dll", NULL);

		if(!ret) {
			char err_str[200];
			sprintf(err_str, "Game Start %s Failed", AppName.c_str());
			MessageBox(NULL, err_str, "Error", MB_OK);
		}

		return 0;
	}

	

	return 0;
}


