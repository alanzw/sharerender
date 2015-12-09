#include "Utility.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

static void RaiseToDebugP(){
	HANDLE hToken;
	HANDLE hProcess = GetCurrentProcess();
	if ( OpenProcessToken(hProcess, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken) )
	{
		TOKEN_PRIVILEGES tkp;
		if ( LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &tkp.Privileges[0].Luid) )
		{
			tkp.PrivilegeCount = 1;
			tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

			BOOL bREt = AdjustTokenPrivileges(hToken, FALSE, &tkp, 0, NULL, 0) ;
		}
		CloseHandle(hToken);
	}
}


SOCKET DuplicateSocketFormProcess(DWORD processId, SOCKET old){
	cg::core::infoRecorder->logTrace("[ToolFunc]: DuplicateSocketFromProcess called.\n");
	SOCKET ret = NULL;
	if(old == -1){
		cg::core::infoRecorder->logError("[ToolFunc]: duplicate socket, invalid old socket handle.\n");
		return NULL;
	}else{
		RaiseToDebugP();
		HANDLE sourceHandle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, processId);
		if(sourceHandle == NULL){
			// open process failed.
		}
		else{
			DuplicateHandle(sourceHandle, (HANDLE)old, GetCurrentProcess(), (LPHANDLE)&ret, 0, FALSE, DUPLICATE_SAME_ACCESS);
			CloseHandle(sourceHandle);
		}

	}
	return ret;
}