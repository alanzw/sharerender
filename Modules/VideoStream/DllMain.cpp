// This is the DLL main entry file

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

// the video stream use the event socket

// dll main entry for pure video stream
DWORD WINAPI DllMain(_In_ HANDLE _HDllHandle, _In_ DWORD _Reason, _In_opt_ LPVOID _Reserved){
	// get socket from cmd line
	char * cmdLine = GetCommandLine();


}