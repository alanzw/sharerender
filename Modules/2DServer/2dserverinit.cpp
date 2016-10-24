#include "2dserver.h"
#include "hook-function.h"

// this file is for window hooker, to hook create window function to get the window handle.

// each 2d server has a global video generator
VideoGenerator * gVideoGenerator = NULL;

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

HWND WINAPI CreateWindowCallback(DWORD dwExStyle,LPCSTR lpClassName,LPCSTR lpWindowName, DWORD dwStyle,int x,int y,int nWidth,int nHeight,HWND hWndParent, HMENU hMenu,HINSTANCE hInstance,LPVOID lpParam) {
	infoRecorder->logError("CreateWindowCallback() called, width:%d, height:%d\n", nWidth, nHeight);
	HWND ret = NULL;
	ret =  CreateWindowNext(dwExStyle,lpClassName,lpWindowName,dwStyle,x,y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);

	// if the width and the height is valid, then, create the thread for video stream
	if(nWidth > 10 && nWidth < 10000 && nHeight > 10 && nHeight < 10000){
		// valid rect for the game window
		// source, filter, encoder
		if(gVideoGenerator == NULL){
			gVideoGenerator = new VideoGenerator();

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
		if(gVideoGenerator == NULL){
			gVideoGenerator = new VideoGenerator();

		}else{
			infoRecorder->logError("[CreateWindow]: multiple valid window?");
		}
	}

	return ret;
}


DWORD GetParentProcessid(){
	DWORD ret = 0;
	char * cmdLine;
	cmdLine = GetCommandLine();
	int len = strlen(cmdLine);

	if (len < 100) {
		infoRecorder->logError("GetSocketsFromCmd(), cmdLine=%s\n", cmdLine);
		string str = cmdLine;
		istringstream in(str);
		DWORD command_socket_handle, input_socket_handle, loader_process_id, frame_index;
		string appname;

		//string local, port;
		in >> appname >> need_dump_mesh >> command_socket_handle >> input_socket_handle >> loader_process_id >> frame_index;

		ret = loader_process_id;
	}
	else {
		infoRecorder->logError("GetSocketsFromCmd(), cmd len >= 100\n");
	}
	return ret;
}

void GetSocketsFromCmd() {
	infoRecorder->logError("GetSocketsFromCmd() called\n");

	char * cmdLine;
	cmdLine = GetCommandLine();
	int len = strlen(cmdLine);

	if(len < 100) {
		infoRecorder->logError("GetSocketsFromCmd(), cmdLine=%s\n", cmdLine);
		string str = cmdLine;
		istringstream in(str);
		DWORD command_socket_handle, input_socket_handle, loader_process_id, frame_index;
		string appname;

		//string local, port;
		in >> appname >> need_dump_mesh >> command_socket_handle >> input_socket_handle >> loader_process_id >> frame_index;

		g_frame_index = frame_index;
		if(command_socket_handle == -1) {
			cs.set_connect_socket(-1);
			infoRecorder->logError("[server init]: error command socket handle.\n");
			//dis.set_connect_socket(-1);
		}
		else {
			cs.set_connect_socket(GetProcessSocket(command_socket_handle, loader_process_id));
			//dis.set_connect_socket(GetProcessSocket(input_socket_handle, loader_process_id));
			infoRecorder->logError("[server init]: new command socket handle:%d.\n", cs.get_connect_socket());
		}

	}
	else {
		infoRecorder->logError("GetSocketsFromCmd(), cmd len >= 100\n");
	}
}

SOCKET GetProcessSocket(SOCKET oldsocket, DWORD source_pid) {
	RaiseToDebugP();
	HANDLE source_handle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, source_pid);
	HANDLE new_handle = 0;
	if(source_handle == NULL) {
		Log::log("GetProcessSocket(), error happen\n");
	}
	DuplicateHandle(source_handle, (HANDLE)oldsocket, GetCurrentProcess(), &new_handle, 0, FALSE, DUPLICATE_SAME_ACCESS);
	CloseHandle(source_handle);
	Log::log("GetProcessSocket(), pid: %d,old sock: %d, socket: %d\n",source_pid,oldsocket,new_handle);
	return (SOCKET)new_handle;
}

void WINAPI ExitProcessCallback(UINT uExitCode) {
	infoRecorder->logError("exit process called\n");

	

	//do the clean job here

	ExitProcessNext(uExitCode);
}

void RaiseToDebugP()
{
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
