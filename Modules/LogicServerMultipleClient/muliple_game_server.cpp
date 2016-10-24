#include <process.h>
//#include "common_net.h"
#include "muliple_game_server.h"
//#include "InputServer.h"
#include "ccg_win32.h"
#include "ccg_config.h"
#include "rtspconf.h"
#include "ctrlconfig.h"
#include "controller.h"
#include "ctrl-sdl.h"
#include "cthread.h"

int need_dump_mesh = 0;
CommandServer cs(Max_Buf_Size);

bool enableRender = true;
bool F9Pressed = false;
bool synSign = false;
bool f10pressed = false;

CRITICAL_SECTION f9;
//CommonNet dis(1);

IDirect3D9* (WINAPI* Direct3DCreate9Next)(UINT SDKVersion) = Direct3DCreate9;
HRESULT (WINAPI * DirectInput8CreateNext)(HINSTANCE hinst,DWORD dwVersion,REFIID riidltf,LPVOID *ppvOut,LPUNKNOWN punkOuter) = DirectInput8Create;
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

//½ØÈ¡ExitProcess
void (WINAPI* ExitProcessNext)(UINT uExitCode) = ExitProcess;

void StartHook() {
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());

	DetourAttach((PVOID*)&CreateWindowNext, CreateWindowCallback);
	DetourAttach((PVOID*)&CreateWindowExWNext, CreateWindowExWCallback);
	DetourAttach((PVOID*)&Direct3DCreate9Next, Direct3DCreate9Callback);

	DetourAttach(&(PVOID&)ExitProcessNext, ExitProcessCallback);

	DetourTransactionCommit();
}
static int StartHookCalled = 0;
int g_frame_index = 0;

const char * LogicFileMap = "LogicSharedData";

int readSharedData(char * dst, int size){
	// see if a memory-mapped file named LogicSharedData already exists
	HANDLE fileMapT = OpenFileMapping(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, LogicFileMap);
	if (fileMapT != NULL){
		// the MMF does exist, map it into the process's address space
		PVOID pView = MapViewOfFile(fileMapT, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);
		if (pView != NULL){
			// read the shared data to dst 
			memcpy(dst, pView, size);
			UnmapViewOfFile(pView);
		}
		CloseHandle(fileMapT);
		return size;
	}
	else{
		Log::log("--------- cannot open mapping ---------\n");
	}

	return 0;
}

char tempbuffer[7500000] = { '1' };

BOOL APIENTRY DllMain( HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved ) {

	static int first = 0;
	char shareData[1000];

	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		{
			Log::init("game_server.log");
			Log::log(GetCommandLine());
			
			cs.start_up();
#if 1
			//GetSocketsFromCmd();
			// get the socket from shared memory file
			int toRead = 1000;
			int ret = readSharedData(shareData, toRead);
			int renders = *(int *)shareData;
			
			SOCKET sock = *(SOCKET *)(shareData + sizeof(int));
			SOCKET newSock = GetProcessSocket(sock, GetParentProcessid());

			cs.set_connect_socket(newSock);

			// frame index is set
#else
			GetSocketsFromSharedFileMap();
#endif
			Log::log("Dllmain(), connect_socket=%d, input socket = %d\n", cs.get_connect_socket(), -1);
#if 0
			cs.send_raw_buffer("YAHOO,start the graphic stream.");
#else
			// set thes socket to no block io
			u_long ul = 1;
			ioctlsocket(cs.get_connect_socket(), FIONBIO, (u_long *)&ul);
			char temp[50];
			fd_set fdSocket;
			FD_ZERO(&fdSocket);
			FD_SET(cs.get_connect_socket(), &fdSocket);
			int nRet = select(0, &fdSocket, NULL, NULL, NULL);
			if (nRet > 0){
				int r = recv(cs.get_connect_socket(), temp, 50, 0);
				temp[r] = 0;
				//cs.recv_raw_buffer(temp, 50);
				Log::log("[game_server]: get %s from render proxy.\n", temp);
				char * message = "YAHOO, start the graphic stream.";
				r = send(cs.get_connect_socket(), message, strlen(message), 0);
				//r = send(cs.get_connect_socket(), tempbuffer, 7500000 - 1, 0);
				if (r < strlen(message)){
					Log::log("[game_server]: send %d chars, but total:%d. error code:%d.\n", r, strlen(message), WSAGetLastError());
				}
				else{
					Log::log("[game_server]: succeeded sending %d chars, total: %d.\n", r, strlen(message));
				}
				Log::log("[game_server]: error code: %d.\n", WSAGetLastError());
			}
			
#endif
			enableRender = true;
			if (StartHookCalled == 0){
				StartHook();
				StartHookCalled = 1;
			}
			
			//SetKeyboardHook(NULL, GetCurrentThreadId());

			// create the input server thread
			InitializeCriticalSection(&f9);
			DWORD dwThreadId;

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
				enableRender = conf->enableRender;
			} while (0);
#if 0
			if (!ctrlServer->init(conf, NULL)){
				Log::slog("[SERVER]: cannot start the input thread.\n");
			}
#endif
#ifdef USE_CLIENT_INPUT
			HANDLE hThreadInput = chBEGINTHREADEX(NULL, 0, ctrl_server_thread, &dis, 0, &dwThreadId);
#endif
			first = 0;

			break;
		}
	case DLL_THREAD_ATTACH: break;
	case DLL_THREAD_DETACH: break;
	case DLL_PROCESS_DETACH:
		{
			//don't do anything here
			break;
		}
		WM_ACTIVATE;
	}
	return TRUE;
}
