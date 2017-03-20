#include <process.h>
#include "CommandServerSet.h"
#include "../LibCore/CThread.h"
#include "LogicContext.h"
#include "../LibDistrubutor/DistributorForLogic.h"
#include "GameClient.h"
#include "../LibCore/CmdHelper.h"
#include "../VideoGen/generator.h"
#include "../LibInput/Controller.h"
#include "KeyboardHook.h"

#ifndef EVENT_NETWORK
//#undef EVENT_NETWORK
#define EVENT_NETWORK
#endif
using namespace std;
using namespace cg;
using namespace cg::core;
//#define ENABLE_CLIENT_CONTROL


#pragma comment(lib, "nvapi.lib")
#pragma comment(lib, "d3d9.lib")

#ifndef MULTI_CLIENTS
CommandServer cs(Max_Buf_Size);
#else // MULTI_CLIENTS
CommandServerSet * csSet = NULL;
cg::core::PTimer * pTimer = NULL;
bool sceneBegin = false;
#endif // MULTI_CLIENTS


/// temp variables for GameServer
static int StartHookCalled = 0;
int g_frame_index = 0;
char tempbuffer[7500000] = { '1' };

 /* hook D3D create */
IDirect3D9* (WINAPI* Direct3DCreate9Next)(
	UINT SDKVersion
	) = Direct3DCreate9;

/* hook DirectInput */
HRESULT (WINAPI * DirectInput8CreateNext)(
	HINSTANCE hinst,
	DWORD dwVersion,
	REFIID riidltf,
	LPVOID *ppvOut,
	LPUNKNOWN punkOuter
	) = DirectInput8Create;

/* hook create window */
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

#ifdef ENABLE_BACKGROUND_RUNNING
// hook the register class function for windows
ATOM (WINAPI *RegisterClassANext)(_In_ const WNDCLASSA * lpwc) = RegisterClassA;
ATOM (WINAPI *RegisterClassWNext)(_In_ const WNDCLASSW * lpwc) = RegisterClassW;

ATOM (WINAPI *RegisterClassExANext)(_In_ const WNDCLASSEXA *lpwcx) = RegisterClassExA;
ATOM (WINAPI *RegisterClassExWNext)(_In_ const WNDCLASSEXW *lpwcx) = RegisterClassExW;


#endif  // ENABLE_BACKGROUND_RUNNING

/* hook ExitProcess, release resources when exiting. */
void (WINAPI* ExitProcessNext)(
	UINT uExitCode
	) = ExitProcess;

BOOL (WINAPI  *ShowWindowNext)(
	__in HWND hWnd,
	__in int nCmdShow
	) = ShowWindow;

void StartHook(bool enableBackRunning) {
	DetourTransactionBegin();
	DetourUpdateThread(GetCurrentThread());

	DetourAttach((PVOID*)&CreateWindowNext, CreateWindowCallback);
	DetourAttach((PVOID*)&CreateWindowExWNext, CreateWindowExWCallback);
	DetourAttach((PVOID*)&Direct3DCreate9Next, Direct3DCreate9Callback);
	//DetourAttach((PVOID*)&ShowWindowNext, ShowWindowCallback);

#ifdef ENABLE_BACKGROUND_RUNNING

	// if the game is CastleStorm, no need to replace wnd proc
	if(enableBackRunning){
		DetourAttach((LPVOID *)&RegisterClassANext, RegisterClassACallback);
		DetourAttach((LPVOID *)&RegisterClassWNext, RegisterClassWCallback);
		DetourAttach((LPVOID *)&RegisterClassExANext, RegisterClassExACallback);
		DetourAttach((LPVOID *)&RegisterClassExWNext, RegisterClassExWCallback);
	}
#endif // ENABLE_BACKGROUND_RUNNING

	DetourAttach(&(PVOID&)ExitProcessNext, ExitProcessCallback);
	DetourTransactionCommit();
}
DWORD WINAPI RenderConnectionLitener(LPVOID param){
	infoRecorder->logTrace("[RenderConnectionLitener]: to listen render connection.\n");
#if 0
	int renderPort = (int)param;
	SOCKET listenSock = socket(AF_INET, SOCK_STREAM, 0);
	SOCKADDR_IN addr;
	memset(&addr, 0, sizeof(addr));
	addr.sin_addr.S_un.S_addr = htonl(INADDR_ANY);
	addr.sin_family = AF_INET;
	addr.sin_port = htons(renderPort);

	bind(listenSock, (SOCKADDR*)&addr, sizeof(SOCKADDR));
	listen(listenSock, 10);

	int addrLen = sizeof(SOCKADDR);
	SOCKADDR_IN addrClient;
	while(1){
		SOCKET sockConn = accept(listenSock, (SOCKADDR *)&addrClient, &addrLen);
		// get a render connection
		// get the info and add to server
		if(!csSet){
			csSet = CommandServerSet::GetServerSet();
		}

		// running the sending window
		do{
			int sndwnd = 8 * 1024 * 1024; // 8MB
			if(setsockopt(sockConn, SOL_SOCKET, SO_SNDBUF, (const char *)&sndwnd, sizeof(sndwnd))){
				infoRecorder->logError("**** set TCP sending buffer failed, ERROR code:%d.\n", GetLastError());
			}

			const char chOpt = 1;
			if(setsockopt(sockConn, IPPROTO_TCP, TCP_NODELAY, &chOpt, sizeof(char))){
				infoRecorder->logError("**** set TCP send no delay failed, ERROR code: %d.\n", GetLastError());
			}

		}while(true);

		csSet->addServer(sockConn);
	}
#else
	ListenServer * server = new ListenServer();
	event_base * base = event_base_new();
	server->setEventBase(base);
	server->setCSSet(csSet);

	server->startListen(7000); /// 7000 port is specialized for render proxy connection
	server->dispatch();
	delete server;
	server = NULL;

#endif
	return 0;
}

DWORD WINAPI GameClientEventProc(LPVOID param){

#if 0
	infoRecorder->logTrace("[GameClientProc]: enter the event dealing thread for game client.\n");
	// connect to loader
	CmdController * ctrl = (CmdController *)param;
	GameClient * gameClient = new GameClient();
	char * name = (char *)(ctrl->getObjName().c_str() + 1);
	gameClient->setName(name);

	IDENTIFIER taskId  = (IDENTIFIER)atoi(cmdCtrl->getIdentifier().c_str());
	gameClient->setTaskID(taskId);
#else
	GameClient * gameClient = (GameClient *)param;
#endif
	event_base * base = event_base_new();
	gameClient->setEventBase(base);
	gameClient->connectToLogicServer();   // connect to logic server
	
#if 0
	gameClient->getCtx()->writeCmd(GAME_READY);
	gameClient->getCtx()->writeIdentifier(taskId);
	gameClient->getCtx()->writeToNet();
#else
	//gameClient->setTaskID(taskId);
	DWORD ret = WaitForSingleObject(gameClient->getClientEvent(), INFINITE);
	switch(ret){
	case WAIT_OBJECT_0:
		infoRecorder->logTrace("[GameClientProc]: client event triggered.\n");
		if(!gameClient->notifyGameReady()){
			infoRecorder->logError("[GameClientProc]: notify GAME READY failed.\n");
			return -1;
		}
		break;
	case WAIT_TIMEOUT:
		infoRecorder->logError("[GameClientProc]: client event time out.\n");
		return -1;
		break;
	case WAIT_FAILED:
		//GameClient::Release();
		return -1;
		break;

	}
#endif

	infoRecorder->logTrace("[DllMain]: enter the game process, task id:%p\n", gameClient->getTaskID());
	gameClient->dispatch();   // dispatch the event

	infoRecorder->logTrace("[GameClientProc]: before exit the event thread, to free GameClient.\n");
	return 0;
}



// get the socket from command, and connect to the logic manager port( we can use the port 8759)
BOOL APIENTRY DllMain( HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved ) {
	WSADATA wsaData;
	WORD sockVersion = MAKEWORD(2,2);
	static int first = 0;
	static bool b = true;
	char shareData[1000] = {0};

	if(infoRecorder == NULL){
		infoRecorder = new InfoRecorder("GameServer");
		infoRecorder->init();
	}
	if(NULL == pTimer){
		pTimer = new cg::core::PTimer();
	}
	DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();
	cg::VideoGen::Initialize();

	if(WSAStartup(sockVersion, &wsaData) != 0){
		infoRecorder->logError("[LogicServer]: WSAStartup failed.\n");
		if(infoRecorder){
			delete infoRecorder;
			infoRecorder = NULL;
		}
		return 0;
	}

#ifdef MULTI_CLIENTS
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		{ 
			char * cmdLine = GetCommandLine();
			infoRecorder->logTrace("[Global]: ");
			infoRecorder->logTrace(cmdLine);
			infoRecorder->logTrace("\n");

			//enableRender = true;
#if 1  // no clients
			// get the task ID, the second argv, add a new parameter to command line, to identify the start mode for game
			string exeName;
			bool enableBackRunning = true;
			infoRecorder->logTrace("[DllMain]: cmd line :%s.\n", cmdLine);

			if(cmdCtrl == NULL){
				infoRecorder->logTrace("[DllMain]: to create the cmd controller with:%s.\n", cmdLine);
				cmdCtrl = CmdController::GetCmdCtroller(cmdLine);
				cmdCtrl->parseCmd();

				infoRecorder->logTrace("[DllMain]: cmd ctrl: %s.\n", cmdCtrl->toString().c_str());
				// open the mapping and the mutex
				exeName = cmdCtrl->getExeName();
				infoRecorder->initMapping(exeName);
				// set the logger
			}
			if(keyCmdHelper == NULL){
				// install keyboard hook
				keyCmdHelper = cg::core::KeyCommandHelper::GetKeyCmdHelper();
				keyCmdHelper->setPrefix((char*)cmdCtrl->getObjName().c_str());
				keyCmdHelper->setSendStep(cmdCtrl->getSendStep());
				keyCmdHelper->installKeyHook(GetCurrentThreadId());
			}
			if (StartHookCalled == 0){
				infoRecorder->logTrace("[Global]: start to hook.\n");
				if(exeName == string("CastleStorm.exe") || exeName == string("castlestorm.exe") || exeName == string("castlestorm") || exeName == string("CastleStorm") || exeName == string("TransGame.exe") || exeName == string("transgame.exe") || exeName == string("ShadowRun.exe") || exeName ==  string("shadowrun.exe")
					){
					enableBackRunning = false;
				}
				enableBackRunning = false;
				StartHook(enableBackRunning);
				StartHookCalled = 1;
			}
			// get the command server set
			// check server set	
			if(!cmdCtrl->is2DGame()){
				if(!csSet){
					csSet = CommandServerSet::GetServerSet();
				}
			}else{
				// for 2D games
			}
			// disable rendering
			// init the rtsp config here
			cg::RTSPConf * config = cg::RTSPConf::GetRTSPConf("config/server.logic.conf");


			//cmdCtrl->setFrameStep(0);
			infoRecorder->logTrace("[DllMain]: render step:%d.\n", cmdCtrl->getFrameStep());

			if(cmdCtrl->isListenMode()){
				infoRecorder->logTrace("[DllMain]: listen mode: :%d.\n", cmdCtrl->getMode());
				IDENTIFIER taskId = NULL;
				// create the event dealing thread
				DWORD clientThreadId = 0;
				HANDLE clientThreadHandle = NULL;
				
				// get the listen mode
				if(cmdCtrl->getMode() == 1){
					// connect to logic
					if(cmdCtrl->isUrlReady()){
						// connect to logic server use the logic url
						string url = cmdCtrl->getLogicUrl();
					}
					else{
						// connect to local host
						taskId = (IDENTIFIER)atoi(cmdCtrl->getIdentifier().c_str());
						infoRecorder->logTrace("[DllMain]: task id:%p", taskId);
						char * name = (char *)(cmdCtrl->getObjName().c_str() + 1);
						GameClient * gClient=  GameClient::GetGameClient();
						gClient->setName(name);
						gClient->setTaskID(taskId);

						clientThreadHandle = chBEGINTHREADEX(NULL, 0, GameClientEventProc, gClient, FALSE, &clientThreadId);
						cmdCtrl->setTaskId((HANDLE)taskId);
					}
				}
				else if(cmdCtrl->getMode() == 2){
					// listen to render proxy, to listen 70000 port
					infoRecorder->logTrace("[DllMain]: to create server for render proxy.\n");
					clientThreadHandle = chBEGINTHREADEX(NULL, 0, RenderConnectionLitener, cmdCtrl, FALSE, &clientThreadId);
				}
				else if(cmdCtrl->getMode() == 3){
					// test mode, render every frame and render proxy reqeust logic server directly

				}
				else{
					infoRecorder->logError("[DllMain]: invalid listen mode:%d.\n", cmdCtrl->getMode());
				}
				// how to set cooperate work mode, render proxy and logic server both do the rendering

			}
			else{
				infoRecorder->logTrace("[DllMain]: stand alone mode.\n");
				if(cmdCtrl->hasRenderConnection()){
					SOCKET old = cmdCtrl->getRenderSocket();
					DWORD ppid = cmdCtrl->getPPid();
					SOCKET sock = GetProcessSocket(old, ppid);
					csSet->addServer(sock);

				}
				// the stand alone mode, setup 
				if(cmdCtrl->getFrameStep() > 0){
					if(cmdCtrl->getFrameStep() > 1){
						// add extra context to CommandServerSet
						int CountToAdd = cmdCtrl->getFrameStep() - 1;
					}
					if(cmdCtrl->enableGenerateVideo()){
						// write video to filed

					}
					else if(cmdCtrl->enableRTSPService()){
						// create rtsp service
					}
					else{
						// configure wroing, must generate video or enabel rtsp, here, default to generate video
					}
				}
				else{
					// no video to generate
					infoRecorder->logError("[DllMain]: no image to render, so save video or rtsp service is invalid.\n");
				}
			}
  // EVENT_NETWORK

#else  // no clients
			
#endif  // no clients

			first = 0;
			infoRecorder->logTrace("[DllMain]: finish dll main.\n");
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
#else   // MULTI_CLIENTS
	
#endif   // MULTI_CLIENTS

	return TRUE;
}

void IdentifierBase::print(){
	infoRecorder->logError("[IdentifierBase]: %s %d is table:%s, is sync: %s, created: 0x%x, updated: 0x%x, frame check flag:0x%x.\n", typeid(*this).name(), id, stable ? "true" : "false", sync ? "true" : "false", creationFlag, updateFlag, frameCheckFlag);

}