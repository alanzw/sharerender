#include "../libCore/CommonNet.h"
#include <process.h>
#include "../LibVideo/Config.h"
#include <WinSock2.h>
#include "../LibRender/LibRenderAPI.h"
#include "../LibCore/DisNetwork.h"
#include "RenderProxy.h"
#include "../LibCore/InfoRecorder.h"
#include "../LibDistrubutor/Context.h"
#include "DisForRender.h"

//#include "../LibDistrubutor/DistributorForRender.h"

#ifndef _DEBUG
#pragma comment(lib, "event.lib")
#pragma comment(lib, "event_core.lib")
#pragma comment(lib, "event_extra.lib")
#else
#pragma comment(lib, "event.d.lib")
#pragma comment(lib, "event_core.d.lib")
#pragma comment(lib, "event_extra.d.lib")
#endif

//libs for video
#ifndef _DEBUG
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.lib")
#pragma comment(lib, "libgroupsock.lib")
#pragma comment(lib, "libBasicUsageEnvironment.lib")
#pragma comment(lib, "libUsageEnvironment.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#else

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.d.lib")
#pragma comment(lib, "libgroupsock.d.lib")
#pragma comment(lib, "libBasicUsageEnvironment.d.lib")
#pragma comment(lib, "libUsageEnvironment.d.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "nvcuvenc.lib")

#endif



#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"d3dx9.lib")

//#define DIS_CLIENT

#define maxl 10010

char buffer[maxl] = "hello, world";
char b2[maxl];
char b3[maxl];


bool client_render = true;
bool fromserver = false;

HHOOK kehook = NULL;
LRESULT CALLBACK HookProc(int nCode, WPARAM wParam, LPARAM lParam){

	//MessageBox(NULL,"Enter hook", "WARNING", MB_OK);
	if (wParam == VK_F1)
		//if( lParam & 0x80000000) // pressed
	{
		if (lParam & 0x80000000) // f10 pressed
		{
			if (fromserver){
				infoRecorder->logError("Client recv the f10 from server!\n");
				fromserver = false;
				return TRUE;
			}
			else
				return TRUE;
		}
		else
			return TRUE;
	}
	//
	return CallNextHookEx(kehook, nCode, wParam, lParam);
}

void SetKeyboardHook(HINSTANCE hmode, DWORD dwThreadId) {
	// set the keyboard hook
	//MessageBox(NULL,"Enter hook", "WARNING", MB_OK);
	//infoRecorder->logError("set the keyboard hook!\n");
	kehook = SetWindowsHookEx(WH_KEYBOARD, HookProc, hmode, dwThreadId);
}

void usage() {
	puts("\nUsage:");
	puts("\tgame_client game_name [window_style y/n] [client_id(1-8)]");

}
#if 0
int main(int argc, char** argv) {
	init_fptable();

	infoRecorder = new InfoRecorder(argv[1]);
	Log::init("game_client.log");

	if(argc == 1) {
		//printf("start game: %s\n", argv[1]);
		cc.load_port(1);
		//dic = new CommonNet(2, 1);
		game_name = "SprillRichi";
	}
	else if(argc == 2) {
		printf("start game: %s\n", argv[1]);
		cc.load_port(1);
		//dic = new CommonNet(2, 1);
		game_name = argv[1];
	}
	else if(argc == 3) {
		printf("start game: %s\n", argv[1]);
		use_server_style = true;
		
		game_name = string(argv[1]);

		cc.load_port(1);
		//dic = new CommonNet(2, 1);
	}
	else if(argc == 4) {
		printf("start game: %s\n", argv[1]);
		use_server_style = true;
		
		game_name = string(argv[1]);

		int c_id = atoi(argv[3]);

		if(c_id > 1) {
			client_render = false;
		}

		cc.load_port(c_id);
	}
	else {
		// listen the video client to connect

		usage();
		return 0;
	}
	
	cc.init();
	
	WaitForSingleObject(videoInitMutex, INFINITE);
	ReleaseMutex(videoInitMutex);
	///devicehandle = CreateMutex(NULL, false, "devicehandle");
	presentMutex = CreateMutex(NULL, FALSE, NULL);

	cc.send_raw_buffer(game_name);

	char tm[100];
	cc.recv_raw_buffer(tm, 100);
	infoRecorder->logError("recv:%s\n", tm);

	// create the present event
	if (presentEvent == NULL){
		presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		if (!presentEvent){
			infoRecorder->logTrace("[main]: create Present Event failed.\n");
		}
	}
	DWORD dwThreadId;
	HANDLE videoThread = chBEGINTHREADEX(NULL, 0, VideoServer, NULL, 0, &dwThreadId);

	MSG xmsg = {0};
	while(xmsg.message != WM_QUIT) {
		if(PeekMessage(&xmsg, NULL, 0U, 0U, PM_REMOVE)) {
			TranslateMessage(&xmsg);
			DispatchMessage(&xmsg);
		}
		else {
			cc.take_command(op_code, obj_id);
			//infoRecorder->logTrace("op_code=%d, obj_id=%d\n", op_code, obj_id);
			
			if(op_code >= 0 && op_code < MaxSizeUntilNow_Opcode) {
				if(client_render)
					(*(funcs[op_code].func))();
			}
			else {
				if(op_code == MaxSizeUntilNow_Opcode) {
					infoRecorder->logError("game_client exit normally.");
				}
				else {
					infoRecorder->logError("game client exit, unexpected op_code");
				}
				break;
			}	
		}
	}

	cc.~CommandClient();
	if (presentEvent)
		CloseHandle(presentEvent);
	if (videoInitMutex)
		CloseHandle(videoInitMutex);
	//dic->~CommonNet();
	return 0;
}
#else
//DisClientForRenderProxy * gDisClient = NULL;
void cleanup(){
#ifdef DIS_CLIENT
	if (gDisClient){
		gDisClient->closeClient();
		delete gDisClient;
		gDisClient = NULL;
	}
#endif
}
#if 0

// the main function for renderproxy
int main(int argc, char ** argv){
	//Log::init("renderproxy.log");
	infoRecorder = new InfoRecorder("RenderProxy");
#ifdef DIS_CLIENT
	DisClientForRenderProxy * disClient = NULL;
	disClient = DisClientForRenderProxy::GetDisForRenderProxy();
	if (disClient == NULL){
		infoRecorder->logTrace("[RenderProxy]: get NULL distributor client for render.\n");
		return -1;
	}
	gDisClient = disClient;
#endif
	atexit(cleanup);

	init_fptable();
#ifdef DIS_CLIENT
	disClient->init(0);   // init the distributor client for render 
	disClient->enterLogicLoop();
#endif

	// only for test


	// destory
#if 0
	disClient->closeClient();
	if (disClient){
		//DisClientForRenderProxy::Release();
		delete disClient;
		disClient = NULL;
	}
#endif
	return 0;
}
#else
int main(int argc, char ** argv){
	infoRecorder = new InfoRecorder("RenderProxy");
	// init the ftable
	// start the network
	WSADATA WSAData;
	WSAStartup(0x101, &WSAData);

	atexit(cleanup);
	init_fptable();

	// no other argv, work in distributed mode
	if(argc == 1){

		RenderProxy * proxy = NULL;
		proxy = RenderProxy::GetProxy();
		event_base * base = event_base_new();
		proxy->setEventBase(base);

		proxy->start();

		// start to listen to RTSP port
		//proxy->listenRTSP();
		proxy->dispatch();
	}else if(argc == 2){
		// specify the dis server url in argv[1]
		RenderProxy * proxy = NULL;
		proxy = RenderProxy::GetProxy();
		event_base * base = event_base_new();
		proxy->setEventBase(base);
		proxy->start(argv[1]);
		proxy->dispatch();
	}else if(argc == 3){
		// specify the dis server url in argv[1], the encoder option in argv[2]
		RenderProxy *proxy = NULL;
		proxy = RenderProxy::GetProxy();
		event_base * base = event_base_new();
		proxy->setEventBase(base);
		proxy->setEncodeOption(atoi(argv[2]));
		proxy->start(argv[1]);
		proxy->dispatch();
	}else if(argc == 4){
		// argv: RenderProxy [logic url] [name] [encode option]
		char * url  = NULL;
		char * object = NULL;
		int encodeOption = atoi(argv[3]);

		url = argv[1];
		object = argv[2];
		char * id = argv[3];

		evutil_socket_t socketForCmd = NULL;
		socketForCmd = connectToGraphic(url, 60000);

		RenderChannel *ch = new RenderChannel();
		ch->rtspObject = _strdup(object);
		ch->taskId = 0;
		ch->setEncoderOption(encodeOption);


		/// send start task cmd
		char cmd[1024] = {0};
		strcpy(cmd, START_GAME);
		strcat(cmd, "+");
		strcat(cmd, object);
		printf("[RenderProxy]: send cmd '%s'\n.", cmd);
		int n = send(socketForCmd, cmd, strlen(cmd), 0);


		if(!ch->initRenderChannel(0, object, socketForCmd)){
			infoRecorder->logError("[Main]: create render channel failed.\n");
			return -1;
		}
		// do the rendering 
		ch->startChannelThread();
		//wait the render channel to exit
		WaitForSingleObject(ch->channelThreadHandle, INFINITE);

	}
	else if(argc == 6){
		// argv: RenderProxy [url] [port] [GameName] [enable rtsp] [rtspPort]
		char * url = argv[1];
		int port = atoi(argv[2]);
		char * gameName = argv[3];
		int enableRTSP = atoi(argv[4]);
		int rtspServerPort = atoi(argv[5]);

		// single render mode
		evutil_socket_t sockForCmd = NULL;
		sockForCmd = connectToGraphic(url, port);
		// create the connection
		RenderChannel * ch = new RenderChannel();
		ch->rtspObject = _strdup(gameName);
		if(!ch->initRenderChannel(0, gameName, sockForCmd)){
			infoRecorder->logError("[Main]: create render channel failed.\n");
			return -1;
		}
		// do the rendering
		ch->startChannelThread();
		// wait the render channel to exit

		WaitForSingleObject(ch->channelThreadHandle, INFINITE);

	}else{
		printf("[Usage:\n");
		printf("RenderProxy\tWork in distributed mode.\n");
		printf("RenderProxy [url] [port] [GameName] [enableRtsp]\tWork in single render mode.\n");
		
	}
	return 0;
}

#endif
#endif