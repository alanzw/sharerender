#include "../libCore/CommonNet.h"
#include <process.h>
//#include "../LibVideo/Config.h"
#include <WinSock2.h>
#include "../LibRender/LibRenderAPI.h"
#include "../LibCore/DisNetwork.h"
#include "RenderProxy.h"
#include "../LibCore/InfoRecorder.h"
#include "../LibDistrubutor/Context.h"
#include "DisForRender.h"

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
	kehook = SetWindowsHookEx(WH_KEYBOARD, HookProc, hmode, dwThreadId);
}

void cleanup(){
#ifdef DIS_CLIENT
	if (gDisClient){
		gDisClient->closeClient();
		delete gDisClient;
		gDisClient = NULL;
	}
#endif
}

/*cmd option:
-u: url option
-e: encoder option
-p: request port
-n: requested game name
-r: enable rtsp or not
-v: rtsp port
-m: work mode, 0 for distributer mode, request the distributor, 1 for request game loader, 2 for request game process

*/
enum RENDERMODE{
	DIS_MODE,
	REQ_LOADER,
	REQ_PROCESS,
	TEST_PSNR
};

void printHelp(){
	// two work mode, each has special arguments
	printf("RenderProxy --help or RenderProxy -h\n");
	printf("RenderProxy default works in DIS_MODE(0), use -m to change the work mode.\n");
	printf("arguments:\n");
	printf("\t-u: the url to request, in DIS_MODE, the url means the distributor url, in REQ_LOADER or REQ_PROCESS, the url means the graphic server.\n");
	printf("\t-e: the encoder option, 1 for X264, 2 for cuda, 3 for nvenc.\n");
	printf("\t-p: the request port, only used in REQ_LOADER or REQ_PROCESS mode.\n");
	printf("\t-n: the request game name, only use in REQ_LOADER or REQ_PROCESS.\n");
	printf("\t-m: specific the work mode, 0 for DIS_MODE, 1 for REQ_LAODER, 2 for REQ_PROCESS.\n");
}

bool dealCmd(int argc, char ** argv){
	char * url = NULL;
	int encoderOption = -1;
	int requestPort = 6000;
	char * requestName = NULL;
	bool enableRtsp = false;
	int rtspPort = 0;
	RENDERMODE mode = DIS_MODE;
	bool enableEncoding = false;
	char * rtspConfFile = NULL;
	
	for(int i = 0; i < argc; i++){
		if(!strcmp(argv[i], "-v") || ! strcmp(argv[i], "-V")){
			// the rtsp port
			rtspPort = atoi(argv[i+1]);
		}
		else if(!strcmp(argv[i], "-u") || !strcmp(argv[i], "-U")){
			// the url
			url = argv[i+1];
		}else if(!strcmp(argv[i], "-e") || !strcmp(argv[i], "-E")){
			// encoder option
			encoderOption = atoi(argv[i+1]);
			enableEncoding = true;
		}else if(!strcmp(argv[i], "-p") || !strcmp(argv[i], "-P")){
			// the port to request
			requestPort = atoi(argv[i+1]);
		}else if(!strcmp(argv[i], "-n") || !strcmp(argv[i], "-N")){
			// the requested game name
			requestName = argv[i+1];
		}else if(!strcmp(argv[i], "-r") || !strcmp(argv[i], "-R")){
			// the rtsp option
			enableRtsp = (atoi(argv[i+1]) ? true : false );
		}else if(!strcmp(argv[i], "-m") || !strcmp(argv[i], "-M")){
			// the work mode
			mode = (RENDERMODE)atoi(argv[i+1]);
		}
		else if(!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")){
			printHelp();
		}
		else if(!strcmp(argv[i], "-c") || !strcmp(argv[i], "-C")){
			rtspConfFile = _strdup(argv[i+1]);
		}
		else{
			// invalid arguments, ignore
		}
	}
	if(rtspConfFile == NULL && url == NULL){
		std::cout << "[RenderProxy]: should add param for url via -u [url] or add param for rtsp config file via -c [filename]." << std::endl;
		return -1;
	}
	cg::RTSPConf *rtspConf = NULL;
	if(rtspConfFile){
		rtspConf= cg::RTSPConf::GetRTSPConf(rtspConfFile);
	}
	else{
		rtspConf = cg::RTSPConf::GetRTSPConf("config/server.render.conf");
	}

	// build the render proxy with given arguments
	RenderProxy * proxy = RenderProxy::GetProxy();
	event_base * base = NULL; // not used when directly request graphic server
	// for request graphic mode
	evutil_socket_t socketForCmd = NULL;
	RenderChannel * ch = NULL;
	char cmd[1024] = {0};
	int n = 0;

	switch(mode){
	case DIS_MODE:
		{
			base = event_base_new();
			proxy->setEventBase(base);
			if(encoderOption != -1){
				proxy->setEncodeOption(encoderOption);
			}
			// use the url if any
			
			if(rtspConf){
				proxy->setRTSPConf(rtspConf);
				//char * surl = rtspConf->getDisUrl();
				//char * s1url = rtspConf->disServerName;
				//char * dups = _strdup(rtspConf->disServerName);
				proxy->start(rtspConf->getDisUrl());
			}
			else if(url){
				proxy->start(url);
			}
			// start to listen to RTSP port
			proxy->dispatch();
		}
		break;
	case REQ_LOADER:
		{
			socketForCmd = connectToGraphic(url, requestPort);
			ch = new RenderChannel();

			ch->rtspObject = _strdup(requestName);
			ch->taskId = 0;
			if(encoderOption != -1)
				ch->setEncoderOption(encoderOption);

			/// send start task cmd
			strcpy(cmd, START_GAME);
			strcat(cmd, "+");
			strcat(cmd, requestName);
			printf("[RenderProxy]: send cmd '%s'\n.", cmd);
			n = send(socketForCmd, cmd, strlen(cmd), 0);

			if(!ch->initRenderChannel(0, requestName, socketForCmd)){
				infoRecorder->logError("[Main]: create render channel failed.\n");
				break;
			}
			// do the rendering 
			ch->startChannelThread();
			//wait the render channel to exit
			WaitForSingleObject(ch->channelThreadHandle, INFINITE);
		}
		break;
	case REQ_PROCESS:
		{
			if(!url){
				std::cout << "[RenderProxy]: missing URL for logic server." << std::endl;
			}
			socketForCmd = connectToGraphic(url, requestPort);
			ch = new RenderChannel();

			ch->rtspObject = _strdup(requestName);
			ch->taskId = 0;
			if(encoderOption != -1)
				ch->setEncoderOption(encoderOption);

			if(!ch->initRenderChannel(0, requestName, socketForCmd)){
				infoRecorder->logError("[Main]: create render channel failed.\n");
				break;
			}
			// do the rendering 
			ch->startChannelThread();
			//wait the render channel to exit
			WaitForSingleObject(ch->channelThreadHandle, INFINITE);
		}
		break;
	case TEST_PSNR:
		{
			socketForCmd = connectToGraphic(url, requestPort);
			ch = new RenderChannel();

			ch->rtspObject = _strdup(requestName);
			ch->taskId = 0;
			if(encoderOption != -1)
				ch->setEncoderOption(encoderOption);

			/// send start task cmd
			//strcpy(cmd, START_GAME);
			//strcat(cmd, "+");
			strcat(cmd, requestName);
			printf("[RenderProxy]: send cmd '%s'.\n", cmd);
			n = send(socketForCmd, cmd, strlen(cmd), 0);

			if(!ch->initRenderChannel(0, requestName, socketForCmd)){
				infoRecorder->logError("[Main]: create render channel failed.\n");
				break;
			}
			// do the rendering 
			ch->startChannelThread();
			//wait the render channel to exit
			WaitForSingleObject(ch->channelThreadHandle, INFINITE);
		}
		break;
	default:
		{
			infoRecorder->logError("[Main]: invalid work mode.\n");
		}
	break;
	}
	// clean up when exit
	if(proxy){
		delete proxy;
		proxy = NULL;
	}
	if(ch){
		delete ch;
		ch = NULL;
	}
	return true;
}

int main(int argc, char ** argv){
	// init the logger
	infoRecorder = new InfoRecorder("RenderProxy");
	infoRecorder->init();
	// start the network
	WSADATA WSAData;
	WSAStartup(0x101, &WSAData);
	// register the clean method
	atexit(cleanup);
	// init the function table
	init_fptable();

#if 0
	int t_argc = 11;
	char * t_argv[] = {
		"RenderProxy.exe",
		"-m",
		"2",
		"-u",
		"127.0.0.1",
		"-p",
		"7000",
		"-e",
		"1",
		"-n",
		"SprillRichi.exe"
	};

	dealCmd(t_argc, t_argv);
#else

	dealCmd(argc, argv);
#endif
	return 0;
}