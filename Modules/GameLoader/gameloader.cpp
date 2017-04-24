#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>
#include "gameloader.h"
#include "../LibCore/Log.h"
#include "detours/detours.h"
#include "../LibDistrubutor/Distributor.h"
#include "../LibCore/InfoRecorder.h"
//#include <KtmW32.h>
#include <tchar.h>
//#include "../LibCore/InfoRecorder.h"
#pragma comment(lib, "detours.lib")
//#pragma comment(lib, "detours/detours.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "event.lib")
#pragma comment(lib, "event_core.lib")
#pragma comment(lib, "event_extra.lib")
#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"d3dx9.lib")
//#pragma comment(lib, "KtmW32.lib")

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


#if 0
// command for logic server
extern const char * INFO;   // recv info cmd from dis, collect the domain info
extern const char * ADD_RENDER;  // recv add render cmd from dis
extern const char * DECLINE_RENDER; // recv decline render cmd from dis
extern const char * GAME_EXIT; // recv game exit cmd from dis
extern const char * CANCEL_TASK;   // recv cancel task cmd from DIS
extern const char * START_TASK;
#endif


// global functions
void onBufferEventRead(struct bufferevent * bev, void *ctx);
void onBufferEventEvent(struct bufferevent * bev, short events, void * ctx);
void listenerCB(struct evconnlistener * listerner, evutil_socket_t sock, struct sockaddr * sddr, int len, void * ctx);
void acceptErrorCB(struct evconnlistener * listener, void *ctx);

// this is for the game loader

#if 0

GameLoader::GameLoader(){
	gameSpider = new GameSpider();

}
GameLoader::~GameLoader(){
	if(gameSpider){
		delete gameSpider;
		gameSpider = NULL;
	}
}
// load a 2D game, and start the video seerver
HANDLE GameLoader::load2DGame(char * gameName){
	// find the path
	//char gamePath[MAX_PATH];  // to store the path
	char * path = NULL;    // remember to free

	path = gameSpider->getPath(gameName);
	if(!gameSpider->changeCurrentDirectory(path)){
		// failed to set path
	}

	char * dllName = NULL;
	// start the game?
	dllName = gameSpider->getHookDllName();

	HANDLE ret = this->startGameWidthDll(gameName, dllName, gamePath);

	return ret;

	// notify the distributor that the game source.

}


/// loading the 3D game may need the cmdline?
HANDLE GameLoader::loadD3DGame(char * gameName){

	char * gamePath = NULL;   // remember to free
	gamePath = gameSpider->getPath(gameName);

	char * dllName = NULL;
	dllName = gameSpider->getHookDllName();
	// start the game?

	/// seemed OK, for now
	HANDLE ret = this->startGameWidthDll(gameName, dllName, gamePath);

	return ret;

}

// first, judge the game type, whether it is D3D or 2D game. then call the load2DGame or loadD3DGame to start the game
HANDLE GameLoader::loadGame(char * gameName){
	int isD3DGame = 0;
	HANDLE ret = NULL;
	isD3DGame = gameSpider->isD3DSupported(gameName);
	if(isD3DGame){
		ret = this->loadD3DGame(gameName);
	}else{
		ret = this->load2DGame(gameName);
	}

	return ret;
}


HANDLE GameLoader::start2DGame(char * gameName, char * gamePath){
	infoRecorder->logTrace("[GameLoader]: start 2D game '%s'.\n", gameName);

	PROCESS_INFORMATION pi = { 0 };
	STARTUPINFO si = { 0 };
	si.cb = sizeof(si);

	HANDLE hnd = GetCurrentProcess();

	printf("Listening\nhandle %d\n", hnd);

	LPSECURITY_ATTRIBUTES lp_attributes;
	LPSECURITY_ATTRIBUTES lpThreadAttributes;
	STARTUPINFO startupInfo = { sizeof(startupInfo) };
	memset(&startupInfo, 0, sizeof(STARTUPINFO));
	startupInfo.cb = sizeof(STARTUPINFO);
	startupInfo.dwFlags = 0;
	startupInfo.wShowWindow = SW_HIDE;

	PROCESS_INFORMATION processInformation;
	char cmdLine[100];

	DWORD id = GetCurrentProcessId();
	sprintf(cmdLine, "%s", gameName);
	printf("cmd line is %s\n", cmdLine);
	bool ret = CreateProcess(NULL, cmdLine, NULL, NULL, TRUE, CREATE_DEFAULT_ERROR_MODE, NULL, gamePath, &si, &pi);
	if (!ret) {
		char err_str[200];
		sprintf(err_str, "Game Start %s Failed", gameName);
		MessageBox(NULL, err_str, "Error", MB_OK);
	}

	return pi.hProcess;

}

HANDLE GameLoader::startGameWidthDll(char * gameName, char * dllName, char * gamePath){
	infoRecorder->logError("[GameLoader]: start game with dll.\n");

	PROCESS_INFORMATION pi = { 0 };
	STARTUPINFO si = { 0 };
	si.cb = sizeof(si);

	HANDLE hnd = GetCurrentProcess();

	printf("Listening\nhandle %d\n", hnd);

	LPSECURITY_ATTRIBUTES lp_attributes;
	LPSECURITY_ATTRIBUTES lpThreadAttributes;
	STARTUPINFO startupInfo = { sizeof(startupInfo) };
	memset(&startupInfo, 0, sizeof(STARTUPINFO));
	startupInfo.cb = sizeof(STARTUPINFO);
	startupInfo.dwFlags = 0;
	startupInfo.wShowWindow = SW_HIDE;

	PROCESS_INFORMATION processInformation;
	char cmdLine[100];

	DWORD id = GetCurrentProcessId();
	sprintf(cmdLine, "%s", gameName);
	printf("cmd line is %s\n", cmdLine);
	bool ret = DetourCreateProcessWithDll(NULL, cmdLine, NULL, NULL, TRUE, CREATE_DEFAULT_ERROR_MODE,
		NULL, gamePath, &si, &pi, dllName, NULL);
	
	if (!ret) {
		char err_str[200];
		sprintf(err_str, "Game Start %s Failed", gameName);
		MessageBox(NULL, err_str, "Error", MB_OK);
	}

	return pi.hProcess;

}
#else

GameLoader * GameLoader::loader;
ProcessManager * ProcessManager::mgr;
RenderManager * RenderManager::mgr;

GameLoader::GameLoader(){
	gameInfo = GameInfo::GetGameInfo();
	if(!gameInfo->loadInfo()){
		printf("[GameLoader]: load game information failed.\n");
	}
	else{
		gameInfo->showAllInfo();
	}
}
GameLoader::GameLoader(char * mapFile){
	gameInfo = GameInfo::GetGameInfo(mapFile);
	if(!gameInfo->loadInfo()){
		printf("[GameLoader]: load game information failed.\n");
	}
	else{
		gameInfo->showAllInfo();
	}
}
GameLoader::~GameLoader(){
	
}
// load a 2D game, and start the video server
HANDLE GameLoader::load2DGame(char * gameName){
	// find the path
#if 0
	//char gamePath[MAX_PATH];  // to store the path
	const char * path = NULL;    // remember to free
	path = gameInfo->findGamePath(gameName).c_str();
	
	const char * dllName = NULL;
	// start the game?
	dllName = gameInfo->findGameDll(gameName).c_str();
	HANDLE ret = this->startGameWithDll(gameName, (char *)dllName, (char *)path);

	return ret;
#else

	GameInfoItem * item = gameInfo->findGameInfo(gameName);

	/// seemed OK, for now
	HANDLE ret = this->startGameWithDll((char *)item->gameName.c_str(), (char *)item->hookDll.c_str(), (char *)item->gamePath.c_str());

	return ret;

#endif
	// notify the distributor that the game source.

}
// load a 2D game, and start the video server
HANDLE GameLoader::load2DGame(char * gameName,char * arg, char * dllName){
	// find the path
#if 0
	//char gamePath[MAX_PATH];  // to store the path
	const char * path = NULL;    // remember to free
	path = gameInfo->findGamePath(gameName).c_str();

	const char * dllName = NULL;
	// start the game?
	dllName = gameInfo->findGameDll(gameName).c_str();
	HANDLE ret = this->startGameWithDll(gameName, (char *)dllName, (char *)path, arg);
#else

	GameInfoItem * item = gameInfo->findGameInfo(gameName);

	/// seemed OK, for now
	HANDLE ret = NULL;
	if(dllName)
		ret = this->startGameWithDll((char *)item->gameName.c_str(), (char *)item->hookDll.c_str(), (char *)item->gamePath.c_str(), arg);
	else
		ret = startGameWithDll((char *)item->gameName.c_str(), dllName, (char *)item->gamePath.c_str(), arg);

#endif
	return ret;

	// notify the distributor that the game source.

}
/// loading the 3D game may need the cmd line?
HANDLE GameLoader::loadD3DGame(char * gameName){
#if 0
	char * gamePath = NULL;   // remember to free
	gamePath = (char *)gameInfo->findGamePath(gameName).c_str(); 

	char * dllName = NULL;
	dllName = (char *)gameInfo->findGameDll(gameName).c_str();
	// start the game?

	/// seemed OK, for now
	HANDLE ret = this->startGameWithDll(gameName, dllName, gamePath);
#else
	GameInfoItem * item = gameInfo->findGameInfo(gameName);

	/// seemed OK, for now
	HANDLE ret = this->startGameWithDll((char *)item->gameName.c_str(), (char *)item->hookDll.c_str(), (char *)item->gamePath.c_str());

#endif
	return ret;

}
/// loading the 3D game may need the cmd line?
HANDLE GameLoader::loadD3DGame(char * gameName,char * arg, char * dllName){
#if 0
	char * gamePath = NULL;   // remember to free
	gamePath = (char *)gameInfo->findGamePath(gameName).c_str(); 

	char * dllName = NULL;
	dllName = (char *)gameInfo->findGameDll(gameName).c_str();
	HANDLE ret = this->startGameWithDll(gameName, dllName, gamePath, arg);
#else
	// start the game?
	GameInfoItem * item = gameInfo->findGameInfo(gameName);

	/// seemed OK, for now
	HANDLE ret = NULL;
	if(NULL == dllName){
		ret = this->startGameWithDll((char *)item->gameName.c_str(), (char *)item->hookDll.c_str(), (char *)item->gamePath.c_str(), arg);
	}
	else{
		cout << "[GameLoader]: use user dll: " << dllName <<endl;
		ret = startGameWithDll((char *)item->gameName.c_str(), dllName, (char *)item->gamePath.c_str(), arg);
	}


#endif
	return ret;

}
// first, judge the game type, whether it is D3D or 2D game. then call the load2DGame or loadD3DGame to start the game
HANDLE GameLoader::loadGame(char * gameName){
	bool isD3DGame = false;
	HANDLE ret = NULL;
	isD3DGame = gameInfo->isD3DSupported(gameName);
	if(isD3DGame){
		printf("[GameLoader]: game '%s' supported d3d.\n", gameName);
		ret = this->loadD3DGame(gameName);
	}else{
		printf("[GameLoacder]: game '%s' not support d3d.\n", gameName);
		ret = this->load2DGame(gameName);
	}

	return ret;
}
// load game with arg
HANDLE GameLoader::loadGame(char * gameName, char * arg, char * dllName){
	bool isD3DGame = false;
	HANDLE ret = NULL;
	isD3DGame = gameInfo->isD3DSupported(gameName);
	if(isD3DGame){
		printf("[GameLoader]: game '%s' supported d3d.\n", gameName);
		ret = this->loadD3DGame(gameName, arg, dllName);
	}else{
		printf("[GameLoacder]: game '%s' not support d3d.\n", gameName);
		ret = this->load2DGame(gameName, arg, dllName);
	}

	// create usage logger for each process

	return ret;
}
// start a 2D game, not used
HANDLE GameLoader::start2DGame(char * gameName, char * gamePath){
	//infoRecorder->logTrace("[GameLoader]: start 2D game '%s'.\n", gameName);

	PROCESS_INFORMATION pi = { 0 };
	STARTUPINFO si = { 0 };
	si.cb = sizeof(si);

	HANDLE hnd = GetCurrentProcess();

	printf("Listening\nhandle %d\n", hnd);

	LPSECURITY_ATTRIBUTES lp_attributes;
	LPSECURITY_ATTRIBUTES lpThreadAttributes;
	STARTUPINFO startupInfo = { sizeof(startupInfo) };
	memset(&startupInfo, 0, sizeof(STARTUPINFO));
	startupInfo.cb = sizeof(STARTUPINFO);
	startupInfo.dwFlags = 0;
	startupInfo.wShowWindow = SW_HIDE;

	PROCESS_INFORMATION processInformation;
	char cmdLine[100];

	DWORD id = GetCurrentProcessId();
	sprintf(cmdLine, "%s", gameName);
	printf("cmd line is %s\n", cmdLine);
	

	return pi.hProcess;

}
// start the game with given dll
HANDLE GameLoader::startGameWithDll(char * gameName, char * dllName, char * gamePath){
	//infoRecorder->logError("[GameLoader]: start game with dll.\n");
	return startGameWithDll(gameName,dllName,gamePath, NULL);
}

// search the current path, whether the dll exist, if exist, copy it to new path
bool GameLoader::copyFile(char * curPath, char * newPath, char * dllName){
	bool ret = true;
	char fileName[MAX_PATH] = {0};
	char newFileName[MAX_PATH] = {0};
	sprintf(fileName, "%s\\%s", ".", dllName);
	sprintf(newFileName, "%s\\%s", newPath, dllName);
	//LPWSTR note = _T("copy dll file trans");

#if 0

	HANDLE trans = CreateTransaction(NULL, 0, TRANSACTION_DO_NOT_PROMOTE, 0, 0,INFINITE, NULL);
	if(trans == INVALID_HANDLE_VALUE){
		infoRecorder->logError("[GameLoader]: create transaction failed with:%d.\n", GetLastError());
		return false;
	}
	BOOL bRet = CopyFileTransacted(fileName, newFileName, NULL, NULL, FALSE, COPY_FILE_FAIL_IF_EXISTS | COPY_FILE_RESTARTABLE, trans);
	if(!bRet){
		infoRecorder->logError("[GameLoader]: copy file failed, %d.\n", GetLastError());
		ret = false;
	}

	CloseHandle(trans);
	return ret;
#else
	return false;

#endif
}
// check the file eixst in path or not
bool GameLoader::checkFile(char * path, char * name){
	WIN32_FIND_DATA findFileData;
	LPTSTR dir;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	bool ret = true;

	dir = (LPTSTR)malloc(MAX_PATH);
	sprintf(dir, "%s\\%s", path,  name);


	hFind = FindFirstFile(dir, &findFileData);

	if(hFind == INVALID_HANDLE_VALUE){
		infoRecorder->logError("[GameLoader]: find the dll '%s' in path '%s', full name '%s', failed.\n", name, path, dir);
		
		FindClose(hFind);
		return false;
	}
	infoRecorder->logTrace("[GameLoader]: dll '%s' exist in '%s'.\n", name, path);

	FindClose(hFind);
	return true;
}
// check the dll file, whether it exist in the game path, if not, copy it
bool GameLoader::checkDllFile(char * gamePath, char *dllName){
	bool ret = true;
	if(!checkFile(gamePath, dllName)){
		if(checkFile(".", dllName)){
			// valid file
			return copyFile(".", gamePath, dllName);
		}
		return false;
	}
	else{
		// dll exist in the game path
	}

	return ret;
}

// start the game with given dll and arg
HANDLE GameLoader::startGameWithDll(char * gameName, char * dllName, char * gamePath, char * arg){
	//infoRecorder->logError("[GameLoader]: start game with dll.\n");
	printf("[GameLoader]: star game '%s' with dll '%s', path '%s'.\n", gameName, dllName, gamePath);
	if(!checkDllFile(gamePath, dllName)){
		infoRecorder->logError("[GameLoader]: resolve the dll file failed.\n");
		printf("[GameLoader]: resolve the dll file '%s' in path '%s' failed.\n", dllName, gamePath);
		return NULL;
	}

	PROCESS_INFORMATION pi = { 0 };
	STARTUPINFO si = { 0 };
	si.cb = sizeof(si);

	HANDLE hnd = GetCurrentProcess();

	printf("Listening\nhandle %d\n", hnd);

	LPSECURITY_ATTRIBUTES lp_attributes;
	LPSECURITY_ATTRIBUTES lpThreadAttributes;
	STARTUPINFO startupInfo = { sizeof(startupInfo) };
	memset(&startupInfo, 0, sizeof(STARTUPINFO));
	startupInfo.cb = sizeof(STARTUPINFO);
	startupInfo.dwFlags = 0;
	startupInfo.wShowWindow = SW_HIDE;

	PROCESS_INFORMATION processInformation;
	char cmdLine[MAX_PATH];

	DWORD id = GetCurrentProcessId();
	if(arg)
		sprintf(cmdLine, "%s/%s %s",gamePath, gameName, arg);
	else
		sprintf(cmdLine, "%s/%s",gamePath, gameName);

	printf("cmd line is %s\n", cmdLine);
	BOOL ret = DetourCreateProcessWithDll(NULL, cmdLine, NULL, NULL, TRUE, CREATE_DEFAULT_ERROR_MODE,
		NULL, gamePath, &si, &pi, dllName, NULL);

	if (!ret) {
		char err_str[200];
		sprintf(err_str, "Game Start %s Failed", gameName);
		MessageBox(NULL, err_str, "Error", MB_OK);
	}

	return pi.hProcess;
}

// Factory
LogicFactory* LogicFactory::logic;
// enter the libevent loop
void LogicFactory::enterLoop(){
	printf("[GameLoader]: enter loop.\n");
	event_base_dispatch(base);

	// before exit
	printf("[GameLoader]: before exit, clean up.\n");
	if(processListener){
		evconnlistener_free(processListener);
		processListener = NULL;
	}
}
// init the game loader, include initing the socket
bool LogicFactory::init(){
	printf("[GameLoader]: init.\n");

	// create the event base, the base is for logic factory only
	base = NULL;
	base = event_base_new();
	if(!base){
		fprintf(stderr, "Couldn't create an event_base: exiting.\n");
		return false;
	}
	return true;
}
// connect to the DisServer
bool LogicFactory::connectDis(char * ip, short port){
	printf("[LogicFactory]: connect to dis server.\n");
	bool ret = true;
	
	sockaddr_in sin;
	evutil_socket_t sock = NULL;

	int sin_size = sizeof(sin);
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr(ip);
	sin.sin_port = htons(port);

	struct bufferevent * bev = NULL;
#if 0
	bev = bufferevent_socket_new(base, -1, BEV_OPT_CLOSE_ON_FREE);
	bufferevent_setcb(bev, onBufferEventRead, NULL, onBufferEventEvent, NULL);

	// connect to dis
	if(bufferevent_socket_connect(bev, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		// error starting connection
		bufferevent_free(bev);
		return false;
	}
#else
	// create the bufferevent
	sock = socket(AF_INET, SOCK_STREAM, 0);
	
	if(evutil_make_socket_nonblocking(sock) < 0){
		infoRecorder->logError("[LogicFactory]: make socket non blocking failed.\n");
		return false;
	}
	frobSocket(sock);

	infoRecorder->logTrace("[LogicFactory]: connection established.\n");
	bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE);
	if(!ctx){
		ctx = new BaseContext();
	}
	ctx->sock = sock;
	ctx->bev = bev;
	
	// set the callback function
	bufferevent_setcb(bev, onBufferEventRead, NULL, onBufferEventEvent, ctx);

	//connect to dis
	if(bufferevent_socket_connect(bev, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		infoRecorder->logError("[LogicFactory]: error starting connection.\n");
		bufferevent_free(bev);
		return false;
	}

	bufferevent_enable(bev, EV_READ | EV_WRITE);

	


#endif
	return ret;
}
bool LogicFactory::registerLogic(){
	infoRecorder->logTrace("[LogicFactory]: register the logic server.\n");

	ctx->writeCmd(REGISTER);
	ctx->writeData((void *)LOGIC, strlen(LOGIC));
	ctx->writeToNet();

	return true;
}
// start listen to render
bool LogicFactory::startInternalListen(){
	printf("[LogicFactory]: start to listen to game process.\n");
	unsigned short port = INTERNAL_PORT;
	if(conf){
		port = conf->loaderPort;
	}
	// create loader listener for game process to connect
	sockaddr_in sin;
	int sin_size = sizeof(sin);
	memset(&sin, 0, sin_size);
	sin.sin_family = AF_INET;
	sin.sin_addr.S_un.S_addr = htonl(0);
	sin.sin_port = htons(port);   // the graphic port

	// create listener with processManager
	processListener = evconnlistener_new_bind(base, listenerCB, pmgr, LEV_OPT_LEAVE_SOCKETS_BLOCKING | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size); // = evcon
	if(!processListener){
		perror("couldn't create listener.\n");
		return false;
	}
	evconnlistener_set_error_cb(processListener, acceptErrorCB);
	return true;
}
// start the ctrl listen
bool LogicFactory::startCtrlListen(){
	printf("[LogicFactory]: start to listen to render.\n");

	sockaddr_in sin;
	int sin_size = sizeof(sin);
	memset(&sin, 0, sin_size);
	sin.sin_family = AF_INET;
	sin.sin_addr.S_un.S_addr = htonl(0);
	sin.sin_port = htons(60001);

	ctrlListener = evconnlistener_new_bind(base, listenerCB, cmgr, LEV_OPT_CLOSE_ON_FREE | LEV_OPT_LEAVE_SOCKETS_BLOCKING | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size);
	if(!ctrlListener){
		perror("couldn't create render listener.\n");
		return false;
	}
	evconnlistener_set_error_cb(ctrlListener, acceptErrorCB);
	return true;
}

// start listen to process
bool LogicFactory::startRenderListen(){
	printf("[LogicFactory]: start to listen to render.\n");
	unsigned port = 60000;
	if(conf){
		port = conf->graphicPort;
	}

	sockaddr_in sin;
	int sin_size = sizeof(sin);
	memset(&sin, 0, sin_size);
	sin.sin_family = AF_INET;
	sin.sin_addr.S_un.S_addr = htonl(0);
	sin.sin_port = htons(port);

	renderListener = evconnlistener_new_bind(base, listenerCB, rmgr, LEV_OPT_CLOSE_ON_FREE | LEV_OPT_LEAVE_SOCKETS_BLOCKING | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size);
	if(!renderListener){
		perror("couldn't create render listener.\n");
		return false;
	}
	evconnlistener_set_error_cb(renderListener, acceptErrorCB);
	return true;
}

// start to listen
bool LogicFactory::startListen(){
	printf("[LogicFactory]: start to listen.\n");
	bool ret = true;
	if(!startInternalListen() || !startRenderListen()){
		printf("[GameLoader]: start listen failed.\n");
		return false;
	}

	return ret;
}
////////////////////// for context //////////////////////

// logic deal all the cmd from dis, render, client and even the game process
bool LogicFactory::dealCmd(BaseContext * ctx){
	infoRecorder->logError("[LogicFactory]: deal cmd.\n");
	bool ret = true;

	char  feedback[1000] = { 0 };
	ctx->readCmd();
	int len = 0;
	char * cmd = ctx->getCmd();
	if (!strncasecmp(cmd, INFO, strlen(INFO))){
		infoRecorder->logError("[LogicFactory]: INFO.\n");

		// collect information and feedback, the cmd is from dis
		float cpuUsage = 0.0f, gpuUsage = 0.0f, memUsage = 0.0f;
		
		collectInfo(cpuUsage, gpuUsage, memUsage);	
		ctx->writeCmd(cg::INFO);
		ctx->writeFloat(cpuUsage);
		ctx->writeFloat(gpuUsage);
		ctx->writeFloat(memUsage);
		ctx->writeToNet();
	}
	else if (!strncasecmp(cmd, ADD_RENDER, strlen(ADD_RENDER))){
		infoRecorder->logError("[LogicFactory]: ADD RENDER, the new render will auto added.\n");
		// add a render to exist connection, the cmd is from render proxy
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;  // to find the task
		IDENTIFIER renderId = *(IDENTIFIER *)(data + sizeof(IDENTIFIER));

		LogicTask * task = findTask(id);
		// find the logic task

		// add render
		task->addRender(ctx);

		BaseContext * game = NULL;
		game = task->processCtx;

		// tell game to add
		game->writeCmd(ADD_RENDER);
		// write current renders' connection
		game->writeIdentifier(ctx->sock);
		DWORD curPid = GetCurrentProcessId();
		infoRecorder->logTrace("[LogicFactory]: current process id:%p.\n", curPid);
		infoRecorder->logError("[LogicFactory]: current process id:%d.\n", curPid);
		game->writeData(&curPid, sizeof(DWORD));
		game->writeToNet();  // write to game process

	}
	else if (!strncasecmp(cmd, DECLINE_RENDER, strlen(DECLINE_RENDER))){
		// the cmd is from dis
		printf("[LogicFactory]: DECLINE RENDER.\n");
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;    // to find the task
		// distinguish render
		IDENTIFIER renderId = *(IDENTIFIER *)(data + sizeof(IDENTIFIER));

		LogicTask * task = findTask(id);
		if(!task){
			infoRecorder->logError("[LogicFactaory]: did not find the task.\n");
			return false;
		}
		task->removeRender(renderId);

		// tell the game process to decline the given render
		BaseContext * game = task->processCtx;
		game->writeCmd(DECLINE_RENDER);
		game->writeIdentifier(renderId);
		game->writeToNet();
	}
	else if (!strncasecmp(cmd, GAME_EXIT, strlen(GAME_EXIT))){
		infoRecorder->logError("[LogicFactory]: GAME EXIT, TODO.\n");

	}
	else if (!strncasecmp(cmd, CANCEL_TASK, strlen(CANCEL_TASK))){
		infoRecorder->logError("[LogicFactory]: CANCEl TASK.\n");
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;
		data += sizeof(IDENTIFIER) + 1;

		// TODO, cancel the task
		// tell the process to exit
		LogicTask * task = findTask(id);
		if(!task){
			infoRecorder->logError("[LogicFactory]: cannot find the task \n");
			return false;
		}
		BaseContext * game = task->processCtx;
		game->writeCmd(GAME_EXIT);
		game->writeIdentifier(id);
		game->writeToNet();

		// remove the task
		removeTask(id);

		// feedback with domain information
		//collect information
		float cpuUsage = 0.0f, gpuUsage = 0.0f, memUsage = 0.0f;

		collectInfo(cpuUsage, gpuUsage, memUsage);

		ctx->writeCmd(INFO);
		ctx->writeFloat(cpuUsage);
		ctx->writeFloat(gpuUsage);
		ctx->writeFloat(memUsage);
		ctx->writeToNet();
	}else if(!strncasecmp(cmd, START_GAME, strlen(START_GAME))){
		// start game, the socket is the render proxy
		char * data = ctx->getData();
		char * gameName = data;
		HANDLE processHandle = NULL;

		char * arg = (char *)malloc(sizeof(char) *1024);
		sprintf(arg, "-m 0 -r %d -e %d -a %d -p %d", 0, 1, ctx->sock, GetCurrentProcessId());
		processHandle = loader->loadGame(gameName, arg);
		if(!processHandle){
			infoRecorder->logError("[LogicFactory]: load game '%s' '%s' failed.\n", gameName, arg);
			return false;
		}
		free(arg);

	}
	else if (!strncasecmp(cmd, START_TASK, strlen(START_TASK))){
		infoRecorder->logError("[LogicFactory]: START TASK.\n");
		// start a game with no render
		char * data = ctx->getData();
		//data++;
		
		IDENTIFIER id = *(IDENTIFIER *)data;
		TASK_MODE mode = (TASK_MODE)*(short *)(data + sizeof(IDENTIFIER));
		char * gameName = (data + sizeof(IDENTIFIER) + sizeof(short));
		infoRecorder->logError("[LogicFactory]: start game '%s', task id:%p.\n", gameName, id);


		HANDLE processHandle = NULL;
		/*
		char arg[20] = {0};
		// create the full parameters to run the game
		sprintf(arg, "%d", id);
		processHandle = loader->loadGame(gameName, arg);
		*/
		char * arg = (char *)malloc(sizeof(char) * 1024);
		if(mode == MODE_FULL_OFFLOAD){
			sprintf(arg, "-i %d -m 1 -r %d -e %d", id, 0, encoderOption);
		}
		else if(mode == MODE_PARTIAL_OFFLOAD){
			sprintf(arg, "-i %d -m 1 -r %d -e %d", id, 2, encoderOption);
		}else if(mode == MODE_NO_OFFLOAD){
			sprintf(arg, "-i %d -m 1 -r %d -e %d", id, 1, encoderOption);
		}

		// listen mode set to 1, means listen to distributor
		processHandle = loader->loadGame(gameName, arg);
		if(!processHandle){
			infoRecorder->logError("[LogicFactory]: load game failed.\n");
			return false;
		}
		free(arg);
		// create the task 
		LogicTask * task = new LogicTask();
		task->name = _strdup(gameName);
		task->id = id;

		task->processCtx = NULL;
		if(!this->addTask(task)){
			// add task failed
			infoRecorder->logError("[RenderContext]: add task to factory failed.\n");
			return false;
		}
	}
	else if (!strncasecmp(cmd, CLIENT_CONNECTED, strlen(CLIENT_CONNECTED))){
		// game client connected with client id
		infoRecorder->logError("[LogicFactory]: CLIENT_CONNECTED, TODO.\n");
		char * data = ctx->getData();
		IDENTIFIER cid = *(IDENTIFIER *)data;
		// this is for control connection
		infoRecorder->logError("[LogicFactory]: CLIENT CONNECTED.\n");

		// find the game process
		map<IDENTIFIER, BaseContext *>::iterator it = gameMap.find(cid);
		if (it != gameMap.end()){
			// find the game
		}
		else{
			infoRecorder->logError("[LogicFactory]: not find the game process.\n");
		}

	}
	else if (!strncasecmp(cmd, CANCEL_SOCKEVENT, strlen(CANCEL_SOCKEVENT))){
		// the game process take over the render connection, tell the logic server to release the event
		infoRecorder->logTrace("[LogicFactory]: cancel socket event.\n");
		char *data = ctx->getData();
		evutil_socket_t s = *(evutil_socket_t *)data;
		// to cancel the socket event
		BaseContext * snet = NULL;
		map<evutil_socket_t, BaseContext *>::iterator it = netCtxMap.find(s);
		if (it != netCtxMap.end()){
			// cancel the event
			snet = it->second;
			//bufferevent_disable(snet->bev, EV_READ | EV_WRITE);
		}
		else{
			// error
			infoRecorder->logError("[LogicFactory]: cannot find the BaseContext associated with %p.\n", s);
			// print all
			map<IDENTIFIER, BaseContext *>::iterator mi;
			for(mi = netCtxMap.begin(); mi!= netCtxMap.end(); mi++){
				infoRecorder->logTrace("[LogicFactory]: context key:%p.\n", mi->first);
			}
			return false;
		}
		infoRecorder->logTrace("[LogicFactory]: cancel sock event for %p done.\n", s);
	}
	else if (!strncasecmp(cmd, DESTROY_SOCKEVENT, strlen(DESTROY_SOCKEVENT))){
		infoRecorder->logError("[LogicFactory]: recv 'DESTROY_SOCKEVENT'\n");
		char * data = ctx->getData();
		evutil_socket_t s = *(evutil_socket_t*)data;
		BaseContext * snet = NULL;
		map<evutil_socket_t, BaseContext *>::iterator it = netCtxMap.find(s);
		if (it != netCtxMap.end()){
			// destroy the event
			snet = it->second;
			bufferevent_free(snet->bev);
			delete snet;
		}
	}
	else if(!strncasecmp(cmd, GAME_READY, strlen(GAME_READY))){
		// game ready from game process
		infoRecorder->logError("[LogicFactory]: recv 'GAME_READY' from game client.\n");
		char * data= ctx->getData();
		IDENTIFIER tid = *(IDENTIFIER *)data;
		LogicTask * task = findTask(tid);

		if(!task){
			infoRecorder->logError("[LogicFactory]: cannot find the task:%p.\n", tid);
			return false;
		}
		task->processCtx = ctx;  // this context is the task's process context

		// send dis server "logic ready.\n"
		// send back LOGIC READY
		this->getCtx()->writeCmd(LOGIC_READY);
		this->getCtx()->writeIdentifier(tid);
		this->getCtx()->writeToNet();

		ctx->writeCmd("TEST");
		ctx->writeToNet();
	}else if(!strncasecmp(cmd, START_RTSP_SERVICE, strlen(START_RTSP_SERVICE))){
		// deal the logic server providing the rtsp services, cmd from dis

		infoRecorder->logError("[LogicFactory]: ERROR, should never be here,\n");
#if 0
		printf("[LogicFactory]: the logic server provide the rtsp service.\n");
		char * data = ctx->getData();
		IDENTIFIER tid = *(IDENTIFIER *)data;

		LogicTask * task = findTask(tid);
		if(!task){
			printf("[LogicFactory]: cannot find the task to start the rtsp.\n");
			return false;
		}
		// find the task
		// to get the handle of the window, ask the game process to send back the handle of window and present event
		BaseContext * game = task->processCtx;
		game->writeCmd(ADD_RTSP_CONNECTION);
		game->writeToNet();

		//this->startRTSP(NULL);
		// TODO
#endif

	}
	else if(!strncasecmp(cmd, ADD_RTSP_CONNECTION, strlen(ADD_RTSP_CONNECTION))){
		// to start the rtsp service
		infoRecorder->logTrace("[LogicFactory]: ADD RTSP CONNECTION.\n");
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)(data);
		short portOff = *(short *)(data + sizeof(IDENTIFIER));
		LogicTask * task = findTask(id);

		if(!task){
			infoRecorder->logTrace("[LogicFactory]: cannot find the task to star the rtsp.\n");
			return false;
		}

		infoRecorder->logTrace("[LogicFactory]: send add rtsp connection to game process, id: '%p', port offset:%d.\n", id, portOff);
		// tell the process to start rtsp service
		BaseContext * game  =  task->processCtx;
		game->writeCmd(ADD_RTSP_CONNECTION);
		game->writeIdentifier(id);
		game->writeData((void *)&portOff, sizeof(short));
		game->writeToNet();

	}else if(!strncasecmp(cmd, RTSP_READY, strlen(RTSP_READY))){
		// recv rtsp ready cmd from game process, tell the dis manager
		infoRecorder->logTrace("[LogicFactory]: RTSP READY.\n");
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)(data);
		getCtx()->writeCmd(RTSP_READY);
		getCtx()->writeIdentifier(id);
		getCtx()->writeToNet();
	}else if(!strncasecmp(cmd, ADD_CTRL_CONNECTION, strlen(ADD_CTRL_CONNECTION))){
		infoRecorder->logError("[LogicFactory]: ADD CTRL CONNECTION, TODO.\n");

	}
	else if(!strncasecmp(cmd, OPTION, strlen(OPTION))){
		// the option for game process
		cg::core::infoRecorder->logTrace("[LogicFactory]: get CMD Option.\n");
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;
		LogicTask * task = findTask(id);

		if(!task){
			cg::core::infoRecorder->logError("[LogicFactory]: cannot find the logic task '%p' for OPTION.\n", id);
			return false;
		}

		cg::core::infoRecorder->logError("[LogicFactory]: OPTION, id:%p.\n", id);
		BaseContext * gameCtx = task->processCtx;
		short optionCount = *(short *)(data + sizeof(IDENTIFIER));

		if(gameCtx){
			gameCtx->writeCmd(OPTION);
			gameCtx->writeIdentifier(id);
			gameCtx->writeShort(optionCount);
			for(int i =0; i < optionCount; i++){
				CMD_OPTION option = *(CMD_OPTION *)(data + sizeof(IDENTIFIER) + sizeof(short) + i * (sizeof(CMD_OPTION) + sizeof(short)));
				short value = *(short *)(data + sizeof(IDENTIFIER) + sizeof(short) + i * (sizeof(CMD_OPTION) + sizeof(short)) + sizeof(CMD_OPTION));
				// write to net
				gameCtx->writeData((void *)&option, sizeof(CMD_OPTION));
				gameCtx->writeShort(value);
			}

			gameCtx->writeToNet();
		}else{
			cg::core::infoRecorder->logError("[LogicFactory]: not find the task '%p'.\n", id);
			return false;
		}
	}
	else{
		infoRecorder->logTrace("[LogicFactory]: unknown cmd: %s.\n", cmd);
		return false;
	}

	return ret;
}


//
LogicTask * LogicFactory::findTask(IDENTIFIER tid){
	infoRecorder->logTrace("[LogicFactory]: to find the task for %p.\n", tid);
	LogicTask * task = NULL;

	map<IDENTIFIER, LogicTask *>::iterator it = taskMap.find(tid);
	if(it != taskMap.end())
	{
		task = it->second;
	}
	else{
		infoRecorder->logError("[LogicFactory]: cannot find the task %p.\n", tid);
	}

	return task;
}

bool LogicFactory::addTask(LogicTask * task){
	infoRecorder->logTrace("[LogicFactory]: add task\n");
	IDENTIFIER id = task->id;
	map<IDENTIFIER, LogicTask *>::iterator it  = taskMap.find(id);
	if(it != taskMap.end()){
		infoRecorder->logError("[LogicFactory]: add task failed, task already exist.\n");
		return false;
	}
	else{
		taskMap[id] = task;
	}
	return true;
}
bool LogicFactory::removeTask(IDENTIFIER tid){
	infoRecorder->logTrace("[LogicFactory]: remove task for %p.\n", tid);
	map<IDENTIFIER, LogicTask *>::iterator it= taskMap.find(tid);
	if(it != taskMap.end()){
		taskMap.erase(tid);
		return true;
	}
	else{
		infoRecorder->logError("[LogicFactory]: remove task failed. task not exist.\n");
		return false;
	}

	return true;
}

// the regluator logic function
// return false if any errors
bool LogicFactory::regulationCall(){
	infoRecorder->logTrace("[LogicFactory]: regulation logic, the cpu increment:%d, the gpu increment:%d.\n", cpuIncreament, gpuIncreament);

	// collect the information
	float cpuUsage = 0.0f, gpuUsage = 0.0f, memUsage = 0.0f;
	collectInfo(cpuUsage, gpuUsage, memUsage);
	if(cpuUsage - gpuUsage > REGULATION_THRESHOLD){
		// cpu usage level is higher





	}
	else if(gpuUsage - cpuUsage > REGULATION_THRESHOLD){
		// gpu usage level is higher







	}
	else{
		// nothing happend
	}


	return true;
}

// the thread proc for regulation to adjust the CPU and GPU usage level
DWORD LogicFactory::RegulatorProc(LPVOID param){
	infoRecorder->logTrace("[LogicFactory]: regulator proc.\n");
	LogicFactory * factory = (LogicFactory *)param;

	if(!factory){
		// error
	}
	
	while(factory->isRunning()){
		// the factory is working, get the usage for CPU and GPU and determine what to do according to the usage level

		if(!factory->regulationCall()) // do the regulator logic
		{
			infoRecorder->logTrace("[LogicFactory]: regulation failed.\n");
			break;
		}

		// sleep for interval
		Sleep(factory->getRegulatorInterval());
	}
	infoRecorder->logTrace("[LogicFacotry]: to exit the regulation proc.\n");

	return 0;
}



///////////////// net work use libevent //////////////////

// bufferevent event callback
void onBufferEventEvent(struct bufferevent * bev, short events, void * ctx){
	printf("[OnBufferEventEvent]: event occur.\n");
	BaseContext * baseCtx = (BaseContext *)ctx;
	if(events & BEV_EVENT_ERROR){
		perror("Error from bufferevent.\n");
	}
	if(events & BEV_EVENT_EOF){
		bufferevent_free(bev);
		baseCtx->bev = NULL;
	}
}

char readBuffer[1024] = {0};
//
void onBufferEventRead(struct bufferevent * bev, void *ctx){
	struct evbuffer * input = bufferevent_get_input(bev);
	BaseContext * baseCtx = (BaseContext *)ctx;
	evutil_socket_t sock = baseCtx->sock;

	LogicFactory * logicFactory = LogicFactory::GetFactory();
	int data_len = evbuffer_get_length(input);
	

	evbuffer_copyout(input, readBuffer, data_len);
	printf("[OnBufferRead]: loader get '%s', context type:%d.\n", readBuffer, baseCtx->contextType);
	baseCtx->setData(readBuffer, data_len);
	memset(readBuffer, 0, 1024);
	logicFactory->dealCmd(baseCtx);
	evbuffer_drain(input, data_len);
}

void listenerCB(struct evconnlistener * listerner, evutil_socket_t sock, struct sockaddr * sddr, int len, void * ctx){
	printf("[ListenerCB]: callback for listen.\n");
	// we got a new connection ! Set up a bufferevent for it
	struct event_base * base = evconnlistener_get_base(listerner);
	struct bufferevent * bev = NULL;

	// sock is the new socket for connections
	LogicFactory * logic = LogicFactory::GetFactory();

	ManagerBase * mgr = (ManagerBase *)ctx;
	if(mgr->type == DIS_CONTEXT){
		bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE);
		BaseContext * disContext = new BaseContext();
		disContext->bev = bev;
		disContext->sock = sock;
		disContext->contextType = DIS_CONTEXT;
		RenderManager * mgr = (RenderManager *)ctx;
		mgr->addCtx(disContext);

		logic->addNetCtxMap(sock, disContext);

		bufferevent_setcb(bev, onBufferEventRead, NULL, onBufferEventEvent, (void *)disContext);
	}
	else if(mgr->type == PROCESS_CONTEXT){
		bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE);
		BaseContext * processCtx = new BaseContext();
		processCtx->bev = bev;
		processCtx->sock = sock;
		processCtx->contextType = PROCESS_CONTEXT;
		ProcessManager * mgr = (ProcessManager *)ctx;
		mgr->addCtx(processCtx);

		logic->addNetCtxMap(sock, processCtx);

		bufferevent_setcb(bev, onBufferEventRead, NULL, onBufferEventEvent, (void *)processCtx);
	}
	else if(mgr->type == RENDER_CONTEXT){
		// a render is connected, create the game process ?
		infoRecorder->logError("[ListenerCB]: render connected.\n");
		BaseContext * renderCtx = new BaseContext();
		//renderCtx->bev = bev;
		renderCtx->sock = sock;
		renderCtx->contextType = RENDER_CONTEXT;
		RenderManager * mgr = (RenderManager *)ctx;
		mgr->addCtx(renderCtx);

		logic->addNetCtxMap(sock, renderCtx);

		u_long ul = 0;
		if(ioctlsocket(sock, FIONBIO, (u_long *)&ul) == SOCKET_ERROR){
			infoRecorder->logError("[ListenerCB]: set socket to blocking mode failed with:%d.\n", WSAGetLastError());
			printf("[ListenerCB]: set socket to blocking mode failed with:%d.\n", WSAGetLastError());
		}
		//bufferevent_setcb(bev, onBufferEventRead, NULL, onBufferEventEvent, (void *)renderCtx);
		// get the ADD_RENDER CMD from render
		char msg[1024] = {0};
		int dlen = 0;
		if((dlen = recv(sock, msg, 1024, 0))!= SOCKET_ERROR){
			renderCtx->setData(msg, dlen);
			logic->dealCmd(renderCtx);
		}else{
			infoRecorder->logError("[ListenerCB]: failed to recv command from render proxy, code:%d.\n", WSAGetLastError());
			Log::slog("[ListenerCB]: failed to recv command from render proxy, code:%d.\n", WSAGetLastError());
		}

	}else if(mgr->type == CTRL_CONTEXT){
		infoRecorder->logError("[ListenerCB]: ctrl connected.\n");
		BaseContext * ctrlCtx = new BaseContext();
		ctrlCtx->sock = sock;
		ctrlCtx->contextType = CTRL_CONTEXT;
		CtrlManager * cmgr = (CtrlManager *)ctx;
		logic->addNetCtxMap(sock, ctrlCtx);

		u_long ul = 0;
		if(ioctlsocket(sock, FIONBIO, (u_long *)&ul) == SOCKET_ERROR){
			infoRecorder->logError("[ListenerCB]: set socket to blocking mode failed with:%d.\n", WSAGetLastError());
			printf("[ListenerCB]: set socket to blocking mode failed with:%d.\n", WSAGetLastError());
		}
	}
	if(bev)
		bufferevent_enable(bev, EV_READ );
}

void acceptErrorCB(struct evconnlistener * listener, void *ctx){
	struct event_base * base = evconnlistener_get_base(listener);
	int err = EVUTIL_SOCKET_ERROR();
	fprintf(stderr, "Got an error %s (%s) on the listener." "Shutting down.\n", err, evutil_socket_error_to_string(err));
	event_base_loopexit(base, NULL);
}


//// callback for rtsp connection
void RTSPReadCB(struct bufferevent * bev, void * arg){
	printf("[RTSPReadCB]: get rtsp cmd.\n");
	struct evbuffer * input = bufferevent_get_input(bev);
	//DisClient * disClient = (DisClient *)arg;
	size_t n = evbuffer_get_length(input);

	char * data = (char *)malloc(sizeof(char )*n);
	evbuffer_copyout(input, data, n);

	// get the describe
	if(data[0] == '$'){
		// ignore
		printf("[RTSPReadCB]: ignore '%s'\n", data);
		goto DONE;
	}
	// REQUEST line
	printf("[RTSPReadCB]: deal '%s'\n", data);

DONE:
	free(data);
	data = NULL;
	evbuffer_drain(input, n);
}

void RTSPEventCB(struct bufferevent * bev, short what, void * arg){
	printf("[RTSPEventCB]: event occurred.\n");
	if(what & BEV_EVENT_ERROR){
		int err = EVUTIL_SOCKET_ERROR();
		printf("[RTSPEventCB]: error. err:%d (%s)\n", err, evutil_socket_error_to_string(err));
		infoRecorder->logError("[RTSPEventCB]: error. err:%d (%s)\n", err, evutil_socket_error_to_string(err));

	}
	if(what & BEV_EVENT_EOF){
		printf("[RTSPEventCB]: EOF of the connection.\n");
		infoRecorder->logError("[RTSPEventCB]: EOF of the connection.\n");
	}

	bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
	bufferevent_disable(bev, EV_READ|EV_WRITE);
	bufferevent_free(bev);
}

void RTSPListenerCB(struct evconnlistener * listener, evutil_socket_t sock, struct sockaddr * addr, int len, void * ctx){
	printf("[RTSPListenerCB]: listener callback, got a new connection.\n");
	DisClient * dc = (DisClient *)ctx;
	dc->startRTSP(sock);
}

void RTSPAcceptErrorCB(struct evconnlistener * listener, void * ctx){
	printf("[RTSPAcceptErrorCB]: error for listen.\n");
	struct event_base * base = evconnlistener_get_base(listener);
	int err = EVUTIL_SOCKET_ERROR();
	fprintf(stderr, "Got an error %d (%s) on the listener." "Shutting down.\n", err, evutil_socket_error_to_string(err));
	event_base_loopexit(base, NULL);
}


LoaderLogger::LoaderLogger(std::string _processName):
processName(_processName), processHandle(NULL), mappingHandle(NULL), mutexHandle(NULL), cpuWatcher(NULL), gpuWatcher(NULL), recorder(NULL){
	// create the file mapping and the mutex
	std::string mappingName = processName + string("_mapping");
	std::string mutexName = processName + string("_mutex");

	printf("[LoaderLogger]: create logger with process name:%s, mapping name:%s, mutex name:%s.\n", processName.c_str(), mappingName.c_str(), mutexName.c_str());

	//mutexHandle = CreateMutex(NULL, FALSE, mutexName.c_str());
	mutexHandle = CreateEvent(NULL, FALSE, FALSE, mutexName.c_str());
	mappingHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, mappingName.c_str());
	if(mappingHandle){
		// open mapping success.
		mappingAddr = MapViewOfFile(mappingHandle, FILE_MAP_ALL_ACCESS, 0, 0 ,0);
	}
	else{
		// to create new
		mappingHandle = CreateFileMapping((HANDLE)0xFFFFFFFF, NULL, PAGE_READWRITE, 0, 64, mappingName.c_str());
		mappingAddr = MapViewOfFile(mappingHandle, FILE_MAP_ALL_ACCESS, 0, 0 ,0);
	}
}
LoaderLogger::~LoaderLogger(){
	if(mappingAddr){
		UnmapViewOfFile(mappingAddr);
		mappingAddr  = NULL;
	}
	if(mutexHandle){
		CloseHandle(mutexHandle);
		mutexHandle = NULL;
	}
	if(mappingHandle){
		CloseHandle(mappingHandle);
		mappingHandle = NULL;
	}
}
bool LoaderLogger::initLogger(){
	
	
	// create the watcher
	cpuWatcher = new CpuWatch();
	gpuWatcher = GpuWatch::GetGpuWatch();
	// create the recorder
	recorder = new LightWeightRecorder((char*)processName.c_str());
	printf("[LoaderLogger]: init logger, create the recorder file: %p.\n", recorder);
	return true;

}
void LoaderLogger::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){

}
BOOL LoaderLogger::onThreadStart(){
	initLogger();
	return TRUE;
}
BOOL LoaderLogger::run(){
	printf("loader logger run...\n");
	DWORD ret = WaitForSingleObject(mutexHandle, 3000);
	if(ret == WAIT_OBJECT_0){
		// read the fps in shared file mapping
		int index = *(int*)mappingAddr;
		int fps = *(((int*)mappingAddr)+1);
		float cpuUsage = (float)cpuWatcher->GetProcessCpuUtilization(processHandle);
		int gpuUsage = gpuWatcher->GetGpuUsage();
		recorder->log("%d %d %f %d\n", index, fps, cpuUsage, gpuUsage);
		printf("%d %d %f %d\n", index, fps, cpuUsage, gpuUsage);
		recorder->flush(true);
	}
	else if(ret == WAIT_TIMEOUT){
		printf("time out.\n");
	}else{
		printf("unknown: %d.\n", ret);
	}
	return TRUE;
}
void LoaderLogger::onQuit(){

}
#endif