#include "gameloader.h"
#include "../LibCore/InfoRecorder.h"
#include "../VideoUtility/rtspconf.h"
//#include "../VideoUtility/videocommon.h"
//#define INCLUDE_DIS_CLIENT   // include the dis client for game loader, the game loader will connect the dis server and start the game, otherwise, start a given game, wait render to connect

GameInfo * GameInfo::gameInfo = NULL;

void printHelp(){
	// the game loader can work in many modes
	printf("GameLoader --help or GameLoader -h\n");



}

DWORD WINAPI TestRequestFromRenderProxy(LPVOID param){

	char args[1024] = {0};
	char buf[1024] = {0};
	SOCKET sock = (SOCKET)param;


	GameLoader * loader = GameLoader::GetLoader();
	// the args is 
	DWORD processId = GetCurrentProcessId();
	int len =  recv(sock, buf, 1024, 0);
	if(len > 0){
		printf("To start game: %s.\n", buf);
	}
	sprintf(args, " -m 0 -a %d -p %d", sock, processId);

	HANDLE gameProcess = loader->loadGame(buf, args);

	return 0;
}

static void WaitRenderProxy(int port){
	// listen the port
	sockaddr_in sin, clientSin;
	int sinSize = sizeof(sin);
	memset(&sin, 0, sinSize);
	memset(&clientSin, 0, sinSize);
	sin.sin_family = AF_INET;
	sin.sin_addr.S_un.S_addr = htonl(0);
	sin.sin_port = htons(port);

	GameLoader * loader = GameLoader::GetLoader();
	DWORD processId = GetCurrentProcessId();

	SOCKET listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	if(bind(listenSock, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		printf("[WaitRenderProxy]: bind error.\n");
		return;
	}
	if(listen(listenSock, 5) < 0){
		printf("[WaitRenderProxy]: listen failed.\n");
		return;
	}

	SOCKET sock = NULL;
	HANDLE process[100] = {NULL};
	int processCount = 0;
	DWORD threadId = 0;
	char args[1024] = {0};
	char buf[1024] = {0};

	while(1){
		printf("[WaitRenderProxy]: listen port:%d, accept ...\n", port);
		sock = accept(listenSock, (sockaddr *)&clientSin, &sinSize);
		if(sock < 0){
			printf("[WaitRenderProxy]: accept failed.\n");
			return;
		}
		else{
			printf("[WaitRenderProxy]: a render has connected.\n");
		}

		int len =  recv(sock, buf, 1024, 0);
		if(len > 0){
			printf("To start game: %s.\n", buf);
		}
		sprintf(args, " -m 0 -a %d -p %d", sock, processId);

		HANDLE gameProcess = loader->loadGame(buf, args);

		process[processCount++] = gameProcess;
	}

	DWORD ret = WaitForMultipleObjects(processCount, process, TRUE, INFINITE);
	
	return;
}

static void dealCmd(int argc, char **argv){
	int mode = 0;
	char configFile[64];


	char * args = (char *)malloc(sizeof(char) * 1024);
	char gameMapFile[50] = {0};
	bool useMap = false;
	bool useDll = false;
	bool gameNameReady = false;
	char externDllName[50] = {0};
	char gameName[50] = {0};
	int port = 6000;
	memset(args, 0, 1024);

	cg::RTSPConf * rtspConf = NULL;
	int encodeOption = 1;

	strcpy(configFile, "config/server.logic.conf");

	for(int i = 0; i < argc; i++){
		if(!strcmp(argv[i], "-m") || !strcmp(argv[i], "-M")){
			// the work mode, mode 0 only listen the render proxy, mode 1 start the logic server standalone, mode 2 work in distributed mode, mode 3 is test mode, that accept the render proxy but do the rendering as well and stores the image to file
			mode = atoi(argv[i+1]);
		}
		else if(!strcmp(argv[i], "-c") || !strcmp(argv[i], "-C")){
			// rtsp config file anme
			strcpy(configFile, argv[i+1]);
		}
		else if(!strcmp(argv[i], "-e") || !strcmp(argv[i], "-E")){
			encodeOption = atoi(argv[i+1]);	
		}
		// for mode 1
		else if(!strcmp(argv[i], "-g") || !strcmp(argv[i], "-G")){
			// game map
			strcpy(gameMapFile, argv[i + 1]);
			useMap = true;
		}
		else if(!strcmp(argv[i], "-d") || !strcmp(argv[i], "-D")){
			// use dll name
			useDll = true;
			strcpy(externDllName, argv[i + 1]);
			strcat(args, " ");
			strcat(args, argv[i]);
			strcat(args, " ");
			strcat(args, argv[i+1]);
		}
		else if(!strcmp(argv[i], "-a") || !strcmp(argv[i], "-A")){
			// the request game name
			strcpy(gameName, argv[i+1]);
			gameNameReady = true;
			strcat(args, " ");
			strcat(args, argv[i]);
			strcat(args, " ");
			strcat(args, argv[i+1]);
		}
		else if(!strcmp(argv[i], "-p") || !strcmp(argv[i], "-P")){
			port = atoi(argv[i+1]);
		}
	}

	if(mode != 1){
		WSADATA WSAData;
		WSAStartup(0x101, &WSAData);
	}

	LogicFactory * factory = NULL;
	GameLoader * loader = NULL;
	HANDLE gameProcess = NULL;
	LoaderLogger * loaderLogger = NULL;
	DWORD ret = 0;

	switch(mode){
	case 0:  // only listen the render proxy, but the render proxy is added after the whole game is running
		factory = LogicFactory::GetFactory();
		factory->init();
		// only listen the render
		factory->startRenderListen();
		factory->enterLoop();
		break;

	case 1: // server standalone
		{
			// init the loader
			loader = GameLoader::GetLoader(useMap ? gameMapFile : NULL);
			if(gameNameReady){
				printf("[Main]: will start the game with cmd line:%s %s.\n", \
				argv[1], args);

				loaderLogger = new LoaderLogger(gameName);
				gameProcess = loader->loadGame(gameName, args, useDll ? externDllName : NULL);

				// set the process handle
				loaderLogger->setProcessHandle(gameProcess);
				loaderLogger->start();
			}
			else{
				cout << "[Main]: cannot find the game name in the args. To exit." << endl;
			}

			//use extern logger to record the usage
			ret = WaitForSingleObject(gameProcess, INFINITE);
			if(ret == WAIT_OBJECT_0){
				loaderLogger->stop();
				delete loaderLogger;
				loaderLogger = NULL;
			}

			// free resources
			free(args);
			args = NULL;
		}
		break;

	case 2: // distributed mode

		factory = LogicFactory::GetFactory();
		factory->init();
		rtspConf = cg::RTSPConf::GetRTSPConf(configFile);
		factory->setRTSPConf(rtspConf);
		factory->setEncoderOption(encodeOption);
		//factory->connectDis(argv[1], DIS_PORT_DOMAIN);
		factory->connectDis(rtspConf->getDisUrl(), rtspConf->disPort);
		factory->registerLogic();
		// enter the network dealing loop
		factory->startListen();
		// dispatch
		factory->enterLoop();
		break;

	case 3:   // test mode, add the render proxy from the beginning.

		// reuse the args
		WaitRenderProxy(port);
		break;
	default:
		printf("You must specific the working mode via -m or -M, (0-3 is candidate, refer to the help.\n");
		printHelp();
	}

}


int main(int argc, char ** argv){
#if 0
	// debug the GameInfo
	GameInfo * info = GameInfo::GetGameInfo();
	if(!info->loadInfo()){
		printf("[main]: game info load failed.\n");
		getchar();
		return -1;
	}
	info->showAllInfo();

#else

	dealCmd(argc, argv); 
	return 0;

	char * args = (char *)malloc(sizeof(char) * 1024);
	char gameMapFile[50] = {0};
	bool useMap = false;
	bool useDll = false;
	bool gameNameReady = false;
	char externDllName[50] = {0};
	char gameName[50] = {0};
	memset(args, 0, 1024);

	infoRecorder = new InfoRecorder("Loader");
	infoRecorder->init();
	// choose the work mode according to argv
	if(argc == 1){
		// only listen the render proxy
		WSADATA WSAData;
		WSAStartup(0x101, &WSAData);

		// debug the logic factory
		LogicFactory * factory = LogicFactory::GetFactory();
		factory->init();
		// only listen the render
		factory->startRenderListen();
		factory->enterLoop();
	}
	else if(argc > 4){
		// got another start option
		// argv[1] is the game name
		for(int i = 1; i < argc; i++){
			// filter the loader args
			if(!strcmp(argv[i], "-g")){
				// the game map file
				strcpy(gameMapFile, argv[i + 1]);
				useMap = true;
			}
			else if(!strcmp(argv[i], "-d")){
				// use dll
				useDll = true;
				strcpy(externDllName, argv[i + 1]);
			}
			else if(!strcmp(argv[i], "-a")){
				// the game name
				strcpy(gameName, argv[i+1]);
				gameNameReady = true;
			}
			strcat(args, " ");
			strcat(args, argv[i]);
		}
		printf("[Main]: start the local game process alone, work with Form ????.\n");

		// init the loader
		GameLoader * loader = NULL;
		if(useMap){
			loader = GameLoader::GetLoader(gameMapFile);
		}
		else{
			loader = GameLoader::GetLoader();
		}

		HANDLE gameProcess = NULL;
		LoaderLogger * loaderLogger = NULL;

		if(gameNameReady){
			printf("[Main]: will start the game with cmd line:%s %s.\n", \
				argv[1], args);

			loaderLogger = new LoaderLogger(gameName);

			if(!useDll)
				gameProcess = loader->loadGame(gameName, args);
			else
				gameProcess = loader->loadGame(gameName, args, externDllName);

			// set the process handle
			loaderLogger->setProcessHandle(gameProcess);
			loaderLogger->start();
		}
		else{
			cout << "[Main]: cannot find the game name in the args. To exit." << endl;
		}

		//use extern logger to record the usage
		DWORD ret = WaitForSingleObject(gameProcess, INFINITE);
		if(ret == WAIT_OBJECT_0){
			loaderLogger->stop();
			delete loaderLogger;
			loaderLogger = NULL;
		}

		// free resources
		free(args);
		args = NULL;
	}
	else{
#if 1
		// start the network
		WSADATA WSAData;
		WSAStartup(0x101, &WSAData);
		//infoRecorder = new InfoRecorder("Logic");
		// debug the logic factory
		LogicFactory * factory = LogicFactory::GetFactory();
		factory->init();
		cg::RTSPConf * conf = NULL;
		// if argc == 2, means that argv[1] is the config file
		if(argc == 2){
			// set encoder option
			conf = cg::RTSPConf::GetRTSPConf(argv[1]);
			factory->setRTSPConf(conf);
			factory->setEncoderOption(1);
			//factory->connectDis(argv[1], DIS_PORT_DOMAIN);
			factory->connectDis(conf->getDisUrl(), conf->disPort);
			factory->registerLogic();
			// enter the network dealing loop
			factory->startListen();
		}
		else if(argc == 3){
			// set encoder option
			// argv[1] is config file
			conf = cg::RTSPConf::GetRTSPConf(argv[1]);
			factory->setRTSPConf(conf);
			factory->setEncoderOption(atoi(argv[2]));
			//factory->connectDis(argv[1], DIS_PORT_DOMAIN);
			factory->connectDis(conf->getDisUrl(), conf->disPort);
			factory->registerLogic();
			// enter the network dealing loop
			factory->startListen();
		}
		// REGISTER logic server


#ifdef INCLUDE_DIS_CLIENT
		// set up the dis part
		loader->connectDis();
#else
		// to debug, start the game process 

#endif

		// dispatch
		factory->enterLoop();
#else
		printf("[LogicServer]: enter the logic server.\n");
		printf("[LogicServer]: to load the game information.\n");

		GameInfo * info = GameInfo::GetGameInfo();
		if(!info->loadInfo()){
			printf("[LogicServer]: game info load failed.\n");
			getchar();
			return -1;
		}
		info->showAllInfo();



		printf("[LogicServer]: to start the network.\n");
		LogicServer * logic = LogicServer::GetLogicServer();

		event_base * base = event_base_new();

		// start the logic server
		logic->start();

		logic->startGraphic();
		logic->startListenGameProcess();
		// dispatch all

		logic->dispatch();

		printf("[LogicServer]: before exit, to clean up.\n");
		getchar();
		getchar();


#endif
#endif

		GameLoader * loader = GameLoader::GetLoader();


	}
	return 0;
}