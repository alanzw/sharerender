#include "gameloader.h"
#include "../LibCore/InfoRecorder.h"
//#include "../VideoUtility/videocommon.h"
//#define INCLUDE_DIS_CLIENT   // include the dis client for game loader, the game loader will connect the dis server and start the game, otherwise, start a given game, wait render to connect

GameInfo * GameInfo::gameInfo = NULL;

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
	infoRecorder = new InfoRecorder("Logic");
	// debug the logic factory
	LogicFactory * factory = LogicFactory::GetFactory();
	factory->init();
	if(argc == 2){
		// set encoder option
		factory->setEncoderOption(1);
		factory->connectDis(argv[1], DIS_PORT_DOMAIN);
		factory->registerLogic();
		// enter the network dealing loop
		factory->startListen();
	}
	else if(argc == 3){
		// set encoder option
		
		factory->setEncoderOption(atoi(argv[2]));
		factory->connectDis(argv[1], DIS_PORT_DOMAIN);
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