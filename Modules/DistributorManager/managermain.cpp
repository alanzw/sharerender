#include "ccg_win32.h"
#include "ccg_config.h"
#include "cthread.h"
#include <queue>

#include "disnetwork.h"
#include "distributormanager.h"
#include "log.h"

// the main entry,
/*
listen port and wait clients
*/

int main(int argc, char ** argv){
	DistributorServerManager * manager = NULL;

	// init the log
	Log::init("distributor.log");

	if (argc == 2){
		manager = new DistributorServerManager(argv[1]);
	}
	else if (argc == 1){
		// use default configure file
		manager = new DistributorServerManager(SERVER_DIS_CONFIG_FILE);
	}
	else{
		printf("should never be here.\n");
	}

	// start the network
	WORD version;
	WSAData wsaData;
	//version = MAKEWORD(1, 1);
	version = MAKEWORD(2, 2);

	int err = WSAStartup(version, &wsaData);
	if (err){
		Log::log("[DistributorManager]: socket start failed.\n");
		return -1;
	}
	else{
		Log::log("[DistributorManager]: socket start success.\n");
	}

	//init the manager
	manager->init();

	// start the logic server and render server thread
	manager->startUp();

	HANDLE clientEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	HANDLE userEvent = manager->getUserServer()->getNotifier();
	HANDLE logicEvent = manager->getLogicServer()->getNotifier();
	HANDLE renderEvent = manager->getRenderServer()->getNotifier();
	HANDLE taskEvent = manager->getTaskManager()->getNotifier();   // task manager notifier
	HANDLE eventList[] = { userEvent, logicEvent, renderEvent, taskEvent };

	DWORD waitRet = 0;
	bool isRunning = true;
	while (isRunning){
		// enter the manage loop
		// waith the event
		waitRet = WaitForMultipleObjects(4, eventList, FALSE, 5000);
		switch (waitRet){
		case WAIT_FAILED:
			// bad call to funtion (invalid handle?)
			Log::slog("[DistributorManager main]: wait for multiple obejct failed.\n");
			
			break;
		case WAIT_TIMEOUT:
			// none of the object became signaled within 5000 milliseconds
			Log::log("[DistirbutorManager mian]: wait for multiple object time out.\n");
			break;
		case WAIT_OBJECT_0 + 0:
			// an user event cames, always mean that a client has sent a game request.
			Log::log("[DistributorManager main]: deal the client request.\n");
			manager->dealClientGameReq();
			break;
		case WAIT_OBJECT_0 + 1:
			// a logic server event cames
			Log::log("[DistributorManager main]: deal the logic server's feedback.\n");
			manager->dealServerReq(manager->getLogicServer());
			break;
		case WAIT_OBJECT_0 + 2:
			//  a render server event cames
			Log::log("[DistributorManager main]: deal the render server's feedback.\n");
			manager->dealServerReq(manager->getRenderServer());
			break;
		case WAIT_OBJECT_0 + 3:
			// a task event signaled.
			Log::log("[DistributorManager main]: deal the task event.\n");
			manager->dealTask();
			break;
		default:
			// nothing 

			break;
		}

		isRunning = manager->isRunning();
		Sleep(1);
		//Log::log("")
	}

	//destory manager
	if (manager){
		delete manager;
		manager = NULL;
	}

	return 0;
}