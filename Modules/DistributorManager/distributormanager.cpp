#include "ccg_win32.h"
#include "ccg_config.h"
#include "cthread.h"
#include <queue>
#include "disnetwork.h"
#include "distributormanager.h"
#include "log.h"

TaskManager::TaskManager(){
	InitializeCriticalSection(&section);
	pSection = &section;
	totalTasks = 0;
	
	notifier = CreateEvent(NULL, FALSE, FALSE, NULL);
}

TaskManager::~TaskManager(){
	if (pSection){
		DeleteCriticalSection(&section);
		pSection = NULL;
	}
	if (notifier){
		CloseHandle(notifier);
		notifier = NULL;
	}
}

bool TaskManager::addTask(Task * task){
	allTaskMap[task->client->s] = task;
	taskQueue.push(task);
	return true;
}

bool TaskManager::changeStatus(SOCKET s, TASK_STATUS status){
	if (allTaskMap[s]->status != status){
		allTaskMap[s]->status = status;
		return true;
	}
	else{
		return false;
	}
}

TASK_STATUS TaskManager::getTaskStatus(SOCKET s){
	return allTaskMap[s]->status;
}

Task * TaskManager::popTask(){
	Task * ret = taskQueue.front();
	taskQueue.pop();
	return ret;

}

int TaskManager::allMoveOneStep(){
	// do nothing

	return 0;
}

Task * TaskManager::getAssignedTask(SOCKET s){
	return allTaskMap[s];
}

bool DistributorServerManager::isRunning(){
	bool ret = true; 
	EnterCriticalSection(pSection);
	ret = running; 
	LeaveCriticalSection(pSection); 
#if 0
	Log::log("[DistributorServerManager]: running = %d.\n", ret);
	printf("is runing:%d\n", ret);
#endif
	return ret;
}

///////////////// for distributor server manager ///////////////////
DistributorServerManager::DistributorServerManager(char * configFileName){
	if (configFileName == NULL){
		conf = new DistributorConfig(SERVER_DIS_CONFIG_FILE);
	}
	else{
		conf = new DistributorConfig(configFileName);
	}
	userServer = new UserServer();
	logicServer = new DistributorServer();
	renderServer = new DistributorServer();
	if (conf == NULL || userServer == NULL || logicServer == NULL || renderServer == NULL){
		Log::log("[DistributorServerManager]: constructor failed.\n");
	}

	//pSection = NULL;
	logicServers = 0;
	renderServers = 0;
	users = 0;
	tasks = 0;

	running = true;

	taskManager = new TaskManager();
	//notifier = CreateEvent(NULL, FALSE, FALSE, NULL);

	InitializeCriticalSection(&section);
	pSection = &section;

}

DistributorServerManager::~DistributorServerManager(){
	if (conf){
		delete conf;
		conf = NULL;
	}
	if (userServer){
		delete userServer;
		userServer = NULL;
	}
	if (logicServer){
		delete logicServer;
		logicServer = NULL;
	}
	if (renderServer){
		delete renderServer;
		renderServer = NULL;
	}
	if (taskManager){
		delete taskManager;
		taskManager = NULL;
	}
}

bool DistributorServerManager::startUp(){
	bool ret = false;
	userServer->startThread();
	logicServer->startThread();
	renderServer->startThread();

	this->startWatchdog();
	return true;
}

void DistributorServerManager::checkManager(){
	EnterCriticalSection(pSection);
	/*
	if (logicServers == 0 && renderServers == 0 && users == 0)
		running = false;
	*/
	LeaveCriticalSection(pSection);
}

int DistributorServerManager::startWatchdog(){
	watchdogThread = chBEGINTHREADEX(NULL, 0, Watchdog, this, FALSE, &watchdogThreadId);
	return 0;
}

DWORD WINAPI DistributorServerManager::Watchdog(LPVOID param){
	DistributorServerManager * manager = (DistributorServerManager *)param;
	if (manager == NULL){
		Log::slog("[DistributorServerManager]: NULL manager specified for watchdog.\n");

		return -1;
	}
	while (manager->isRunning()){
		manager->collectInfomation();
		Sleep(manager->getSleepInterval());
		manager->checkManager();
	}
	return 0;
}

// init the distributor server manager
bool DistributorServerManager::init(){
	bool ret = false, ret1 = false, ret2 = false;

	ret  = this->userServer->init(conf->getUserPort());
	ret1 = this->logicServer->init(conf->getLogicPort());
	ret2 = this->renderServer->init(conf->getRenderPort());

	this->sleepInterval = 5000;
	this->sleepInterval = conf->getTimter();
	return ret;
}

// collecte the server's information, send the command
bool DistributorServerManager::collectInfomation(){
	logicServer->collectInfomation();
	renderServer->collectInfomation();
	return true;
}

// when client recved logic url and the render url, the client will feedback
// the user server will notify the render to request logic, 
int DistributorServerManager::dealClientFeedback(){

	return 0;
}

// deal with server req
int DistributorServerManager::dealServerReq(DistributorServer * server){
#if 0
	DISSERVER_EVENT evet = server->getEventType();
	//get the server req
	switch (evet){
	case DISSERVER_EVENT::DIS_CLOSE:
		// server notified the manager, the server will be closed.

		break;
	case DISSERVER_EVENT::DIS_GRAPHIC:
		// logic has established graphic connection

		break;
	case DISSERVER_EVENT::DIS_INFO:
		// the server send back the machine info

		break;
	default:
		break;
	}
#else
	//deal the request in reqEngine
	char cmd[512];
	char buf[512];
	int eventCount = 0;
	ClientReq * req = NULL;
	// find the server's request
	WaitForSingleObject(server->getEventTrigger(), INFINITE);
	eventCount = server->reqEngine->getReqCount();
	for (int i = 0; i < eventCount; i++){
		req = server->reqEngine->getReqAndRemove();
		if (req){
			if (!strncasecmp(req->reqInfo, LOGIC_STARTED, strlen(LOGIC_STARTED))){
				// the logic server feed back, game process started.
				// find the task and change the task status
				char * p = req->reqInfo + strlen(LOGIC_STARTED);
				SOCKET clientSock = *(SOCKET *)p;
				Log::slog("[DistributorServerManager]: logic started, client socket:%p.\n", clientSock);
				taskManager->changeStatus(clientSock, TASK_STATUS::ASSIGNED);
				// send the start client cmd to user
				Task * task = taskManager->getAssignedTask(clientSock);

				// format the START_CLIENT cmd
				// [START_CLIENT:renderCount+url1+url2+..+]
				int sendLen = 0;
				strcpy(cmd, START_CLIENT);
				p = cmd + strlen(START_CLIENT);
				sendLen += strlen(START_CLIENT);

				unsigned short * sp = (unsigned short *)p;
				*sp = task->renderCount;
				p += sizeof(unsigned short);
				*p = '+';
				sendLen += sizeof(unsigned short)+1;
				p++;
				for (int j = 0; j < task->renderCount; j++){
					sprintf(p, "%s+", task->renderServer[j]->url);
					p += strlen(task->renderServer[j]->url) + 1;
					sendLen += strlen(task->renderServer[j]->url) + 1;
				}
				Log::slog("[DistributorServerManager]: send START CLIENT cmd to client, len:%d.\n", sendLen);
				send(clientSock, cmd, sendLen, 0);
			}
			else if (!strncasecmp(req->reqInfo, RENDER_STARTED, strlen(RENDER_STARTED))){
				// the render started feed back, the render has been running.
				// find the task and change the status
				char * p = req->reqInfo + strlen(RENDER_STARTED);
				SOCKET clientSock = *(SOCKET *)p;
				Log::slog("[DistributorServerManager]: render started, client socket:%p.\n", clientSock);
				if (!taskManager->changeStatus(clientSock, TASK_STATUS::STARTED)){
					// notify the user
					Log::slog("[DistributorServerManager]: send render started cmd to client:%p.\n", clientSock);
					send(clientSock, RENDER_STARTED, strlen(RENDER_STARTED), 0);

				}
				else{
					Log::slog("[DistributorServerManager]: render already started. may error.\n");
				}
			}
		}
	}
#endif

	return 0;
}

// deal the task in the task manager
int DistributorServerManager::dealTask(){
	char cmd[100];
	char buf[100];
	char logicUrl[100] = { 0 };
	char renderUrl[100] = { 0 };
	NetDomain * logicSer = NULL, *renderSer = NULL;
	int recved = 0;
	// all task in taskQueue is INIT, or may error
	int taskToDeal = this->taskManager->getUndoneTaskCount();
	for (int i = 0; i < taskToDeal; i++){
		Task * toDo = taskManager->popTask();
		if (toDo == NULL){
			Log::log("[DistributorServerManager]: get NULL from queue.\n");
			return -1;
		}
		// assign servers to task
		if (toDo->status == TASK_STATUS::INIT){
			logicSer = findLogicCandidateDomain(toDo->require);
			int renders = findRenderCandidateDomain(toDo->require,1, &(toDo->renderServer));

			if (NULL == logicSer){
				Log::log("[DistributorServerManager]: get NULL logic server candidate.\n");
				return -1;
			}
			if (renders <= 0){
				Log::log("[DistributorserverManager]: get 0 render server candidate.\n");
				return -1;
			}

			toDo->logicServer = logicSer;
			// render server has been set
			toDo->renderCount = renders;

			// add the task to server's task list
			logicServer->getServerNode(logicSer->s)->taskList.push_back(toDo);
			
			/////////start the servers directly? new idea
			// 
			// start logic command: [START_LOGIC:gameName+SOCKET+UINT+url+url+url+url+...]
			//strcpy(cmd, START_LOGIC);
			char * cur = cmd;
			int size = 0;
			sprintf(cmd, "%s%s+", START_LOGIC, toDo->taskName);

			//(cmd + strlen(cmd));
			cur = cmd + strlen(cmd);
			size += strlen(cmd);

			SOCKET * pSock = (SOCKET *)cur;
			*pSock = toDo->client->s;
			cur += sizeof(SOCKET);
			size += sizeof(SOCKET);
			*cur = '+';
			cur++;
			size++;
			//char * cur = 
			unsigned short * sp = (unsigned short *)cur;
			*sp = (unsigned short)renders;

			char * p = cur + sizeof(unsigned short);
			size += sizeof(unsigned short);

			*p = '+';
			size++;
			p++;
			//send(logicSer->s, cmd, strlen(cmd), 0);
			for (int i = 0; i < toDo->renderCount; i++){
				strcpy(p, toDo->renderServer[i]->url);
				p += strlen(toDo->renderServer[i]->url);
				size += strlen(toDo->renderServer[i]->url);
				*p = '+';
				p++;
				size++;

				//sprintf(cmd, "%s:%s ", RENDER, toDo->renderServer[i]->url);
				//send(logicSer->s, cmd, strlen(cmd), 0);
				renderServer->getServerNode(toDo->renderServer[i]->s)->taskList.push_back(toDo);
			}
			
			Log::slog("[DistributorServerManager]: to send start logic command to logic server, dst socket:%p.\n", logicSer->s);
			// send, cmd format [RENDER:count+url1+url2+...
			send(logicSer->s, cmd, size, 0);
#if 0
			//////// wait the servers' feedback

			recved = recv(logicSer->s, buf, sizeof(buf), 0);
			if (recved <= 0){
				Log::log("[DistributorServerManager]: wait logic server feedback failed.\n");
			}
			// format the command and send

			// tell client, the logic server and the render server
			// client cmd like [LOGIC:url] [RENDER:count] [RENDER:url] [RENDER:url] ..., when get all, client will send a feedback, when manager get the 
			// feedback, manager will let the render know, start graphic stream
			// the render server will notify manager when create window.
			// then, manager notify client to start request
			// [consider the contorller config]
			strcpy(cmd, LOGIC);
			strcat(cmd, toDo->logicServer->url);
			send(toDo->client->s, cmd, strlen(cmd), 0);

			strcpy(cmd, RENDER);
			sprintf(cmd, "%s:%d ", RENDER, toDo->renderCount);
			send(toDo->client->s, cmd, strlen(cmd), 0);

			for (int i = 0; i < toDo->renderCount; i++){
				sprintf(cmd, "%s:%s ", RENDER, toDo->renderServer[i]);
				send(toDo->client->s, cmd, strlen(cmd), 0);
			}
			//strcat(cmd, toDo->renderServer);

			taskManager->changeStatus(toDo->client->s, TASK_STATUS::ASSIGNED);
#endif
		}
		else{
			Log::slog("[DistributorServerManager]: task status is not init.\n");

		}
	}
	// check every task, to move one step farther
	//taskManager->allMoveOneStep();
}

// deal the client game request
int DistributorServerManager::dealClientGameReq(){
	int eventCount = 0;
	int newTask = 0;
	char gameName[100] = { 0 };
	char logicUrl[100] = { 0 };
	char renderUrl[100] = { 0 };

	SOCKET s = NULL;
	UserServer * user = this->userServer;
	if (user == NULL){
		Log::log("[DistributorServerManager]: get NULL user server.\n");
		return -1;
	}
	ClientReq * req = NULL;
	// find the client request.
	WaitForSingleObject(getUserServer()->getEventTrigger(), INFINITE);
	eventCount = getUserServer()->reqEngine->getReqCount();
	char cmd[100];
	Task * temTask = NULL;
	for (int i = 0; i < eventCount; i++){
		// assign logic server and render server
		req = getUserServer()->reqEngine->getReqAndRemove();
		if (req){
			if (!strncasecmp(req->reqInfo, REQ_GAME, strlen(REQ_GAME))){
				// requst like this: REQ_GAME:gameName
				strcpy(gameName, req->reqInfo + strlen(REQ_GAME) + 1);
				Log::slog("[DistributorServerManager]: client request game '%s', client socket: %p.\n", gameName, req->s);
				
				ResourceRequire * usage = estimateResourceOccupy(gameName);
#if 0
				s = findLogicCandidate(usage);
				char * url = logicServer->serverUrlMap.find(s)->second;
				strcpy(logicUrl, url);
				url = findRenderCandidateUrl(usage);
				strcpy(renderUrl, url);
#endif
				// genrate a task, add to task manager
				temTask = (Task *)malloc(sizeof(Task));
				if (temTask == NULL){
					Log::slog("[DistributorServerManager]: generate task failed.\n");
					break;
				}
				NetDomain * client = (NetDomain*)malloc(sizeof(NetDomain));

				// init the task
				//temTask->init();
				memset(temTask, 0, sizeof(temTask));
				temTask->taskName = _strdup(gameName);
				temTask->status = TASK_STATUS::INIT;
				temTask->require = usage;
				temTask->client = client;
				temTask->client->s = req->s;
				
				newTask++;

				taskManager->addTask(temTask);
				// free client request
				delete req;
				req = NULL;
			}
			else if(!strncasecmp(req->reqInfo, START_RENDER, strlen(START_RENDER))){
				// get the client feedback to start render
				char tcmd[100] = { 0 };
				strcpy(tcmd, START_RENDER);
				Log::slog("[DistributorServerManager]: start render request from %p.\n", req->s);

				char * p = tcmd + strlen(tcmd);
				char * tp = p;
				int sendLen = strlen(tcmd);  // the cmd length
				if (taskManager->getTaskStatus(req->s) == TASK_STATUS::ASSIGNED){
					Log::log("[DistributorServerManager]: req from %p get a assigned task.\n", req->s);
					taskManager->changeStatus(req->s, TASK_STATUS::STARTED);

					// notify the render with client socket
					Task * toDo = taskManager->getAssignedTask(req->s);
					Log::slog("[DistributorServerManager]: task info, client sock:%p, renders:%d. send start render cmd to renders.\n", req->s, toDo->renderCount);
#if 0
					for (int i = 0; i < toDo->renderCount; i++){
						tp = p;
						SOCKET * sp = (SOCKET *)tp;
						*sp = req->s;
						
						// notify each render
						send(toDo->renderServer[i]->s, tcmd, sendLen + sizeof(SOCKET), 0);
						Log::log("[DistributorServerManager]: send to %p the start render command.\n", toDo->renderServer[i]->s);
					}
#endif

					// send start render to logic
					tp = p;
					SOCKET * sp = (SOCKET *)tp;
					*sp = req->s;
					Log::slog("[DistributorServerManager]: send to logic server '%p' the start render command.\n", toDo->logicServer->s);
					send(toDo->logicServer->s, tcmd, sendLen + sizeof(SOCKET), 0);
				}
				else{
					Log::log("[DistributorServerManager]: task status is not assigned, but try to start.\n");
				}
			}

			// set the task manager event
			if (newTask){
				Log::log("[DistributorServerManager]: new task trigger the event.\n");
				taskManager->triggerNotifier();
			}
		}
		else{
			Log::slog("[DistributorServerManager]: get client request failed.\n");
			return -1;
		}
	}
	return 0;
}

ResourceRequire * DistributorServerManager::estimateResourceOccupy(char * game){
	ResourceRequire * require = new ResourceRequire();
	require->logicGpu = 30.0f;
	require->logicCpu = 20.0f;
	require->renderCpu = 20.0f;
	require->renderGpu = 30.0f;

	return require;
}
//return the domain for game logic
NetDomain * DistributorServerManager::findLogicCandidateDomain(ResourceRequire * upbound){
	logicServers = getLogicServer()->nEventTotal - 1;
	Log::log("[DistributorServerManager]: %d logic server in total.\n", logicServers);
	if (logicServers){
		return logicServer->findCandidate(SERVER_TYPE::LOGIC_SERVER, upbound, 0);
	}
	else
	{
		Log::log("[DistributorServerManager]: 0 logic servers.\n");
		return NULL;
	}
}

// return the found servers number
int DistributorServerManager::findRenderCandidateDomain(ResourceRequire * upbound, int count, NetDomain *** ret){
	// devide the resource requirement
	ResourceRequire * subRequire = new ResourceRequire();
	subRequire->logicCpu = upbound->logicCpu;
	subRequire->logicGpu = upbound->logicGpu;

	subRequire->renderCpu = upbound->renderCpu / count;
	subRequire->renderGpu = upbound->renderGpu / count;

	Log::log("[DistributorServerManager]: render candidate require cpu:%f, gpu:%f.\n", subRequire->renderCpu, subRequire->renderGpu);

	return renderServer->findNCandidate(SERVER_TYPE::RENDER_SERVER, subRequire, count, ret);
}