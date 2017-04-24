//#include "network.h"
#define WIN32_LEAN_AND_MEAN
#include "../VideoUtility/videocommon.h"
#include "../VideoUtility/ccg_config.h"
#include "../LibCore/CThread.h"
#include <queue>
#include "Distributor.h"

#include "../LibCore/DisNetwork.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"
#include "../LibInput/Controller.h"

#ifndef USE_LIBEVENT
#define USE_LIBEVENT
#endif


#define FULL_OFFLOAD_MODE_TEST

namespace cg{

	BaseContext * ctxToDis = NULL;

	// use libevent
	const char * INFO				= "INFO";
	const char * REGISTER			= "REGISTER";
	const char * REQ_GAME			= "REQ_GAME";
	const char * RENDER_OVERLOAD	= "RENDER_OVERLOAD";
	const char * LOGIC_OVERLOAD		= "LOGIC_OVERLOAD";
	const char * CLIENT_EXIT		= "CLIENT_EXIT";
	const char * LOGIC_READY		= "LOGIC_READY";
	const char * RENDER_READY		= "RENDER_READY";
	const char * RENDER_CONTEXT_READY = "RENDER_CONTEXT_READY";
	const char * RTSP_READY			= "RTSP_READY";

	const char * ADD_RENDER			= "ADD_RENDER";
	const char * DECLINE_RENDER		= "DECLINE_RENDER";
	const char * GAME_EXIT			= "GAME_EXIT";
	const char * CLIENT_CONNECTED	= "CLIENT_CONNECTED";

	const char * START_TASK			= "START_TASK";
	const char * START_GAME			= "START_GAME";
	const char * CANCEL_TASK		= "CANCEL_TASK";

	const char * ADD_LOGIC			= "ADD_LOGIC";
	const char * ADD_CTRL_CONNECTION = "ADD_CTRL_CONNECTION";
	const char * OPTION = "OPTION";

	const char * LOGIC				= "LOGIC";
	const char * RENDER				= "RENDER";
	const char * CLIENT				= "CLIENT";


	void DisClient::collectInfo(float & cpuUsage, float & gpuUsage, float & memUsage){
		cg::core::infoRecorder->logTrace("[DisClient]: collect information.\n");
		static int callCount = 0;
		cpuUsage = callCount * 10;
		gpuUsage = callCount * 10;
		memUsage = callCount * 10;

		callCount++;
	}

	////////  DisServer

	DisServer * DisServer::server = NULL;

	DisServer::DisServer(){
		watchdoaRunning = false;
		collectionInterval = 30;    // temp set 30

		// create the mutex
		mutex = CreateMutex(NULL, FALSE, NULL);
	}

	DisServer::~DisServer(){
		if(disServerUrl){
			free(disServerUrl);
			disServerUrl = NULL;
		}
	}

	// collect the domain information
	void DisServer::collectInfo(){
		cg::core::infoRecorder->logTrace("[DisServer]: collection all domains' information.\n");
	}


	//watchdog thread
	DWORD WINAPI WatchdogProc(LPVOID * param){
		cg::core::infoRecorder->logTrace("[WatchDog]: enter proc.\n");
		DisServer * server = (DisServer *)param;
		while (server->isWatchdogRunning()){
			server->collectInfo();

			Sleep(server->getSleepInterval());
		}

		return 0;
	}


	short DisServer::offsetBase = 0;

	void DisServer::startWatchdog(){
		cg::core::infoRecorder->logTrace("[DisServer]: start watch dog.\n");
		DWORD threadId;
		HANDLE watchdog = chBEGINTHREADEX(NULL, 0, WatchdogProc, this, 0, &threadId);
		cg::core::infoRecorder->logTrace("[DisServer]: watch dog:0x%p, id:%d.\n", watchdog, threadId);
	}



	// deal the event from given BaseContext

	//bool DisServer::dealEvent(BaseContext * ctx){
	bool DisServer::dealEvent(DomainInfo * ctx){
		ctx->readCmd();
		char * cmd = ctx->getCmd();
		char feedback[512] = { 0 };
		int len = 0;

		if (!strncasecmp(cmd, INFO, strlen(INFO))){
			// get the information of this domain
			cg::core::infoRecorder->logTrace("[DisServer]: INFO for %s.\n", ctx->toString().c_str());
			evutil_socket_t id = ctx->sock;
			char * data = ctx->getData();
			float * usage = (float *)data;

			DomainInfo * info = domainMap[id];
			info->cpuUsage = *usage;
			usage++;
			info->gpuUsage = *usage;
			usage++;
			info->memUsage = *usage;
			// update the list

			//scheduleWhenUpdate(info);
			// do the schedule at run time, merge green domains and deal with overload domains, do the prediction here
		}
		else if (!strncasecmp(cmd, REGISTER, strlen(REGISTER))){
			cg::core::infoRecorder->logTrace("[DisServer]: REGISTER from '%s'.\n", ctx->toString().c_str());
			// a new logic server or a new render proxy connected
			char * data = ctx->getData();
			evutil_socket_t id = ctx->sock;

			// add map
			domainMap[id] = ctx;

			//data++;

			if (!strncasecmp(data, LOGIC, strlen(LOGIC))){
				cg::core::infoRecorder->logTrace("[DisServer]: a new Logic connected, id:%p, url:%s.\n", id, ctx->url);
				logicMap[id] = ctx;
				ctx->domainType = string("LOGIC");
			}
			else if (!strncasecmp(data, RENDER, strlen(RENDER))){
				cg::core::infoRecorder->logTrace("[DisServer]: a new render connected, id:%p, url:%s.\n",id, ctx->url);
				renderMap[id] = ctx;
				ctx->domainType = string("RENDER");
			}
			else if (!strncasecmp(data, CLIENT, strlen(CLIENT))){
				cg::core::infoRecorder->logTrace("[DisServer]: a new Client connected, ERROR, id:%p, url:%s.\n", id, ctx->url);
				ctx->domainType = string("CLIENT");
			}
			else{
				// error?
				cg::core::infoRecorder->logTrace("[DisServer]: data is %s\n", data);
				return false;
			}
		}
		else if (!strncasecmp(cmd, REQ_GAME, strlen(REQ_GAME))){
			char * data = ctx->getData();
			ctx->domainType = string("client");
			clientMap[ctx->sock] = ctx;
			//data++;
			// get the game name
			cg::core::infoRecorder->logError("[DisServer]: REQ_GAME, get game name '%s' from client '%s'.\n", data, ctx->toString().c_str());

			// TODO, determine the rtsp service is on logic server or render server
			TaskInfo * task = buildTask(string(data), ctx);
			if(task){
				if(!dispatchToLogic(task)){
					// failed
					cg::core::infoRecorder->logTrace("[DisServer]: failed to dispatch task to logic.\n");
					return false;
				}
			}else{
				// failed to build task
				cg::core::infoRecorder->logTrace("[DisServer]: failed to build task.\n");
				return false;
			}
			return true;
			
		}
		else if (!strncasecmp(cmd, RENDER_OVERLOAD, strlen(RENDER_OVERLOAD))){
			cg::core::infoRecorder->logTrace("[DisServer]: RENDER_OVERLAOD for '%s'.\n", ctx->toString().c_str());
			evutil_socket_t sock = ctx->sock;
			DomainInfo * render = ctx;// renderMap[sock];
			if (render){
				return solveRenderOverload(render);
			}
			return false;
		}
		else if (!strncasecmp(cmd, LOGIC_OVERLOAD, strlen(LOGIC_OVERLOAD))){
			cg::core::infoRecorder->logTrace("[DisServer]: LOGIC_OVERLOAD for %s.\n", ctx->toString().c_str());
			evutil_socket_t sock = ctx->sock;
			DomainInfo * logic = ctx; //logicMap[sock];
			// how to figure out logic overload problem?????
			if (logic){
				return solveLogicOverload(logic);

			}
			return false;
		}
		else if (!strncasecmp(cmd, CLIENT_EXIT, strlen(CLIENT_EXIT))){
			cg::core::infoRecorder->logTrace("[DisServer]: CLIENT_EXIT for '%s'.\n", ctx->toString().c_str());
			IDENTIFIER id = ctx->sock;
			DomainInfo * client = ctx; //clientMap[id];
			// update all domains

			// find the task
			map<IDENTIFIER, TaskInfo *>::iterator it;
			TaskInfo * task = NULL;
			it = taskMap.find(id);
			if (it != taskMap.end()){
				task = it->second;
				taskMap.erase(id);

				DomainInfo * dom = (DomainInfo *)task->logicCtx;
				dom->writeCmd(CANCEL_TASK);
				dom->writeIdentifier(id);
				dom->writeToNet();
				
				// send Game exit cmd to logic

				dom->taskMap.erase(id);

				// for each render
				for(int i = 0; i< task->renderCount; i++){
					dom = (DomainInfo *)task->renderCtx[i];
					dom->writeCmd(CANCEL_TASK);
					dom->writeIdentifier(id);
					dom->writeToNet();
					dom->taskMap.erase(id);
				}

				delete task;
			}
			else{
				// not find
				cg::core::infoRecorder->logTrace("[DisServer]: ERROR, not find task when client '%s' exit.\n", ctx->toString().c_str());
				return false;
			}

			// remove the task from  map
			clientMap.erase(id);
			// free data structure
			delete client;
		}
		else if (!strncasecmp(cmd, LOGIC_READY, strlen(LOGIC_READY))){
			cg::core::infoRecorder->logTrace("[DisServer]: LOGIC_READY from '%s'.\n", ctx->toString().c_str());
			// find the task
			char * data = ctx->getData();
			//data++;
			IDENTIFIER  id = *(IDENTIFIER *)data;
			// game is started at server side, send logic IP to render

			TaskInfo * task = NULL;
			cg::core::infoRecorder->logTrace("[DisServer]: task id:%p.\n", id);
			// find the task first
			map<IDENTIFIER , TaskInfo *>::iterator it = taskMap.find(id);
			if(it != taskMap.end()){
				task = it->second;
			}else{
				task = NULL;
				cg::core::infoRecorder->logTrace("[DisServer]: error, not find the task.\n");
				return false;
			}
			task->logicReady = true;

#ifdef ENABLE_CLIENT_CONTROL
			// send ADD_LOGIC to client
			cg::core::infoRecorder->logError("[DisServer]: send ADD LOGIC to client:%s.\n", task->client->getNetName().c_str());
			task->client->writeCmd(ADD_LOGIC);
			task->client->writeData(task->logicCtx->url, strlen(task->logicCtx->url));
			task->client->writeToNet();

#endif

			//sendCmdToRender(task, feedback, cmdlen);
			// cmd: START_TASK+pid+renderId(render socket in dis server) + [Logic url]
			if(task->mode == MODE_FULL_OFFLOAD || task->mode == MODE_PARTIAL_OFFLOAD){
				// send start task command to renders
				sendCmdToRender(task, START_TASK);
				return true;	
			}
			if(task->mode != MODE_FULL_OFFLOAD){
				// MODE_NO_OFFLOADING, send ADD RTSP CONNECTION cmd.
				BaseContext * lg = task->logicCtx;

				// tell logic server to start the rtsp 
				lg->writeCmd(ADD_RTSP_CONNECTION);
				lg->writeIdentifier(id);
				lg->writeData((void *)&task->portOffset, sizeof(short));
				lg->writeToNet();
				cg::core::infoRecorder->logTrace("[DisServer]: send add rtsp connection, id:%p, port:%d.\n", id, task->portOffset);

				// collect information
				lg->writeCmd(INFO);
				lg->writeToNet();
			}else{
				cg::core::infoRecorder->logError("[DisServer]: task mode is unknown.\n");
				return false;
			}
		}
		else if (!strncasecmp(cmd, RENDER_READY, strlen(LOGIC_READY))){
			cg::core::infoRecorder->logError("[DisServer]: RENDER_READY from '%s'.\n", ctx->toString().c_str());
			char * data = ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;

			TaskInfo * task = taskMap[id];
			DomainInfo * client = clientMap[id];
			// if there a render need to cancel ??? migration

			int i = 0;
			if (task->logicReady){
				for (i = 0; i < task->renderCount; i++){
					if (task->renderCtx[i] == ctx){
						// find the ctx
						break;
					}
				}

				task->renderReady[i] = true;
			}
			else{
				cg::core::infoRecorder->logTrace("[DisServer]: error, logic is NOT ready when receiving RENDER READY cmd.\n");
				return false;
			}
			// check the migrate task
			//IDENTIFIER renderId = ctx->sock;
			if (task->isMigrating(id)){
				cg::core::infoRecorder->logTrace("[DisServer]: the render has a old render.\n");
				// the ready render has a old render, remove the old render
				BaseContext * old = task->migrateCtx[id];
				task->removeCtx(old);
				task->migrateCtx.erase(id);
				// tell the logic to cancel old task
				// cancel the old task with old render id
				cancelRender(task, old->sock);

				// find the render domain
				DomainInfo * oldRender = NULL;
				map<evutil_socket_t, DomainInfo *>::iterator it = renderMap.find(old->sock);
				if (it != renderMap.end()){
					// find
					oldRender = it->second;
					if (oldRender->status == RENDER_MIGRATE){
						// migrate done
						oldRender->status = RENDER_SERVE;
					}
				}
				else{
					// not find
					cg::core::infoRecorder->logError("[DisServer]: ERROR, not find the valid render.\n");

				}

			}

			//tell the render to start the rtsp service

			//ask info
			// logic
			task->logicCtx->writeCmd(INFO);
			task->logicCtx->writeToNet();
			// render
#if 1
			cg::core::infoRecorder->logError("[DisServer]: RENDER_READY, to send rtsp command, render count:%d\n", task->renderCount);
			BaseContext * rt = NULL;
			for (int i = 0; i < task->renderCount; i++){
				rt = task->renderCtx[i];
				if (task->renderReady[i] == true && task->mode != MODE_NO_OFFLOAD){
					// tell each render to start the rtsp 
					rt->writeCmd(ADD_RTSP_CONNECTION);
					rt->writeIdentifier(id);
					rt->writeData((void *)&task->portOffset, sizeof(short));
					rt->writeToNet();
					cg::core::infoRecorder->logError("[DisServer]: ADD rtsp connection to %d-th render.,id:%p, port:%d.\n", i, id, task->portOffset);

					// collect information
					rt->writeCmd(INFO);
					rt->writeToNet();
				}
			}
#endif
		}
		else if(!strncasecmp(cmd, RTSP_READY, strlen(RTSP_READY))){
			// rtsp service ready, tell the client request rtsp service
			cg::core::infoRecorder->logTrace("[DisServer]: RTSP Ready from '%s'.\n", ctx->toString().c_str());
			char * data = ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;

			TaskInfo * task = taskMap[id];
			DomainInfo * client = clientMap[id];
#if 1
			task->ctrlReady = true;
#endif
			// TODO, check the logic
			if(!task->ctrlReady){
				client->writeCmd(ADD_CTRL_CONNECTION);
				//client->writeData()
				client->writeData((void *)task->logicCtx->url, sizeof(task->logicCtx->url));
				client->writeToNet();
			}
			
			client->writeCmd(ADD_RENDER);
			client->writeData((void *)&task->portOffset, sizeof(short));
			client->writeData((void *)ctx->url, strlen(ctx->url));
			client->writeToNet();
		}
		else{
			cg::core::infoRecorder->logTrace("[DisServer]: unknown cmd :'%s' from '%s'.\n", cmd, ctx->toString().c_str());
			return false;
		}
		return true;

	}
	// Helper API
	void DisServer::AddRenderToTask(IDENTIFIER taskId){
		DomainInfo * render = getRenderCandidate(10, 10);
		map<IDENTIFIER, TaskInfo *>::iterator it = taskMap.find(taskId);
		if(it != taskMap.end()){
			// find the task
		}
		else{
			// error
			cg::core::infoRecorder->logError("[DisServer]: cannot find the task for '%d'.\n", taskId);
			return;
		}
		TaskInfo * task = it->second;
		if(render){
			// find a render
			// find the task, add the render to task
			if(task->mode == MODE_NO_OFFLOAD)
				task->mode = MODE_PARTIAL_OFFLOAD;

			task->addRender(render);
			// send cmd to render
			if(!sendCmdToRender(task, START_TASK, render)){
				cg::core::infoRecorder->logError("[DisServer]: send cmd '%s' to render: %p failed.\n", START_TASK, render->sock);
			}

		}else{
			cg::core::infoRecorder->logError("[DisServer]: add render did not get a proper render domain.\n");
			return;
		}
	}
	void DisServer::ChangeOffloadLevel(IDENTIFIER taskId, TASK_MODE mode /*= MODE_FULL_OFFLOAD*/){
		map<IDENTIFIER, TaskInfo *>::iterator it = taskMap.find(taskId);
		TaskInfo * task = NULL;
		if(it != taskMap.end()){
			// find the task
			task = it->second;
		}else{
			cg::core::infoRecorder->logError("[DisServer]: cannot find task '%d'.\n", taskId);
			return;
		}
		// send option cmd to logic server
		if(!task->logicReady){
			// error
			cg::core::infoRecorder->logError("[DisServer]: task '%d' logic '%p' server is not ready.\n", taskId, task->logicCtx);
			return;
		}
		// send cmd
		task->logicCtx->writeCmd(OPTION);
		task->logicCtx->writeIdentifier(taskId);
		task->logicCtx->writeShort(1);
		task->logicCtx->writeData((void *)CMD_OPTION::SETOFFLOAD, sizeof(CMD_OPTION));
		task->logicCtx->writeShort(mode);
		task->logicCtx->writeToNet();

	}
	// send CHANGE_ENCODER cmd to given task in given domain
	void DisServer::ChangeEncoderType(IDENTIFIER taskId, IDENTIFIER domainId){
		map<IDENTIFIER, TaskInfo *>::iterator it = taskMap.find(taskId);
		if(it == taskMap.end()){
			// not find the task
			cg::core::infoRecorder->logError("[DisServer]:ChangeEncoderType(), not find the task for '%p'.\n", taskId);
			return;
		}
		TaskInfo * task = it->second;
		BaseContext * ctx = NULL;
		CMD_OPTION option;
		cg::core::infoRecorder->logTrace("[DisServer]:ChangeEncoderType(), logic id:%d.\n", task->logicCtx->sock);
		if(task->logicCtx->sock == domainId){
			// find the domain, it is the logic
			cg::core::infoRecorder->logError("[DisServer]: the domain is a logic server: %s.\n", task->logicCtx->getNetName().c_str());
			option = ENCODEROPTION;
			ctx = task->logicCtx;
			ctx->writeCmd(OPTION);
			ctx->writeIdentifier(taskId);
			ctx->writeShort(1);
			ctx->writeData((void *)&option, sizeof(CMD_OPTION));
			ctx->writeShort(0);
			ctx->writeToNet(0);
			return;
		}
		else{
			for(int i = 0; i < task->renderCount;i++){
				cg::core::infoRecorder->logTrace("[DisServer]:ChangeEncoderType(), '%dth' render id:%d.\n", i, task->renderCtx[i]->sock);
				if(domainId == task->renderCtx[i]->sock){
					// find the domain, it is a render
					cg::core::infoRecorder->logError("[DisServer]: the domain is the '%dth' render.\n", i);
					ctx = task->renderCtx[i];
					ctx->writeCmd(OPTION);
					ctx->writeIdentifier(taskId);
					ctx->writeShort(1);
					ctx->writeData((void *)&option, sizeof(CMD_OPTION));
					ctx->writeShort(0);
					ctx->writeToNet(0);
					return;
				}
			}
		}
		// 
		cg::core::infoRecorder->logError("[DisServer]: cannot find the task '%d' in domain '%d'.\n", taskId, domainId);
		return;
	}

	// find a logic candidate that can satisfy the requirement, especially CPU
	DomainInfo * DisServer::getLogicCandidate(float cpuRe, float gpure){
		cg::core::infoRecorder->logTrace("[DisServer]: get logic candidate.\n");
		DomainInfo * ret = NULL, * t = NULL;
		map<evutil_socket_t, DomainInfo *>::iterator it;
		int size = logicMap.size();
		DomainInfo ** arr = (DomainInfo **)malloc(sizeof(DomainInfo *) * (size + 2));
		int index = 0;
		// now only a few domains, here, so travel all
		for (it = logicMap.begin(); it != logicMap.end(); it++){
			// for each domain
			t = it->second;
			arr[index++] = t;
		}
		// sort first

		for (int i = 0; i < size; i++){
			for (int j = i; j < size; j++){
				if (arr[i]->cpuUsage < arr[j]->cpuUsage){
					t = arr[i];
					arr[i] = arr[j];
					arr[j] = t;
				}
			}
		}
		// >=
		// best fit
#ifdef SCHEDULE_USE_BEST_FIT
		for (int i = size - 1; i >= 0; i++){
			if(arr[i]->cpuUsage + cpuRe <= OVERLOAD_THRESHOLD){
				ret = arr[i];
				break;
			}
		}
#endif
		// or worst fit?
		for (int i = 0; i < size; i++){
			if (arr[i]->cpuUsage + cpuRe <= OVERLOAD_THRESHOLD){
				// find the worth fit
				ret = arr[i];
				break;
			}
		}
		// or first fit ? no use !

		if(arr){
			free(arr);
		}
		if(ret){
			cg::core::infoRecorder->logTrace("[DisServer]: get logic candidate with sock:%p, url:%s.\n",ret->sock, ret->url);
		}

		return ret;
	}
	// find a render candidate that can satisfy the requirement, especially GPU
	DomainInfo * DisServer::getRenderCandidate(float cpuRe, float gpuRe){
		cg::core::infoRecorder->logTrace("[DisServer]: get render candidate.\n");
		DomainInfo * ret = NULL, *t = NULL;
		map<evutil_socket_t, DomainInfo *>::iterator it;
		int size = renderMap.size();
		DomainInfo ** arr = (DomainInfo **)malloc(sizeof(DomainInfo *) * (size + 2));
		int index = 0;
		// now only a few domains, here, so travel all
		for (it = renderMap.begin(); it != renderMap.end(); it++){
			// for each domain
			t = it->second;
			arr[index++] = t;
		}
		// sort first

		for (int i = 0; i < size; i++){
			for (int j = i; j < size; j++){
				if (arr[i]->gpuUsage < arr[j]->gpuUsage){
					t = arr[i];
					arr[i] = arr[j];
					arr[j] = t;
				}
			}
		}
		// >=
		// best fit
#ifdef SCHEDULE_USE_BEST_FIT
		for (int i = size - 1; i >= 0; i++){
			if (arr[i]->gpuUsage + gpuRe <= OVERLOAD_THRESHOLD){
				ret =  arr[i];
				break;
			}
		}
#endif
		// or worst fit?
		for (int i = 0; i < size; i++){
			if (arr[i]->gpuUsage + gpuRe <= OVERLOAD_THRESHOLD){
				// find the worth fit
				ret = arr[i];
				break;
				return arr[i];
			}
		}
		// or first fit ? no use !

		if(arr){
			free(arr);
		}

		if(ret)
			cg::core::infoRecorder->logTrace("[DisServer]: get render candidate with sock:%p, url:%s.\n",ret->sock, ret->url);

		return ret;
	}
	bool DisServer::sendCmdToRender(TaskInfo * task, const char * cmd, DomainInfo * render){
		cg::core::infoRecorder->logTrace("[DisServer]: send cmd to render '%s'.\n", render->toString().c_str());
		if(task->status != ASSIGNED){
			cg::core::infoRecorder->logError("[DisServer]: task is not assigned.\n)");
			return false;
		}
		cg::core::infoRecorder->logTrace("[DisServer]: send to render, task id: %p, render id: %p, port:%d, logic url:%s.\n", task->id, render->sock, task->portOffset, task->logicCtx->url);
		short nameLen = task->taskName.length();
		// send cmd to given server
		render->writeCmd(cmd);
		render->writeIdentifier(task->id);
		render->writeIdentifier(render->sock);
		render->writeData((void *)&task->portOffset, sizeof(short));
		
		render->writeData((void *)&nameLen, sizeof(short));
		render->writeData((void *)task->taskName.c_str(), nameLen);
		render->writeData((void *)&task->logicCtx->url, strlen(task->logicCtx->url));
		render->writeToNet(0);
		return true;
	}

	bool DisServer::sendCmdToRender(TaskInfo * task, const char * cmd){
		cg::core::infoRecorder->logTrace("[DisServer]: send cmd to render.\n");
		if (task->status != ASSIGNED){
			cg::core::infoRecorder->logError("[DisServer]: task is not assigned.\n");
			return false;
		}

		short nameLen = task->taskName.length();
		// cmd format: START_TASK+pid+renderId(render socket in dis server)+ port offset + [Logic url]
		BaseContext * ctx = NULL;
		for(int i = 0;i< task->renderCount; i++){
			ctx = task->renderCtx[i];
			ctx->writeCmd(cmd);

			ctx->writeIdentifier(task->id);
			ctx->writeIdentifier(ctx->sock);
			ctx->writeData((void *)&task->portOffset, sizeof(short));

			ctx->writeData((void *)&nameLen, sizeof(short));
			ctx->writeData((void *)task->taskName.c_str(), nameLen);

			ctx->writeData((void *)task->logicCtx->url, strlen(task->logicCtx->url));
			ctx->writeToNet();
			cg::core::infoRecorder->logTrace("[DisServer]: send START TASK to render with logic url:%s.\n", task->logicCtx->url);
		}
		return true;
	}

	// NOT USED. send the given command to render those are assigned to given task
	bool DisServer::sendCmdToRender(TaskInfo * task, char * cmddata, int len){
		cg::core::infoRecorder->logTrace("[DisServer]: send given cmd to render.\n");
		if (task->status != ASSIGNED){
			// error
			cg::core::infoRecorder->logTrace("[DisServer]: task is not assigned.\n");
			return false;
		}
		//int sendCount = 0;
		for (int i = 0; i < task->renderCount; i++){
			if (task->renderCtx[i] && task->renderReady[i] && task->isSent[i] == false){
				cg::core::infoRecorder->logTrace("[DisServer]: send START TASK command to render proxy.\n");
				task->isSent[i] = true;
				task->renderCtx[i]->writeToNet(cmddata, len);
			}
		}

	}
	// solve the overload warning from a render
	bool DisServer::solveRenderOverload(DomainInfo * render){
		cg::core::infoRecorder->logTrace("[DisServer]: solve render overload.\n");
		bool ret = false;
		// find a task in the render to release resource, by migrate a part of the rendering load to other render domains

		TaskInfo * taskToMigrate = NULL;
		DomainInfo * dstRender = NULL;

		// TODO , find the task and render
		// try to find a undivided task
		if (!(taskToMigrate = render->getUndivideTask())){
			cg::core::infoRecorder->logTrace("[DisServer]: overload render does not have undivided task.\n");
			// to find a divided task that has the least child
			TaskInfo * t = render->getLeastDivideTask();
			// to migrate the task
			render->status = RENDER_MIGRATE;
			DomainInfo * newRender = getRenderCandidate(0, 0);

			IDENTIFIER tid = t->id;  // get the task id

			if (!newRender->isTaskExist(tid)){
				// 1. migrate to a render do not contain the task
				// send START TASK to the new render

				migrateRendering(render, newRender, t);
			}
			else{
				// 2. dst render contains the same task, merge it( actually, just release this)
				// tell the logic server to release, DECLINE RENDER with client id and render id
				cancelRender(t, render);
			}

			return true;
		}

		// to divide the task
		render->status = RENDER_MIGRATE;   // migrate to release the load
		cg::core::infoRecorder->logTrace("[DisServer]: Divide task.\n");
		// calculate resource calculate
		float cpuRe = 0.0f, gpuRe = 0.0f, memRe = 0.0f;


		dstRender = getRenderCandidate(cpuRe, gpuRe);

		if (!dstRender){
			cg::core::infoRecorder->logTrace("[DisServer]: get a new render to release the load, but get No render.\n");
			return false;
		}
		ret = true;

		// add a render to the task
		addRender(taskToMigrate, dstRender);
		return ret;
	}
	// solve the overload warning from a logic, nothing to do right now
	bool DisServer::solveLogicOverload(DomainInfo * logic){
		cg::core::infoRecorder->logTrace("[DisServer]: solve logic overload for '%s', nothing we can do, ERROR.\n", logic->toString().c_str());
		bool ret = false;

		return ret;
	}
	// merge the green logic server ??? nothing to do right now
	bool DisServer::mergeLogic(DomainInfo * logic){

		cg::core::infoRecorder->logTrace("[DisServer]: merge logic for '%s', nothing we can do, ERROR.\n", logic->toString().c_str());
		bool ret = false;

		return ret;
	}
	// merge the green render server, migrate the render thread and then shut down the render domain. additionally, we need to package the render as much as possible
	bool DisServer::mergeRender(DomainInfo * render){
		cg::core::infoRecorder->logTrace("[DisServer]: merge render.\n");
		bool ret = false;

		render->status = RENDER_GREEN;   // tag the render with RENDER MIGRATE

		// for each task runs on the render
		map<IDENTIFIER, TaskInfo *>::iterator it;
		for (it = render->taskMap.begin(); it != render->taskMap.end(); it++){
			IDENTIFIER id = it->second->id;
			TaskInfo * task = it->second;
			// allocate a new render for this task
			float cpuRe = 0.0f, gpuRe = 0.0f, memRe = 0.0f;

			// calculate the resource requirement

			// first to find the render that runs sub task for the Task, is the render is available, just merge the task

			// get the render candidate
			DomainInfo * newRender = getRenderCandidate(cpuRe, gpuRe);
			if (newRender){
				// a part of the task exist on the render
				if (newRender->isTaskExist(id)){
					cg::core::infoRecorder->logTrace("[DisServer]: a part of the task exist, just release the old render.\n");
					cancelRender(task, render);
					continue;
				}

				migrateRendering(render, newRender, task);
			}
			else{
				cg::core::infoRecorder->logTrace("[DisServer]: migrate find NO capable render, TEMP ERROR Now, the solution it to start new render, TODO.\n");
				return false;
			}
		}

		return ret;
	}

	// add a new render to the task
	bool DisServer::addRender(TaskInfo * task, DomainInfo * dstRender){
		cg::core::infoRecorder->logTrace("[DisServer]: add render to '%s'.\n", dstRender->toString().c_str());
		BaseContext * logicForTask = task->logicCtx;
		IDENTIFIER id = *(IDENTIFIER *)task->id;
		IDENTIFIER renderId = dstRender->sock;

		dstRender->addTask(task);
		dstRender->writeCmd(START_TASK);
		dstRender->writeIdentifier(id);
		dstRender->writeIdentifier(renderId);
		dstRender->writeData(logicForTask->url, strlen(logicForTask->url));
		dstRender->writeToNet();

		return true;
	}

	// migrate rendering 'task' from domain 'src' to destination 'dst'
	bool DisServer::migrateRendering(DomainInfo * src, DomainInfo * dst, TaskInfo * task){
		cg::core::infoRecorder->logTrace("[DisServer]: migrate rendering task from '%s' to '%s'.\n", src->toString().c_str(), dst->toString().c_str());
		bool ret = false;
		// send START TASK to the new render
		IDENTIFIER id = task->id;
		IDENTIFIER renderId = dst->sock;

		dst->addTask(task);
		dst->writeCmd(START_TASK);
		dst->writeIdentifier(id);
		dst->writeIdentifier(renderId);
		dst->writeData(task->logicCtx->url, strlen(task->logicCtx->url));
		dst->writeToNet();

		// TODO
		task->migrateCtx[id] = src;

		return ret;
	}

	bool DisServer::cancelRender(TaskInfo * task, IDENTIFIER oldRenderId){
		cg::core::infoRecorder->logTrace("[DisServer]: cancel render.\n");
		bool ret = true;

		IDENTIFIER id = task->id;
		BaseContext * logicCtx = task->logicCtx;
		logicCtx->writeCmd(DECLINE_RENDER);
		logicCtx->writeIdentifier(id);
		logicCtx->writeIdentifier(oldRenderId);
		logicCtx->writeToNet();
		return ret;
	}

	bool DisServer::cancelRender(TaskInfo * task, DomainInfo * oldRender){
		cg::core::infoRecorder->logTrace("[DisServer]: cancel render for '%s'.\n", oldRender->toString().c_str());
		bool ret = true;

		IDENTIFIER id = task->id;
		IDENTIFIER renderId = oldRender->sock; // old render id

		BaseContext * logicCtx = task->logicCtx;
		logicCtx->writeCmd(DECLINE_RENDER);
		logicCtx->writeIdentifier(id);
		logicCtx->writeIdentifier(renderId);
		logicCtx->writeToNet();

		return ret;
	}

	// schedule when update, package the heavy-loaded domain, merge the green domain
	bool DisServer::scheduleWhenUpdate(DomainInfo * domain){
		cg::core::infoRecorder->logTrace("[DisServer]: schedule when update.\n");
		bool ret = false;
		//check the updated domain
		if (domain->status == RENDER_MIGRATE){
			// ignore
		}
		else{
			if (domain->isOverload()){
				// need to do something?
				solveRenderOverload(domain);
			}
			else if (domain->isGreen()){
				// need to merge ????
				mergeRender(domain);
			}
		}

		// first to packet heavy-loaded domain
		map<evutil_socket_t, DomainInfo *>::iterator it;
		// for each render domain
		for (it = renderMap.begin(); it != renderMap.end(); it++){

		}
		// merge the green domain if any

		return ret;
	}

	void DisServer::printDomain(){
		map<evutil_socket_t, DomainInfo *>::iterator it = domainMap.begin();
		std::cout << "\t System has " << domainMap.size() << " servers registered." << std::endl;
		int count = 0;
		for(; it != domainMap.end(); it++){
			cg::core::infoRecorder->logTrace("[Domain:%p]\n", it->first);
			std::cout << "\t" << count ++ << " th server: " << std::endl;
			it->second->print();
		}
	}
	void DisServer::printTask(){
		map<IDENTIFIER, TaskInfo *>::iterator it;
		std::cout << "\t system total has " << taskMap.size() << " tasks." << std::endl;
		for(it = taskMap.begin(); it != taskMap.end(); it++){
			// print the task
			it->second->print();
		}
	}

	// build the task fro game request, first, estimate resource requirement, determine the task mode, allocate logic domain and render domain
	TaskInfo * DisServer::buildTask(string taskName, DomainInfo * client){
		cg::core::infoRecorder->logTrace("[DisServer]: create a task with id:%p.\n", client->sock);
		float cpuRe = 10, gpuRe = 10;
		bool needRender = false;
		DomainInfo * render = NULL;
		TaskInfo * task = NULL;
		DomainInfo * logic = getLogicCandidate(cpuRe, gpuRe);

		// determine the task mode according to usage level


		task = new TaskInfo();
		task->setLogic(logic);
		task->taskName = taskName;

		task->client = client;
		task->portOffset = offsetBase++;
		taskMap[client->sock] = task;
		logic->addTask(task);
		task->status = ASSIGNED;
		task->id = client->sock;
		task->mode = MODE_NO_OFFLOAD;
		// if not enough, allocate a helper render

#ifdef FULL_OFFLOAD_MODE_TEST

		needRender = true;
#endif

		if(needRender){
#ifdef FULL_OFFLOAD_MODE_TEST
			task->mode = MODE_FULL_OFFLOAD;
#else
			task->mode = MODE_PARTIAL_OFFLOAD;
#endif
			//task->mode = MODE_FULL_OFFLOAD;
			render = getRenderCandidate(cpuRe, gpuRe);
			task->addRender(render);
			render->addTask(task);
		}
		return task;
	}

	bool DisServer::dispatchToLogic(TaskInfo * task){
		cg::core::infoRecorder->logTrace("[DisServer]: dispatch task.\n");
		DomainInfo * logic = NULL;
		DomainInfo * render = NULL;

		logic = (DomainInfo *)task->logicCtx;
		logic->writeCmd(START_TASK);
		logic->writeIdentifier(task->id);
		logic->writeShort(task->mode);
		logic->writeData((void *)task->taskName.c_str(), task->taskName.length());
		logic->writeToNet();

		if(task->mode == MODE_NO_OFFLOAD){
			// only logic server
			
		}else if(task->mode == MODE_PARTIAL_OFFLOAD){
			// logic server also render
			
		}else if(task->mode == MODE_FULL_OFFLOAD){
			// all rendering goes to renders
		}

		return true;
	}

	///////////////  DisClient //////////////

	bool DisClient::dealEvent(BaseContext * ctx){
		cg::core::infoRecorder->logTrace("[DisClient]: deal event.\n");
		ctx->readCmd();
		char * cmd = ctx->getCmd();
		//if (!strncasecmp())


		return true;
	}

	bool DisClient::start(){
		cg::core::infoRecorder->logTrace("[DisClient]: start.\n");


		return true;
	}

	bool DisClient::connectToServer(char * ip, short port){
		cg::core::infoRecorder->logTrace("[DisClient]: connect to dis server, ip:%s, port:%d.\n", ip, port);
		int iResult = 0;
		bool ret = true;
		evutil_socket_t sock = NULL;

		//char * url = NULL;
		//unsigned short port = DIS_PORT_CTRL;   // use control port
		sockaddr_in sin;

		int sin_size = sizeof(sin);
		sin.sin_family = AF_INET;
		sin.sin_addr.s_addr = inet_addr(ip);
		sin.sin_port = htons(port);

		struct bufferevent * bev = NULL;

		sock = socket(AF_INET, SOCK_STREAM, 0);
		if(evutil_make_socket_nonblocking(sock) < 0){
			cg::core::infoRecorder->logTrace("[DisClient]: make socket non blocking failed\n");
			return false;
		}
		//frobSocket(sock);

		bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE);
		// set callback function
		bufferevent_setcb(bev, DisClientReadCB, DisClientWriteCB, DisClientEventCB, this);

		// connect
		if(bufferevent_socket_connect(bev, (struct sockaddr *)&sin, sizeof(sin)) < 0){
			cg::core::infoRecorder->logTrace("[DisCliet]: error starting connection.\n");
			bufferevent_free(bev);
			return false;
		}
		cg::core::infoRecorder->logTrace("[DisClient]: connection established.\n");

		bufferevent_enable(bev, EV_READ | EV_WRITE);

		if (!ctx){
			ctx = new BaseContext();
			ctxToDis = ctx;
		}

		ctx->sock = sock;
		ctx->bev = bev;


		return true;
	}

	// listen the rtsp connection
	bool DisClient::listenRTSP(short portOffset){
		cg::core::infoRecorder->logTrace("[RenderProxy]: listen to RTSP connection port.\n");
		sockaddr_in sin;
		int sin_size = sizeof(sin);
		memset(&sin, 0, sin_size);
		sin.sin_family = AF_INET;
		sin.sin_addr.S_un.S_addr = htonl(0);
		sin.sin_port = htons(DIS_PORT_RTSP + portOffset);   // listen to rtsp port

		connlistener = evconnlistener_new_bind(base, RTSPListenerCB, this, LEV_OPT_LEAVE_SOCKETS_BLOCKING | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size);
		if (!connlistener){
			perror("couldn't create listener.\n");
			return false;
		}
		evconnlistener_set_error_cb(connlistener, RTSPAcceptErrorCB);

		return true;
	}


	// callback for DisClient

	void DisClientReadCB(struct bufferevent * bev, void * arg){
		DisClient * disClient = (DisClient *)arg;
		struct evbuffer * input = bufferevent_get_input(bev);
		size_t n = evbuffer_get_length(input);
		char * data = (char *)malloc(sizeof(char)* n + 1);
		memset(data, 0, n + 1);
		evbuffer_copyout(input, data, n);

		// deal event
		BaseContext * ctx = disClient->getCtx();
		ctx->setData(data, n);

		disClient->dealEvent(ctx);
		free(data);
		data = NULL;
		evbuffer_drain(input, n);
	}

	// callback for DisClient
	void DisClientWriteCB(struct bufferevent * bev, void * arg){
		struct evbuffer * output = bufferevent_get_output(bev);
		if(evbuffer_get_length(output) == 0){
			cg::core::infoRecorder->logTrace("[DisClientWriteCB]: flushed answer.\n");
		}
	}

	// event callback
	void DisClientEventCB(struct bufferevent * bev, short what, void * arg){
		DisClient * disClient = (DisClient *)arg;

		if(what & (BEV_EVENT_EOF | BEV_EVENT_ERROR)){
			perror("[DisClient]: error EOF | ERROR");
			bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
			delete disClient;
			bufferevent_disable(bev, EV_READ | EV_WRITE);
			bufferevent_free(bev);
		}
	}

	// for rtsp connection

	void frobSocket(evutil_socket_t sock){
#ifdef HAVE_SO_LINGER
		struct linger l;
#endif
		int one = 1;
		if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *)&one, sizeof(one)) < 0){
			perror("set socketopt(SO_RESUADDR)");

		}
#ifdef HAVE_SO_LINGER

		l.l_onoff = 1;
		l.l_linger = 0;
		if (setsockopt(sock, SOL_SOCKET, SO_LINGER, (void *)&l, sizeof(l)) < 0)
			perror("setsocketopt(SO_LINGER)");
#endif
	}

}