#include "DistributorForLogic.h"
#include "../LibCore/InfoRecorder.h"

namespace cg{
#ifndef USE_LIBEVENT

#else
	// use libevent

	const char * CANCEL_SOCKEVENT = "CANCEL_SOCKEVENT";
	const char * DESTROY_SOCKEVENT = "DESTROY_SOCKEVENT";
	const char * COPY_HANDLE = "COPY_HANDLE";
	//const char * ADD_CTRL_CONNECTION = "ADD_CTRL_CONNECTION";
	const char * ADD_RTSP_CONNECTION = "ADD_RTSP_CONNECTION";
	//const char * RTSP_READY = "RTSP_READY";

	const char * GAME_READY = "GAME_READY";
	const char * START_RTSP_SERVICE = "START_RTSP_SERVICE";
	const char * CANCEL_RTSP_SERVICE = "CANCEL_RTSP_SERVICE";
	


	///////////// LogicServer ////////////////

	LogicServer *LogicServer::logicServer;

	bool LogicServer::dealEvent(BaseContext * ctx){
		char  feedback[1000] = { 0 };
		ctx->readCmd();
		int len = 0;
		char * cmd = ctx->getCmd();
		if (!strncasecmp(cmd, INFO, strlen(INFO))){
			cg::core::infoRecorder->logTrace("[LogicServer]: INFO.\n");


			// collect information and feedback
			float cpuUsage = 0.0f, gpuUsage = 0.0f, memUsage = 0.0f;

			collectInfo(cpuUsage, gpuUsage, memUsage);
#if 0
			sprintf(feedback, "%s+", INFO);
			len = strlen(feedback);
			float * pf = (float *)(feedback + len);
			*pf = cpuUsage;
			*pf = gpuUsage;
			*pf = memUsage;
			len += sizeof(float) * 3;

			ctx->writeToNet(feedback, len);
#else
			ctx->writeCmd(INFO);
			ctx->writeFloat(cpuUsage);
			ctx->writeFloat(gpuUsage);
			ctx->writeFloat(memUsage);
			ctx->writeToNet();
#endif

		}
		else if (!strncasecmp(cmd, ADD_RENDER, strlen(ADD_RENDER))){
			cg::core::infoRecorder->logTrace("[LogicServer]: ADD RENDER, the new render will auto added.\n");
			// add a render to exist connection, the cmd is from render proxy
			char * data = ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;  // to find the task
			//char * 
			IDENTIFIER renderId = *(IDENTIFIER *)(data + sizeof(IDENTIFIER));

			BaseContext * game = NULL;
			// find the game process and tell it to add render
			map<IDENTIFIER, BaseContext *>::iterator it = gameMap.find(id);
			if(it != gameMap.end()){
				// find the game
				game = it->second;
			}
			else{
				// not find the game
				cg::core::infoRecorder->logTrace("[LogicServer]: cannot find the task running.\n");
				return false;
			}
			// tell game to add
			game->writeCmd(ADD_RENDER);
			// write current renders' connection
#if 0
			game->writeData((void *)ctx->sock, sizeof(IDENTIFIER));
#else
			game->writeIdentifier(ctx->sock);
#endif
			game->writeToNet();  // write to game process

		}
		else if (!strncasecmp(cmd, DECLINE_RENDER, strlen(DECLINE_RENDER))){
			cg::core::infoRecorder->logTrace("[LogicServer]: DECLINE RENDER.\n");
			char * data = ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;    // to find the task
			// distinguish render
			IDENTIFIER renderId = *(IDENTIFIER *)(data + sizeof(IDENTIFIER));
		}
		else if (!strncasecmp(cmd, GAME_EXIT, strlen(GAME_EXIT))){
			cg::core::infoRecorder->logTrace("[LogicServer]: GAME EXIT.\n");

		}
		else if (!strncasecmp(cmd, CANCEL_TASK, strlen(CANCEL_TASK))){
			cg::core::infoRecorder->logTrace("[LogicServer]: CANCEl TASK.\n");
			char * data = ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;
			data += sizeof(IDENTIFIER) + 1;

			// TODO, cancel the task







			// feedback with domain information
			//collect information
			float cpuUsage = 0.0f, gpuUsage = 0.0f, memUsage = 0.0f;

			collectInfo(cpuUsage, gpuUsage, memUsage);

#if 0
			sprintf(feedback, "%s+", INFO);
			int len = strlen(feedback);
			float * pf = (float *)(feedback + len);
			*pf = cpuUsage;
			pf++;
			*pf = gpuUsage;
			pf++;
			*pf = memUsage;

			len += sizeof(float) * 3;
			ctx->writeToNet(feedback, len);
#else
			ctx->writeCmd(INFO);
			ctx->writeFloat(cpuUsage);
			ctx->writeFloat(gpuUsage);
			ctx->writeFloat(memUsage);
			ctx->writeToNet();
#endif


		}
		else if (!strncasecmp(cmd, START_TASK, strlen(START_TASK))){
			cg::core::infoRecorder->logTrace("[LogicServer]: START TASK.\n");
			// start a game with no render
			char * data = ctx->getData();
			//data++;

			IDENTIFIER id = *(IDENTIFIER *)data;
			char * gameName = (data + sizeof(IDENTIFIER) + 1);
			cg::core::infoRecorder->logTrace("[LogicServer]: start game '%s', task id:%p.\n", gameName, id);

			startGame(gameName, id);

#if 0
			// send back LOGIC READY
			ctx->writeCmd(LOGIC_READY);
			ctx->writeData(&id, sizeof(IDENTIFIER));
			ctx->writeToNet();
#endif

#if 0
			sprintf(feedback, "%s+", LOGIC_READY);
			IDENTIFIER * pid = (IDENTIFIER *)(feedback + strlen(feedback));
			int len = strlen(feedback);
			*pid = id;
			len += sizeof(IDENTIFIER);

			ctx->writeToNet(feedback, len);
#endif

		}
		else if (!strncasecmp(cmd, CLIENT_CONNECTED, strlen(CLIENT_CONNECTED))){
			// game client connected with client id
			char * data = ctx->getData();
			IDENTIFIER cid = *(IDENTIFIER *)data;
			// this is for control connection
			cg::core::infoRecorder->logTrace("[LogicServer]: CLIENT CONNECTED.\n");

			// find the game process
			map<IDENTIFIER, BaseContext *>::iterator it = gameMap.find(cid);
			if (it != gameMap.end()){
				// find the game
			}
			else{
				cg::core::infoRecorder->logTrace("[LogicServer]: not find the game process.\n");
			}

		}
		else if (!strncasecmp(cmd, CANCEL_SOCKEVENT, strlen(CANCEL_SOCKEVENT))){
			// the game process take over the render connection, tell the logic server to release the event
			char *data = ctx->getData();
			evutil_socket_t s = *(evutil_socket_t *)data;
			// to cancel the socket event
			BaseContext * snet = NULL;
			map<evutil_socket_t, BaseContext *>::iterator it = netCtxMap.find(s);
			if (it != netCtxMap.end()){
				// cancel the event
				snet = it->second;
				bufferevent_disable(snet->bev, EV_READ | EV_WRITE);
			}
			else{
				// error
				cg::core::infoRecorder->logTrace("[LogicServer]: cannot find the BaseContext associated with %p.\n", s);
				return false;
			}
		}
		else if (!strncasecmp(cmd, DESTROY_SOCKEVENT, strlen(DESTROY_SOCKEVENT))){
			cg::core::infoRecorder->logTrace("[LogicServer]: recv 'DESTROY_SOCKEVENT'\n");
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
			// game ready from game client
			cg::core::infoRecorder->logTrace("[LogicServer]: recv 'GAME_READY' from game client.\n");
			char * data= ctx->getData();
			IDENTIFIER tid = *(IDENTIFIER *)data;

			// send dis server "logic ready.\n"
			// send back LOGIC READY
			this->getCtx()->writeCmd(LOGIC_READY);
#if 0
			this->getCtx()->writeData(&tid, sizeof(IDENTIFIER));
#else
			this->getCtx()->writeIdentifier(tid);
#endif
			this->getCtx()->writeToNet();

			ctx->writeCmd("TEST");
			ctx->writeToNet();
		}else if(!strncasecmp(cmd, ADD_RTSP_CONNECTION, strlen(ADD_RTSP_CONNECTION))){
			// deal the logic server providing the rtsp services
			cg::core::infoRecorder->logTrace("[LogicServer]: the logic server provide the rtsp service.\n");
			//this->startRTSP(NULL);

			// dispatch the cmd to game process

			// find the process
			char * data = ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;
			short portOff = *(short *)(data + sizeof(IDENTIFIER));

			BaseContext * gameCtx = gameMap[id];
			if(gameCtx){
				//
				gameCtx->writeCmd(ADD_RTSP_CONNECTION);
				gameCtx->writeIdentifier(id);
				gameCtx->writeData((void *)&portOff, sizeof(short));
				gameCtx->writeToNet();
			}

		}else if(!strncasecmp(cmd, CANCEL_RTSP_SERVICE, strlen(CANCEL_RTSP_SERVICE))){
			// cancel the rtsp service in the game process
			// find the process
			char * data= ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;
			BaseContext * gameCtx = gameMap[id];
			if(gameCtx){
				gameCtx->writeCmd(CANCEL_RTSP_SERVICE);
				gameCtx->writeToNet();
			}
		}
		else if(!strncasecmp(cmd, OPTION, strlen(OPTION))){
			// the option for game process
			char * data = ctx->getData();
			IDENTIFIER id = *(IDENTIFIER *)data;
			BaseContext * gameCtx = gameMap[id];

			short optionCount = *(short *)(data + sizeof(IDENTIFIER));


			if(gameCtx){
				gameCtx->writeCmd(OPTION);
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
				cg::core::infoRecorder->logError("[LogicServer]: not find the task '%'.\n", id);
				return false;
			}
		}
		else{
			cg::core::infoRecorder->logTrace("[LogicServer]: unknown cmd.\n");
			return false;
		}
		return true;

	}
	bool LogicServer::start(){
		// star the logic server
		// connect to dis and register
		connectToServer(DIS_URL_DISSERVER, DIS_PORT_DOMAIN);
		// register as logic

		cg::core::infoRecorder->logTrace("[LogicServer]: to register.\n");
		ctx->writeCmd(REGISTER);
		ctx->writeData((void *)LOGIC, strlen(LOGIC));

		ctx->writeToNet();

		return true;
	}
#if 0
	bool LogicServer::listenRTSP(){
		printf("[LogicServer]: listen to RTSP connection port.\n");



		return true;
	}
#endif

	// after the rtsp client connected, get the describe information, find the game
	bool LogicServer::startRTSP(evutil_socket_t sock){
		cg::core::infoRecorder->logTrace("[LogicServer]: start RTSP.\n");
		//get the RTSP request, find the service and notify game process

		char buffer[1024] = { 0 };
		char rtspObject[50] = { 0 };   // request rtsp object

		int n = recv(sock, buffer, sizeof(buffer), 0);
		if (n > 0){
			// get object


			// find the game process that hold this service

			// notify the game process, deal the rtsp cmd
		}
		else{
			cg::core::infoRecorder->logTrace("[LogicServer]: failed to recv description from rtsp client.\n");
			return false;
		}

		return true;
	}

	// start the game process
	void LogicServer::startGame(char * gameName, IDENTIFIER tid){
		cg::core::infoRecorder->logTrace("[LogicServer]: start the game '%s'.\n", gameName);

		char cmdline[100] = {0};

		PROCESS_INFORMATION pi = {0};
		STARTUPINFO si = {0};
		si.cb = sizeof(si);
		si.wShowWindow = SW_NORMAL;
		si.cbReserved2 = NULL;
		si.lpDesktop = NULL;
		si.dwFlags = STARTF_USESHOWWINDOW;

		sprintf(cmdline, "cmd.exe /k %s %d > %s.result.log", "TestFakeGame", tid, "TestFakeGame");
		cg::core::infoRecorder->logTrace("[LogicServcer]: start game cmdline:%s\n", cmdline);

		BOOL ret = CreateProcess(NULL, cmdline, NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi);
		cg::core::infoRecorder->logTrace("[LogicServer]: end of process creation. RET:%d\n", ret);
	}

	// callback functions for dealing graphic connection
	// read callback
	static void GraphicReadCB(struct bufferevent * bev, void * arg){
		cg::core::infoRecorder->logTrace("[GraphicReadCB]: read data from render proxy.\n");
		LogicServer * logicServer = LogicServer::GetLogicServer();

		BaseContext * ctx = (BaseContext *)arg;

		struct evbuffer * input = bufferevent_get_input(bev);
		size_t n = evbuffer_get_length(input);
		char * data = (char *)malloc(sizeof(char ) *n);
		evbuffer_copyout(input, data, n);

		// deal the event
		ctx->setData(data, n);
		logicServer->dealEvent(ctx);

		// free temp dataS
		free(data);
		data = NULL;
		// remove current data from input buffer
		evbuffer_drain(input, n);
	}

	// write callback
	static void GraphicWriteCB(struct bufferevent * bev, void *arg){
		cg::core::infoRecorder->logTrace("[GraphicWriteCB]: write data to render proxy.\n");
		struct evbuffer * output= bufferevent_get_output(bev);
		if(evbuffer_get_length(output) == 0){
			cg::core::infoRecorder->logTrace("[GraphicWriteCB]: flushed answer.\n");
			//bufferevent_free(bev);
		}
	}

	// event callback
	static void GraphicEventCB(struct bufferevent * bev, short events, void * arg){
		cg::core::infoRecorder->logTrace("[GraphciEventCB]: deal network event from render proxy.\n");

		if(events & BEV_EVENT_EOF){
			cg::core::infoRecorder->logTrace("[GraphicEventCB]: connection closed.\n");
		}
		else if(events & BEV_EVENT_ERROR){
			cg::core::infoRecorder->logTrace("[GraphicEventCB]: got an error on the connection :%s.\n", strerror(errno));
		}
		bufferevent_free(bev);

	}


	// callback functions for dealing game connection
	// read callback
	static void GameReadCB(struct bufferevent * bev, void * arg){
		cg::core::infoRecorder->logTrace("[GameReadCB]: read data from Game client.\n");
		LogicServer * logicServer = LogicServer::GetLogicServer();

		BaseContext * ctx = (BaseContext *)arg;

		struct evbuffer * input = bufferevent_get_input(bev);
		size_t n = evbuffer_get_length(input);
		char * data = (char *)malloc(sizeof(char ) *n);
		evbuffer_copyout(input, data, n);

		// deal the event
		ctx->setData(data, n);
		logicServer->dealEvent(ctx);

		// free temp dataS
		free(data);
		data = NULL;
		// remove current data from input buffer
		evbuffer_drain(input, n);
	}

	// write callback
	static void GameWriteCB(struct bufferevent * bev, void *arg){
		cg::core::infoRecorder->logTrace("[GameWriteCB]: write data to Game client.\n");
		struct evbuffer * output= bufferevent_get_output(bev);
		if(evbuffer_get_length(output) == 0){
			cg::core::infoRecorder->logTrace("[GameWriteCB]: flushed answer.\n");
			//bufferevent_free(bev);
		}
	}

	// event callback
	static void GameEventCB(struct bufferevent * bev, short events, void * arg){
		cg::core::infoRecorder->logTrace("[GameEventCB]: deal network event from Game client.\n");

		if(events & BEV_EVENT_EOF){
			cg::core::infoRecorder->logTrace("[GameEventCB]: connection closed.\n");
		}
		if(events & BEV_EVENT_CONNECTED){
			cg::core::infoRecorder->logTrace("[GameEventCB]: game client connected.\n");
		}
		else if(events & BEV_EVENT_ERROR){
			cg::core::infoRecorder->logTrace("[GameEventCB]: got an error on the connection :%s.\n", strerror(errno));
		}
		bufferevent_free(bev);

	}

	// lstener callback
	static void
		GraphicListenerCB(struct evconnlistener * listener, evutil_socket_t fd, struct sockaddr * sa, int socklen, void * arg){
			cg::core::infoRecorder->logTrace("[GraphicListenerCB]: got an new connection.\n");
			LogicServer * logicServer = (LogicServer *)arg;
			event_base * base = logicServer->getBase();

			struct bufferevent * bev = NULL;
			bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);
			if(!bev){
				cg::core::infoRecorder->logTrace("[GraphicListenerCB]: error constructing bufferevent.\n");
				event_base_loopbreak(base);
				return;
			}

			// create new context for graphic
			BaseContext * ctx = new BaseContext();
			ctx->bev = bev;
			ctx->sock = fd;

			logicServer->addRenderContext(ctx);


			bufferevent_setcb(bev, GraphicReadCB, GraphicWriteCB, GraphicEventCB, ctx);
			bufferevent_enable(bev, EV_WRITE | EV_READ);
			// graphic connection established

	}

	// lstener callback
	static void
		GameListenerCB(struct evconnlistener * listener, evutil_socket_t fd, struct sockaddr * sa, int socklen, void * arg){
			cg::core::infoRecorder->logTrace("[GameListenerCB]: got an new connection.\n");
			LogicServer * logicServer = (LogicServer *)arg;
			event_base * base = logicServer->getBase();

			struct bufferevent * bev = NULL;
			bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);
			if(!bev){
				cg::core::infoRecorder->logTrace("[GameListenerCB]: error constructing bufferevent.\n");
				event_base_loopbreak(base);
				return;
			}

			// create new context for graphic
			BaseContext * ctx = new BaseContext();
			ctx->bev = bev;
			ctx->sock = fd;

			//logicServer->addRenderContext(ctx);


			bufferevent_setcb(bev, GameReadCB, GameWriteCB, GameEventCB, ctx);
			bufferevent_enable(bev, EV_WRITE | EV_READ);
			// Game connection established
			cg::core::infoRecorder->logTrace("[LogicServer]: game connection established, FD:%p.\n", fd);

	}

	static void GraphicAcceptErrorCB(struct evconnlistener * listener, void *ctx){
		struct event_base * base = evconnlistener_get_base(listener);
		int err = EVUTIL_SOCKET_ERROR();
		cg::core::infoRecorder->logTrace("[GraphicAcceptErrorCB]: Got an error %d (%s) on the listener. " "Shutting down.\n", err, evutil_socket_error_to_string(err));
		event_base_loopexit(base, NULL);
	}

	// listen to game process connection
	bool LogicServer::startListenGameProcess(){
		cg::core::infoRecorder->logTrace("[LogicServer]: start listen game process connection.\n");
		sockaddr_in sin;
		int sin_size = sizeof(sin);
		memset(&sin, 0, sin_size);
		sin.sin_family = AF_INET;
		sin.sin_addr.S_un.S_addr = htonl(0);
		sin.sin_port = htons(DIS_PORT_CLIENT);


		gameListener = evconnlistener_new_bind(base, GameListenerCB, this, /*LEV_OPT_LEAVE_SOCKETS_BLOCKING |*/ LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size);

		if(!gameListener)
		{
			cg::core::infoRecorder->logTrace("[LogicServer]: couldn't create game listener.\n");
			return false;
		}
		evconnlistener_set_error_cb(gameListener, GraphicAcceptErrorCB);

		return true;
	}

	// start to listen to graphic port
	bool LogicServer::startGraphic(){
		cg::core::infoRecorder->logTrace("[LogicServer]: start to listen to graphic port:%d, wait render proxy to connect.\n", DIS_PORT_GRAPHIC);
		sockaddr_in sin;
		int sin_size = sizeof(sin);
		memset(&sin, 0, sin_size);
		sin.sin_family = AF_INET;
		sin.sin_addr.S_un.S_addr = htonl(0);
		sin.sin_port = htons(DIS_PORT_GRAPHIC);


		graphicListener = evconnlistener_new_bind(base, GraphicListenerCB, this, LEV_OPT_LEAVE_SOCKETS_BLOCKING | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size);

		if(!graphicListener)
		{
			cg::core::infoRecorder->logTrace("[LogicServer]: couldn't create listener.\n");
			return false;
		}
		evconnlistener_set_error_cb(graphicListener, GraphicAcceptErrorCB);
		return true;
	}

	// the ctx is the render connection's contxt, the socket is the IDENTIFIER of this render
	bool LogicServer::addRenderContext(BaseContext * _ctx){
		cg::core::infoRecorder->logTrace("[LogicServer]: add render connection context.\n");

		map<IDENTIFIER, BaseContext *>::iterator it = renderMap.find(_ctx->sock);
		if(it != renderMap.end()){
			// exist

		}
		else{
			// create new map
			cg::core::infoRecorder->logTrace("[LogicServer]: add new render connection.\n");
			renderMap[_ctx->sock] = _ctx;
			netCtxMap[_ctx->sock] = _ctx;
		}

		return true;
	}

#endif

}