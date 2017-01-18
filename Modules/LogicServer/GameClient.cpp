#include "GameClient.h"
#include "LogicContext.h"

#include "../VideoGen/generator.h"
//#include "../LibDistrubutor/Context.h"

#include "CommandServerSet.h"
#include "../LibCore/CmdHelper.h"

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

#endif

#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"d3dx9.lib")

extern CommandServerSet * csSet;

///////////// GameClient /////////////////
// callback for game client
void GameClientReadCB(struct bufferevent * bev, void * arg){
	GameClient * client = (GameClient*)arg;
	struct evbuffer * input = bufferevent_get_input(bev);
	size_t n = evbuffer_get_length(input);
	char * data = (char *)malloc(sizeof(char)* n);
	evbuffer_copyout(input, data, n);
	printf("[GameClientReadCB]: read '%s'\n", data);
	// deal event
	BaseContext * ctx = client->getCtx();
	ctx->setData(data, n);

	client->dealEvent(ctx);
	free(data);
	data = NULL;
	// remove the data from buffer
	evbuffer_drain(input, n);
}

// callback for DisClient
void GameClientWriteCB(struct bufferevent * bev, void * arg){
	struct evbuffer * output = bufferevent_get_output(bev);
	int len = 0;
	if((len = evbuffer_get_length(output)) == 0){
		printf("[GameClientWriteCB]: flushed answer.\n");
	}
	else{
		printf("[GameClientWriteCB]: write %d bytes.\n", len);
	}
}

// event callback
void GameClientEventCB(struct bufferevent * bev, short what, void * arg){
	GameClient * client = (GameClient*)arg;

	if(what & BEV_EVENT_ERROR){
		// error
		perror("[GameClientEventCB]: error from bufferevent.");
		int err = EVUTIL_SOCKET_ERROR();
		printf("[DisClientEventCB]: error occur. err: %d (%s)\n", err, evutil_socket_error_to_string(err));
	}

	if(what & BEV_EVENT_CONNECTED){
		printf("[GameClientEventCB]: connection created.\n");
	}
#if 1
	if(what & (BEV_EVENT_EOF)){
		printf("[GameClientEventCB]: BEV_EVENT_EOF may error:%d.\n", WSAGetLastError());
		perror("[GameClientEventCB]: error EOF.\n");
		//DebugBreak();
		bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
		delete client;
		bufferevent_disable(bev, EV_READ | EV_WRITE);
		bufferevent_free(bev);
	}
#endif
}

GameClient * GameClient::gameClient = NULL;
bool GameClient::initialized = false;

GameClient::GameClient(){
	sock = NULL;
	ctx = NULL;
	gameName = NULL;
	base = NULL;
	ctrlSock = NULL;
	ctrlThreadHandle = NULL;
	rtspThreadHandle = NULL;
	ctrlThreadId = 0;
	rtspThreadID = 0;
	clientEvent =  CreateEvent(NULL, FALSE, FALSE, NULL);
}
GameClient::~GameClient(){
	if(gameName){
		free(gameName);
		gameName = NULL;
	}
	if(clientEvent){
		CloseHandle(clientEvent);
		clientEvent = NULL;
	}
}

bool GameClient::dealEvent(BaseContext * ctx){
	bool ret = true;
	char feedback[1024] = { 0 };
	ctx->readCmd();
	int len = 0;
	char * cmd = ctx->getCmd();
	char * data = ctx->getData();

	if (!strncasecmp(cmd, COPY_HANDLE, strlen(COPY_HANDLE))){
		// copy the old handle, should feedback to cancel listen
	}
	else if (!strncasecmp(cmd, ADD_RENDER, strlen(ADD_RENDER))){
		// add render to context
		evutil_socket_t oldRenderSock = *(evutil_socket_t *)data;
		DWORD ppid = *(DWORD*)(data + sizeof(evutil_socket_t));
		// get the new socket 
		infoRecorder->logTrace("[GameClient]: add render, old sock:%p, process id:%d.\n", oldRenderSock, ppid);
		SOCKET newRenderSock = GetProcessSocket(oldRenderSock, ppid);
		handleMap[oldRenderSock] = newRenderSock;

		if (!addRenderConnection(newRenderSock)){
			printf("[GameClient]: add render connection failed.\n");
			return false;
		}
		// feedback
		ctx->writeCmd(CANCEL_SOCKEVENT);
		ctx->writeIdentifier(oldRenderSock);
		ctx->writeToNet();
	}
	else if (!strncasecmp(cmd, DECLINE_RENDER, strlen(DECLINE_RENDER))){
		//decline a render
		evutil_socket_t oldRenderSock = *(evutil_socket_t *)data;

		// get the new socket
		evutil_socket_t renderSock = NULL;
		map<IDENTIFIER, IDENTIFIER>::iterator it = handleMap.find(oldRenderSock);

		if (it != handleMap.end()){
			if (!declineRenderConnection(it->second)){
				printf("[GameClient]: decline render connection failed.\n");
				return false;
			}
			ctx->writeCmd(DESTROY_SOCKEVENT);
			ctx->writeIdentifier(oldRenderSock);
			ctx->writeToNet();
		}
		else{
			printf("[GameClient]: no handle map to old %p.\n", oldRenderSock);
			return false;
		}
		// feedback

	}
	else if (!strncasecmp(cmd, ADD_CTRL_CONNECTION, strlen(ADD_CTRL_CONNECTION))){
		// add the ctrl connection
		evutil_socket_t oldCtrlSock = *(evutil_socket_t *)data;
		DWORD ppid = *(DWORD*)(data + sizeof(evutil_socket_t) + 1);
		// copy the socket handle

		SOCKET ctrolSock = GetProcessSocket(oldCtrlSock, ppid);
		handleMap[oldCtrlSock] = ctrlSock;

		if (!startControlThread(ctrolSock)){
			printf("[GameClient]: start control failed.\n");

			return false;
		}

		// feedback
		ctx->writeCmd(CANCEL_SOCKEVENT);
		ctx->writeIdentifier(oldCtrlSock);
		ctx->writeToNet();
	}
	else if (!strncasecmp(cmd, ADD_RTSP_CONNECTION, strlen(ADD_RTSP_CONNECTION))){
		// add rtsp connection to this process
#if 0
		evutil_socket_t oldRtspSock = *(evutil_socket_t *)data;
		DWORD ppid = *(DWORD *)(data + sizeof(evutil_socket_t) + 1);
		// copy the socket handle
		SOCKET rtspSock = GetProcessSocket(oldRtspSock, ppid);
		handleMap[oldRtspSock] = rtspSock;

		// start the rtsp thread
		if(!startRTSPThread(rtspSock)){
			printf("[GameClient]: start rtsp failed.\n");
			return false;
		}

		// feedback
		ctx->writeCmd(CANCEL_SOCKEVENT);
		ctx->writeData((void *)&oldRtspSock, sizeof(evutil_socket_t));
		ctx->writeToNet();
#else
		// get the port offset and start listen
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;
		short portOff = *(short *)(data + sizeof(IDENTIFIER )+ 1);
		printf("[GameClient]: rtsp port offset is :%d.\n", portOff);
		
		event_base * _base = bufferevent_get_base(ctx->bev);

		VideoGen * gen = VideoGen::findVideoGen(id);
		if(!gen){
			infoRecorder->logError("[GameClient]: the video generator for '%p' is not created yet.\n",id);
			return false;
		}else{
			infoRecorder->logTrace("[GameClient]: find video generator '%p' for id:%p.\n", gen, id);
		}
		infoRecorder->logTrace("[GameClient]: add rtsp connection for task '%p', port offset:%d, rtsp context:%p.\n", id, portOff, gen->getContext());
		evconnlistener * rtspListener = listenPort(DIS_PORT_RTSP + portOff, _base, gen->getContext());

		// TODO, manage the listener
		// notify the game loader
		ctx->writeCmd(RTSP_READY);
		ctx->writeIdentifier(id);
		ctx->writeToNet();
#endif
	}else if(!strncasecmp(cmd, CANCEL_RTSP_SERVICE, strlen(CANCEL_RTSP_SERVICE))){
		// cancel the rtsp service for the game process
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;
		VideoGen * gen = VideoGen::findVideoGen(id);
		if(!gen){
			// error
			infoRecorder->logError("[GameClient]: the rtsp service for '%d' has not created yet.\n", id);
		}else{
			gen->getContext()->enableGen = false;  // stop to generate video
		}
	}else if(!strncasecmp(cmd, OPTION, strlen(OPTION))){
		// deal with option command
		// format:[cmd][id][OPTION count]{[option][value] ... [option][value]
		infoRecorder->logError("[GameClient]: get Option.\n");
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;
		VideoGen *gen = VideoGen::findVideoGen(id);
		if(!gen){
			// error
			infoRecorder->logError("[GameClient]: the generator for '%d' has not created yet.\n", id);
			return false;
		}
		infoRecorder->logError("[GameClient]: find the generator:%p.\n", gen);
		// now 
		short optionCount = *(short *)(data + sizeof(IDENTIFIER));
		for(int i = 0; i < optionCount; i++){
			CMD_OPTION option = *(CMD_OPTION *)(data + sizeof(IDENTIFIER) + sizeof(short) + i * (sizeof(CMD_OPTION) + sizeof(short)));
			short value = *(short *)(data + sizeof(IDENTIFIER) + sizeof(short) + i * (sizeof(CMD_OPTION) + sizeof(short)) + sizeof(CMD_OPTION));
			if(!dealCmdOption(option, value, gen)){
				infoRecorder->logError("[GameClient]: deal cmd option %d failed.\n");
			}
#if 0
			switch(CMD_OPTION){
			case SETOFFLOAD:

				break;
			case ENCODEROPTION:

				break;
			default:
				infoRecorder->logError("[GameClient]: unknown cmd option:%d.\n", option);
				break;
			}
#endif
		}
	}
	else if(!strncasecmp(cmd, "TEST", strlen("TEST"))){
		infoRecorder->logError("[GameClient]: recv TEST after send GAME_READY\n");
		return true;
	}
	else{
		infoRecorder->logError("[GameClient]: get unknown cmd:%s.\n", cmd);
		return false;
	}

	return ret;
}
// deal with the cmd option for game process, include change the encoder type and set offloading level
bool GameClient::dealCmdOption(cg::CMD_OPTION option, short value, cg::VideoGen * gen){
	cg::core::CmdController * cmd = NULL;
	switch(option){
	case ENCODEROPTION:
		infoRecorder->logTrace("[GameClient]: deal cmd option: ENCODER OPTION, val:%d.\n", value);

		// just change the encoder type just now
#if 0
		gen->postThreadMsg(WM_USR_ENCODER, 0, 0);
#else
		gen->onThreadMsg(WM_USR_ENCODER, 0, 0);
#endif
		return true;
		break;
	case SETOFFLOAD:
		infoRecorder->logTrace("[GameClient]: deal cmd option: SET OFFLOAD, val:%d.\n", value);
		cmd = cg::core::CmdController::GetCmdCtroller();
		if(value)
			cmd->setFullOffload();
		return true;
		break;
	default:
		infoRecorder->logError("[GameClient]: deal cmd option: unknown %d.\n", option);
		return false;
		break;
	}
}

// connect to game loader process when dll is injected to game process
void GameClient::connectToLogicServer(){
	infoRecorder->logTrace("[GameClient]: connect to logic server.\n");
	//evutil_socket_t sock = NULL;
	cg::RTSPConf * config = cg::RTSPConf::GetRTSPConf();


	sockaddr_in sin;
	int sin_size = sizeof(sin);
	sin.sin_family = AF_INET;
	sin.sin_addr.S_un.S_addr = inet_addr(LOGCAL_HOST);
	//sin.sin_port = htons(INTERNAL_PORT);
	sin.sin_port = htons(config->loaderPort);

	struct bufferevent * bev = NULL;

	sock = socket(AF_INET, SOCK_STREAM, 0);
#if 0
	// connet
	if(connect(sock, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		int e = errno;
		printf("[GameClient]: connect to logic server failed with:%d.\n", WSAGetLastError());
		char msg[1024] = {0};
		sprintf(msg, "connect logic failed with:%d.", WSAGetLastError());
		MessageBox(NULL, msg, "WARNING", MB_OK);
		return;
	}
	if(evutil_make_socket_nonblocking(sock) < 0){
		printf("[GameClient]: make socket non blocking failed.\n");
		return;
	}
	//frobSocket(sock);
#endif

	bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE);
	// set callback function
	infoRecorder->logTrace("[GameClient]: try to connect.\n");
	if(bufferevent_socket_connect(bev, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		// error starting connection
		printf("[GameClient]: connect to logic server failed.\n");
		infoRecorder->logError("[GameClient]: connect to logic server failed.\n");
		bufferevent_free(bev);
		return ;
	}
	bufferevent_setcb(bev, GameClientReadCB, GameClientWriteCB, GameClientEventCB, this);

	bufferevent_enable(bev, EV_READ | EV_WRITE);
	//sock = bufferevent_getfd(bev);
	if(!ctx){
		ctx = new BaseContext();
	}
	else{
		infoRecorder->logError("[GameClient]: ctx is not NULL.\n");
		printf("[GameClient]: ctx is not NULL.\n");
		MessageBox(NULL, "ctx is not NULL.", "WARNING", MB_OK);
	}
	ctx->sock = sock;
	ctx->bev = bev;
	infoRecorder->logTrace("[GameClient]: connect to logic server succeeded.\n");
}

// add the given connection, send the command stream
bool GameClient::addRenderConnection(SOCKET sock){
	infoRecorder->logTrace("[GameClient]: add render connection to game '%s' process.\n", gameName);
	BaseContext * ctx = NULL;
	struct bufferevent * bev = NULL;
	map<IDENTIFIER, BaseContext *>::iterator it = renderCtxMap.find(sock);
	if(it != renderCtxMap.end()){
		// find
		ctx = it->second;
	}
	else{
		// create new context
		ctx = new BaseContext();
		ctx->sock = sock;
		renderCtxMap[sock] = ctx;
	}
	// change the logic context
	if(!csSet){
		csSet = CommandServerSet::GetServerSet();
	}

	// tunning sending window
#if 1   // set send buffer size
	do {
		int sndwnd = 8 * 1024 * 1024;	// 8MB
		if(setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char *)&sndwnd, sizeof(sndwnd)) == 0) {
			infoRecorder->logTrace("*** set TCP sending buffer success.\n");
		} else{
			infoRecorder->logTrace("*** set TCP sending buffer failed.\n");

		}
#if 0    // set non-blocking
		// set nonblocking??????
		u_long iMode = 1;  // non-blocking mode is enabled
		if(ioctlsocket(sock, FIONBIO, &iMode) == -1){
			infoRecorder->logTrace("*** set TCP send non-blocking failed.\n");
		}
		else{
			infoRecorder->logTrace("*** set TCP send non-blocking succ.\n");
		}
#endif   // set non-blocking

		const char chOpt = 1;
		if(setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,&chOpt, sizeof(char))){
			infoRecorder->logTrace("**** set TCP send no delay failed, ERROR CODE:%d.\n", GetLastError());
		}
		else{
			infoRecorder->logTrace("**** set TCP send no delay succ.\n");
		}


	} while(0);
	//
#endif

	csSet->addServer(sock);

	return true;
}

// decline the given connection of a render
bool GameClient::declineRenderConnection(SOCKET sock){
	printf("[GameClient]: decline render connection to game '%s' process.\n");
	BaseContext * ctx = NULL;
	struct bufferevent * bev = NULL;
	map<IDENTIFIER, BaseContext *>::iterator it = rtspCtxMap.find(sock);
	if(it != rtspCtxMap.end()){
		/// find
		ctx = it->second;
	}
	else{
		// create new context
		ctx = new BaseContext();
		ctx->sock = sock;
		bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE);
		ctx->bev = bev;
		rtspCtxMap[sock] = ctx;
	}

	csSet->declineServer(sock);

	return true;
}
bool GameClient::startControlThread(SOCKET sock){
	printf("[GameClient]: control for game '%s' is from remote socket %p.\n", gameName, sock);


	return true;
}

bool GameClient::startRTSPThread(SOCKET sock){
	printf("[GameClient]: RTSP service for game '%s'.\n", gameName);
	// create a new thread for rtsp service
	rtspThreadHandle = chBEGINTHREADEX(NULL, 0, RTSPThreadProc, sock, FALSE, &rtspThreadID);

	return true;
}

// thread proc for control
DWORD GameClient::CtrlThreadProc(LPVOID param){
	printf("[GameClient]: Control thread.\n");


	return 0;
}

// thread proc for RTSP thread
DWORD GameClient::RTSPThreadProc(LPVOID param){
	printf("[GameClient]: RTSP thread.\n");
	evutil_socket_t fd = (evutil_socket_t)param;
	char buf[1024] = {0};


	return 0;
}



/////////////////////for listernServer /////////////////////

static void ListenServerEventCB(struct bufferevent * bev, short events, void * ctx){
	if(events & BEV_EVENT_ERROR){
		int err = GetLastError();
		infoRecorder->logError("[ListenServerEventCB]: error from bufferevent, last err:%d.\n", err);
		if(err == 10054){
			// closed by others
			infoRecorder->logError("[ListenServerEventCB]: connection reset by others.\n");
			ListenServer  * server=  (ListenServer *)ctx;
			server->declineRenderConnection(bufferevent_getfd(bev));
		}
	}
	if(events & BEV_EVENT_EOF){
		// connection closed
		infoRecorder->logError("[ListenServerEventCB]: connection closed.\n");
		ListenServer * server = (ListenServer *)ctx;
		// remove the closed context
		server->declineRenderConnection(bufferevent_getfd(bev));
	}
	if(events & (BEV_EVENT_EOF | BEV_EVENT_ERROR)){
		bufferevent_free(bev);
	}
	infoRecorder->logError("[ListenServerEventCB]: event deal done.\n");
}

static void ListenServerAcceptConnCB(struct evconnlistener * listener, evutil_socket_t fd, struct sockaddr *address, int socklen, void * ctx){
	// got a new connection for rendering
	infoRecorder->logError("[ListenerServerAcceptConnCB]: get a new connection: %p.\n", fd);
	ListenServer * server = (ListenServer *)ctx;
	struct event_base * base = evconnlistener_get_base(listener);
	struct bufferevent * bev = bufferevent_socket_new(base, fd, BEV_OPT_CLOSE_ON_FREE);

	// set callbacks for buffer event ???
	bufferevent_setcb(bev, NULL, NULL, ListenServerEventCB, ctx);
	bufferevent_enable(bev, EV_READ|EV_WRITE);

	// add context
	server->addRenderConnection(fd);

}
static void ListenerServerAcceptErrorCB(struct evconnlistener * listener, void *ctx){
	struct event_base * base = evconnlistener_get_base(listener);
	int err =  EVUTIL_SOCKET_ERROR();
	// log the error
	infoRecorder->logError("[ListenerServerAcceptErrorCB]: error %d (%s) on listener. Shutting down.\n", err, evutil_socket_error_to_string(err));
	event_base_loopexit(base, NULL);
}


bool ListenServer::addRenderConnection(SOCKET sock){
	if(!_csSet)
		return false;
	CommandServerSet * c = (CommandServerSet *)_csSet;
	c->addServer(sock);
	return true;
}

bool ListenServer::declineRenderConnection(SOCKET sock){
	if(!_csSet)
		return false;
	CommandServerSet * c = (CommandServerSet *)_csSet;
	c->declineServer(sock);
	return true;
}
bool ListenServer::startListen(int port){
	// the base is set, build the connlistener
	if(!base){
		// error
		return false;
	}
	struct sockaddr_in sin;
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_addr.S_un.S_addr = htonl(INADDR_ANY);  // listen any connection
	sin.sin_port = htons(port);
	listener = evconnlistener_new_bind(base, ListenServerAcceptConnCB, this, LEV_OPT_CLOSE_ON_FREE |LEV_OPT_REUSEABLE, -1, (struct sockaddr *)&sin, sizeof(sin));
	if(!listener){
		infoRecorder->logError("[ListenServer]: couldn't create listener.\n");
		return false;
	}
	evconnlistener_set_error_cb(listener, ListenerServerAcceptErrorCB);

	return true;
}
ListenServer::ListenServer():sock(NULL), ctx(NULL), listener(NULL), _csSet(NULL), base(NULL){ }
ListenServer::~ListenServer(){ }
void ListenServer::dispatch(){
	event_base_dispatch(base);
}
bool ListenServer::dealEvent(cg::BaseContext * ctx){
	bool ret = true;
	char feedback[1024] = {0};
	ctx->readCmd();
	int le = 0;
	char * cmd = ctx->getCmd();
	char * data = ctx->getData();

	return true;
}