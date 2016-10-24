#include "disforclient.h"
#include "../LibCore/Log.h"

//#include "../LibInput/CtrlSdl.h"

#include "client.h"
#include "../LibCore/InfoRecorder.h"

#ifndef USE_LIBEVENT

#else

using namespace cg;
using namespace cg::core;

/////////////// UserClient /////////////
bool UserClient::dealEvent(BaseContext * ctx){
	ctx->readCmd();
	char * cmd = ctx->getCmd();
	char * data = ctx->getData();
	int len = 0;

	if (!strncasecmp(cmd, ADD_RENDER, strlen(ADD_RENDER))){
		infoRecorder->logTrace("[UserClient]: ADD RENDER.\n");
		// get the render url, and start the slave thread to receive RTSP stream
		short portOff = *(short *)data;
		char * renderUrl = data + sizeof(short);

		startRTSP(renderUrl, DIS_PORT_RTSP + portOff);

	}
	else if (!strncasecmp(cmd, DECLINE_RENDER, strlen(DECLINE_RENDER))){
		infoRecorder->logTrace("[UserClient]: DECLINE RENDER.\n");
		// remove the given render
		IDENTIFIER rid = *(IDENTIFIER *)data;
		data += sizeof(IDENTIFIER) + 1;

		char *  renderUrl = data;
		cancelRTSP(rid, renderUrl);
		// TODO, client cancel render, feed back or not?
	}
	else if (!strncasecmp(cmd, ADD_LOGIC, strlen(ADD_LOGIC))){
		infoRecorder->logError("[UserClient]: ADD LOGIC.\n");
		//create the control thread
		char * logicUrl = data;

		addLogic(logicUrl);
	}
	else{
		infoRecorder->logTrace("[UserClient]: unknown cmd.\n");
		return false;
	}
	return true;
}

bool UserClient::startRTSP(char * url, int port){

	char rtspUrl[100] = {0};

	infoRecorder->logTrace("[UserClient]: start rtsp thread, url:%s, port:%d.\n", url, port);
	if (rtspCount + 1 <= MAX_RENDER_COUNT){
		//how to create the connection  ??????????
		// create sub game stream and start the sub game stream thread
		infoRecorder->logTrace("[Client]: rtsp index:%d, url:%s, port:%d, name:%s.\n", rtspCount, url, port, gameName);
		
		sprintf(rtspUrl, "rtsp://%s:%d/%s", url, port, gameName);
		SubGameStream * subStream = new SubGameStream(rtspUrl);
		
		gameStream->addSubStream(subStream);

		DWORD tid = 0;
		HANDLE handle = chBEGINTHREADEX(NULL, 0, rtspThreadForSubsession, subStream, FALSE, &tid);

		subStream->setThread(handle);

		threadIDs[rtspCount] = tid;
		threadHandles[rtspCount] = handle;
		renderUrls[rtspCount] = _strdup(url);
		rtspCount++;
	}
	else{
		infoRecorder->logTrace("[UserClient]: rtsp connection is FULL.\n");
		return false;
	}

	return true;
}

bool UserClient::cancelRTSP(IDENTIFIER rid, char * url){
	infoRecorder->logTrace("[UserClient]: cancel the rtsp connection.\n");

	// search the rid
	int i = 0;
	for (i = 0; i < rtspCount; i++){
		if (threadIDs[i] == rid && !strcmp(url, renderUrls[i])){
			// find the render to cancel
			break;
		}
	}
	// do something to cancel the rtsp connection
	shutdownRTSP(rid);
	{
		// release something
		free(renderUrls[i]);
	}

	// if no 
	if (rtspCount < 1){
		infoRecorder->logTrace("[UserClient]: error, no rtsp connection.\n");
		return false;
	}
	// remove the connection info
	for (int j = i; j < rtspCount - 1; j++){
		threadIDs[j] = threadIDs[j + 1];
		threadHandles[j] = threadHandles[j + 1];
		renderUrls[j] = renderUrls[j + 1];
	}

	return true;
}

cg::input::CtrlMessagerClient * gCtrlClient = NULL;

// mainly handle the control ????
bool UserClient::addLogic(char * url){
	infoRecorder->logError("[UserClient]: add logic %s.\n", url);
#ifdef ENABLE_CLIENT_CONTORL
	ctrlConf = CtrlConfig::GetCtrlConfig(STREAM_CLIENT_CONFIG);
	ctrlClient = new CtrlMessagerClient();
	do{
		if(conf->ctrlenable){
			if(ctrlClient->initQueue(32768, sizeof(sdlmsg_t)) < 0){
				conf->ctrlenable = 0;
				break;
			}
			if(!ctrlClient->init(conf, CTRL_CURRENT_VERSION)){
				infoRecorder->logError("[clinet]: cannot init the contorller");
			}

		}
	}while(0);

#endif

#if 0
	//connect to the logic url
	bool ret = true;
	evutil_socket_t ctrlSock = NULL;

	//char * url = NULL;
	unsigned short port = DIS_PORT_CTRL;   // use control port
	sockaddr_in sin;

	int sin_size = sizeof(sin);
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr(url);
	sin.sin_port = htons(port);

	struct bufferevent * bev = NULL;

	ctrlSock = socket(AF_INET, SOCK_STREAM, 0);
	evutil_make_socket_nonblocking(ctrlSock);

	// connect to logic
	if (connect(ctrlSock, (struct sockaddr*)&sin, sizeof(sin)) < 0) {
		int e = errno;
		if (!EVUTIL_ERR_CONNECT_RETRIABLE(e)) {
			infoRecorder->logTrace("[UsrClient]: connect to logic server failed.\n");
			evutil_closesocket(sock);
			return -1;
		}
	}
	// create new BaseContext
	BaseContext * ctrlCtx = new BaseContext();

	// send the CLIENT_CONNECTED cmd with identifier
	ctrlCtx->sock = ctrlSock;
	ctrlCtx->writeCmd(CLIENT_CONNECTED);
	ctrlCtx->writeData((void *)&this->sock, sizeof(IDENTIFIER));
	ctrlCtx->writeToNet();
#endif

	cg::input::CtrlConfig * ctrlConf = cg::input::CtrlConfig::GetCtrlConfig(CTRL_CLIENT_CONFIG);
	if (!ctrlConf){
		infoRecorder->logError("[client]: create new ctrl config failed.\n");
	}
	// init the ctrl messager
	//cg::input::CtrlMessagerClient * ctrlClient = new cg::input::CtrlMessagerClient();
	ctrlClient = new cg::input::CtrlMessagerClient();
	
	//ctrlClient->init(rtspConf, CTRL_CURRENT_VERSION);
	// launch controller?

	ctrlConf->ctrlenable = true;  // temp set false;
	ctrlConf->ctrl_servername = _strdup(url);

	do{
		if (ctrlConf->ctrlenable){
			if (ctrlClient->initQueue(32768, sizeof(cg::input::sdlmsg_t)) < 0){
				rtsperror("Cannot initialize controller queue, controller disabled.\n");
				infoRecorder->logError("[Client]: cannot initialize controller queue, controller disable.\n");
				ctrlConf->ctrlenable = 0;
				break;
			}
			ctrlClient->init(ctrlConf, CTRL_CURRENT_VERSION);
			if (!ctrlClient->start()){
				rtsperror("Cannot create controller thread, controller disabled.\n");
				infoRecorder->logError("[Client]: cannot create controller thread, controller disabled.\n");
				ctrlConf->ctrlenable = 0;
				break;
			}
		}
	} while (0);

	infoRecorder->logError("[Client]: after create the controller thread, ctrlClient: %p.\n",ctrlClient);
	gCtrlClient = ctrlClient;
	return true;
}



DWORD UserClient::ClientThreadProc(LPVOID param){
	infoRecorder->logTrace("[UserClient]: enter client thread proc.\n");


	return 0;
}

bool UserClient::shutdownRTSP(IDENTIFIER rid){
	infoRecorder->logTrace("[UserClient]: shutdown the rtsp connection.\n");


	return true;
}

bool UserClient::launchRequest(char * disServerUrl, int port, char * gameName){
	infoRecorder->logTrace("[UserClient]: launch the request, server:%s:%d, name:%s.\n", disServerUrl, port, gameName);
	bufferevent * bev = NULL;

	memset(&sin, 0, sizeof(sin));

	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr(disServerUrl);
	sin.sin_port = htons(DIS_PORT_DOMAIN);
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0){
		infoRecorder->logTrace("[UserClient]: create socket failed.\n");
		return false;
	}

	frobSocket(sock);

	if (connect(sock, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		int e = errno;
		if (!EVUTIL_ERR_CONNECT_RETRIABLE(e)){
			infoRecorder->logTrace("[UserClient]: connect to logic server failed.\n");
			evutil_closesocket(sock);
			return false;
		}
	}

	bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE);

	bufferevent_setcb(bev, clientReadCB, NULL, clientErrorCB, this);
	bufferevent_enable(bev, EV_READ | EV_WRITE);

	// send the request
	if(!ctx)
		ctx = new BaseContext();
	ctx->sock = sock;
	ctx->bev = bev;

	ctx->writeCmd(REQ_GAME);
	ctx->writeData(gameName, strlen(gameName));
	ctx->writeToNet();

	return true;
}




// callback functions for user client
void clientReadCB(struct bufferevent * bev, void * arg){
	UserClient * client = (UserClient *)arg;
	struct evbuffer * input = bufferevent_get_input(bev);
	size_t n = evbuffer_get_length(input);
	char *data = (char *)malloc(sizeof(char) * n);
	evbuffer_copyout(input, data, n);

	//deal something
	BaseContext * ctx = client->getCtx();
	ctx->setData(data, n);

	client->dealEvent(ctx);
	if(data){
		free(data);
		data = NULL;
	}

	// 
	evbuffer_drain(input, n);
}
void clientErrorCB(struct bufferevent * bev, short what, void * arg){
	UserClient * client = (UserClient *)arg;
	infoRecorder->logTrace("[clientErrorCB]:\n");
	if(what & BEV_EVENT_EOF){

	}
	if(what & BEV_EVENT_ERROR){

	}
	bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
	if(client){
		delete client;
		client = NULL;
	}
	bufferevent_disable(bev, EV_READ | EV_WRITE);
	bufferevent_free(bev);

}

DWORD WINAPI NetworkThreadProc(LPVOID param){
	// connect to dis
	GameStreams * streams = (GameStreams *)param;
	infoRecorder->logTrace("[NetWorkProc]: gameStreams: %p.\n", streams);
	char * gameName = streams->name;
	UserClient * client = new UserClient();
	// set the rtps name
	client->setName(gameName);
	event_base * base = event_base_new();
	client->setEventBase(base);

	client->launchRequest(streams->getDisUrl(), DIS_PORT_DOMAIN, gameName);

	// dispatch
	client->dispatch();
	return 0;
}

#endif
