#include "disforclient.h"
//#include "../LibInput/CtrlSdl.h"

#include "client.h"
#include "../LibCore/InfoRecorder.h"

#ifndef USE_LIBEVENT

#else

using namespace cg;
using namespace cg::core;

///// global control client
cg::input::CtrlMessagerClient * gCtrlClient = NULL;

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
		RTSPConf * conf = RTSPConf::GetRTSPConf();
		int port = DIS_PORT_RTSP;
		if(conf){
			port = conf->serverPort;
		}

		startRTSP(renderUrl, port+ portOff);

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
		// create sub game stream and start the sub game stream thread
		infoRecorder->logTrace("[Client]: rtsp index:%d, url:%s, port:%d, name:%s.\n", rtspCount, url, port, gameName);
		
		sprintf(rtspUrl, "rtsp://%s:%d/%s", url, port, gameName);
		printf("[Client]: request url: %s.", rtspUrl);
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


// mainly handle the control ???? (yes)
bool UserClient::addLogic(char * url){
	infoRecorder->logError("[UserClient]: add logic %s.\n", url);
	RTSPConf * rtspConf = RTSPConf::GetRTSPConf();
	if (!rtspConf){
		infoRecorder->logError("[client]: get RTSP config failed.\n");
	}
	// init the ctrl message
	ctrlClient = new cg::input::CtrlMessagerClient();

	// launch controller?
	// set the logic server name for controller
	//ctrlConf->ctrl_servername = _strdup(url);
	_strdup(url);

	do{
		if (rtspConf->ctrlEnable){
			if (ctrlClient->initQueue(32768, sizeof(cg::input::sdlmsg_t)) < 0){
				rtsperror("Cannot initialize controller queue, controller disabled.\n");
				infoRecorder->logError("[Client]: cannot initialize controller queue, controller disable.\n");
				rtspConf->ctrlEnable = 0;
				break;
			}
			ctrlClient->init(rtspConf, url, CTRL_CURRENT_VERSION);
			if (!ctrlClient->start()){
				rtsperror("Cannot create controller thread, controller disabled.\n");
				infoRecorder->logError("[Client]: cannot create controller thread, controller disabled.\n");
				rtspConf->ctrlEnable = 0;
				break;
			}
		}
	} while (0);

	infoRecorder->logError("[Client]: after create the controller thread, ctrlClient: %p.\n",ctrlClient);
	gCtrlClient = ctrlClient;
	return true;
}

// not used
DWORD UserClient::ClientThreadProc(LPVOID param){
	infoRecorder->logTrace("[UserClient]: enter client thread proc.\n");


	return 0;
}

// shutdown rtsp connection
bool UserClient::shutdownRTSP(IDENTIFIER rid){
	infoRecorder->logTrace("[UserClient]: shutdown the rtsp connection.\n");


	return true;
}

// launch rtsp request to logic server
bool UserClient::launchRequest(char * disServerUrl, int port, char * gameName){
	infoRecorder->logTrace("[UserClient]: launch the request, server:%s:%d, name:%s.\n", disServerUrl, port, gameName);
	bufferevent * bev = NULL;

	memset(&sin, 0, sizeof(sin));

	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr(disServerUrl);
	sin.sin_port = htons(port);
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
	infoRecorder->logTrace("[NetWorkProc]: gameStreams: %p, game name:%s.\n", streams, streams->name);
	char * gameName = streams->name;
	UserClient * client = new UserClient();

	// set the rtsp name
	client->setName(gameName);
	event_base * base = event_base_new();

	struct RTSPConf * rtspConf = RTSPConf::GetRTSPConf();
	int disPort = DIS_PORT_DOMAIN;
	if(rtspConf){
		disPort = rtspConf->disPort;
	}
	client->setEventBase(base);
	client->launchRequest(streams->getDisUrl(), disPort, gameName);
	client->dispatch();
	return 0;
}
DWORD WINAPI UserClientThreadProc(LPVOID param){
	GameStreams * streams = (GameStreams *)param;
	char * gameName = streams->name;
	UserClient * client = new UserClient();
	struct RTSPConf * rtspConf = RTSPConf::GetRTSPConf();

	client->setName(gameName);
	event_base *base = event_base_new();
	client->setEventBase(base);
	client->startRTSP(rtspConf->getDisUrl(), rtspConf->serverPort);

	return 0;
}

#endif
