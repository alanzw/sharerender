#include "DisForRender.h"


const char * ADD_RTSP_CONNECTION = "ADD_RTSP_CONNECTION";

evutil_socket_t connectToGraphic(char * url, int port){
	printf("[Global]: connect to graphic server '%s:%d'\n", url, port);
	evutil_socket_t sock;
	sockaddr_in sin;

	int sin_size = sizeof(sin);
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr(url);
	sin.sin_port = htons(port);

	sock = socket(AF_INET, SOCK_STREAM, 0);

	if(connect(sock, (sockaddr *)&sin, sizeof(sin)) == SOCKET_ERROR){
		printf("[Global]: connect to graphic server failed with %d.\n", WSAGetLastError());
		return NULL;
	}
	return sock;

}


///////////// RenderProxy ////////////////


RenderProxy * RenderProxy::renderProxy;

bool RenderProxy::dealEvent(BaseContext * ctx){
	char feedback[512] = { 0 };
	int len = 0;
	ctx->readCmd();
	char * cmd = ctx->getCmd();
	if (!strncasecmp(cmd, INFO, strlen(INFO))){
		printf("[RenderProxy]: INFO.\n");
		// collect information
		float cpuUsage = 0.0f, gpuUsage = 0.0f, memUsage = 0.0f;

		collectInfo(cpuUsage, gpuUsage, memUsage);
		// write back the information

		ctx->writeCmd(INFO);
		ctx->writeFloat(cpuUsage);
		ctx->writeFloat(gpuUsage);
		ctx->writeFloat(memUsage);
		ctx->writeToNet();

	}
	else if (!strncasecmp(cmd, START_TASK, strlen(START_TASK))){
		printf("[RenderProxy]: START TASK.\n");
		// format: cmd+id+url
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)(data);
		IDENTIFIER renderId = *(IDENTIFIER *)(data + sizeof(IDENTIFIER));
		//char * gameaName = (char *)(data + sizeof(IDENTIFIER) * 2);
		short portOffset = *(short *)(data + sizeof(IDENTIFIER) * 2);
		char * url = data + sizeof(IDENTIFIER) * 2 + sizeof(short);
		// start the render channel, connect to logic and send client id and render id to find the exist task in logic
#if 0
		connectToServer(url, DIS_PORT_GRAPHIC);
#else

		printf("[RenderProxy]: to start render task, graphic server '%s'.\n", url);
		// connect to logic server
		evutil_socket_t sockForCmd = NULL;
		sockForCmd = connectToGraphic(url, DIS_PORT_GRAPHIC);

		// wait for server name from logic
		char serviceName[50] = { 0 };
		//recv((SOCKET)sockForCmd, serviceName, sizeof(serviceName), 0);
		strcpy(serviceName, "Trine");
		// create render 
		RenderChannel * ch = new RenderChannel();
		ch->rtspObject = _strdup(serviceName);

		// TODO, set the object for the rtsp service

		ch->initRenderChannel(0, serviceName, sockForCmd);
		//start render channel thread
		ch->startChannelThread();

#endif

		// send back RENDER READY, not right
		ctx->writeCmd(RENDER_READY);
#if 0
		ctx->writeData((void *)&id, sizeof(IDENTIFIER));
#else

		ctx->writeIdentifier(id);
#endif
		ctx->writeToNet();

	}
	else if (!strncasecmp(cmd, CANCEL_TASK, strlen(CANCEL_TASK))){
		printf("[RenderProxy]: CANCEL TASK.\n");
		// cancel a task

		// TODO, now no cancel task command will be sent to render

		// send back updated information
		float cpuUsage = 0.0f, gpuUsage = 0.0f, memUsage = 0.0f;
		collectInfo(cpuUsage, gpuUsage, memUsage);
		ctx->writeCmd(INFO);
		ctx->writeFloat(cpuUsage);
		ctx->writeFloat(gpuUsage);
		ctx->writeFloat(memUsage);
		ctx->writeToNet();
	}
	else if(!strncasecmp(cmd, ADD_RTSP_CONNECTION, strlen(ADD_RTSP_CONNECTION))){
		// start the rtsp services
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)(data);
		short portOff = *(short *)(data + sizeof(IDENTIFIER) + 1);

		// start listening


		VideoItem * item = VideoContext::GetContext()->findItem(id);
		if(!item){
			printf("[RenderProxy]: cannot find the video context for task:%p.\n", id);
			return false;
		}
		event_base * _base = bufferevent_get_base(ctx->bev);
		
		evconnlistener * rtspListener = listenPort(DIS_PORT_RTSP + portOff, _base, item);

		// TODO , manage the listener

	}
	else{
		printf("[RenderProxy]: unknown cmd.\n");
		return false;
	}
	return true;
}
bool RenderProxy::start(){
	// start the render proxy
	// connect to dis and register
	connectToServer(DIS_URL_DISSERVER, DIS_PORT_DOMAIN);
	// register as render
	ctx->writeCmd(REGISTER);
	ctx->writeData((void *)RENDER, strlen(RENDER));

	ctx->writeToNet();
	return true;
}

bool RenderProxy::startRTSP(evutil_socket_t sock){
	printf("[RenderProxy]: star the rtsp dealing.\n");

	startRTSPThread(sock);
	return true;
}


void RenderProxy::startRTSPThread(evutil_socket_t sock){
	DWORD threadID;
	HANDLE threadHandle = chBEGINTHREADEX(NULL, 0, RTSPThreadProc, sock, FALSE, &threadID);

}

// the RTSP thread proc
DWORD RenderProxy::RTSPThreadProc(LPVOID param){

	RenderProxy * proxy = RenderProxy::GetProxy();
	evutil_socket_t fd = (evutil_socket_t)param;   // get the socket

	char buf[1024] = { 0 };

#ifndef NORMAL // only for test
	// get the rtsp cmd
	recv((SOCKET)fd, buf, sizeof(buf), 0);
	printf("[RenderProxy]: rtsp cmd: '%s'.\n", buf);

	char * obj = "Test";
	list<RenderChannel *>::iterator it;
	RenderChannel * channel = NULL;
	for (it = proxy->serviceMap.begin(); it != proxy->serviceMap.end(); it++){
		channel = (*it);
		if (!strncasecmp(obj, channel->rtspObject, strlen(channel->rtspObject))){
			// find the service
			goto rtsp_find;
		}
	}
	// not find the service
	printf("[RTSPThreadProc]: cannot find the service.\n");
	return -1;

rtsp_find:
	//enter the rtsp logic
	printf("[RTSPThread]: enter the rtsp logic.\n");


	return 0;
#else

	return 0;
#endif

}

#if 0
DWORD RenderProxy::RenderThreadProc(LPVOID param){
	RenderChannel * ch = (RenderChannel *)param;
	printf("[RenderThreadProc]: enter render thread for game: '%s'.\n", ch->rtspObject);

	char buffer[100] = { 0 };
	bool running = true;
	fd_set sockSet;
	FD_ZERO(&sockSet);
	FD_SET(ch->cmdSock, &sockSet);
	timeval tv;
	tv.tv_sec = 2;
	tv.tv_usec = 0;
	do{

		if(select(0, &sockSet,NULL,NULL, NULL) >0){

			if(FD_ISSET(ch->cmdSock, &sockSet)){
				recv(ch->cmdSock, buffer, 100, 0);
				if (!strncasecmp(buffer, "exit", strlen("exit"))){
					running = false;
				}
				else{
					//printf("[RenderChannel]: recv '%s'.\n", buffer);
				}
				memset(buffer, 0, 100);
			}


		}
	} while (running);


	return 0;
}

void RenderProxy::startRenderThread(RenderChannel * ch){
	printf("[RenderProxy]: start render thread.\n");
	DWORD threadId;
	HANDLE thread = chBEGINTHREADEX(NULL, 0, RenderThreadProc, ch, FALSE, &threadId);
}

#endif