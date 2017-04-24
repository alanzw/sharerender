#include "DisForRender.h"
#include "../LibCore/InfoRecorder.h"

using namespace cg;
using namespace cg::core;

evutil_socket_t connectToGraphic(char * url, int port){
	infoRecorder->logError("[Global]: connect to graphic server '%s:%d'\n", url, port);
	evutil_socket_t sock;
	sockaddr_in sin;

	int sin_size = sizeof(sin);
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr(url);
	sin.sin_port = htons(port);

	sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

	if(connect(sock, (sockaddr *)&sin, sizeof(sin)) == SOCKET_ERROR){
		infoRecorder->logError("[Global]: connect to graphic server failed with %d.\n", WSAGetLastError());
		return NULL;
	}
	return sock;
}

///////////// RenderProxy ////////////////

RenderProxy * RenderProxy::renderProxy;

bool RenderProxy::dealEvent(cg::BaseContext * ctx){
	char feedback[512] = { 0 };
	int len = 0;
	ctx->readCmd();
	char * cmd = ctx->getCmd();
	if (!strncasecmp(cmd, INFO, strlen(INFO))){
		infoRecorder->logTrace("[RenderProxy]: INFO.\n");
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
		infoRecorder->logTrace("[RenderProxy]: START TASK.\n");
		// format: cmd+id+url
		char * data = ctx->getData();
		cg::IDENTIFIER id = *(cg::IDENTIFIER *)(data);
		cg::IDENTIFIER renderId = *(cg::IDENTIFIER *)(data + sizeof(cg::IDENTIFIER));
		short portOffset = *(short *)(data + sizeof(cg::IDENTIFIER) * 2);
		short nameLen = *(short*)(data + sizeof(cg::IDENTIFIER) * 2 +sizeof(short));
		char serviceName[100] = {0}, * p = NULL;
		p = data + sizeof(cg::IDENTIFIER) * 2 + sizeof(short) * 2;
		memcpy(serviceName, p, nameLen);

		char * url = data + sizeof(cg::IDENTIFIER) * 2 + sizeof(short) * 2 + nameLen;
		// start the render channel, connect to logic and send client id and render id to find the exist task in logic
#if 0
		connectToServer(url, DIS_PORT_GRAPHIC);
#else
		infoRecorder->logTrace("[RenderProxy]: to start render task, graphic server '%s', id:%p.\n", url ,id);
		// connect to logic server
		evutil_socket_t sockForCmd = NULL;
		int graphicPort = 60000;
		if(conf)
			graphicPort = conf->graphicPort;
		sockForCmd = connectToGraphic(url, graphicPort);
#if 0
		// set nonblocking??????
		u_long iMode = 1;  // non-bnlocking mode is enabled
		if(ioctlsocket(sockForCmd , FIONBIO, &iMode)){
			infoRecorder->logTrace("*** set TCP send non-blocking failed.\n");
		}
		else{
			infoRecorder->logTrace("*** set TCP send non-blocking succ.\n");
		}
#endif
		// wait for server name from logic
		//char serviceName[50] = { 0 };
		//strcpy(serviceName, "Trine");
		// create render 
		RenderChannel * chan = new RenderChannel();
		if(conf){
			chan->rtspConf = conf;
		}
		else{
			chan->rtspConf = cg::RTSPConf::GetRTSPConf("config/server.render.conf");
		}
		chan->rtspObject = _strdup(serviceName);
		chan->taskId = id;
		chan->setEncoderOption(this->getEncodeOption());

		// TODO, set the object for the rtsp service
		chan->initRenderChannel(id, serviceName, sockForCmd);

		// set the id
		// send to logic the ADD_RENDER+task ID + Render ID
		cg::BaseContext * temCtx = new cg::BaseContext();
		temCtx->sock = sockForCmd;
		temCtx->writeCmd(ADD_RENDER);
		temCtx->writeIdentifier(id);
		temCtx->writeIdentifier(renderId);
#if 0
		temCtx->writeToNet();
#else
		if(send(temCtx->sock, temCtx->writeBuffer, temCtx->writeLen, 0) == SOCKET_ERROR){
			infoRecorder->logTrace("[render channel]: send failed with %d.\n", WSAGetLastError());
		}
#endif
		//start render channel thread
		chan->startChannelThread();

#endif

#if 1
		// send back RENDER READY, not right
		ctx->writeCmd(cg::RENDER_READY);
		ctx->writeIdentifier(id);
		//ctx->writeData((void *)&id, sizeof(IDENTIFIER));
		ctx->writeToNet();
#endif

	}
	else if (!strncasecmp(cmd, CANCEL_TASK, strlen(CANCEL_TASK))){
		infoRecorder->logTrace("[RenderProxy]: CANCEL TASK, destroy the task include rtsp service.\n");
		// cancel a task
		// TODO, now no cancel task command will be sent to render
		// cancel the task, the command is sent to logic server, logic server will end the grahpic stream

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
		
#if 1
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)(data);
		short portOff = *(short *)(data + sizeof(IDENTIFIER));
		infoRecorder->logError("[RenderProxy]: ADD RTSP Connection, task id:%p, port:%d.\n", id, portOff);
		// start listening
		event_base * _base = bufferevent_get_base(ctx->bev);
		infoRecorder->logError("[RenderProxy]: get base:%p\n", _base);
		VideoGen::Initialize();

		VideoGen * gen = VideoGen::findVideoGen(id);
		infoRecorder->logError("[RenderProxy]: find gen: %p.\n", gen);
		if(!gen){
			infoRecorder->logError("[RenderProxy]: the video generator for '%d' is not created yet.\n", id);
			return false;
		}else{
			infoRecorder->logError("[RenderProxy]: the video generator for '%p' is '%p'.\nto listen port: %d.\n", id, gen, conf->serverPort + portOff);
		}
		evconnlistener * rtspListener = listenPort(conf->serverPort + portOff, _base, gen->getContext());

		// TODO , manage the listener
		// notify manager to start rtsp connection
		ctx->writeCmd(RTSP_READY);
		ctx->writeIdentifier(id);
		ctx->writeToNet();
#endif
	}else if(!strncasecmp(cmd, OPTION, strlen(OPTION))){
		// deal the option command
		// deal with option command
		// format:[cmd][id][OPTION count]{[option][value] ... [option][value]
		infoRecorder->logError("[RenderProxy]: RENDER OPTION.\n");
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;
		VideoGen *gen = VideoGen::findVideoGen(id);
		if(!gen){
			// error
			infoRecorder->logError("[GameClient]: the rtsp service for '%d' has not created yet.\n", id);
			return false;
		}
		// now 
		short optionCount = *(short *)(data + sizeof(IDENTIFIER));
		for(int i = 0; i < optionCount; i++){
			CMD_OPTION option = *(CMD_OPTION *)(data + sizeof(IDENTIFIER) + sizeof(short) + i * (sizeof(CMD_OPTION) + sizeof(short)));
			short value = *(short *)(data + sizeof(IDENTIFIER) + sizeof(short) + i * (sizeof(CMD_OPTION) + sizeof(short)) + sizeof(CMD_OPTION));
			
			switch(option){
			case ENCODEROPTION:

				break;
			case SETOFFLOAD:
				
				break;

			default:

				break;
			}

		}
	}
	else{
		infoRecorder->logTrace("[RenderProxy]: unknown cmd.\n");
		return false;
	}
	return true;
}
bool RenderProxy::start(char * disUrl){
	// start the render proxy
	// connect to dis and register
	int disPort = DIS_PORT_DOMAIN;
	if(conf){
		disPort = conf->disPort;
	}
	if(!connectToServer(disUrl, disPort)){
		return false;
	}

	// register as render
	ctx->writeCmd(cg::REGISTER);
	ctx->writeData((void *)RENDER, strlen(RENDER));

	ctx->writeToNet();
	return true;
}

#if 0

bool RenderProxy::startRTSP(evutil_socket_t sock){
	infoRecorder->logTrace("[RenderProxy]: star the rtsp dealing.\n");
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
	// get the RTSP cmd
	recv((SOCKET)fd, buf, sizeof(buf), 0);
	infoRecorder->logTrace("[RenderProxy]: RTSP cmd: '%s'.\n", buf);

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
	infoRecorder->logTrace("[RTSPThreadProc]: cannot find the service.\n");
	return -1;

rtsp_find:
	//enter the rtsp logic
	infoRecorder->logTrace("[RTSPThread]: enter the rtsp logic.\n");

	return 0;
#else

	return 0;
#endif

}

// get the cpu and gpu usage and determine the encoding device for each encoding task
bool RenderProxy::regulation(){


	return true;
}


#endif