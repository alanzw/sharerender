#include "ProcessClient.h"

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



const char * CHANGE_TO_X264 = "CHANGE_TO_X264";
const char * CHANGE_TO_NVENC = "CHANGE_TO_NVENC";


bool GameClient::dealEvent(BaseContext * ctx){
	bool ret = true;
	char feedback[1024] = { 0 };
	ctx->readCmd();
	int len = 0;
	char * cmd = ctx->getCmd();
	char * data = ctx->getData();

	if(!strncasecmp(cmd, "TEST", strlen("TEST"))){
		printf("[ProcessClient]: recv TEST after send GAME_READY\n");
		return true;
	}
	else if(!strncasecmp(cmd, CHANGE_TO_NVENC, strlen(CHANGE_TO_NVENC))){
		// change the encoder device to nvenc, in GameClient, this command means to enable the rendering in logic server


	}
	else if(!strncasecmp(cmd, CHANGE_TO_X264, strlen(CHANGE_TO_X264))){
		// change the encoder device to x264, in GameClient, this command means to enable the rendering in logic server


	}
	else if(!strncasecmp(cmd, ADD_RTSP_CONNECTION, strlen(ADD_RTSP_CONNECTION))){
		char * data = ctx->getData();
		IDENTIFIER id = *(IDENTIFIER *)data;
		short portOff = *(short *)(data + sizeof(IDENTIFIER) + 1);

		// get the rtsp context to generate the video
		VideoContext * vctx = VideoContext::GetContext();
		vctx->WaitForNofify();
		VideoItem * item = vctx->findItem(id);
		event_base * rtspListener = listenPort(DIS_PORT_RTSP + portOff, base, item);

	}
	else{
		printf("[ProcessClient]: get unknown cmd.\n");
		return false;
	}

	return ret;
}


// callback for game client
void ProcessClientReadCB(struct bufferevent * bev, void * arg){
	ProcessClient * client = (ProcessClient*)arg;
	struct evbuffer * input = bufferevent_get_input(bev);
	size_t n = evbuffer_get_length(input);
	char * data = (char *)malloc(sizeof(char)* n);
	evbuffer_copyout(input, data, n);
	printf("[ProcessClientReadCB]: read '%s'\n", data);
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
void ProcessClientWriteCB(struct bufferevent * bev, void * arg){
	struct evbuffer * output = bufferevent_get_output(bev);
	int len = 0;
	if((len = evbuffer_get_length(output)) == 0){
		printf("[ProcessClientWriteCB]: flushed answer.\n");
	}
	else{
		printf("[ProcessClientWriteCB]: write %d bytes.\n", len);
	}
}

// event callback
void ProcessClientEventCB(struct bufferevent * bev, short what, void * arg){
	GameClient * client = (GameClient*)arg;

	if(what & BEV_EVENT_ERROR){
		// error
		perror("[ProcessClientEventCB]: error from bufferevent.");
		int err = EVUTIL_SOCKET_ERROR();
		printf("[ProcessClienttEventCB]: error occur. err: %d (%s)\n", err, evutil_socket_error_to_string(err));
	}

	if(what & BEV_EVENT_CONNECTED){
		printf("[ProcessClientEventCB]: connection created.\n");
	}
#if 1
	if(what & (BEV_EVENT_EOF)){
		printf("[ProcessClientEventCB]: BEV_EVENT_EOF may error:%d.\n", WSAGetLastError());
		perror("[ProcessClientEventCB]: error EOF.\n");
		//DebugBreak();
		bufferevent_setcb(bev, NULL, NULL, NULL, NULL);
		delete client;
		bufferevent_disable(bev, EV_READ | EV_WRITE);
		bufferevent_free(bev);
	}
#endif
}
