#include "../LibDistrubutor/Distributor.h"
#include "../LibCore/InfoRecorder.h"
#include "Monitor.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "../LibDistrubutor/Context.h"
#include "CmdController.h"
#include "../VideoUtility/rtspconf.h"

#include <iostream>
#include <string>
#include <sstream>


using namespace std;


#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"d3dx9.lib")

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
using namespace cg;
using namespace cg::core;

// bufferevent event callback
void onBufferEventEvent(struct bufferevent * bev, short events, void *ctx){

	BaseContext *netCtx = (BaseContext  *)ctx;

	
	if (events &(BEV_EVENT_EOF | BEV_EVENT_ERROR)){
		bufferevent_free(bev);
		netCtx->bev = NULL;
	}
}

// bufferevent read callback
void onBufferEventRead(struct bufferevent * bev, void *ctx){
	// This callback is invoked when there is data to read on bev
	cg::core::infoRecorder->logTrace("[OnBufferEventRead]: read data \n");
	struct evbuffer * input = bufferevent_get_input(bev);
	struct evbuffer * output = bufferevent_get_output(bev);


	DomainInfo* netCtx = (DomainInfo*)ctx;
	// here, deal the data received
	evutil_socket_t sock = (evutil_socket_t)netCtx->sock;

	int data_len = evbuffer_get_length(input);
	char * data = (char *)malloc(sizeof(char) * data_len+1);
	memset(data, 0, data_len);
	evbuffer_copyout(input, data, data_len);
	data[data_len] = 0;

	cg::core::infoRecorder->logTrace("[OnBufferEventRead]: read len:%d, '%s'.\n", data_len, data);

	netCtx->setData(data, data_len);
	DisServer * server = DisServer::GetDisServer();

	server->dealEvent(netCtx);

	free(data);
	data = NULL;

	evbuffer_drain(input, data_len);

	return;

	// copy all the data from input buffer to the output buffer
	//evbuffer_add_buffer(output, input);
}

// bufferevent write callback
void onBufferEventWrite(struct bufferevent * bev, void * ctx){
	struct evbuffer * output = bufferevent_get_output(bev);
	if(evbuffer_get_length(output) == 0){
		cg::core::infoRecorder->logTrace("[OnBufferEventWrite]: flushed answer.\n");
		//bufferevent_free(bev);
	}
}


void DisManagerListenerCB(struct evconnlistener * listerner, evutil_socket_t sock, struct sockaddr * addr, int len, void * ctx){
	/* we got a new connection ! Set up a bufferevent for it. */
	cg::core::infoRecorder->logTrace("[DisManagerListenerCB]: got an connection.\n");
	struct event_base * base = NULL;
	base = evconnlistener_get_base(listerner);
	struct bufferevent * bev = NULL;
	bev = bufferevent_socket_new(base, sock, BEV_OPT_CLOSE_ON_FREE/* | BEV_OPT_THREADSAFE*/);

	if(base == NULL){
		cg::core::infoRecorder->logTrace("[DisManagerListenerCB]: get the base failed.\n");
	}
	
	if(bev == NULL){
		cg::core::infoRecorder->logTrace("[DisManagerListenerCB]: get new bufferevent failed\n");
	}

	// sock is the new socket for connection, create a machine info
	DomainInfo* netctx = new DomainInfo();
	netctx->bev = bev;
	netctx->sock = sock;
	netctx->addr = (sockaddr_in *)malloc(sizeof(char *) * len + 10);
	memcpy(netctx->addr , addr, len);

	netctx->url = _strdup(inet_ntoa(netctx->addr->sin_addr));
	cg::core::infoRecorder->logTrace("[DisManagerListernerCB]: get url, addr:%p, url:%s.\n", netctx->url, netctx->url);

	DisServer * server = (DisServer *)ctx;
	//server->addCtx(netctx);
	//cout << "DisManager: get new connection: " << netctx->getNetName() << endl;

	bufferevent_setcb(bev, onBufferEventRead, NULL, onBufferEventEvent, (void *)netctx);

	bufferevent_enable(bev, EV_READ | EV_WRITE);
}

static void acceptErrorCB(struct evconnlistener * listener, void *ctx){
	struct event_base * base = evconnlistener_get_base(listener);
	int err = EVUTIL_SOCKET_ERROR();
	cg::core::infoRecorder->logError("Got an error %d (%s) on the listener. Shutting down.\n", err, evutil_socket_error_to_string(err));
	event_base_loopexit(base, NULL);
}

DWORD WINAPI DisManagerNetProc(LPVOID param){
	DisServer * server = (DisServer *)param;

	event_base * base = event_base_new();
	if(!base){
		cg::core::infoRecorder->logError("[DisManagerThread]: Couldn't create an event base: exiting.\n");
		return -1;
	}

	if(server->getServerUrl() == NULL){
		cg::core::infoRecorder->logError("[DisManagerThread]: get NULL url.\n");
		return -1;
	}

	cg::RTSPConf * conf = cg::RTSPConf::GetRTSPConf();

	// listen the dis port
	sockaddr_in sin;
	int sin_size=  sizeof(sin);
	memset(&sin, 0, sin_size);
	sin.sin_family = AF_INET;
	
	sin.sin_addr.S_un.S_addr = inet_addr(server->getServerUrl()); //htonl(INADDR_ANY);
	sin.sin_port = htons(conf->disPort);
	cg::core::infoRecorder->logTrace("[DisManagerThread]: listen to url:%s, port:%d.\n", server->getServerUrl(), conf->disPort);
	server->startWatchdog();
	server->setEventBase(base);

	// to listen
	struct evconnlistener * connListener = evconnlistener_new_bind(base, DisManagerListenerCB, server, LEV_OPT_LEAVE_SOCKETS_BLOCKING | LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size);
	if(!connListener){
		cg::core::infoRecorder->logTrace("[DisManagerThread]: Couldn't create listener.\n");
		return -1;
	}
	evconnlistener_set_error_cb(connListener, acceptErrorCB);
	cg::core::infoRecorder->logTrace("[DisManagerThread]: dispatch.\n");

	// enter the event loop
	server->dispatch();

	cg::core::infoRecorder->logTrace("[DisManagerThread]: before exit.\n");
	evconnlistener_free(connListener);
}

void printHelp(){
	printf("[Commands supported]:\n");
	printf("\t[CMD]: PRINT_DOMAIN\n\t\t[Option]: print the server's information.\n");
	printf("\t[CMD]: PRINT_TASK\n\t\t[Option]: print all task's information.\n");
	printf("\t[CMD]: ADD_RENDER [TASK ID] [URL]\n\t\t[Option]: add render to given task.\n");
	printf("\t[CMD]: DECLINE_RENDER [TASK ID] [RENDER ID]\n\t\t[Option]: decline the given render for a certain task.\n");
	printf("\t[CMD]: MIGRATE [DOMAIN ID]\n\t\t[Option]: migrate the whole render server.\n");
	printf("\t[CMD]: MIGRATE_RENDER [TASK ID] [RENDER ID] [URL]\n\t\t[Option]: migrate a single render task from [RENDER ID] to [URL].\n");
}

int PrintMachineInfo(){

	// test
	char local[255] = {0};
	char str[32] = {0};
	char ** pptr = NULL;
	gethostname(local, sizeof(local));
	HOSTENT * ph = gethostbyname(local);
	if(ph == NULL){
		// get the ip failed
		return -1;
	}

	cout << "official host name: " << ph->h_name << endl;
	for(pptr = ph->h_aliases; *pptr != NULL; pptr ++){
		cout << "alias: " << *pptr << endl;
	}
	switch(ph->h_addrtype){
	case AF_INET:

		//break;
	case AF_INET6:
		pptr = ph->h_addr_list;
		for(; *pptr != NULL; *pptr++){
			cout << "address: " << inet_ntop(ph->h_addrtype, *pptr, str, sizeof(str)) << endl;
		}
		cout << "first address: " << inet_ntop(ph->h_addrtype, ph->h_addr_list[0], str, sizeof(str)) << endl;
		break;
	default:
		cout << "unknown address type" << endl;
		break;
	}
}


const char * PRINT_DOMAIN = "PRINT_DOMAIN";
const char * PRINT_TASK = "PRINT_TASK";
const char * ADD_RENDER_C = "ADD_RENDER";
const char * DECLINE_RENDER_C = "DECLINE_RENDER";
const char * MIGRATE_C = "MIGRATE";
const char * MIGRATE_RENDER_C = "MIGRATE_RENDER";
const char * CHANGE_ENCODER_C = "CHANGE_ENCODER";
const char * FULL_OFFLOAD_C = "FULL_OFFLOAD";

int main(int argc, char ** argv){
	infoRecorder = new InfoRecorder("distributor");
	// start the network
	WSADATA WSAData;
	WSAStartup(0x101, &WSAData);

	CmdDealer * ctrl = CmdDealer::GetDealer();//::GetController();
	DisServer * server = DisServer::GetDisServer();
	ctrl->setServer(server);
	char * rtspConfigDefault = "config/server.distributor.conf";
	char * rtspConfig = NULL;
	if(argc > 1){
		rtspConfig = _strdup(argv[1]);  // get the config file
	}else{
		rtspConfig = rtspConfigDefault;
	}
	// load the rtsp config file
	cg::RTSPConf * rtspConf = cg::RTSPConf::GetRTSPConf(rtspConfig);
	server->setDisUrl(rtspConf->getDisUrl());

	DWORD netThreadId  = 0;
	HANDLE netThreadHandle = chBEGINTHREADEX(NULL, 0, DisManagerNetProc, server, FALSE, &netThreadId);

	PrintMachineInfo();

	// start the cmd
	string cmd;
	char tcmd[512] = {0};
	bool run = true;
	printHelp();
	do{
		cout << "----- Input the Command -----------" << endl << "CMD: ";
		std::getline(std::cin, cmd);
		//getline(cmd);
		//cin >> cmd;
		cg::core::infoRecorder->logTrace("[DisManager]: input command:%s\n", cmd.c_str());
		//cout << "----- DisManager -----------" << endl << "CMD: " << cmd << endl << "DO: " << endl;
		if(!strncmp(cmd.c_str(), PRINT_DOMAIN, strlen(PRINT_DOMAIN))){
			server->printDomain();
		}else if(!strncmp(cmd.c_str(), PRINT_TASK, strlen(PRINT_TASK))){
			server->printTask();
		}
		else if(!strncmp(cmd.c_str(), ADD_RENDER_C, strlen(ADD_RENDER_C))){
			// add render to given task
			// get the task ID
			//char tcmd[512] = {0};
			IDENTIFIER id = NULL;
			if(sscanf(cmd.c_str(), "%s %d", tcmd, &id) == -1){

				std::cout <<"INVALID cmd or parameter." << std::endl;
				continue;
			}
			std::cout << "Get CMD: " << tcmd << " id: " << id << std::endl;
			server->AddRenderToTask(id);

		}
		else if(!strncmp(cmd.c_str(), DECLINE_RENDER_C, strlen(DECLINE_RENDER_C))){
			// cancel a given render for the given task
		}
		else if(!strncmp(cmd.c_str(), MIGRATE_C, strlen(MIGRATE_C))){
			// migrate the whole render server
		}
		else if(!strncmp(cmd.c_str(), MIGRATE_RENDER_C, strlen(MIGRATE_RENDER_C))){
			// migrate a single render task
		}
		else if(!strncmp(cmd.c_str(), CHANGE_ENCODER_C, strlen(CHANGE_ENCODER_C))){
			// change the given task's encoder type
			IDENTIFIER id = NULL;
			IDENTIFIER domainId = NULL;
			if(sscanf(cmd.c_str(), "%s %d %d", tcmd, &id, &domainId) == -1){
				std::cout << "INVALID cmd or parameter" << std::endl;
				continue;
			}
			std::cout  << "Get CMD: " << tcmd << " id: " << id << " domain id: " << domainId << std::endl;
			server->ChangeEncoderType(id, domainId);

		}
		else if(!strncmp(cmd.c_str(), FULL_OFFLOAD_C, strlen(FULL_OFFLOAD_C))){
			// full offload the given task
			IDENTIFIER id = NULL;
			TASK_MODE mode = MODE_FULL_OFFLOAD;
			if(sscanf(cmd.c_str(), "%s %d %d", tcmd, &id, &mode) == -1){
				std::cout << "INVALID cmd or parameter " << std::endl;
				continue;
			}
			std::cout << "Get CMD: " << tcmd << " id: " << id << " mode: " << 
				(mode == MODE_FULL_OFFLOAD ? "FULL_OFFLOAD" : (mode == MODE_NO_OFFLOAD ? "NO_OFFLOAD" : (mode == MODE_PARTIAL_OFFLOAD ? "PARTIAL_OFFLOAD" : "UNKNOWN")));
			server->ChangeOffloadLevel(id, mode);
		}
		else{
			// invalid cmd
			cout << "----- Error CMD -----" << endl << "CMD: " << cmd << endl;
			printHelp();
		}
	}while(run);
	// before exit
	cg::core::infoRecorder->logTrace("[Main]: before exit the dis manager.\n");

	return 0;
}