
#include "../LibVideo/Config.h"

#include <string.h>
#include <time.h>
#include <map>
#include "../LibVideo/RtspConf.h"
#include "../LibVideo/RtspContext.h"
#include "../LibVideo/Pipeline.h"
#include "../LibVideo/Encoder.h"

#include "../LibVideo/FilterRGB2YUV.h"
#include <queue>
#include "../LibCore/Disnetwork.h"
#include "../LibDistrubutor/DistributorForRender.h"
#include "../LibCore/Log.h"
#include "../LibRender/LibRenderChannel.h"

DisClientForRenderProxy * DisClientForRenderProxy::disForProxy;

DisClientForRenderProxy * DisClientForRenderProxy::GetDisForRenderProxy(){
	if (disForProxy == NULL){
		disForProxy = new DisClientForRenderProxy();
		return disForProxy;
	}
	else
	{
		return disForProxy;
	}
}

DisClientForRenderProxy::DisClientForRenderProxy(){
	// create distributor config

	InitializeCriticalSection(&section);
	pSection = &section;

	notifier = CreateEvent(NULL, FALSE, FALSE, NULL);
	nEventTotal = 0;

	sock = NULL;
	listenSock = NULL;
	threadHandle = NULL;

	disConf = new DistributorConfig(CLIENT_DIS_CONFIG_FILE);
	cpuWatcher = new CpuWatch();
	gpuWatcher = new GpuWatch();

	renderServerNode = new ServerNode();
}

DisClientForRenderProxy::~DisClientForRenderProxy(){

	if (pSection){
		DeleteCriticalSection(pSection);
		pSection = NULL;
	}
	if (notifier){
		CloseHandle(notifier);
		notifier = NULL;
	}

	if (disConf){
		delete disConf;
		disConf = NULL;
	}

	for (int i = 0; i < nEventTotal; i++){
		if (eventArray[i]){
			WSACloseEvent(eventArray[i]);
			eventArray[i] = NULL;
		}
		if(sockArray[i]){
			closesocket(sockArray[i]);
			sockArray[i] = NULL;
		}
	}
#if 0
	if (sock){
		closesocket(sock);
		sock = NULL;
	}
	if (listenSock){
		closesocket(listenSock);
		listenSock = NULL;

	}
#endif
	if (threadHandle){
		TerminateThread(threadHandle, 0);
		threadHandle = NULL;
		//CloseHandle(threadHandle);
	}
	if (renderServerNode){
		delete renderServerNode;
		renderServerNode = NULL;
	}
	map<SOCKET, Task *>::iterator mi;
	for (mi = taskMap.begin(); mi != taskMap.end(); mi++){
		delete mi->second;
		mi->second = NULL;
	}

	if (cpuWatcher){
		delete cpuWatcher;
		cpuWatcher = NULL;
	}
	if (gpuWatcher){
		delete gpuWatcher;
		gpuWatcher = NULL;
	}

}

void DisClientForRenderProxy::Release(){
	if (disForProxy){
		delete disForProxy;
		disForProxy = NULL;
	}
}

int DisClientForRenderProxy::init(int port){
	// start the network

	WORD version;
	WSAData wsaData;
	//version = MAKEWORD(1, 1);
	version = MAKEWORD(2, 2);

	int err = WSAStartup(version, &wsaData);
	if (err){
		infoRecorder->logTrace("[DisClientForRenderProxy]: socket start failed.\n");
		return -1;
	}
	else{
		infoRecorder->logTrace("[DisClientForRenderProxy]: socket start success.\n");
	}

	// create the sock to wait logic server to connect
	listenSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	sin.sin_family = AF_INET;
	sin.sin_port = htons(port);
	sin.sin_addr.S_un.S_addr = INADDR_ANY;

	if (bind(listenSock, (sockaddr *)&sin, sizeof(sin)) == SOCKET_ERROR){
		infoRecorder->logTrace("[DisClientForRenderProxy]: failed bind().\n");
		return -1;
	}
	listen(listenSock, 5);

	// init the socket for manager
	sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	msin.sin_family = AF_INET;
	msin.sin_addr.S_un.S_addr = inet_addr(disConf->getServerUrl());

	msin.sin_port = htons(disConf->getRenderPort());
	if (connect(sock, (SOCKADDR *)&msin, sizeof(msin)) == -1){
		infoRecorder->logTrace("[DisClientForRenderProxy]: connect manager failed.url:%s, port:%d\n", disConf->getServerUrl(), disConf->getRenderPort());
		return -1;
	}
	return 0;

}

void DisClientForRenderProxy::closeClient(){
	infoRecorder->logTrace("[DisClientForRenderProxy]: send STOP command.\n");
	send(sock, STOP, strlen(STOP), 0);
}

int DisClientForRenderProxy::collectInformation(){
	this->renderServerNode->serverInfo->cpuUsage = cpuWatcher->GetSysCpuUtilization();
	this->renderServerNode->serverInfo->gpuUsage = gpuWatcher->GetGpuUsage();
	
	infoRecorder->logTrace("[DisCLientForRenderProxy]: cpu usage: %f, gpu usage: %f.\n", renderServerNode->serverInfo->cpuUsage, renderServerNode->serverInfo->gpuUsage);
	return 0;
}
//deal the cmd from manager and 
int DisClientForRenderProxy::dealCmd(char * cmd, SOCKET s){
	// if the cmd is start render, remove the socket event
	if (!strncasecmp(cmd, START_RENDER, strlen(START_RENDER))){
		// start render cmd from logic server
		char tem[] = { "example" };
#if 0
		// cmd format: [START_RENDER:int+gameName]
		char tem[100];
		sscanf(cmd, "START_RENDER:%s", tem);
		infoRecorder->logTrace("DisClientForRenderProxy]: get a start render cmd from %p.\n", s);
#else
		SOCKET clientSock = *(SOCKET *)(cmd + strlen(START_RENDER));
		infoRecorder->logError("[DistributorForRender]: get cmd start render, client socket:%p.\n", clientSock);
#endif
		
		// remove the graphic socket from listen array
		for (int i = 0; i < this->nEventTotal; i++){
			if (sockArray[i] == s){

				infoRecorder->logTrace("[DistributorForRender]: find the socket: %p\n", s);
				// find the graphic socket
				WSAEVENT wEvent = eventArray[i];
				for (int j = i; j < nEventTotal - 1; j++){
					eventArray[j] = eventArray[j + 1];
					sockArray[j] = sockArray[j + 1];
				}
				sockArray[nEventTotal - 1] = NULL;
				eventArray[nEventTotal - 1] = NULL;
				WSACloseEvent(wEvent);
				nEventTotal--;
				break;
			}
		}

		// create the render channel with the game name and the socket
		RenderChannel * rch = new RenderChannel();
		rch->initRenderChannel(0, tem, s);
		// start the render thread
#if 0
		rch->start();
#else
		rch->startChannelThread();
#endif

		// notify the manager
#if 0
		send(sock, RENDER_STARTED, strlen(RENDER_STARTED), 0);
#else
		char back[50] = { 0 };
		char * tp = NULL;
		strcpy(back, RENDER_STARTED);
		tp = back + strlen(back);
		int sendLen = strlen(back);

		SOCKET * sp = (SOCKET *)tp;
		*sp = clientSock;
		sendLen += sizeof(SOCKET);
		//send(sock, back, sendLen, 0);
		send(sock, back, sendLen, 0);
		infoRecorder->logError("[DistributorForRender]: send %p the RENDER_STARTED cmd, (graphic socket:%p).\n", sock, s);

#endif
	}
	else if (!strncasecmp(cmd, INFO, strlen(INFO))){
		// manager need render server's information
		collectInformation();
		// send information back to manager
		char back[512], *p = back;
		int sendSize = 0;
		sprintf(back, "%s:", INFO);
		p += strlen(back);
		sendSize = strlen(back);
		memcpy(p, renderServerNode->serverInfo, sizeof(MachineInfo));
		sendSize += sizeof(MachineInfo);

		send(sock, back, sendSize, 0);
	}
	else if (!strncasecmp(cmd, STOP_RENDER, strlen(STOP_RENDER))){
		return -1;
	}

	return 0;
}

DWORD DisClientForRenderProxy::enterLogicLoop(){

	infoRecorder->logTrace("[DisClientForRenderProxy]: enter the logic loop.\n");
	DWORD ret = 0;
	//int renderport = disConf->getRenderPort();

	//init(renderport);
	init(disConf->getGraphicPort());

	WSAEVENT evet = WSACreateEvent();
	WSAEventSelect(listenSock, evet, FD_ACCEPT | FD_READ | FD_CLOSE);
	eventArray[nEventTotal] = evet;
	sockArray[nEventTotal] = listenSock;
	nEventTotal++;
	WSAEVENT evet1 = WSACreateEvent();
	WSAEventSelect(sock, evet1, FD_READ | FD_CLOSE | FD_WRITE);
	eventArray[nEventTotal] = evet1;
	sockArray[nEventTotal] = sock;
	nEventTotal++;

	while (true){
		infoRecorder->logTrace("[DisClientForRenderProxy]: wait for '%d' events.\n", nEventTotal);
		// wait all net event
		int nIndex = WSAWaitForMultipleEvents(nEventTotal, eventArray, FALSE, WSA_INFINITE, FALSE);
		nIndex = nIndex - WSA_WAIT_EVENT_0;
		infoRecorder->logTrace("[DisClientForRenderProxy]: wait for multiple events returned: %d.\n", nIndex);
		// call WSAWaitForMultipleEvents for each evet
		for (int i = nIndex; i < nEventTotal; i++){
			nIndex = WSAWaitForMultipleEvents(1, &eventArray[i], TRUE, 1000, FALSE);
			if (nIndex == WSA_WAIT_FAILED || nIndex == WSA_WAIT_TIMEOUT){
				infoRecorder->logTrace("[DisClientForRenderProxy]: wait for '%d' events failed or timed out.\n", nEventTotal);
				continue;
			}
			else{
				WSANETWORKEVENTS event;
				WSAEnumNetworkEvents(sockArray[i], eventArray[i], &event);
				if (event.lNetworkEvents & FD_ACCEPT){
					// deal FD_ACCEPT msg
					if (event.iErrorCode[FD_ACCEPT_BIT] == 0){
						if (nEventTotal > WSA_MAXIMUM_WAIT_EVENTS){
							infoRecorder->logTrace("[DisClientForRenderProxy]: too many connections.\n");
							continue;
						}
						SOCKET snew = accept(sockArray[i], NULL, NULL);

						// set the send buffer size and the recv buffer size
						setTcpBuffer(snew);
						setNBIO(snew);
						// the socket is for a render thread.
#if 0
#if 0
						char tem[50];
						sprintf(tem, "%s:%p", START_RENDER, snew);
						dealCmd(tem, snew);
#endif
#else
						infoRecorder->logTrace("[DisClientForRender]: loader connected to %p.\n", snew);
						WSAEVENT evet = WSACreateEvent();
						WSAEventSelect(snew, evet, FD_READ | FD_CLOSE | FD_WRITE);
						eventArray[nEventTotal] = evet;
						sockArray[nEventTotal] = snew;
						nEventTotal++;
#endif
					}
				}
				else if (event.lNetworkEvents & FD_READ){
					// deal with FD_READ message
					if (event.iErrorCode[FD_READ_BIT] == 0){
						int nRecv = recv(sockArray[i], buffer, sizeof(buffer), 0);
						if (nRecv > 0){
							buffer[nRecv] = 0;
							infoRecorder->logTrace("[DisClientForRenderProxy]: %p recved: %s.\n", sockArray[i], buffer);
							if (dealCmd(buffer, sockArray[i]) == -1){
								infoRecorder->logTrace("[DisClientForRenderProxy]: recved close render cmd.\n");
								break;
							}
						}
						else{
							infoRecorder->logTrace("[DisClientForRenderProxy]: recv failed.\n");
						}
					}
				}
				else if (event.lNetworkEvents & FD_CLOSE){
					// deal with FD_CLOSE msg
					if (event.iErrorCode[FD_CLOSE_BIT] == 0){
						closesocket(sockArray[i]);
						for (int j = i; j < nEventTotal; j++){
							eventArray[j] = eventArray[j + 1];
							sockArray[j] = sockArray[j + 1];
						}
						nEventTotal--;
					}
				}
				else if (event.lNetworkEvents & FD_WRITE){
					// deal with FD_WRITE msg
					// TODO
					infoRecorder->logError("[DisClientForRenderProxy]: socket write event, but do nothing.\n");
				}
				else{
					infoRecorder->logTrace("[DisClientForRenderProxy]: something else happened, events code:%d", event.lNetworkEvents);
				}
			}
		}
	}

	// close 
	///WSACloseEvent()

	return ret;
}

DWORD WINAPI DisClientForRenderProxy::DisClientThreadForRenderProxy(LPVOID param){
	DWORD ret = 0;
	DisClientForRenderProxy * client = (DisClientForRenderProxy*)param;
	//int renderport = client->disConf->getRenderPort();

	client->init(client->disConf->getGraphicPort());

	WSAEVENT evet = WSACreateEvent();
	WSAEventSelect(client->listenSock, evet, FD_ACCEPT | FD_READ | FD_CLOSE);
	client->eventArray[client->nEventTotal] = evet;
	client->sockArray[client->nEventTotal] = client->listenSock;
	client->nEventTotal++;
	WSAEVENT evet1 = WSACreateEvent();
	WSAEventSelect(client->sock, evet1, FD_READ);
	client->eventArray[client->nEventTotal] = evet1;
	client->sockArray[client->nEventTotal] = client->sock;

	client->nEventTotal++;

	while (true){
		// wait all net event
		int nIndex = WSAWaitForMultipleEvents(client->nEventTotal, client->eventArray, FALSE, WSA_INFINITE, FALSE);
		nIndex = nIndex - WSA_WAIT_EVENT_0;
		// call WSAWaitForMultipleEvents for each evet
		for (int i = nIndex; i < client->nEventTotal; i++){
			nIndex = WSAWaitForMultipleEvents(1, &client->eventArray[i], TRUE, 1000, FALSE);
			if (nIndex == WSA_WAIT_FAILED || nIndex == WSA_WAIT_TIMEOUT){
				continue;
			}
			else{
				WSANETWORKEVENTS event;
				WSAEnumNetworkEvents(client->sockArray[i], client->eventArray[i], &event);
				if (event.lNetworkEvents & FD_ACCEPT){
					// deal FD_ACCEPT msg
					if (event.iErrorCode[FD_ACCEPT_BIT] == 0){
						if (client->nEventTotal > WSA_MAXIMUM_WAIT_EVENTS){
							infoRecorder->logTrace("[DisClientForRenderProxy]: too many connections.\n");
							continue;
						}
						SOCKET snew = accept(client->sockArray[i], NULL, NULL);
						
						// the socket is for a render thread.
#if 0
						char tem[50];
						sprintf(tem, "%s:%p", START_RENDER, snew);
						client->dealCmd(tem, snew);
#else
						WSAEVENT evet = WSACreateEvent();
						WSAEventSelect(snew, evet, FD_READ | FD_CLOSE | FD_WRITE);
						client->eventArray[client->nEventTotal] = evet;
						client->sockArray[client->nEventTotal] = snew;
						client->nEventTotal++;
#endif
					}
				}
				else if (event.lNetworkEvents & FD_READ){
					// deal with FD_READ message
					if (event.iErrorCode[FD_READ_BIT] == 0){
						int nRecv = recv(client->sockArray[i], client->buffer, sizeof(client->buffer), 0);
						if (nRecv > 0){
							client->dealCmd(client->buffer, client->sockArray[i]);
						}
						else{
							infoRecorder->logTrace("[DisClientForRenderProxy]: recv failed.\n");
						}
					}
				}
				else if (event.lNetworkEvents & FD_CLOSE){
					// deal with FD_CLOSE msg
					if (event.iErrorCode[FD_CLOSE_BIT] == 0){
						closesocket(client->sockArray[i]);
						for (int j = i; j < client->nEventTotal; j++){
							client->eventArray[j] = client->eventArray[j + 1];
							client->sockArray[j] = client->sockArray[j + 1];
						}
						client->nEventTotal--;
					}
				}
				else if (event.lNetworkEvents & FD_WRITE){
					// deal with FD_WRITE msg

				}
			}
		}
	}
	return ret;
}

void DisClientForRenderProxy::startThread(){
	threadHandle = chBEGINTHREADEX(NULL, 0, DisClientThreadForRenderProxy, this, FALSE, &threadId);
}

int DisClientForRenderProxy::connectManager(char * url, int port){
	return 0;
}