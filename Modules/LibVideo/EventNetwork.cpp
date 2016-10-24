#include <WinSock2.h>
#include "EventNetwork.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>


// this is for the event network, it is the common functions for both server and  client
// constructor and destructor
EventNet::EventNet(){
	// construct the base
	eventArray = NULL;
	sockArray = NULL;

	// store the event and socket
	eventArray = (WSAEVENT *)malloc(sizeof(WSAEVENT)*WSA_MAXIMUM_WAIT_EVENTS);
	sockArray = (SOCKET *)malloc(sizeof(SOCKET)*WSA_MAXIMUM_WAIT_EVENTS);

	if(eventArray == NULL || sockArray == NULL){
		infoRecorder->logError("[NetEvent]: contructor, eventArray:%p, sockArray:%p.\n",eventArray, sockArray);

	}
	connectSock = NULL;
	nEventTotal = 0;

}

EventNet::~EventNet(){
	if(eventArray){
		free(eventArray);
		eventArray = NULL;
	}
	if(sockArray){
		free(sockArray);
		sockArray = NULL;
	}
	if(connectSock){
		closesocket(connectSock);
		connectSock = NULL;
	}
	nEventTotal = 0;
}

// override the init function
int EventNet::init(int port){

	infoRecorder->logError("[EventNet]: call init(port).\n");

	connectSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(connectSock == INVALID_SOCKET){
		return -1;
	}
}

// init the event network as a server
int EventNet::initAsServer(int port){
	infoRecorder->logError("[EventNet]: call initAsServer.\n");

	sin.sin_family = AF_INET;
	sin.sin_port = htons(port);
	sin.sin_addr.S_un.S_addr = INADDR_ANY;

	if(bind(connectSock, (sockaddr *)&sin, sizeof(sin)) == SOCKET_ERROR){
		infoRecorder->logError("[EventNet]: failed bind().\n");
		return -1;
	}
	listen(connectSock, 5);

	// create event object
	WSAEVENT eve = WSACreateEvent();
	WSAEventSelect(connectSock, eve, FD_ACCEPT|FD_CLOSE|FD_READ);
	eventArray[nEventTotal] = eve;
	sockArray[nEventTotal] = connectSock;
	nEventTotal++;

}

int EventNet::initAsClient(char * ip, int port){
	infoRecorder->logError("[EventNet]: call initAsClient.\n");

	sin.sin_family = AF_INET;
	sin.sin_port = htons(port);
	sin.sin_addr.S_un.S_addr = inet_addr(ip);
	
	if(connect(connectSock,(sockaddr *)&sin, sizeof(sockaddr_in)) == SOCKET_ERROR){
		infoRecorder->logError("[EventNet]: failed connect.\n");
		return -1;
	}

	WSAEVENT eve = WSACreateEvent();
	WSAEventSelect(connectSock, eve, FD_READ|FD_WRITE|FD_CLOSE);
	eventArray[nEventTotal] = eve;
	sockArray[nEventTotal] = connectSock;
	nEventTotal ++;
	
}


// the  thread  precedure for the event network
DWORD EventNet::NetThread(LPVOID param){
	EventNet * net = (EventNet *)param;
	
	while(true){

		int nIndex = WSAWaitForMultipleEvents(net->nEventTotal, net->eventArray, FALSE, WSA_INFINITE, FALSE);
		
		nIndex = nIndex - WSA_WAIT_EVENT_0;
		for(int i = nIndex; i< net->nEventTotal; i++){

			nIndex = WSAWaitForMultipleEvents(1, &net->eventArray[i], TRUE, 1000, FALSE);
			if(nIndex == WSA_WAIT_FAILED || nIndex == WSA_WAIT_TIMEOUT){
				continue;
			}
			else{
				// get the message, 
				WSANETWORKEVENTS even;
				WSAEnumNetworkEvents(net->sockArray[i], net->eventArray[i], &even);
				if(even.lNetworkEvents & FD_ACCEPT){
					if(even.iErrorCode[FD_ACCEPT_BIT] == 0){
						// deal the accept event
						if(net->nEventTotal > WSA_MAXIMUM_WAIT_EVENTS){
							infoRecorder->logError("[EventNet]: thread precedure, too many connections.\n");
							continue;
						}
						SOCKET sNew = accept(net->sockArray[i], NULL, NULL);
						WSAEVENT eve = WSACreateEvent();
						WSAEventSelect(sNew, eve, FD_READ|FD_CLOSE|FD_WRITE|FD_ACCEPT);
						// add to event array

						net->eventArray[net->nEventTotal] = eve;
						net->sockArray[net->nEventTotal] = sNew;
						net->nEventTotal++;
					}
				}else if(even.lNetworkEvents & FD_READ){
					// deal the read event
					if(even.iErrorCode[FD_READ_BIT] == 0){
						char szText[256];
						int nRecv = recv(net->sockArray[i], szText, strlen(szText), 0);
						if(nRecv > 0){
							szText[nRecv] = 0;
#if 0
							if(&(net->dealCmd))
								net->dealCmd(szText, net->sockArray[i], strlen(szText), NULL);
							else{
								infoRecorder->logError("[EventNet]: dael cmd function is NULL.\n");

							}
#else
							net->dealCmd(szText, net->sockArray[i], strlen(szText), NULL);
#endif
						}
						else{
							infoRecorder->logError("[EventNet]: recv failed.\n");
						}
					}
				}
				else if(even.lNetworkEvents & FD_CLOSE){
					if(even.iErrorCode[FD_CLOSE_BIT] == 0){
						// deal the close event
						closesocket(net->sockArray[i]);
						for(int j = i; j < net->nEventTotal; j++){
							net->sockArray[j] = net->sockArray[j+1];
							net->eventArray[j] = net->eventArray[j+1];
						}
						net->nEventTotal--;
					}
				}else if(even.lNetworkEvents & FD_WRITE){
					// deal the write event
					infoRecorder->logError("[EventNet]: TODO deal the write event.\n");
				}
			}
		}
	}


	return 0;
}

int EventNet::dealCmd(char * cmd, SOCKET s, int len, void * param){
	infoRecorder->logError("[EventNet]: call deal cmd, do nothing.\n");
	return -1;
}

// create the thread for event network
bool EventNet::startEventDealingThread(){
	return true;
}