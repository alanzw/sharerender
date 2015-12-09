// this is for the logic context in logic server

#include "LogicContext.h"
#include "../LibCore/InfoRecorder.h"
using namespace cg;
using namespace cg::core;

// for each client connection, the logic context will show the object creation status in clients in a big global map

extern void RaiseToDebugP();

// get the evutil_socket_t from give process
evutil_socket_t GetProcessSocket(evutil_socket_t old, DWORD pid){
	RaiseToDebugP();
	HANDLE sourceHandle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid);
	HANDLE newHandle = 0;
	if(sourceHandle == NULL){
		infoRecorder->logTrace("[global]: GetProcessSocket(), error open process failed.\n");
		return NULL;
	}
	DuplicateHandle(sourceHandle, (HANDLE)old, GetCurrentProcess(), &newHandle, 0, FALSE, DUPLICATE_SAME_ACCESS);
	infoRecorder->logTrace("[Global]: to close the process handle.\n");
	CloseHandle(sourceHandle);
	infoRecorder->logTrace("[global]: pid:%d, old sock:%p, new sock:%p.\n", pid, old, newHandle);

	return (evutil_socket_t)newHandle;
}
#if 0

bool ProcessContext::dealCmd(){
	if(!strncasecmp(cmd, ADD_RENDER, strlen(ADD_RENDER))){
		// recv add render cmd from loader
		evutil_socket_t sockInLoader = *(evutil_socket_t *)data;
		DWORD loaderProcessId = *(DWORD *)(data + sizeof(evutil_socket_t));
		evutil_socket_t newSock = GetProcessSocket(sockInLoader, loaderProcessId);
		if(newSock){
			// add new connection to CommnadServerSet
			infoRecorder->logTrace("[ProcessContext]: get new socket, add to connection.\n");
			socketMap.insert(map<evutil_socket_t, evutil_socket_t>::value_type(sockInLoader, newSock));

			// TODO

			return true;
		}
		else{
			return false;
		}
	}
	else if(!strncasecmp(cmd, DECLINE_RENDER, strlen(DECLINE_RENDER))){
		// recv decline render cmd  from loader
		evutil_socket_t sockInLoader = *(evutil_socket_t *)data;
		evutil_socket_t newSock = 0;
		map<evutil_socket_t, evutil_socket_t>::iterator it = socketMap.find(sockInLoader);
		if(it != socketMap.end()){
			// find
			newSock = it->second;

			// TODO

			//
			return true;
		}
		else{
			// not find, error
			infoRecorder->logError("[ProcessContext]: DECLINE RENDER, cannot find the socket.\n");
			return false;
		}
	}

}
// init the process context
bool ProcessContext::init(){
	infoRecorder->logTrace("[ProcessContext]: init.\n");

	WSADATA WSAData;
	WSAStartup(0x101, &WSAData);

	// create the event base
	base = NULL;
	base = event_base_new();
	if(!base){
		infoRecorder->logError("[ProcessContext]: couldn't create an event_base.\n");
		fprintf(stderr, "Couldn't create an event_base: exiting.\n");
		return false;
	}

	return true;
}

// connect to loader using libevent
bool ProcessContext::connectToLoader(){
	
	infoRecorder->logTrace("[ProcessContext]: connect to loader.\n");
	unsigned short port= INTERNAL_PORT;
	sockaddr_in sin;

	int sin_size = sizeof(sin);
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = inet_addr("127.0.0.1");   // localhost
	sin.sin_port = htons(port);

	struct bufferevent * bev = NULL;
	bev = bufferevent_socket_new(base, -1, BEV_OPT_CLOSE_ON_FREE);
	bufferevent_setcb(bev, onBufferEventRead, NULL, onBufferEventEvent, NULL);
	
	// connnect
	if(bufferevent_socket_connect(bev, (struct sockaddr *)&sin, sizeof(sin))< 0){
		infoRecorder->logError("[ProcessContext]: connect to loader failed.\n");
		bufferevent_free(bev);
		return false;
	}
	return true;


}
// the thread to communicate with loader
DWORD WINAPI ProcessContextProc(LPVOID param){

}
#endif