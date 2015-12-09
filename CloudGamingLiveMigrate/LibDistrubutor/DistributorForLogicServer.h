// this is for the logic server 
#ifndef __DISTRIBUTORFORLOGICSERVER_H__
#define __DISTRIBUTORFORLOGICSERVER_H__
#include "distributor.h"
#include <Windows.h>
#include <map>

// run on distributor server manager
class DisServerForLogicServer:public DistributorServer{

};

// listen the controller, game request from render proxy, run on logic server
// run on the loader
class DisClientForLogicServer{
	//DistributorClient * disClient;
	DistributorConfig * disConf;
	CRITICAL_SECTION section;
	LPCRITICAL_SECTION pSection;

	DWORD threadId;
	HANDLE threadHandle;
	HANDLE notifier;

	SOCKET sock;
	sockaddr_in sin;

	DisClientForLogicServer();
	static DisClientForLogicServer * disForLogicServer;
	static DWORD WINAPI DisClientThreadForLogicServer(LPVOID param);
public:

	char buffer[5120];
	int nEventTotal;

	WSAEVENT eventArray[WSA_MAXIMUM_WAIT_EVENTS];
	SOCKET sockArray[WSA_MAXIMUM_WAIT_EVENTS];

	ServerNode * logicServerNode;  // record the logic server's information
	map<SOCKET, Task *> taskMap; // local socket to find a task

	static DisClientForLogicServer * GetDisForLogicServer();
	static void Release();
	void startThread();
	~DisClientForLogicServer();

	int connectManager(char * url, int port);
	int init(int port);
	int dealCmd(char * cmd, SOCKET s);

	int collectInformation();
	int connectToRenders(Task * task);

};
#endif