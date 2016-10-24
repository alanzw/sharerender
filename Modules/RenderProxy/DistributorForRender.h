#ifndef __DISTRIBUTORFORRENDER_H__
#define __DISTRIBUTORFORRENDER_H__
// this is for the render proxy
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <map>
#include "../LibCore/CpuWatch.h"
#include "../LibCore/GpuWatch.h"

using namespace std;

// listen the rtsp request from client
// run on render proxy 
class DisClientForRenderProxy : public DistributorBase{
	//DistributorClient * disClient;  // accept the 
	DistributorConfig *			disConf;
	CRITICAL_SECTION			section;
	LPCRITICAL_SECTION			pSection;

	DWORD						threadId;
	HANDLE						threadHandle;
	HANDLE						notifier;

	SOCKET						sock, listenSock;
	sockaddr_in					sin, msin;

	CpuWatch *					cpuWatcher;
	GpuWatch *					gpuWatcher;

	DisClientForRenderProxy();
	static DisClientForRenderProxy * disForProxy;
	static DWORD WINAPI			DisClientThreadForRenderProxy(LPVOID param);
public:

	char						buffer[5120];
	int							nEventTotal;
	WSAEVENT					eventArray[WSA_MAXIMUM_WAIT_EVENTS];
	SOCKET						sockArray[WSA_MAXIMUM_WAIT_EVENTS];

	ServerNode *				renderServerNode; // record the render server's information and task map
	map<SOCKET, Task *>			taskMap; // local socket to find a task
	
	~DisClientForRenderProxy();
	static DisClientForRenderProxy * GetDisForRenderProxy();
	static void					Release();
	void						startThread();
	void						closeClient();
	
	int							connectManager(char * url, int port);
	int							init(int port);
	int							dealCmd(char * cmd, SOCKET s);
	int							collectInformation();
	DWORD						enterLogicLoop();
};

#endif