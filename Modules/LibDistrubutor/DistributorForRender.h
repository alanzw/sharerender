#ifndef __DISTRIBUTORFORRENDER_H__
#define __DISTRIBUTORFORRENDER_H__
// this is for the render proxy
#include "distributor.h"

#include <Windows.h>
#include <map>
#include "../LibCore/CpuWatch.h"
#include "../LibCore/GpuWatch.h"

using namespace std;

// distributor server for render proxy
class DisServerForRenderProxy: public DistributorServer{
	DistributorConfig * disConf;

	DisServerForRenderProxy();
	
	static DisServerForRenderProxy * disForProxy;

public:
	virtual ~DisServerForRenderProxy();
	static DisServerForRenderProxy * GetDisForRenderProxy();
	static void Release();

	int connectManager(char * url, int port);
	int init(int port);
	virtual int dealCmd(char * cmd, SOCKET s);
	int collectInfomation();
	DWORD enterLogicLoop();

};

// listen the rtsp request from client
class DisClientForRenderProxy: public DistributorClient{
	DistributorClient * disClient;

	DisClientForRenderProxy();
	static DisClientForRenderProxy * disForProxy;
public:
	
	static DisClientForRenderProxy * GetDisForRenderProxy();
	static void Release();

	~DisClientForRenderProxy();

};

#endif