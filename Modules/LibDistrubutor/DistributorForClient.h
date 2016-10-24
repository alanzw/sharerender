#ifndef __DISTRIBUTORFORCLIENT_H__
#define __DSITRIBUTORFORCLIENT_H__
// this is for the client part
#include "Distributor.h"
#include <Windows.h>
// run on distributor server manager
class DisServerForClient : public DistributorServer{

	ReqEngine * reqEngine; //only for the client server, record the client game request

public:
	virtual int dealcmd(char * msg, SOCKET s);
};

// run on the client side
class DisClientForClient{
	DistributorClient * disClient;
};
#endif