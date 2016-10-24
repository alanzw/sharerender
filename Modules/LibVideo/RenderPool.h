#ifndef __RENDERPOOL__
#define __RENDERPOOL__
// manage the render pool and monitor the information of local machine
#include "Commonwin32.h"

class RenderTask{
	int task_id;
	char * client_ip;
	int client_port;
	SOCKADDR_IN peer_addr_conn;
	DWORD parent_process_id;
};

class RenderPool{
	RenderTask * task_list;
};

#endif