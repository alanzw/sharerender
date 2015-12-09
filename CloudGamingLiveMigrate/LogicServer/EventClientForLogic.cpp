#include "../LibCore/Utility.h"
#include "EventClientForLogic.h"


/// for the event client

EventClientForProcess::EventClientForProcess():EventNet(EventClient){
	infoRecorder->logTrace("[EventClientForProcess]: constructor called.\n");
}
EventClientForProcess::~EventClientForProcess(){
	infoRecorder->logTrace("[EventClientForProcess]: destructor called.\n");
}

// connect to the loader's event server and get the message from the event server
// the message may contains the changing encoding device request, add render servers and so on.
// this is designed to run on a render server and a logic server
int EventClientForProcess::dealCmd(char * cmd, SOCKET s, int len, void *param){
	infoRecorder->logTrace("[EventClientForProcess]: call dealCmd with:%s.\n", cmd);
	if(!strncmp(cmd, LOGIC_ADD_RENDER, strlen(LOGIC_ADD_RENDER))){
		// the server send the cmd to add a new render for the process
		// msg format is : LOGIC_ADD_RENDER+count+rpocessid+SOCKET
		// get the socket from cmd and duplicate the socket handle
		SOCKET old = NULL;
		char * p = cmd + strlen(LOGIC_ADD_RENDER) +1;
		int count = *(int *)p;
		p+=(sizeof(int)+1);

		DWORD loaderProcessId = *(DWORD*)p;

		p+=(sizeof(DWORD)+1);
		SOCKET sNew = NULL;
		for(int i = 0; i< count; i++){
			old = *(SOCKET*)p;
			sNew = DuplicateSocketFormProcess(loaderProcessId, old);
			addSocketMap(old, sNew);
			csSet->addServer(sNew);
			p+=(sizeof(SOCKET) +1);
		}
		

	}else if(!strncmp(cmd, LOGIC_DECLINE_RENDER, strlen(LOGIC_DECLINE_RENDER))){
		// the server send the cmd to decline the render for the process
		// msg format is: LOGIC_DECLINE_RENDER+count+processid+SOCKET+...
		char * p = cmd + strlen(LOGIC_DECLINE_RENDER) + 1;
		int count = *(int *)p;
		p+=(sizeof(int) +1);
		DWORD loaderProcessId = *(DWORD *)p;
		p+=(sizeof(DWORD) + 1);
		for(int i = 0 ; i < count ; i++){
			SOCKET t = *(SOCKET *)p;
			// find the mapped socket in game process

			csSet->declineServer(getRealSocket(t));

			p+= (sizeof(SOCKET)+1);
		}
	}
	else if(!strncmp(cmd, LOGIC_CONNECT_SERVERS, strlen(LOGIC_CONNECT_SERVERS))){
		// connect to the servers, loader send the game process all the socket
		
		// get the sockets, duplicate it from kernel and create commandserver, the index of each frame is the sequence number of the socket, msg format: LOGIC_CONNECT_SERVERS+[loader process id]+count+SOCKET+...
		char * p = cmd + strlen(LOGIC_CONNECT_SERVERS);
		DWORD loaderProcessId = *(DWORD *)p;
		p+= (sizeof(DWORD) +1);
		int count = *(int *)p;
		p += (sizeof(int) + 1);

		SOCKET t = NULL;
		SOCKET *sockets = NULL;

		sockets = (SOCKET *)malloc(sizeof(SOCKET) * count);

		for(int i = 0; i< count ; i++){
			// get all the socket
			t = *(SOCKET *)p;
			// duplicate the socket from loader process

			sockets[i] = DuplicateSocketFormProcess(loaderProcessId, t);
			addSocketMap(t, sockets[i]);
			p+=(sizeof(SOCKET) + 1);
		}


		// create the commandserverset using the render count
		csSet = new CommandServerSet(count, sockets);
	}
}


// TODO : add code in Present function, to notify each render server their frame index