#ifndef __EVENTNETWORK_H__
#define __EVENTNETWORK_H__
// to achieve a event network

// the event network is to accept and recv by event, it is a common interface to the windows network. I think this class is used among loader and logic server
#include <WinSock2.h>
#define EVENT_NETWORK

// define the cmd strings
extern const char * LOGIC_ADD_RENDER;
extern const char * LOGIC_DECLINE_RENDER;

enum EventNetType{
	EventServer,
	EventClient
};

class EventNet{
private:
	int init(int port);

	SOCKET connectSock;
	EventNetType type;

	char				buffer[512];
	int					nEventTotal;

	struct sockaddr_in sin;

	// the events and sockets
	WSAEVENT *			eventArray;
	SOCKET	*			sockArray;
public:
	// the callback function to deal the cmd
	// cmd: is the cmd to deal
	// s: is the socket which response to the command
	// param: is the user defined parameters to pass
	virtual int dealCmd(char * cmd, SOCKET s, int len, void * param);
	// form base
	
	virtual int initAsServer(int port);
	virtual int initAsClient(char * ip, int port);
	static DWORD NetThread(LPVOID param);

public:

	EventNet();
	EventNet(EventNetType _type);
	~EventNet();

	//set socket
	inline void setSocket(SOCKET s){ connectSock = s; }
	inline SOCKET getSocket(){return connectSock;}
	// get the socket
	//inline void registerCallback(int (*func)(char *cmd, SOCKET s, int len, void * param){	dealCmd = func; }

	// start the event dealing thread
	bool startEventDealingThread();
	// set the socket out side
	inline void setConnectSock(SOCKET s){ connectSock = s; }
};

#endif