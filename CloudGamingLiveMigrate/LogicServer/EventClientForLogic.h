#ifndef __EVENTCLIENTFORLOGIC_H__
#define __EVENTCLIENTFORLOGIC_H__
#include "CommandServerSet.h"
#include "EventNetwork.h"

#include "SmallHash.h"

extern CommandServerSet * csSet;

// new cmd string
extern const char * LOGIC_CONNECT_SERVERS;

// run along with the hook function dll
// the event client need to connect to renders
class EventClientForProcess: public EventNet{
	// lock for main thread and the event thread
	HANDLE notifier;
	map<SOCKET, SOCKET> socketMap; // the loader's socket value to find the duplicated socket in game process

public:


	EventClientForProcess();
	virtual ~EventClientForProcess();
	virtual int dealCmd(char * cmd, SOCKET s, int len, void *param);
	HANDLE getNotifier();
	inline SOCKET getRealSocket(SOCKET old){
		return socketMap[old];
	}
	inline void addSocketMap(SOCKET old, SOCKET real){ socketMap[old] = real;}
	inline void removeSocketMap(SOCKET old){ socketMap.erase(old);}
};

#endif