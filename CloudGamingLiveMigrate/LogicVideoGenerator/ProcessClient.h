#ifndef __PROCESSCLIENT_H__
#define __PROCESSCLIENT_H__

// the process client does the similar work with GameClient, connect to GameLoader and take command from GameLoader, Process Client works for testing the regulation strategy in LogicServer

#include "../LibDistrubutor/Context.h"


class ProcessClient{
	evutil_socket_t sock;   // socket connected to Game Loader
	BaseContext * _ctx;

	char * gameName;
	event_base * base;

	map<IDENTIFIER, IDENTIFIER> handleMap;

	DWORD rtspThreadId;
	HANDLE rtspThreadHandle;

	bool startRTSPThread(SOCKET sock);


	static DWORD RTSPThread(LPVOID param);

public:
	ProcessClient(){
		_ctx = NULL;
		sock = NULL;
		gameName = NULL;
		base = NULL;
		rtspThreadHandle = NULL;
	}
	~ProcessClient(){
		if(gameName){
			free(gameName);
			gameName = NULL;
		}

	}

	BaseContext * getCtx(){ return _ctx;}
	inline void setEventBase(event_base * b){ base = b; }
	void dispatch(){ event_base_dispatch(base); }
	bool dealEvent(BaseContext * ctx);

	map<IDENTIFIER, BaseContext *> rtspCtxMap;
	void setName(char * name){ gameName = _strdup(gameName); }
	
};

#endif