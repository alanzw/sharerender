#ifndef __GAMECLIENT_H__
#define __GAMECLIENT_H__

#include "../LibDistrubutor/Context.h"
#include "../LibDistrubutor/DistributorForLogic.h"
#include "../VideoGen/generator.h"

// the game client runs inside a game process, recv cmd from logic server
// start game, copy socket handle received to game process, add render, decline render, start ctrl module
class GameClient{
	evutil_socket_t sock;
	cg::BaseContext * ctx;

	cg::IDENTIFIER taskId;

	char * gameName;
	event_base * base;

	map<cg::IDENTIFIER, cg::IDENTIFIER> handleMap;

	evutil_socket_t ctrlSock;

	DWORD ctrlThreadId, rtspThreadID;
	HANDLE ctrlThreadHandle, rtspThreadHandle;

	HANDLE clientEvent;

	bool addRenderConnection(SOCKET sock);
	bool declineRenderConnection(SOCKET sock);
	bool startControlThread(SOCKET sock);
	// for rtsp 
	bool startRTSPThread(SOCKET sock);

	bool dealCmdOption(cg::CMD_OPTION option, short value, cg::VideoGen * gen);


	// thread proc

	static DWORD WINAPI CtrlThreadProc(LPVOID param);
	static DWORD WINAPI RTSPThreadProc(LPVOID param);
	static GameClient * gameClient;
	static bool initialized;

	GameClient();
public:

	HANDLE getClientEvent(){ return clientEvent; }


	static GameClient * GetGameClient(){
		if(!gameClient){
			gameClient = new GameClient();
		}
		return gameClient;
	}
	static void Release(){ if(gameClient){ delete gameClient; gameClient = NULL; initialized = false;}}
	static bool IsInitialized(){ return initialized; }
	static void	SetInitialized(bool val){ initialized = val; }
	
	~GameClient();
	cg::BaseContext * getCtx(){	return ctx; }
	void connectToLogicServer();
	inline void setEventBase(event_base * b){ base = b; }
	void dispatch(){ event_base_dispatch(base); }
	bool dealEvent(cg::BaseContext * ctx);

	inline void setTaskID(cg::IDENTIFIER val){ taskId= val;}
	inline cg::IDENTIFIER getTaskID(){ return taskId; }
	bool notifyGameReady(){
		bool ret = false;
		if(ctx){
			ctx->writeCmd(cg::GAME_READY);
			ctx->writeIdentifier(taskId);
			ctx->writeToNet();
			ret = true;
		}
		return ret;
	}
	// only for debug, socket to find the context(new socket in game process)
	map<cg::IDENTIFIER, cg::BaseContext *> renderCtxMap;
	map<cg::IDENTIFIER, cg::BaseContext *> rtspCtxMap;

	void setName(char * name){ gameName = _strdup(name);}
};

class ListenServer{
	evutil_socket_t sock;
	cg::BaseContext * ctx;
	event_base * base;
	evconnlistener * listener;
	void * _csSet;

	
public:
	ListenServer();
	~ListenServer();
	cg::BaseContext * getCtx(){ return ctx; }

	bool addRenderConnection(SOCKET sock);
	bool declineRenderConnection(SOCKET sock);
	bool startListen(int port);
	inline void setCSSet(void * csSet){ _csSet = csSet; }
	inline void setEventBase(event_base * b){ base = b; }
	void dispatch();
	bool dealEvent(cg::BaseContext * ctx);
	
};

#endif