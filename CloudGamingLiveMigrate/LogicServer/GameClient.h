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

	char * gameName;
	event_base * base;

	map<cg::IDENTIFIER, cg::IDENTIFIER> handleMap;

	evutil_socket_t ctrlSock;

	DWORD ctrlThreadId, rtspThreadID;
	HANDLE ctrlThreadHandle, rtspThreadHandle;

	bool addRenderConnection(SOCKET sock);
	bool declineRenderConnection(SOCKET sock);
	bool startControlThread(SOCKET sock);
	// for rtsp 
	bool startRTSPThread(SOCKET sock);

	bool dealCmdOption(cg::CMD_OPTION option, short value, cg::VideoGen * gen);


	// thread proc

	static DWORD WINAPI CtrlThreadProc(LPVOID param);
	static DWORD WINAPI RTSPThreadProc(LPVOID param);

public:

	GameClient();
	~GameClient();
	cg::BaseContext * getCtx(){	return ctx; }
	void connectToLogicServer();
	inline void setEventBase(event_base * b){ base = b; }
	void dispatch(){ event_base_dispatch(base); }
	bool dealEvent(cg::BaseContext * ctx);
	// only for debug, socket to find the context(new socket in game process)
	map<cg::IDENTIFIER, cg::BaseContext *> renderCtxMap;
	map<cg::IDENTIFIER, cg::BaseContext *> rtspCtxMap;

	void setName(char * name){ gameName = _strdup(name);}
};

#endif