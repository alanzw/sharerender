#ifndef __DISFORCLIENT_H__
#define __DISFORCLIENT_H__
#if 0
#include "../../CloudGamingLiveMigrate/LibCore/DisNetwork.h"
#include "../../CloudGamingLiveMigrate/LibDistrubutor/Distributor.h"
//#include "Distributor.h"

class DisClientForClient: public DistributorClient{
	//DistributorClient * disClient;
	int					renders;
	char **				renderList;
	
	int					newRenders;
	char **				newRenderList;

	HANDLE				urlEvent;
	char *				reqGameName;

	DisNetwork * network;
public:
	DisClientForClient();
	~DisClientForClient();

	virtual int			dealCmd(char * cmd, int len);
	inline	int			getRenderCount(){ return renders; }
	char *				getRenderUrl(int index);
	int					init();
	inline HANDLE		getUrlEvent(){ return urlEvent; }
	inline void			setReqGameName(char * name){ reqGameName = _strdup(name); }

	int					mergeRenders();
};

#else

#define USE_LIBEVENT

#include "../LibDistrubutor/Context.h"
#include "../LibDistrubutor/Distributor.h"
#include <stdio.h>
#include "rtspclientmultisource.h"
#include "../LibCore/InfoRecorder.h"
#include "../LibInput/Controller.h"
using namespace cg;
using namespace cg::core;

#if 0
///////////// cmd for client /////////////////////
extern const char * ADD_RENDER;  // recv add render cmd from dis
extern const char * DECLINE_RENDER; // recv decline render cmd from dis
extern const char * ADD_LOGIC; // add the logic domain to receive control
extern const char * CLIENT_CONNECTED;

////////////// const string for DATA ////////////////
extern const char * LOGIC; // dis server get the data with REGISTER
extern const char * RENDER; // dis server get the data with REGISTER
extern const char * CLIENT; // dis server may get the data with REGISTER, not sure
#endif

// the client is for user(user side)
class UserClient{
	evutil_socket_t sock;  // the connection socket
	struct sockaddr_in sin;

	short rtspCount;  // the number of rtsp connection
	DWORD threadIDs[MAX_RENDER_COUNT];
	HANDLE threadHandles[MAX_RENDER_COUNT];
	IDENTIFIER renderIDs[MAX_RENDER_COUNT];
	char * renderUrls[MAX_RENDER_COUNT];


	DWORD clientThreadId;
	HANDLE clientThreadHandle;
	static DWORD WINAPI ClientThreadProc(LPVOID param);

	bool shutdownRTSP(IDENTIFIER rid);
	BaseContext * ctx;
	// 
	event_base * base;  // base for libevent


	GameStreams * gameStream;
	char * gameName;
	

public:
	cg::input::CtrlMessagerClient * ctrlClient;
	inline void setEventBase(event_base * _b){ base = _b; }
	inline void setName(char * name){ gameName = _strdup(name); }
	void startClientThread(){
		clientThreadHandle = chBEGINTHREADEX(NULL, 0, ClientThreadProc, this, FALSE, &clientThreadId);
	}
	inline HANDLE getThreadHandle(){ return clientThreadHandle; }

	inline BaseContext * getCtx(){ return ctx; }
	UserClient(){
		ctrlClient = NULL;
		base = NULL;
		ctx = NULL;
		sock = NULL;
		rtspCount = 0;
		for (int i = 0; i < MAX_RENDER_COUNT; i++){
			threadIDs[i] = 0;
			threadHandles[i] = NULL;
			renderIDs[i] = NULL;
			renderUrls[i] = NULL;
		}
		gameStream = GameStreams::GetStreams();
	}
	~UserClient(){
		for (int i = 0; i < MAX_RENDER_COUNT; i++){
			if (renderUrls[i]){
				free(renderUrls[i]);
				renderUrls[i] = NULL;
			}

		}
		rtspCount = 0;
		if (ctx){
			delete ctx;
			ctx = NULL;
		}
		
	}
	bool init(){
		cg::core::infoRecorder->logTrace("[UserClient]: init.\n");
		base = event_base_new();
		return true;
	}
	bool launchRequest(char * disServerUrl, int port, char * gameName);

	bool dealEvent(BaseContext * ctx);

	// deal the add render cmd
	bool startRTSP(char * url, int port);
	// deal the decline render cmd
	bool cancelRTSP(IDENTIFIER rid, char * url);    //????????
	// deal the add logic cmd
	bool addLogic(char * url);

	void dispatch(){
		cg::core::infoRecorder->logTrace("[UserClient]: dispatch event.\n");
		event_base_dispatch(base);
	}
};

DWORD WINAPI NetworkThreadProc(LPVOID param);

/// used only by direct request rtsp from render proxy
DWORD WINAPI UserClientThreadProc(LPVOID param);  
#if 0
// callback function for client network
void onBufferEventEvent(struct bufferevent * bev, short events, void * ctx);
void onBufferEventRead(struct bufferevent * bev, void *ctx);
#endif
void clientReadCB(struct bufferevent * bev, void * arg);
void clientErrorCB(struct bufferevent * bev, short what, void * arg);
#endif    
#endif  // __DISFORCLIENT_H__