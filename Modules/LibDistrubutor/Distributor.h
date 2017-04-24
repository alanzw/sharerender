#ifndef __DISTRIBUTOR_H__
#define __DISTRIBUTOR_H__

#define USE_LIBEVENT

#include <string>
#include <WinSock2.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <stdio.h>

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/util.h>
#include <event2/listener.h>
#include <event2/keyvalq_struct.h>
#include <map>
#include <list>

#include "Context.h"

using namespace std;

#define MAX_BUFFER_SIZE 1024
#define MAX_CMD_LEN 50
#define MAX_RENDER_COUNT 4

#define OVERLOAD_THRESHOLD 95   // 95% usage is the overload bound
#define HEAVEYLOAD_THRESHOLD 80  // 80% usage is the heavy load bound
#define THREATLOAD_THRESHOLD 60

#define GREEN_THRESHOLD 30

/// define the port
#define DIS_PORT_CLIENT 8556	// game process to connect game loader
#define DIS_PORT_CTRL  8555   // for control connection   
#define DIS_PORT_DOMAIN 8557   // dis server, logic server and render, user client to request games
#define DIS_PORT_GRAPHIC 60000   // graphic connection between logic and render
#define DIS_PORT_RTSP 8554  // rtsp connection between render and client


#define LOCAL_TEST

#ifdef LOCAL_TEST
#define LOGCAL_HOST "127.0.0.1"
#define DIS_URL_DISSERVER "127.0.0.1"

#define DIS_URL_LOGIC "127.0.0.1"
#define DIS_URL_RENDER "127.0.01"
#else  // LOCAL_TEST
#define LOGCAL_HOST "192.168.1.100"
#define DIS_URL_DISSERVER "192.168.1.100"

#define DIS_URL_LOGIC "192.168.1.100"
#define DIS_URL_RENDER "192.168.1.100"
#endif  // LOCAL_TEST

// for regulation, define the regulation threshold
#define REGULATION_THRESHOLD 10

// enable use best-fit strategy, default is worst fit
//#define SCHEDULE_USE_BEST_FIT

#ifndef chBEGINTHREADEX
#include <process.h>
typedef unsigned(__stdcall * PTHREAD_START)(void *);
#define chBEGINTHREADEX(psa, cbStack, pfnStartAddr, \
	pvParam, fdwCreate, pdwThreadID) \
	((HANDLE)_beginthreadex(\
	(void *)(psa), \
	(unsigned)(cbStack), \
	(PTHREAD_START)(pfnStartAddr), \
	(void *)(pvParam), \
	(unsigned)(fdwCreate), \
	(unsigned *)(pdwThreadID)))

#endif

#ifndef EVUTIL_ERR_CONNECT_RETRIABLE
#define EVUTIL_ERR_CONNECT_RETRIABLE(e) \
	((e) == WSAEWOULDBLOCK || \
	(e) == WSAEINTR || \
	(e) == WSAEINPROGRESS || \
	(e) == WSAEINVAL)

#endif

namespace cg{

	///////////////////////// command to deal for each side /////////////////////
	////////////// cmd for dis server//////////////////
	extern const char * INFO;    // recv info from domains
	extern const char * REGISTER; /// recv register cmd from domains
	extern const char * REQ_GAME; // recv game request from client
	extern const char * RENDER_OVERLOAD; // recv render overload cmd from render proxy
	extern const char * LOGIC_OVERLOAD; // recv logic server overload cmd from logic server
	extern const char * CLIENT_EXIT; // recv client exit cmd from client or render proxy
	extern const char * LOGIC_READY; //recv logic ready cmd from logic server, means that the game is started
	extern const char * RENDER_READY; // recv render ready cmd from render server, means that the render thread is started
	extern const char * RENDER_CONTEXT_READY; // when complete the context synchronize in the logic server
	extern const char * RTSP_READY;
	extern const char * RTSP_READY; // rtsp service ready from logic server or render
	extern const char * ADD_RTSP_CONNECTION;
	extern const char * OPTION;   // the option cmd
	extern const char * START_TASK;
	extern const char * START_GAME; // only game server and render, render send start game to request the game
	extern const char * CANCEL_TASK;

	////////////// const string for DATA ////////////////
	extern const char * LOGIC; // dis server get the data with REGISTER
	extern const char * RENDER; // dis server get the data with REGISTER
	extern const char * CLIENT; // dis server may get the data with REGISTER, not sure


	//////////// const string for client///////////////
	extern const char * ADD_RENDER;  // recv add render cmd from dis
	extern const char * DECLINE_RENDER; // recv decline render cmd from dis
	extern const char * ADD_LOGIC; // add the logic domain to receive control
	extern const char * CLIENT_CONNECTED;


	class DisServer{
		bool watchdoaRunning;
		int collectionInterval;  // interval to collect information

		char * disServerUrl;

		// all maps for ctx, domain and task
		//map<evutil_socket_t, BaseContext *> ctxMap;
		map<evutil_socket_t, DomainInfo *> domainMap;
		map<IDENTIFIER, TaskInfo *> taskMap;
		map<evutil_socket_t, DomainInfo *> logicMap;
		map<evutil_socket_t, DomainInfo *> renderMap;
		map<evutil_socket_t, DomainInfo *> clientMap;

		list<DomainInfo *> lightWeightLogic;
		list<DomainInfo *> heavyLoadLogic;
		list<DomainInfo *> overloadLogic;

		list<DomainInfo *> lightweightRender;
		list<DomainInfo *> heavyLoadRender;
		list<DomainInfo *> overloadRender;

		DisServer();
		event_base * base;
		HANDLE mutex;

		// timer
		struct event *clockEvent;
		static DisServer * server;

		// for the domains
		static short offsetBase;
		DomainInfo * getLogicCandidate(float  cpuRe, float  gpure);
		DomainInfo * getRenderCandidate(float  cpuRe, float  gpuRe);

		bool sendCmdToRender(TaskInfo * task, char * cmddata, int len);
		bool sendCmdToRender(TaskInfo * task, const char * cmd);
		// base function for render
		bool migrateRendering(DomainInfo * src, DomainInfo * dst, TaskInfo * task);
		bool addRender(TaskInfo * task, DomainInfo * dstRender);
		// solve the overload of the domains
		bool solveRenderOverload(DomainInfo * render);
		bool solveLogicOverload(DomainInfo * logic);
		// send cmd to logic of the task to decline older render
		bool cancelRender(TaskInfo * task, DomainInfo * oldRender);
		bool cancelRender(TaskInfo * task, IDENTIFIER oldRenderId);
		// send cmd to new render add logic
		bool mergeRender(DomainInfo * render);
		bool mergeLogic(DomainInfo * logic);
		// schedule when a domain updated
		bool scheduleWhenUpdate(DomainInfo * domain);
		// build the task for game request
		TaskInfo * buildTask(string taskName, DomainInfo * client);
		// dispatch new task to logic and renders
		bool dispatchToLogic(TaskInfo * task);
	public:

		inline void setDisUrl(char * url){
			int len = strlen(url);
			disServerUrl = (char *)malloc(sizeof(char) * len + 10);
			strcpy(disServerUrl, url);
		}
		inline char * getServerUrl(){ return disServerUrl;}

		inline void lock(){
			WaitForSingleObject(mutex, INFINITE);
		}
		inline void unlock(){
			ReleaseMutex(mutex);
		}
		virtual ~DisServer();
		inline void setEventBase(event_base * _b){
			base = _b;
		}
		inline void dispatch(){
			event_base_dispatch(base);
		}

		static DisServer * GetDisServer(){
			if (server){
				return server;
			}
			server = new DisServer();
			return server;
		}

		bool dealEvent(DomainInfo * ctx);
		void startWatchdog();
		void collectInfo();
		bool isWatchdogRunning(){ return watchdoaRunning; }
		int getSleepInterval(){ return collectionInterval; }

		bool ableToMigrate();
		void printDomain();
		void printTask();

		// helper API
		void AddRenderToTask(IDENTIFIER id);
		void ChangeOffloadLevel(IDENTIFIER id, TASK_MODE taskMode = MODE_FULL_OFFLOAD);
		void ChangeEncoderType(IDENTIFIER taskId, IDENTIFIER domainId);
		bool sendCmdToRender(TaskInfo * task, const char * cmd, DomainInfo * render);
	};
	// this client is for logic and render
	class DisClient{
		evutil_socket_t sock;   // the connection socket
		struct evconnlistener * connlistener;
	public:
		BaseContext * ctx;
		event_base * base;

		bool		connectToServer(char * ip, short port);  // connect to dis server
		void		collectInfo(float & cpuUsage, float & gpuUsage, float & memUsage);
		inline void setEventBase(event_base * b){ base = b; }
		inline BaseContext * getCtx(){ return ctx;  }
		inline void dispatch(){
			printf("[DisClient]: dispatch.\n");
			event_base_dispatch(base);
		}

		virtual bool dealEvent(BaseContext * ctx);
		virtual bool start();  // start the dis client
		//virtual bool listenRTSP();  // wait for rtsp connection
		virtual ~DisClient(){}
		DisClient(){
			ctx = NULL;
			base = NULL;
			sock = NULL;
			connlistener = NULL;
		}

		// for rtsp
		bool		listenRTSP(short portOffset);
		virtual bool startRTSP(evutil_socket_t sock){
			printf("[DisStartRTSP]:\n");
			return true;
		}
		inline event_base * getBase(){ return base; }
	};

	extern BaseContext * ctxToDis;

	void frobSocket(evutil_socket_t sock);
	void DisClientReadCB(struct bufferevent * bev, void * arg);
	void DisClientEventCB(struct bufferevent * bev, short what, void * arg);
	void DisClientWriteCB(struct bufferevent * bev, void * arg);
}
#endif