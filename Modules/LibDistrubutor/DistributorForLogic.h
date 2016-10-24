// this is for the logic server 
#ifndef __DISTRIBUTORFORLOGICSERVER_H__
#define __DISTRIBUTORFORLOGICSERVER_H__
#include "distributor.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <map>
#include "Context.h"

namespace cg{

#ifndef USE_LIBEVENT

	class DisClientForLogic: public DistributorClient{

	public:

		virtual int dealCmd(char * cmd, SOCKET s, int len, void * param);
	};

	// run on distributor server manager
	class DisServerForLogicServer:public DistributorServer{

	};

	// listen the controller, game request from render proxy, run on logic server
	// run on the loader
	class DisClientForLogicServer{
		//DistributorClient * disClient;
		DistributorConfig * disConf;
		CRITICAL_SECTION section;
		LPCRITICAL_SECTION pSection;

		DWORD threadId;
		HANDLE threadHandle;
		HANDLE notifier;

		SOCKET sock;
		sockaddr_in sin;

		DisClientForLogicServer();
		static DisClientForLogicServer * disForLogicServer;
		static DWORD WINAPI DisClientThreadForLogicServer(LPVOID param);
	public:

		char buffer[5120];
		int nEventTotal;

		WSAEVENT eventArray[WSA_MAXIMUM_WAIT_EVENTS];
		SOCKET sockArray[WSA_MAXIMUM_WAIT_EVENTS];

		ServerNode * logicServerNode;  // record the logic server's information
		map<SOCKET, Task *> taskMap; // local socket to find a task

		static DisClientForLogicServer * GetDisForLogicServer();
		static void Release();
		void startThread();
		~DisClientForLogicServer();

		int connectManager(char * url, int port);
		int init(int port);
		int dealCmd(char * cmd, SOCKET s);

		int collectInformation();
		int connectToRenders(Task * task);

	};
#else



	// use libevent

	///////////// cmd for logic server ///////////////
	extern const char * INFO;   // recv info cmd from dis, collect the domain info
	extern const char * ADD_RENDER;  // recv add render cmd from dis
	extern const char * DECLINE_RENDER; // recv decline render cmd from dis
	extern const char * GAME_EXIT; // recv game exit cmd from dis
	extern const char * CANCEL_TASK;   // recv cancel task cmd from DIS
	extern const char * START_TASK;
	extern const char * CLIENT_CONNECTED;

	extern const char * START_RTSP_SERVICE; // recv start the rtsp service from dis

	// for game client feedback
	extern const char * CANCEL_SOCKEVENT;  // cancel the socket event, just disable the event
	extern const char * DESTROY_SOCKEVENT; // destroy the socket event, release connection as well

	//////////// cmd for game client /////////////////
	extern const char * COPY_HANDLE; // copy the handle with old handle value
	extern const char * ADD_RENDER; // add render socket
	extern const char * DECLINE_RENDER; // decline a render
	extern const char * ADD_CTRL_CONNECTION;  // add a ctrl connection
	extern const char * ADD_RTSP_CONNECTION;  // add a rtsp connection
	extern const char * RTSP_READY; // the rtsp service is ready
	extern const char * GAME_READY; // the game process is started, feedback to logic server
	extern const char * CANCEL_RTSP_SERVICE; // cancel the rtsp service


	



	////////////// const string for DATA ////////////////
	extern const char * LOGIC; // dis server get the data with REGISTER

	extern const char * PROCESS_EXIT;

	


	// run on a server, as logic server
	class LogicServer :public DisClient{
		map<IDENTIFIER, BaseContext *> renderMap;  // render id to find the render context
		map<IDENTIFIER, BaseContext *> gameMap;  // task id to find game process net context
		map<evutil_socket_t, BaseContext *> netCtxMap; // socket to find the BaseContext, all BaseContext

		static DWORD WINAPI LogicThreadProc(LPVOID param);  // the proc for logic server


		// TODO, add task management



		// solve the rtsp and game process problem

		struct evconnlistener * graphicListener, * gameListener;
		LogicServer():DisClient(){}
		static LogicServer * logicServer;


		//

	public:
		void startGame(char * gameName, IDENTIFIER tid);

		static LogicServer * GetLogicServer(){
			if(!logicServer)
				logicServer = new LogicServer();

			return logicServer;
		}

		virtual bool start(); // start the logic server 
		virtual bool dealEvent(BaseContext * ctx);
		//virtual bool listenRTSP();  // wait for rtsp connection
		virtual bool startRTSP(evutil_socket_t sock);

		// for graphic
		bool startGraphic();
		bool addRenderContext(BaseContext * _ctx);

		// for manage the game process
		bool startListenGameProcess();
	};

#endif

}
#endif