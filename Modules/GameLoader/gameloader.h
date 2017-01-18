#ifndef __GAMELOADER_H__
#define __GAMELOADER_H__
// this is for the game loader
// start a 2D game and the video server in a seperated process

//#include "gamespider.h"
#include "GameInfo.h"
#include "../LibDistrubutor/Context.h"
#include "../LibDistrubutor/Distributor.h"
#include "../LibDistrubutor/DistributorForLogic.h"

#include "../LibCore/CpuWatch.h"
#include "../LibCore/GpuWatch.h"
#include "../LibCore/CThread.h"
#include "../LibCore/InfoRecorder.h"
#include "../VideoUtility/rtspconf.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

// use libevent to manage the connection
#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/util.h>
#include <event2/listener.h>

#include <map>

using namespace std;
using namespace cg;
using namespace cg::core;

#if 0
// dis command
extern const char * INFO;   // recv info cmd from dis, collect the domain info
extern const char * ADD_RENDER;  // recv add render cmd from dis
extern const char * DECLINE_RENDER; // recv decline render cmd from dis
extern const char * GAME_EXIT; // recv game exit cmd from dis
extern const char * CANCEL_TASK;   // recv cancel task cmd from DIS
extern const char * START_TASK;

// render cmd
extern const char * START_TASK;  // start game
extern const char * ADD_RENDER; // add the connection to a game


// game cmd
extern const char * PROCESS_STARTED;   // feedback to tell success
extern const char * RENDER_EXIT; // a render has disconnected


extern const char * PROCESS_EXIT;  // loader tell process to exit
#endif



class ManagerBase{
protected:
	ManagerBase(){}
	
public:
	ContextType type;   // dis or render
	map<IDENTIFIER, BaseContext *> ctxMap;

	virtual void addCtx(BaseContext * ctx){

		ctxMap.insert(map<IDENTIFIER, BaseContext*>::value_type(ctx->sock, ctx));
	}
	virtual BaseContext * getCtx(IDENTIFIER id){
		map<IDENTIFIER, BaseContext*>::iterator it = ctxMap.find(id);
		if(it != ctxMap.end()){
			return it->second;
		}
		else
			return NULL;
	}
};
// manage the game process runs on logic server
class ProcessManager : public ManagerBase{
	static ProcessManager * mgr;
	
	ProcessManager(){
		type = PROCESS_CONTEXT;
	}
public:
	
	static ProcessManager * GetManager(){
		if(mgr){
			return mgr;
		}
		else
		{
			mgr = new ProcessManager();
			return mgr;
		}
	}
	bool startManagerThread();

};
class RenderManager : public ManagerBase{
	static RenderManager * mgr;
	RenderManager(){
		type = RENDER_CONTEXT;
	}
public:
	static RenderManager * GetManager(){
		if(!mgr)
			mgr = new RenderManager();
		return mgr;
	}
	

};

class CtrlManager: public ManagerBase{
	static CtrlManager * mgr;
	CtrlManager(){ type = CTRL_CONTEXT;}
public:
	static CtrlManager * GetManager(){
		if(!mgr){
			mgr = new CtrlManager();
		}
		return mgr;
	}
};

DWORD WINAPI ProcessManagerProc(LPVOID param);


// load the game and start
class GameLoader{
	//GameSpider * gameSpider;
	GameInfo * gameInfo;

	HANDLE startGameWithDll(char * gameName, char * dllName, char * gamePath);
	HANDLE start2DGame(char *gameName, char * gamePath);
	HANDLE loadD3DGame(char * gameName);   // load a d3d game
	HANDLE load2DGame(char * gameName);    // load a 2d game and start the video server

	HANDLE startGameWithDll(char * gameName, char * dllName, char * gamePath, char * arg);
	HANDLE start2DGame(char *gameName, char * gamePath, char * arg);
	HANDLE loadD3DGame(char * gameName, char * arg, char * dllName = NULL);   // load a d3d game
	HANDLE load2DGame(char * gameName, char * arg, char * dllName = NULL);    // load a 2d game and start the video server

	bool checkFile(char * path, char * name);
	bool checkDllFile(char * gamePath, char *dllName);
	bool copyFile(char * curPath, char * newPath, char * dllName);
	
	GameLoader();
	GameLoader(char * mapFile);
	static GameLoader * loader;
public:
	static GameLoader * GetLoader(char * mapFile = NULL){
		if(!loader){
			if(mapFile)
				loader = new GameLoader(mapFile);
			else{
				loader = new GameLoader();
			}
		}
		return loader;
	}
	~GameLoader();

	HANDLE loadGame(char * gameName);     // the call entry to the loader
	HANDLE loadGame(char * gameName, char * arg, char * dllName = NULL); // the call entry to the loader
};

struct LogicTask{
	IDENTIFIER id;
	char * name;
	evutil_socket_t processSock;   // process socket for the task
	BaseContext *processCtx;

	size_t renderCount;
	evutil_socket_t renderSock[MAX_RENDER_COUNT]; // store the render socket here
	map<evutil_socket_t,BaseContext *> renderMap;

	// functions
	LogicTask(){
		name = NULL;
		id = 0;
		processCtx = NULL;
		renderCount = 0;
	}
	// add render to the task
	void addRender(BaseContext * ctx){
		infoRecorder->logTrace("[LogicTask]: render count:%d.\n", renderCount);
		printf("[LogicTask]: render count:%d.\n", renderCount);
		renderMap.insert(map<evutil_socket_t, BaseContext *>::value_type(ctx->sock, ctx));
		renderSock[renderCount ++] = ctx->sock;
	}

	// remove the given render of the task
	void removeRender(evutil_socket_t s){
		// remove the RenderContext
		renderMap.erase(s);

		for(int i =0; i< renderCount; i++){
			if(renderSock[i] == s){
				for(int j = i; j< renderCount - 1; j ++)
				{
					renderSock[j] = renderSock[j + 1];
				}
				return;
			}
		}
	}
	// cancel this task
	bool cancel(){
		if(processCtx){
			// tell the process to exit
			processCtx->writeCmd(PROCESS_EXIT);
			processCtx->writeToNet();
		}
	}
	~LogicTask(){
		if(name){
			free(name);
			name = NULL;
		}
		renderCount = 0;

	}
};

class LogicFactory: public DisClient{
	GameLoader * loader;
	ProcessManager * pmgr;
	RenderManager * rmgr;
	CtrlManager * cmgr;
	struct event_base * base;   // the event base

	cg::RTSPConf * conf;
	map<IDENTIFIER, LogicTask *> taskMap;

	struct evconnlistener * processListener;
	struct evconnlistener * renderListener; // listen to the render proxy
	struct evconnlistener * ctrlListener;   // listen the ctrl
	
	static LogicFactory* logic;
	LogicFactory(){
		loader = GameLoader::GetLoader();
		pmgr = ProcessManager::GetManager();
		rmgr = RenderManager::GetManager();
		base = NULL;
		processListener = NULL;
		renderListener = NULL;
		conf = NULL;
	}


	map<evutil_socket_t, BaseContext *> netCtxMap;
	map<IDENTIFIER, BaseContext *> gameMap;

	bool running;// the flag to identify the logic factory is running
	int regulatorInterval; // the interval to schedule for regulation
	DWORD regulatorID;
	HANDLE regulatorHandle;
	int cpuIncreament, gpuIncreament;  // the increment for cpu and gpu when encoding
	int encoderOption;
	static DWORD WINAPI RegulatorProc(LPVOID param);

public:

	inline void setRTSPConf(cg::RTSPConf * c){ conf = c; }

	inline void setEncoderOption(int option){ encoderOption = option;}
	inline int getEncoderOption(){ return encoderOption; }
	// for hardware adaption
	inline void setCpuIncreament(int _val){ cpuIncreament = _val;}
	inline void setGpuINcreament(int _val){ gpuIncreament = _val;}
	inline bool isRunning(){ return running; }
	inline int getRegulatorInterval(){ return regulatorInterval; }
	inline void setRegulatorInterval(int val){ regulatorInterval = val; }
	inline void starRegulator(){
		infoRecorder->logTrace("[LogicFactory]: star the regulator procedure.\n");
		regulatorHandle = chBEGINTHREADEX(NULL, 0, RegulatorProc, this, FALSE, &regulatorID);
	}
	bool regulationCall();



	static LogicFactory * GetFactory(){
		if(!logic)
			logic = new LogicFactory();
		return logic;
	}

	void addNetCtxMap(evutil_socket_t sock, BaseContext * ctx){
		netCtxMap[sock] = ctx;
	}
	bool init();
	bool startListen();
	// start listen to the process
	bool startInternalListen();
	// start listen to the render
	bool startRenderListen();
	bool startCtrlListen();

	bool connectDis(char * ip, short port);
	bool registerLogic();

	void enterLoop();

	// manage the task
	LogicTask * findTask(IDENTIFIER tid);
	bool addTask(LogicTask * task);
	bool removeTask(IDENTIFIER tid);



	// from parent 
	virtual bool dealCmd(BaseContext * ctx);
};



class LoaderLogger: public cg::core::CThread{
	HANDLE processHandle;
	std::string processName;
	HANDLE mappingHandle;
	HANDLE mutexHandle;
	LPVOID mappingAddr;

	CpuWatch * cpuWatcher;
	GpuWatch * gpuWatcher;
	
	LightWeightRecorder * recorder;

	bool initLogger();

public:
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual BOOL onThreadStart();
	virtual void onQuit();
	virtual BOOL run();

	inline void setProcessHandle(HANDLE handle){ processHandle = handle; } 

	LoaderLogger(std::string _processName);

	~LoaderLogger();
	
};

#endif