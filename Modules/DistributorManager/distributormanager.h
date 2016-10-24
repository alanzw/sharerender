#ifndef __DISTRIBUTORMANAGER_H__
#define __DISTRIBUTORMANAGER_H__
// this for the distributor manager
#include "distributor.h"
using namespace std;
#define DISTRIBUTOR_CONFIG "config\\server.distributor.conf"

class TaskManager{
	CRITICAL_SECTION			section;
	LPCRITICAL_SECTION			pSection;  // critical section for task manager
	int							totalTasks;  // the total number of tasks
	map<SOCKET, Task *>			allTaskMap;  // all the task map
	queue<Task *>				taskQueue;  /// the task queue

	HANDLE						notifier;

public:
	TaskManager();
	~TaskManager();

	bool						addTask(Task * task);
	bool						changeStatus(SOCKET s, TASK_STATUS targetStatus);
	TASK_STATUS					getTaskStatus(SOCKET s);
	inline HANDLE				getNotifier(){ return notifier; }
	inline void					triggerNotifier(){ SetEvent(notifier); }
	inline int					getUndoneTaskCount(){ return taskQueue.size(); }
	Task *						popTask();  // pop a task from task queue and remove
	int							allMoveOneStep();

	Task *						getAssignedTask(SOCKET s);
};

class DistributorServerManager{
	CRITICAL_SECTION			section;  // for change the server count
	LPCRITICAL_SECTION			pSection;

	DistributorConfig *			conf;  // read the ports
	//DistributorServer *		disLogic;
#if 0
	DistributorServer *			userServer;   // deal with client connection
	DistributorClient *			logicClient; // send request to logic server
	DistributorClient *			renderClient;  // send request to render server
#else
	UserServer *				userServer;
	DistributorServer *			logicServer;
	DistributorServer *			renderServer;
#endif
	// for distributing managing
	int							logicServers; // the count for the logic servers
	int							renderServers; // the count for the render servers
	int							users;
	int							tasks;

	int							sleepInterval;
	bool						running;

	TaskManager *				taskManager;
	// for manager the task map

	HANDLE watchdogThread;
	DWORD watchdogThreadId;
public:
	DistributorServerManager(char * configFileName = NULL);
	~DistributorServerManager();
#if 0
	inline DistributorServer *	getUserServer(){ return userServer; }
#else
	inline UserServer *			getUserServer(){ return userServer; }
#endif
	inline DistributorServer *	getLogicServer(){ return logicServer; }
	inline DistributorServer *	getRenderServer(){ return renderServer; }
	inline TaskManager *		getTaskManager(){ return taskManager; }
	inline void					setSleepInterval(int s){ sleepInterval = s; }
	inline int					getSleepInterval(){ return sleepInterval * 1000; }

	bool						init();
	bool						collectInfomation();
	bool						isRunning();//{ bool ret = true; EnterCriticalSection(pSection); ret = running; LeaveCriticalSection(pSection); return ret; }
	//SOCKET acceptClient
	bool						startUp();   // start the logic thread and the render thread, and the watchdog
	int							sendCmd(char * cmd, SOCKET s);

	// the watchdog is for timely collecting information
	int							startWatchdog();
	static DWORD WINAPI			Watchdog(LPVOID param);
#if 0
	// from parent class
	virtual void				onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void				onThreadStart();
	virtual void				onQuit();
	virtual BOOL				stop();
	virtual void				run();
#endif
	// deal client or servers' request
	int							dealClientGameReq();
	int							dealServerReq(DistributorServer * server);
	int							dealClientFeedback();
	int							dealTask();
	ResourceRequire *			estimateResourceOccupy(char * game);  // esitmate the resource requirement for given game
#if 0
	char *						findLogicCandidateUrl(ResourceRequire *upbound);
	char *						findRenderCandidateUrl(ResourceRequire * upbound);
	SOCKET						findLogicCandidate(ResourceRequire * upbound);
	SOCKET						findRenderCandidate(ResourceRequire * upbound);
#endif
	NetDomain *					findLogicCandidateDomain(ResourceRequire * upbound);
	int							findRenderCandidateDomain(ResourceRequire * upbound, int count, NetDomain *** ret);

	
	void						checkManager();
};
#endif