#ifndef __TASKQUEUE_H__
#define __TASKQUEUE_H__

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <process.h>
#include "../libCore/Queue.h"
#include "GameServer.h"
#include "../LibCore/InfoRecorder.h"


#define ENABLE_QUEUE_LOG

#ifndef chBEGINTHREADEX
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

using namespace cg;
using namespace cg::core;

enum QueueStatus{
	QUEUE_INVALID,
	QUEUE_UPDATE, // the queue only responsible for updating
	QUEUE_CREATE   // the queue will create only
};

class TaskQueue{
	int count;

	QueueStatus qStatus;

	void * ctx; // the work context

	CRITICAL_SECTION cs;
	CRITICAL_SECTION mu;

	Queue<IdentifierBase *> taskQueue;

	static DWORD WINAPI QueueProc(LPVOID param);
	HANDLE evt;

	HANDLE mutex;
	bool isLocked;

	void awake(){
#ifdef ENABLE_QUEUE_LOG
		infoRecorder->logTrace("[TaskQueue]: awake.\n");
#endif
		SetEvent(evt);
	}
	static int index;
	DWORD threadId;
	HANDLE threadHandle;

public:
	inline void setStatus(QueueStatus s){ qStatus = s;}
	inline QueueStatus getStatusw(){ return qStatus; }
	inline void setContext(void * c){ ctx = c;}
	// the reset function will clear the queue, destory the thread
	bool reset(){
		if(!CloseHandle(threadHandle)){
			// close the thread get an error
			infoRecorder->logError("[TaskQueue]: exit the working thread error with%:d\n", GetLastError());
		}else{
			return false;
		}

		return true;
	}
	~TaskQueue(){
		reset();
		threadHandle = NULL;
		DeleteCriticalSection(&cs);
		DeleteCriticalSection(&mu);
		if(evt){
			CloseHandle(evt);
			evt = NULL;
		}
		count = 0;
		threadId = 0;
		ctx = NULL;
		isLocked = false;
	}
	TaskQueue(): taskQueue(){
		InitializeCriticalSection((LPCRITICAL_SECTION)&cs);
		InitializeCriticalSection((LPCRITICAL_SECTION)&mu);

		evt = CreateEvent(NULL, FALSE, FALSE, NULL);
		count = 0;
		threadId = 0;
		threadHandle = NULL;
		qStatus = QueueStatus::QUEUE_CREATE;
		ctx = NULL;
		isLocked = false;
	}
	inline void init(){
#ifdef ENABLE_QUEUE_LOG
		infoRecorder->logTrace("[TaskQueue]: init the task queue.\n");
#endif
		mutex = CreateMutex(NULL, TRUE, NULL);
		ReleaseMutex(mutex);
	}

	inline HANDLE getEvent(){return evt;}

	void startThread(){
		threadHandle = chBEGINTHREADEX(NULL, 0, QueueProc, this, FALSE, &threadId);
	}

	void add(IdentifierBase * obj){
		if(qStatus == QUEUE_INVALID){
			return;    // the queue status is INVALID, not need to check
		}else if(qStatus == QUEUE_UPDATE){
			//DebugBreak();
		}
		EnterCriticalSection(&cs);
		count++;
		taskQueue.push(obj);
		if(count == 1){
			awake();
		}

		LeaveCriticalSection(&cs);
#ifdef ENABLE_QUEUE_LOG
		infoRecorder->logTrace("[TaskQueue]: add object, obj count:%d.\n", count);
#endif
	}
	inline IdentifierBase * getObj(){
		return taskQueue.front();
	}
	inline void popObj(){
#ifdef ENABLE_QUEUE_LOG
		infoRecorder->logTrace("[TaskQueue]: pop object.\n");
#endif

		taskQueue.pop();
		EnterCriticalSection(&cs);
		count--;
		LeaveCriticalSection(&cs);
	}
	inline bool isDone(){
		bool ret = false;
		EnterCriticalSection(&cs);
		ret = count == 0 ? true: false;
		LeaveCriticalSection(&cs);
#ifdef ENABLE_QUEUE_LOG
		infoRecorder->logTrace("[TaskQueue]: is done ? %s\n", ret ? "true" : "false");
#endif
		return ret;
	}
	inline void lock(){
#ifdef ENABLE_QUEUE_LOG
		infoRecorder->logTrace("[TaskQueue]: lock the queue.\n");
#endif
		WaitForSingleObject(mutex, INFINITE);
		isLocked = true;
	}
	inline void unlock(){
#ifdef ENABLE_QUEUE_LOG
		infoRecorder->logTrace("[TaskQueue]: unlock the queue.\n");
#endif
		if(isLocked){
			ReleaseMutex(mutex);
			isLocked = false;
		}
	}
};

#endif