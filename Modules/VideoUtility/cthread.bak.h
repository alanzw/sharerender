#ifndef __CTHREAD_H__
#define __CTHREAD_H__
/// this is for a message driven thread 
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

class CThread{
public:
	CThread();
	virtual ~CThread();

public:
	BOOL			isRunning(){ return running; }
	BOOL			start(); // start the thread
	BOOL			stop(); // stop the thread

	virtual BOOL	run() = 0; // run the thread func, called multiple time over and over again, if FALSE, the thread will exit
	virtual void	onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam) = 0; // deal the thread message
	virtual BOOL	onThreadStart() = 0; // call once when start the thread, if FALSE, exit
	virtual void	onQuit() = 0; // call when quitting thread

	BOOL			postThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam); // send thread message
	BOOL			isStart(); // is start or not
	DWORD			getThreadId(){ return this->threadId; }
	HANDLE			getThread(){ return this->threadHandle; }

	void			enableSleep(BOOL bEnable); // allow thread to sleep or not
private :
	HANDLE			threadHandle;
	DWORD			threadId;
	BOOL			running;
	int				sleepTime;
	static DWORD WINAPI ThreadProc(LPVOID param);
	
};

// use PostThreadMessage to achieve the message thread

#endif
