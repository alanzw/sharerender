#ifndef __CTHREAD_H__
#define __CTHREAD_H__
/// this is for a message driven thread 
#include <process.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>


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

namespace cg{
	namespace core{

		class CThread{
		public:
			CThread();
			~CThread();

		public:
			BOOL start(); // start the thread
			virtual BOOL stop(); // stop the thread
			virtual BOOL run() = 0; // run the thread func, called multiple time over and over again

			void enableSleep(BOOL bEnable); // allow thread to sleep or not

			virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam) = 0; // deal the thread message
			virtual BOOL onThreadStart() = 0; // call once when start the thread
			virtual void onQuit() = 0; // call when quitting thread

			BOOL postThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam); // send thread message
			BOOL isStart(); // is start or not
			DWORD getThreadId(){ return this->threadId; }
			HANDLE getThread(){ return this->thread; }

		private :
			HANDLE thread;
			DWORD threadId;
			static DWORD WINAPI ThreadProc(LPVOID param);
			int sleepTime;
		};

		// use PostThreadMessage to achieve the message thread

	}
}
#endif
