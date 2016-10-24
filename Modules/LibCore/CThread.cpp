#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "CThread.h"

namespace cg{
	namespace core{
		CThread::CThread(){
			sleepTime = 1;
			threadId = -1;
		}

		CThread::~CThread(){

		}

		BOOL CThread::isStart(){
			return threadId != -1;
		}

		BOOL CThread::start(){
			// create the thread proc
			this->thread = chBEGINTHREADEX(NULL, 0, ThreadProc, this, NULL, &this->threadId);
			return threadId != -1;
		}

		BOOL CThread::postThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
			if (threadId != -1){
				return PostThreadMessage(threadId, msg, wParam, lParam);
			}
			return FALSE;
		}

		BOOL CThread::stop(){
			if (threadId != -1){
				return PostThreadMessage(threadId, WM_QUIT, 0, 0);
			}
			return FALSE;
		}


		void CThread::enableSleep(BOOL bEnable){
			if (bEnable)
				sleepTime = 1;
			else
				sleepTime = 0;
		}

		DWORD CThread::ThreadProc(LPVOID param){
			CThread * pThread = (CThread *)param;
			if(FALSE == pThread->onThreadStart()){
				// init failed
				return -1;
			}

			MSG msg;
			while (true){
				while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)){
					switch (msg.message){
					case WM_QUIT:
						pThread->onQuit();
						return 0;
					default:
						pThread->onThreadMsg(msg.message, msg.wParam, msg.lParam);
						continue;
					}
				}

				if(FALSE == pThread->run()){
					// error
					pThread->stop();
				}

				//if (pThread->sleepTime > 0)
				Sleep(pThread->sleepTime);  // Sleep(0) will give up the thread time slice
				//pThread->sleepTime = 1;
			}
			return 0;
		}
	}
}