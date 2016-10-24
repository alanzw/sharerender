#include <Windows.h>
#include "cthread.h"

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

void CThread::onThreadStart(){

}

void CThread::onQuit(){}

void CThread::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
	//TranslateMessage()
}
void CThread::run(){

}



void CThread::enableSleep(BOOL bEnable){
	if (bEnable)
		sleepTime = 1;
	else
		sleepTime = 0;
}

DWORD CThread::ThreadProc(LPVOID param){
	CThread * pThread = (CThread *)param;
	pThread->onThreadStart();

	MSG msg;
	while (true){
		while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)){
			switch (msg.message){
			case WM_QUIT:
				pThread->onQuit();
				return 0;
				break;
			default:
				pThread->onThreadMsg(msg.message, msg.wParam, msg.lParam);
				continue;
			}
		}

		pThread->run();

		if (pThread->sleepTime > 0)
			Sleep(pThread->sleepTime);
		pThread->sleepTime = 1;
	}
	return 0;
}