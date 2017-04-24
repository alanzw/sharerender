#include "TaskQueue.h"
#include "../LibCore/InfoRecorder.h"
#include "CommandServerSet.h"

namespace cg{
	namespace core{
#ifndef USE_CLASS_THREAD
DWORD TaskQueue::QueueProc(LPVOID param){
	TaskQueue * taskq = (TaskQueue *)param;

	IdentifierBase * t = NULL;
	taskq->init();

	infoRecorder->logError("[QueueProc]: wait the event.\n");
	WaitForSingleObject(taskq->getEvent(), INFINITE);
	taskq->lock();

	while(true){
		while(!taskq->isDone()){
#ifdef ENABLE_QUEUE_LOG
			infoRecorder->logTrace("[TaskQueue]: loop deal task, context:%p, task count:%d.\n", taskq->ctx, taskq->count);
#endif
			infoRecorder->logError("[TaskQueue]: loop deal task, context:%p, task count:%d.\n", taskq->ctx, taskq->count);
			t = taskq->getObj();
			if(!t){
				infoRecorder->logError("[TaskQueue]: get NULL front, task queue has:%d.\n", taskq->getCount());
				break;

			}
			if(t->sync == true){
				taskq->unlock();
				infoRecorder->logError("[TaskQueue]: the task is synchronized, %s.\n", typeid(*t).name());
			}
			else{
				// do the work here

				if(taskq->qStatus == QUEUE_CREATE){
#ifdef ENABLE_QUEUE_LOG
					infoRecorder->logTrace("[QueueProc]: create, check creation.\n");
					t->print();
#endif
					t->checkCreation(taskq->ctx);
					if(t->stable){
						t->checkUpdate(taskq->ctx);
					}
					//t->sendCreation();
				}
				else if(taskq->qStatus == QUEUE_UPDATE){
#ifdef ENABLE_QUEUE_LOG
					infoRecorder->logTrace("[QueueProc]: update, check update.\n"); 
					t->print();
#endif
					t->checkCreation(taskq->ctx);
					t->checkUpdate(taskq->ctx);
				}else{
					infoRecorder->logError("[QueueProc]: task status is INVALID.\n");
				}
				//t->sendUpdate();
			}

			taskq->popObj();
		}
		//infoRecorder->logError("[QueueProc]: wait the task event.\n");
		WaitForSingleObject(taskq->getEvent(), INFINITE);
		taskq->lock();
	}
}
#else  // USE_CLASS_THREAD
		TaskQueue::TaskQueue(){
			InitializeCriticalSection((LPCRITICAL_SECTION)&cs);
			evt = CreateMutex(NULL, FALSE, NULL);
			mutex = CreateMutex(NULL, FALSE, NULL);
			count = 0;
			qStatus = QUEUE_CREATE;
			ctx = NULL;
			isLocked = false;
			awakeTime = 0;
			totalObjects = 0;
			createTimes = 0;
			updateTimes = 0;
			pPTimer = new PTimer();
			pThreadTimer = new PTimer();
			timeCounter = 0;
			isWaiting = false;
		}
		TaskQueue::~TaskQueue(){
			DeleteCriticalSection(&cs);
			if(mutex){
				CloseHandle(mutex);
				mutex = NULL;
			}
			if(evt){
				CloseHandle(evt);
				evt = NULL;
			}
			ctx = NULL;
			isLocked = false;
			count = 0;
			totalObjects = 0;
			createTimes = 0;
			updateTimes = 0;
			if(pPTimer){
				delete pPTimer;
				pPTimer = NULL;
			}
			if(pThreadTimer){
				delete pThreadTimer;
				pThreadTimer = NULL;
			}
			timeCounter = 0;
		}
		inline void TaskQueue::awake(){
			awakeTime++;
			//infoRecorder->logError("[TaskQueue]: thread %d awake, count:%d.\n", getThreadId(), count);
			ReleaseMutex(evt);
		}

		inline IdentifierBase *TaskQueue::getObj(){
			return taskQueue.front();
		}
		inline void TaskQueue::popObj(){
			taskQueue.pop();
			EnterCriticalSection(&cs);
			count--;
			LeaveCriticalSection(&cs);
		}
		int TaskQueue::getCount(){
			int ret = 0;
			EnterCriticalSection(&cs);
			ret = count;
			LeaveCriticalSection(&cs);
			return ret;
		}
		void TaskQueue::setStatus(QueueStatus s){
			qStatus = s;
		}
		inline QueueStatus TaskQueue::getStatus(){
			return qStatus;
		}
		void TaskQueue::setContext(void *c){
			ctx = c;
		}

		
		// called by main thread
		void TaskQueue::add(IdentifierBase *obj){
			if(QUEUE_INVALID == qStatus){
				return;
			}else if(QUEUE_UPDATE == qStatus){

			}
			EnterCriticalSection(&cs);
			count++;
			totalObjects++;
			taskQueue.push(obj);
			if(isWaiting){
				awake();
			}
			LeaveCriticalSection(&cs);
		}
		void TaskQueue::framePrint(){
			infoRecorder->logError("[TaskQueue]: proc %d current has %d objects to deal, last period totally awake %d times, deal %d objects, actually create:%d, actually update:%d, time counter:%f ms.\n", getThreadId(), count, awakeTime, totalObjects, createTimes, updateTimes, timeCounter * 1000.0 / pThreadTimer->getFreq());
			awakeTime = 0;
			totalObjects = 0;
			createTimes = 0;
			updateTimes = 0;
			timeCounter = 0;

#if 0
			static bool tobreak = true;
			if(tobreak){
				DebugBreak();
			}
#endif
		}

		inline bool TaskQueue::isDone(){
			bool ret = false;
			EnterCriticalSection(&cs);
			ret = taskQueue.empty();
			LeaveCriticalSection(&cs);
			return ret;
		}
		inline void TaskQueue::lock(){
			WaitForSingleObject(mutex, 1000);
			isLocked = true;
		}
		inline void TaskQueue::unlock(){
			if(isLocked){
				ReleaseMutex(mutex);
				isLocked = false;
			}
		}
		void TaskQueue::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){}
		void TaskQueue::onQuit(){}
		// on start, do the initialization
		BOOL TaskQueue::onThreadStart(){
			//infoRecorder->logError("[TaskQueue]: thread %d has been started.\n", getThreadId());
			ReleaseMutex(mutex);
			return TRUE;
		}
		// the function should be done in a thread slice
		BOOL TaskQueue::run(){
			IdentifierBase * ib = NULL;
			
			int ret = 0;
			int tmpTime = 0;
			//infoRecorder->logError("[TaskQueue]: thread %d run.\n", getThreadId());
			isWaiting = true;
			DWORD waitRet = WaitForSingleObject(evt, INFINITE);
			isWaiting = false;
			pThreadTimer->Start();

			switch(waitRet){
			case WAIT_OBJECT_0:
				// object has entered the queue
				//infoRecorder->logError("[TaskQueue]: thread %d wait has been triggered, that means at least an object has been pushed.\n", getThreadId());
				break;
			case WAIT_FAILED:
				// error
				infoRecorder->logError("[TaskQueue]: thread %d wait failed, error:%d.\n", getThreadId(), GetLastError());
				return FALSE;
			case WAIT_ABANDONED:
				// thread exit with out release mutex
				infoRecorder->logError("[TaskQueue]: thread %d exit without releasing mutex.\n");
				ReleaseMutex(evt);
				return FALSE;
			case WAIT_TIMEOUT:
				//infoRecorder->logError("[TaskQueue]: thread %d wait timeout.\n");
				return TRUE;
			default:
				infoRecorder->logError("[TaskQueue]: thread %d wait return %d, unknown.\n", getThreadId(), ret);
				// even timeout, is OK
				return TRUE;
			}

			while(!isDone()){
#ifdef ENABLE_QUEUE_LOG
				infoRecorder->logTrace("[TaskQueue]: loop deal task, count: %d.\n", count);
#endif // ENABLE_QUEUE_LOG
				ib = getObj();
				ContextAndCache * context = (ContextAndCache *)ctx;
				
				if(!ib){
					infoRecorder->logError("[TaskQueue]: get NULL front, task queue has: %d.\n", count);
					break;
				}
				//context->resetChecked(ib->frameCheckFlag);

				if(true == ib->sync){
					infoRecorder->logError("[TaskQueue]: the task is synchronized, %s.\n", typeid(*ib).name());
				}
				else{
					// do the work
					if(QUEUE_CREATE == qStatus){
						// do the creation
#ifdef ENABLE_QUEUE_LOG
						infoRecorder->logTrace("[TaskQueue]: create, check creation.\n");
						ib->print();
#endif  // ENABLE_QUEUE_LOG
						pPTimer->Start();
						ret = ib->checkCreation(ctx);
						tmpTime = pPTimer->Stop();
						if(ret){
							createTimes++;
							ret =0;
							infoRecorder->logError("create time: %f\t", tmpTime * 1000.0 / pPTimer->getFreq());
							ib->print();
						}
						if(ib->stable){
							pPTimer->Start();
							ret = ib->checkUpdate(ctx);
							tmpTime = pPTimer->Stop();
							if(ret){
								updateTimes++;
								ret = 0;
								infoRecorder->logError("update stable time: %f\t", tmpTime * 1000.0 / pPTimer->getFreq());
								ib->print();
							}
						}
						//context->resetChecked(ib->frameCheckFlag);
					}else if(QUEUE_UPDATE == qStatus){
#ifdef ENABLE_QUEUE_LOG
						infoRecorder->logTrace("[TaskQueue]: update, check update.\n");
						ib->print();
#endif
						pPTimer->Start();
						ret = ib->checkCreation(ctx);
						tmpTime = pPTimer->Stop();
						if(ret){
							createTimes++;
							ret = 0;
							infoRecorder->logError("crate time: %f\t", tmpTime * 1000.0 / pPTimer->getFreq());
							ib->print();
						}
						pPTimer->Start();
						ret = ib->checkUpdate(ctx);
						tmpTime = pPTimer->Stop();
						if(ret){
							updateTimes++;
							ret  = 0;
							infoRecorder->logError("update non-stable time: %f\t", tmpTime * 1000.0 / pPTimer->getFreq());
							ib->print();
						}
					}
					else{
						infoRecorder->logError("[TaskQueue]: task status is invalid.\n");
					}
				}
				//context->resetChecked(ib->frameCheckFlag);
				popObj();
			}
			//infoRecorder->logError("[TaskQueue]: actually")
			timeCounter += (int)pThreadTimer->Stop();

			return TRUE;
			
		}
		
#endif  // USE_CLASS_THREAD
	}
}