#include "TaskQueue.h"
#include "../LibCore/InfoRecorder.h"

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