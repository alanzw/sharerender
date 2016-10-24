#include <sstream>
#include <fstream>

using namespace std;
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"
#include <time.h>
#include "../LibCore/TimeTool.h"
#include "PipelineBase.h"

PipelineBase::PipelineBase(int privdata_size) {
	//this->condMutex = CreateMutex(NULL, false, "cond mutex");

	InitializeCriticalSection(&condMutex);
	
	this->poolmutex = CreateMutex(NULL, FALSE, NULL);

	bufpool = NULL;
	datahead = datatail = NULL;
	privdata = NULL;
	privdata_size = 0;
	if (privdata_size > 0) {
		alloc_privdata(privdata_size);
	}
	return;
}

PipelineBase::~PipelineBase(){
	//bufpool = datapool_free(bufpool);
	//datahead = datapool_free(datahead);
	datatail = NULL;
	if (privdata) {
		free(privdata);
		privdata = NULL;
		privdata_size = 0;
	}
	DeleteCriticalSection(&condMutex);
	if (waitMutex){
		CloseHandle(waitMutex);
		waitMutex = NULL;
	}
	if (waitEvent){
		CloseHandle(waitEvent);
		waitEvent = NULL;
	}

	map<long, HANDLE>::iterator mi;
	for (mi = condMap.begin(); mi != condMap.end(); mi++){
		if (mi->second){
			CloseHandle(mi->second);
			mi->second = NULL;
		}
	}

	if (poolmutex){
		CloseHandle(poolmutex);
		poolmutex = NULL;
	}
	
	
	return;
}

const char * PipelineBase::name() {
	return myname.c_str();
}
struct pooldata *
	PipelineBase::allocate_data() {
		// allocate a data from buffer pool
		struct pooldata *data = NULL;
		WaitForSingleObject(poolmutex, INFINITE);

		if (bufpool == NULL) {
			// no more available free data - force to release the eldest one
			data = load_data_unlocked();
			if (data == NULL) {
				infoRecorder->logError("data pool: FATAL - unexpected NULL data returned (pipe '%s', data=%d, free=%d).\n",
					this->name(), datacount, bufcount);
				exit(-1);
			}
		}
		else {
			data = bufpool;
			bufpool = data->next;
			data->next = NULL;
			bufcount--;
		}
		ReleaseMutex(poolmutex);

		return data;
	}

void
PipelineBase::store_data(struct pooldata *data) {
	// store a data into data pool (at the end)
	infoRecorder->logTrace("[pipeline]: pipe '%s' store data.\n", name());
	data->next = NULL;
	WaitForSingleObject(poolmutex, INFINITE);

	if (datatail == NULL) {
		// data pool is empty
		datahead = datatail = data;
	}
	else {
		// data pool is not empty
		datatail->next = data;
		datatail = data;
	}
	datacount++;
	infoRecorder->logTrace("[pipeline]: stored data, data count:%d\n", datacount);
	ReleaseMutex(poolmutex);
	return;
}



struct pooldata *
	PipelineBase::load_data_unlocked() {
		// load a data from data (work) pool
		infoRecorder->logTrace("[pipeline]: to load data, data count:%d\n", datacount);
		struct pooldata *data;
		if (datatail == NULL) {
			return NULL;
		}
		data = datahead;
		datahead = data->next;
		data->next = NULL;
		if (datahead == NULL)
			datatail = NULL;
		datacount--;
		return data;
	}


struct pooldata *
	PipelineBase::load_data() {
		infoRecorder->logTrace("[pipeline]: pipe '%s' load data.\n", name());
		struct pooldata *data;

		WaitForSingleObject(poolmutex, INFINITE);
		data = load_data_unlocked();
		ReleaseMutex(poolmutex);
		return data;
	}

void
PipelineBase::release_data(struct pooldata *data) {
	infoRecorder->logTrace("[pipeline]: pipe '%s' release data.\n", name());
	// return a data to buffer pool
	WaitForSingleObject(poolmutex, INFINITE);
	data->next = bufpool;
	bufpool = data;
	bufcount++;
	ReleaseMutex(poolmutex);
	return;
}

int
PipelineBase::data_count() {
	return datacount;
}

int
PipelineBase::buf_count() {
	return bufcount;
}

void *
PipelineBase::alloc_privdata(int size) {
	if (privdata == NULL) {
	alloc_again:
		if ((privdata = malloc(size)) != NULL) {
			privdata_size = size;
		}
		else {
			privdata_size = 0;
		}
		return privdata;
	}
	else if (size <= privdata_size) {
		return privdata;
	}
	// privdata != NULL & size > privdata_size
	free(privdata);
	goto alloc_again;
	// never return from here
	return NULL;
}

void *
PipelineBase::set_privdata(void *ptr, int size) {
	if (size <= privdata_size) {
		memcpy(privdata, ptr, size);
		//bcopy(ptr, privdata, size);
		return privdata;
	}
	return NULL;
}

void *
PipelineBase::get_privdata() {
	return privdata;
}

int
PipelineBase::get_privdata_size() {
	return privdata_size;
}


void
PipelineBase::client_register(long tid, HANDLE cond) {
	//WaitForSingleObject(condMutex,INFINITE);
	EnterCriticalSection(&condMutex);
	waitEvent = cond;
	clientThreadId = tid;
	condMap[tid] = cond;
	//waitMutex = cond;
	//ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return;
}

void
PipelineBase::client_unregister(long tid) {
	EnterCriticalSection(&condMutex);
	//WaitForSingleObject(condMutex, INFINITE);
	condMap.erase(tid);
	//ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return;
}

int
PipelineBase::wait(HANDLE cond, HANDLE mutex) {
	DWORD ret;
	infoRecorder->logTrace("[pipeline]: pipe %s is waiting...\n", this->name());
	WaitForSingleObject(mutex, INFINITE);

	ret = WaitForSingleObject(cond, INFINITE);
	if (ret == WAIT_OBJECT_0){
		Log::logscreen("[pipeline]: wait single is signed.\n");
		infoRecorder->logTrace("[pipeline]: pipe %s wait single is signed.\n", name());
	}
	else if (WAIT_FAILED == ret){
		Log::logscreen("[pipeline]: wait failed. last error:%d.\n", GetLastError());
		infoRecorder->logTrace("[pipeline]: pipe %s wait failed.\n", name());
	}
	//ret = pthread_cond_wait(cond, mutex);

	ReleaseMutex(mutex);
	infoRecorder->logTrace("[pipeline]: pipe %s exit waiting..\n", name());
	return ret;
}

int
PipelineBase::timedwait(HANDLE cond, HANDLE mutex, const struct timespec *abstime) {
	int ret = 0;
	infoRecorder->logTrace("[pipeline]: pipe %s is timed waiting, time :%d ms... %ds:%dns\n", this->name(), abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000, abstime->tv_sec,abstime->tv_nsec);
	Log::logscreen("[pipeline]: pipe %s is timed waiting, time :%d ms...\n", this->name(), abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000);
	WaitForSingleObject(mutex, INFINITE);
	ret = WaitForSingleObject(cond, abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000);
	if (ret == WAIT_OBJECT_0){
		infoRecorder->logTrace("[pipeline]: pipe %s timed wait single is signed.\n", name());
	}
	else if (ret == WAIT_TIMEOUT){
		infoRecorder->logTrace("[pipeline]: pipe %s timed wait timed out.\n", name());
	}
	else if (ret == WAIT_FAILED){
		infoRecorder->logTrace("[pipeline]: pipe %s timed wait failed.\n", name());
	}

	//ret = pthread_cond_timedwait(cond, mutex, abstime);

	ReleaseMutex(mutex);
	infoRecorder->logTrace("[pipeline]: pipe %s exit timed waiting..\n", name());
	return ret;
}

void
PipelineBase::notify_all() {
	map<long, HANDLE>::iterator mi;
	//WaitForSingleObject(condMutex, INFINITE);
	EnterCriticalSection(&condMutex);
#if 0
	for (mi = condmap.begin(); mi != condmap.end(); mi++) {
		SetEvent(mi->second);

	}
#else
	infoRecorder->logTrace("[pipeline]: notify event:%p.\n", waitEvent);
	SetEvent(waitEvent);
#endif
	//ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return;
}

void
PipelineBase::notify_one(long tid) {
	map<long, HANDLE>::iterator mi;
	infoRecorder->logTrace("[pipeline]: pipe %s is notifying thread %d...\n", this->name(), tid);
	//WaitForSingleObject(condMutex, INFINITE);
	EnterCriticalSection(&condMutex);

	if ((mi = condMap.find(tid)) != condMap.end()) {
		SetEvent(mi->second);
	}
	//ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return;
}

int
PipelineBase::client_count() {
	int n;
	//WaitForSingleObject(condMutex, INFINITE);
	EnterCriticalSection(&condMutex);
	n = (int)condMap.size();
	// ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return n;
}