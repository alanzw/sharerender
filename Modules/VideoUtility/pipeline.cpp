#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <sstream>
#include <fstream>
#include "../LibCore/log.h"
#include "pipeline.h"
#include "../libcore/InfoRecorder.h"
#include "../LibCore/TimeTool.h"

//#define ENABLE_PIPELINE_LOG


using namespace std;
using namespace cg;
using namespace cg::core;

static HANDLE pipelinemutex = NULL;
static std::map<std::string, pipeline *> pipelinemap;


pipeline::pipeline(int privdata_size, char * prefix){
	if(pipelinemutex == NULL){
		pipelinemutex = CreateMutex(NULL, FALSE, NULL);
	}
	InitializeCriticalSection(&condMutex);
	poolmutex = NULL;
	poolmutex = CreateMutex(NULL, FALSE, NULL);
	if(poolmutex == NULL){
		infoRecorder->logError("[pipeline]: pool mutex creation failed.\n");
	}
	WaitForSingleObject(poolmutex, INFINITE);
	ReleaseMutex(poolmutex);
#ifdef ENABLE_PIPELINE_LOG
	infoRecorder->logError("[pipeline]: pool mutex:%p.\n", poolmutex);
#endif

	int i = 0; 
	char t[100] = {0};

	i++;
	bufpool = NULL;
	datahead = datatail = NULL;
	privdata = NULL;

	if(privdata_size > 0){
		alloc_privdata(privdata_size);
	}

	if(!prefix)
		sprintf(t, "pipe-%d", GetCurrentThreadId());
	else
		sprintf(t,"pipe-%s-%d", prefix, GetCurrentThreadId());
#ifdef ENABLE_PIPELINE_LOG
	infoRecorder->logError("[pipeline]: constructor called, name:%s.\n", t);
#endif
	do_register(t, this);
}

pipeline::~pipeline(){
	if(bufpool){
		datapool_free(bufpool);
		bufpool = NULL;
	}

	datatail = NULL;

	if(privdata){
		free(privdata);
		privdata = NULL;
		privdata_size = 0;
	}

	DeleteCriticalSection(&condMutex);

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
}

const char * pipeline::name(){
	return myname.c_str();
}
struct pooldata * pipeline::allocate_data(){
	// allocate a data from buffer pool
	struct pooldata *data = NULL;
	WaitForSingleObject(poolmutex, INFINITE);

	if (bufpool == NULL) {
		// no more available free data - force to release the eldest one
		data = load_data_unlocked();
		if (data == NULL) {
			infoRecorder->logError("data pool: FATAL - unexpected NULL data returned (pipe '%s', data=%d, free=%d).\n", this->name(), datacount, bufcount);
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
	pipeline::store_data(struct pooldata *data) {
		// store a data into data pool (at the end)
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logError("[pipeline]: pipe '%s' store data.\n", name());
#endif
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
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: %s stored data, data count:%d\n", name(), datacount);
#endif
		ReleaseMutex(poolmutex);
		return;
}


struct pooldata *
	pipeline::load_data_unlocked() {
		// load a data from data (work) pool
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: to load data, data count:%d\n", datacount);
		if(datacount == 0){
			infoRecorder->logError("[pipeline]: %s to load data, but data count is %d.\n",name(), datacount);
		}
#endif
		struct pooldata *data = NULL;

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
	pipeline::load_data() {
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: pipe '%s' load data.\n", name());
#endif
		struct pooldata *data = NULL;

		WaitForSingleObject(poolmutex, INFINITE);
		data = load_data_unlocked();
		ReleaseMutex(poolmutex);
		return data;
}

void
	pipeline::release_data(struct pooldata *data) {
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logError("[pipeline]: pipe '%s' release data.\n", name());
#endif
		// return a data to buffer pool
		WaitForSingleObject(poolmutex, INFINITE);
		data->next = bufpool;
		bufpool = data;
		bufcount++;
		ReleaseMutex(poolmutex);
		return;
}

int
	pipeline::data_count() {
		return datacount;
}

int
	pipeline::buf_count() {
		return bufcount;
}

void *
	pipeline::alloc_privdata(int size) {
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
	pipeline::set_privdata(void *ptr, int size) {
		if (size <= privdata_size) {
			memcpy(privdata, ptr, size);
			//bcopy(ptr, privdata, size);
			return privdata;
		}
		return NULL;
}

void *
	pipeline::get_privdata() {
		return privdata;
}

int
	pipeline::get_privdata_size() {
		return privdata_size;
}


void
	pipeline::client_register(long tid, HANDLE cond) {
		EnterCriticalSection(&condMutex);
		waitEvent = cond;
		clientThreadId = tid;
		condMap[tid] = cond;
		LeaveCriticalSection(&condMutex);
		return;
}

void
	pipeline::client_unregister(long tid) {
		EnterCriticalSection(&condMutex);
		condMap.erase(tid);
		LeaveCriticalSection(&condMutex);
		return;
}

int
	pipeline::wait(HANDLE cond, HANDLE mutex) {
		DWORD ret;
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: pipe %s is waiting for event:0x%p\n", this->name(), cond);
#endif
		WaitForSingleObject(mutex, INFINITE);
		ret = WaitForSingleObject(cond, INFINITE);
		if (ret == WAIT_OBJECT_0){
#ifdef ENABLE_PIPELINE_LOG
			infoRecorder->logTrace("[pipeline]: %s's wait single is signed.\n", name());
#endif
		}
		else if (WAIT_FAILED == ret){
			infoRecorder->logError("[pipeline]: %s wait failed. last error:%d.\n", name(), GetLastError());
		}
		ReleaseMutex(mutex);
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: pipe %s exit waiting..\n", name());
#endif
		return ret;
}

int
	pipeline::timedwait(HANDLE cond, HANDLE mutex, const struct cg::core::timespec *abstime) {
		int ret = 0;
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: pipe %s is timed waiting, time :%d ms...\n", this->name(), abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000);
#endif
		WaitForSingleObject(mutex, INFINITE);
		ret = WaitForSingleObject(cond, abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000);
		if (ret == WAIT_OBJECT_0){
#ifdef ENABLE_PIPELINE_LOG
			infoRecorder->logTrace("[pipeline]: pipe %s timed wait single is signed.\n", name());
#endif
		}
		else if (ret == WAIT_TIMEOUT){
			infoRecorder->logError("[pipeline]: pipe %s timed wait timed out.\n", name());
		}
		else if (ret == WAIT_FAILED){
			infoRecorder->logError("[pipeline]: pipe %s timed wait failed.\n", name());
		}

		ReleaseMutex(mutex);
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: pipe %s exit timed waiting..\n", name());
#endif
		return ret;
}

void
	pipeline::notify_all() {
		EnterCriticalSection(&condMutex);
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: %s notifier the waited event:0x%p.\n", name(), waitEvent);
#endif
		SetEvent(waitEvent);
		LeaveCriticalSection(&condMutex);
		return;
}

void
	pipeline::notify_one(long tid) {
		map<long, HANDLE>::iterator mi;
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("[pipeline]: pipe %s is notifying thread %d...\n", this->name(), tid);
#endif
		EnterCriticalSection(&condMutex);
		if ((mi = condMap.find(tid)) != condMap.end()) {
			SetEvent(mi->second);
		}
		LeaveCriticalSection(&condMutex);
		return;
}

int
	pipeline::client_count() {
		int n;
		EnterCriticalSection(&condMutex);
		n = (int)condMap.size();
		LeaveCriticalSection(&condMutex);
		return n;
}
// not allocate the data area
struct pooldata * pipeline::datapool_init(int n){
	int i = 0;
	struct pooldata *data = NULL;
	//
	if (n <= 0 )
		return NULL;
	//
	bufpool = NULL;

	for (i = 0; i < n; i++) {
		if ((data = (struct pooldata*) malloc(sizeof(struct pooldata))) == NULL)
		{
			bufpool = datapool_free(bufpool);
			return NULL;
		}
		memset(data, 0, sizeof(struct pooldata));

		data->ptr = NULL;		
		data->next = bufpool;
		bufpool = data;
	}
	datacount = 0;
	bufcount = n;
	return bufpool;
}

struct pooldata * pipeline::datapool_init(int n, int datasize) {
	int i;
	struct pooldata *data;
	//
	if (n <= 0 || datasize <= 0)
		return NULL;
	//
	bufpool = NULL;


	// the data is IDirect3DSurface9 *
	for (i = 0; i < n; i++) {
		if ((data = (struct pooldata*) malloc(sizeof(struct pooldata) + datasize)) == NULL)
		{
			bufpool = datapool_free(bufpool);
			return NULL;
		}
		memset(data, 0, sizeof(struct pooldata) + datasize);

		//bzero(data, sizeof(struct pooldata) + datasize);
		data->ptr = ((unsigned char*)data) + sizeof(struct pooldata);


		data->next = bufpool;
		bufpool = data;
	}
	datacount = 0;
	bufcount = n;
	return bufpool;
}

struct pooldata * pipeline::datapool_free(struct pooldata *head) {
	struct pooldata *next;
	//
	if (head == NULL)
		return NULL;
	//
	do {
		next = head->next;
		free(head);
		head = next;
	} while (head != NULL);
	//
	bufpool = datahead = datatail = NULL;
	datacount = bufcount = 0;
	//
	return NULL;
}

//////////////////////// static function //////////////////

int pipeline::do_register(const char * provider, pipeline * pipe){
	DWORD ret = WaitForSingleObject(pipelinemutex, INFINITE);  // lock
#ifdef ENABLE_PIPELINE_LOG
	infoRecorder->logTrace("[pipeline]: register called, provider:%s.\n", provider);
#endif
	if (pipelinemap.find(provider) != pipelinemap.end()){
		// already registered
		ReleaseMutex(pipelinemutex);
		infoRecorder->logError("pipeline: duplicated pipeline '%s'\n", provider);
		return -1;
	}
	pipelinemap[provider] = pipe;
	ReleaseMutex(pipelinemutex);
	pipe->myname = provider;
#ifdef ENABLE_PIPELINE_LOG
	infoRecorder->logTrace("pipeline: new pipeline '%s' registered.\n", provider);
#endif
	return 0;
}

void
	pipeline::do_unregister(const char *provider) {
		DWORD ret = WaitForSingleObject(pipelinemutex, INFINITE);  // lock
		pipelinemap.erase(provider);
		ReleaseMutex(pipelinemutex);
#ifdef ENABLE_PIPELINE_LOG
		infoRecorder->logTrace("pipeline: pipeline '%s' unregistered.\n", provider);
#endif
		return;
}

pipeline *
	pipeline::lookup(const char *provider) {
		map<string, pipeline*>::iterator mi;
		pipeline *pipe = NULL;
		DWORD ret = WaitForSingleObject(pipelinemutex, INFINITE);
		if ((mi = pipelinemap.find(provider)) == pipelinemap.end()) {
			ReleaseMutex(pipelinemutex);
			return NULL;
		}
		pipe = mi->second;
		ReleaseMutex(pipelinemutex);
		return pipe;
}

void pipeline::Release(){
	if (pipelinemutex){
		CloseHandle(pipelinemutex);
		pipelinemutex = NULL;
	}
}