#include <sstream>
#include <fstream>

using namespace std;

#include "log.h"

#include "pipeline.h"
#include "inforecoder.h"
#include "TimeTool.h"

extern InfoRecorder * infoRecorder;

static HANDLE pipelinemutex = NULL;
static map<string, pipeline *> pipelinemap;

#if 0

int pipeline::do_register(const char * provider, pipeline * pipe){
	DWORD ret = WaitForSingleObject(pipelinemutex, INFINITE);  // lock
	infoRecorder->logError("[pipeline]: register called, provider:%s.\n", provider);
	if (pipelinemap.find(provider) != pipelinemap.end()){
		// already registered
		ReleaseMutex(pipelinemutex);
		Log::slog("pipeline: duplicated pipeline '%s'\n", provider);
		return -1;
	}
	pipelinemap[provider] = pipe;
	ReleaseMutex(pipelinemutex);
	pipe->myname = provider;
	Log::log("pipeline: new pipeline '%s' registered.\n", provider);
	return 0;
}

void
	pipeline::do_unregister(const char *provider) {
		DWORD ret = WaitForSingleObject(pipelinemutex, INFINITE);  // lock
		pipelinemap.erase(provider);
		ReleaseMutex(pipelinemutex);
		Log::log("pipeline: pipeline '%s' unregistered.\n", provider);
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

//////////////////////////////////////////////////////////////////////////////

pipeline::pipeline(int privdata_size): PipelineBase(privdata_size) {
	//this->condMutex = CreateMutex(NULL, false, "cond mutex");
	char t[100] = {0};
	sprintf(t, "pipe-%d", GetCurrentThreadId());
	infoRecorder->logError("[pipeline]: constructor called, name:%s.\n", t);
	do_register(t, this);
#if 0
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
#endif
}

pipeline::~pipeline() {
	bufpool = datapool_free(bufpool);
	datahead = datapool_free(datahead);

#if 0
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
#endif
}

void pipeline::Release(){
	if (pipelinemutex){
		CloseHandle(pipelinemutex);
		pipelinemutex = NULL;
	}
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
#else

pipeline::pipeline(int privdata_size){
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

	infoRecorder->logError("[pipeline]: pool mutex:%p.\n", poolmutex);
	//DebugBreak();
	int i = 0; 
	i++;
	bufpool = NULL;
	datahead = datatail = NULL;
	privdata = NULL;
	privdata_size = 0;
	if(privdata_size > 0){
		alloc_privdata(privdata_size);
	}


	char t[100] = {0};
	sprintf(t, "pipe-%d", GetCurrentThreadId());
	infoRecorder->logError("[pipeline]: constructor called, name:%s.\n", t);
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
				Log::slog("data pool: FATAL - unexpected NULL data returned (pipe '%s', data=%d, free=%d).\n",
					this->name(), datacount, bufcount);
				//exit(-1);
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
	Log::log("[pipeline]: pipe '%s' store data.\n", name());
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
	Log::slog("[pipeline]: stored data, data count:%d\n", datacount);
	ReleaseMutex(poolmutex);
	return;
}


struct pooldata *
	pipeline::load_data_unlocked() {
		// load a data from data (work) pool
		Log::slog("[pipeline]: to load data, data count:%d\n", datacount);
	
		if(datacount == 0){
			Log::slog("[pipeline]: error.\n");
		}
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
	pipeline::load_data() {
		Log::log("[pipeline]: pipe '%s' load data.\n", name());
		struct pooldata *data;

		WaitForSingleObject(poolmutex, INFINITE);
		data = load_data_unlocked();
		ReleaseMutex(poolmutex);
		return data;
	}

void
pipeline::release_data(struct pooldata *data) {
	Log::log("[pipeline]: pipe '%s' release data.\n", name());
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
pipeline::client_unregister(long tid) {
	EnterCriticalSection(&condMutex);
	//WaitForSingleObject(condMutex, INFINITE);
	condMap.erase(tid);
	//ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return;
}

int
pipeline::wait(HANDLE cond, HANDLE mutex) {
	DWORD ret;
	infoRecorder->logError("[pipeline]: pipe %s is waiting for event:0x%p\n", this->name(), cond);
	WaitForSingleObject(mutex, INFINITE);

	ret = WaitForSingleObject(cond, INFINITE);
	if (ret == WAIT_OBJECT_0){
		infoRecorder->logError("[pipeline]: wait single is signed.\n");
		Log::log("[pipeline]: pipe %s wait single is signed.\n", name());
	}
	else if (WAIT_FAILED == ret){
		infoRecorder->logError("[pipeline]: wait failed. last error:%d.\n", GetLastError());
		Log::log("[pipeline]: pipe %s wait failed.\n", name());
	}
	//ret = pthread_cond_wait(cond, mutex);

	ReleaseMutex(mutex);
	infoRecorder->logError("[pipeline]: pipe %s exit waiting..\n", name());
	return ret;
}

int
pipeline::timedwait(HANDLE cond, HANDLE mutex, const struct timespec *abstime) {
	int ret = 0;
	Log::log("[pipeline]: pipe %s is timed waiting, time :%d ms... %ds:%dns\n", this->name(), abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000, abstime->tv_sec,abstime->tv_nsec);
	Log::logscreen("[pipeline]: pipe %s is timed waiting, time :%d ms...\n", this->name(), abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000);
	WaitForSingleObject(mutex, INFINITE);
	ret = WaitForSingleObject(cond, abstime->tv_nsec / 1000000 + abstime->tv_sec * 1000);
	if (ret == WAIT_OBJECT_0){
		Log::log("[pipeline]: pipe %s timed wait single is signed.\n", name());
	}
	else if (ret == WAIT_TIMEOUT){
		Log::log("[pipeline]: pipe %s timed wait timed out.\n", name());
	}
	else if (ret == WAIT_FAILED){
		Log::log("[pipeline]: pipe %s timed wait failed.\n", name());
	}

	//ret = pthread_cond_timedwait(cond, mutex, abstime);

	ReleaseMutex(mutex);
	Log::log("[pipeline]: pipe %s exit timed waiting..\n", name());
	return ret;
}

void
pipeline::notify_all() {
	map<long, HANDLE>::iterator mi;
	//WaitForSingleObject(condMutex, INFINITE);
	EnterCriticalSection(&condMutex);
#if 0
	for (mi = condmap.begin(); mi != condmap.end(); mi++) {
		SetEvent(mi->second);

	}
#else
	infoRecorder->logError("[pipeline]: notifier event:0x%p.\n", waitEvent);
	Log::log("[pipeline]: notify event:%p.\n", waitEvent);
	SetEvent(waitEvent);
#endif
	//ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return;
}

void
pipeline::notify_one(long tid) {
	map<long, HANDLE>::iterator mi;
	Log::log("[pipeline]: pipe %s is notifying thread %d...\n", this->name(), tid);
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
pipeline::client_count() {
	int n;
	//WaitForSingleObject(condMutex, INFINITE);
	EnterCriticalSection(&condMutex);
	n = (int)condMap.size();
	// ReleaseMutex(condMutex);
	LeaveCriticalSection(&condMutex);
	return n;
}
// not allocate the data area
struct pooldata * pipeline::datapool_init(int n){
	int i;
	struct pooldata *data;
	//
	if (n <= 0 )
		return NULL;
	//
	bufpool = NULL;


	// the data is IDirect3DSurface9 *
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
	infoRecorder->logError("[pipeline]: register called, provider:%s.\n", provider);
	if (pipelinemap.find(provider) != pipelinemap.end()){
		// already registered
		ReleaseMutex(pipelinemutex);
		Log::slog("pipeline: duplicated pipeline '%s'\n", provider);
		return -1;
	}
	pipelinemap[provider] = pipe;
	ReleaseMutex(pipelinemutex);
	pipe->myname = provider;
	Log::log("pipeline: new pipeline '%s' registered.\n", provider);
	return 0;
}

void
	pipeline::do_unregister(const char *provider) {
		DWORD ret = WaitForSingleObject(pipelinemutex, INFINITE);  // lock
		pipelinemap.erase(provider);
		ReleaseMutex(pipelinemutex);
		Log::log("pipeline: pipeline '%s' unregistered.\n", provider);
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

#endif