#ifndef __PIPELINEBASE_H__
#define __PIPELINEBASE_H__
// this is for the base of pipeline.
#include <map>
#include <string>
#include <Windows.h>

struct pooldata{
	void * ptr;
	struct pooldata * next;
};

class PipelineBase{
public:
	std::string myname;
	CRITICAL_SECTION condMutex;

	HANDLE waitMutex;
	HANDLE waitEvent;
	DWORD clientThreadId;

	int bufcount, datacount;  

	std::map<long, HANDLE> condMap;

	HANDLE poolmutex;
	struct pooldata * bufpool; // unused free pool
	struct pooldata * datahead, * datatail; // occupied data

	struct pooldata * load_data_unlock();   // load one data from work pool w/o lock
	const char * name();

	PipelineBase(int privdata_size);
	virtual ~PipelineBase();

	// buffer pool
	virtual struct pooldata * datapool_init(int n, int datasize) = 0;
	virtual struct pooldata * datapool_free(struct pooldata * head) = 0;

	void store_data(struct pooldata * data); // store one data into work pool
	struct pooldata * load_data_unlocked();
	struct pooldata * load_data();   // load one data from work pool

	virtual struct pooldata * allocate_data(); // allocate one free data from pool
	void release_data(struct pooldata * data); // release one data into free pool
	int data_count();
	int buf_count();

	// private data

	void * privdata;
	int privdata_size;

	void * alloc_privdata(int size);
	void * set_privdata(void * ptr, int size);
	void * get_privdata();
	int get_privdata_size();

	/// work withe cients
	void client_register(long tid, HANDLE cond);
	void client_unregister(long tid);
	int wait(HANDLE cond, HANDLE mutex);
	int timedwait(HANDLE cond, HANDLE mutex, const struct timespec *abstime);

	void notify_all();
	void notify_one(long tid);
	int client_count();
};

#endif