// pipeline for frame manager, 
#ifndef __PIPELINE_H__
#define __PIPELINE_H__
#if 0
#include "PipelineBase.h"

using namespace std;
class pipeline : public PipelineBase{
private:
	
public:
	// static functions
	static int do_register(const char *provider, pipeline *pipe);
	static void do_unregister(const char *provider);
	static pipeline * lookup(const char *provider);
	// constructor & deconstructor
	pipeline(int privdata_size = 0);
	~pipeline();

	static void Release();
	// buffer pool
	//virtual struct pooldata * allocate_data();	  // allocate one free data from free pool
	//virtual void release_data(struct pooldata *data); // release one data into free pool
	virtual struct pooldata * datapool_init(int n, int datasize);
	virtual struct pooldata * datapool_free(struct pooldata * head);
};

#else
#include <map>
#include <string>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

using namespace std;

struct pooldata{
	void * ptr;
	struct pooldata * next;
};


class pipeline{
private:
	string myname;
	CRITICAL_SECTION condMutex;

	HANDLE waitEvent;
	DWORD clientThreadId;

	map<long, HANDLE> condMap;

	HANDLE poolmutex;
	struct pooldata * bufpool; // unused free pool
	struct pooldata * datahead, * datatail; // occupied data

	int bufcount, datacount;
	struct pooldata * load_data_unlock();   // load one data from work pool w/o lock
	void * privdata;
	int privdata_size;
public:
	const char * name();


	pipeline(int privdata_size);
	virtual ~pipeline();
	
	// buffer pool
	virtual struct pooldata * datapool_init(int n, int datasize);
	virtual struct pooldata * datapool_init(int n);
	virtual struct pooldata * datapool_free(struct pooldata * head);

	void store_data(struct pooldata * data); // store one data into work pool
	struct pooldata * load_data_unlocked();
	struct pooldata * load_data();   // load one data from work pool

	virtual struct pooldata * allocate_data(); // allocate one free data from pool
	void release_data(struct pooldata * data); // release one data into free pool
	int data_count();
	int buf_count();

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


	// static functions
	static int do_register(const char *provider, pipeline *pipe);
	static void do_unregister(const char *provider);
	static pipeline * lookup(const char *provider);
	static void Release();
};

#endif
#endif