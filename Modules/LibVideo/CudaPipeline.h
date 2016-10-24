#ifndef __CUDA_PIPELINE_H__
#define __CUDA_PIPELINE_H__
// this is for the pipeline inside gpu using cuda
#include <cuda.h>
#include "types.h"

#include "PipelineBase.h"
#include <d3d9.h>


using namespace std;

class CudaPipeline: public PipelineBase{

	static HANDLE pipelineMutex;
	static map<string, CudaPipeline *> cudaPipeMap;

	// private data
	void * privdata;
	int privdata_size;
public:
	// static functions
	static int do_register(const char * provider, CudaPipeline * pipe);
	static void do_unregister(const char * provider);
	static CudaPipeline * lookup(const char * provider);

	CudaPipeline(int privdata_size = 0);
	~CudaPipeline();

	static void Release();

	// from parent class
	//virtual struct pooldata * allocate_data();
	//virtual void release_data(struct pooldata * data);
	virtual struct pooldata * datapool_init(int n, int datasize);
	struct pooldata * datapool_init(int n, IDirect3DDevice9 * device);
	virtual struct pooldata * datapool_free(struct pooldata * head);
};

#endif