
#include "Commonwin32.h"
#include "CudaPipeline.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"


#include <d3d9.h>


// the function to create surface

IDirect3DSurface9 * CreateSurface(int width, int height, IDirect3DDevice9 * device){
	IDirect3DSurface9 * surface = NULL;
	HRESULT hr = device->CreateOffscreenPlainSurface(width, height, D3DFORMAT::D3DFMT_A8R8G8B8, D3DPOOL::D3DPOOL_MANAGED, &surface, NULL);
	if(FAILED(hr)){
		infoRecorder->logTrace("[CreateSurface]: create surface to store framebuffer data failed.\n");
		return NULL;
	}
	return surface;
}

// this is for the cuda pipeline, stores the IDirect3DSurface9 to the pipeline and notify the encoder.

HANDLE CudaPipeline::pipelineMutex;
map<string, CudaPipeline *> CudaPipeline::cudaPipeMap;


CudaPipeline::CudaPipeline(int privdata_size): PipelineBase(privdata_size){
	
}

CudaPipeline::~CudaPipeline(){

}

int CudaPipeline::do_register(const char * provider, CudaPipeline *pipe){
	DWORD ret = WaitForSingleObject(pipelineMutex, INFINITE);  // lock

	if (cudaPipeMap.find(provider) != cudaPipeMap.end()){
		// already registered
		ReleaseMutex(pipelineMutex);
		infoRecorder->logError("pipeline: duplicated pipeline '%s'\n", provider);
		return -1;
	}
	cudaPipeMap[provider] = pipe;
	ReleaseMutex(pipelineMutex);
	pipe->myname = provider;
	infoRecorder->logTrace("pipeline: new pipeline '%s' registered.\n", provider);
	return 0;
}

void CudaPipeline::do_unregister(const char * provider){
	DWORD ret = WaitForSingleObject(pipelineMutex, INFINITE);  // lock
		cudaPipeMap.erase(provider);
		ReleaseMutex(pipelineMutex);
		infoRecorder->logTrace("pipeline: pipeline '%s' unregistered.\n", provider);
		return;
}

CudaPipeline * CudaPipeline::lookup(const char * provider){
	map<string, CudaPipeline*>::iterator mi;
		CudaPipeline *pipe = NULL;
		DWORD ret = WaitForSingleObject(pipelineMutex, INFINITE);
		if ((mi = cudaPipeMap.find(provider)) == cudaPipeMap.end()) {
			ReleaseMutex(pipelineMutex);
			return NULL;
		}
		pipe = mi->second;
		ReleaseMutex(pipelineMutex);
		return pipe;
}

void CudaPipeline::Release(){
	if (pipelineMutex){
		CloseHandle(pipelineMutex);
		pipelineMutex = NULL;
	}

}
#if 0
struct pooldata * CudaPipeline::datapool_init(int n, IDirect3DDevice9 *device)
{
	int i;
	struct pooldata *data;
	//
	if (n <= 0 || datasize <= 0)
		return NULL;
	//
	bufpool = NULL;
	for (i = 0; i < n; i++) {
		// create some surface
		if ((data = (struct pooldata*) malloc(sizeof(struct pooldata))) == NULL){
			bufpool = datapool_free(bufpool);
			return NULL;
		}
		bzero(data, sizeof(struct pooldata));
		// create the surface
		//data->ptr = ((unsigned char*)data) + sizeof(struct pooldata);
		data->ptr = (void *)(CreateSurface(800, 600, device));
		data->next = bufpool;
		bufpool = data;
	}
	datacount = 0;
	bufcount = n;
	return bufpool;	
}

#endif

// create the data pool, data is surface...
struct pooldata * CudaPipeline::datapool_init(int n, int datasize){
	int i;
	struct pooldata *data;
	//
	if (n <= 0 || datasize <= 0)
		return NULL;
	//
	bufpool = NULL;
	for (i = 0; i < n; i++) {
		// create some surface
		if ((data = (struct pooldata*) malloc(sizeof(struct pooldata) + datasize)) == NULL){
			bufpool = datapool_free(bufpool);
			return NULL;
		}
		bzero(data, sizeof(struct pooldata) + datasize);
		data->ptr = ((unsigned char*)data) + sizeof(struct pooldata);

		// create the surface
		//SurfaceFrame * sframe = (SurfaceFrame *)data->ptr;
		
		//data->ptr = (void *)(CreateSurface(800, 600, NULL));
		data->next = bufpool;
		bufpool = data;
	}
	datacount = 0;
	bufcount = n;
	return bufpool;

}

// free the data pool data, the data is surface.
struct pooldata * CudaPipeline::datapool_free(struct pooldata * head){
	struct pooldata *next;
	//
	if (head == NULL)
		return NULL;
	//
	do {
		next = head->next;
		// free surface
		free(head);
		head = next;
	} while (head != NULL);
	//
	bufpool = datahead = datatail = NULL;
	datacount = bufcount = 0;
	//
	return NULL;
}

