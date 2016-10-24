#include "pipeline.h"
#include "encoder.h"
#include "log.h"
#include "CudaEncoder.h"
#include "TimeTool.h"
#include <Windows.h>


// these are the inline versions for all the SDK helper functions
#if 0
inline void __checkCudaErrors(CUresult err, const char *file, const int line){
	if (CUDA_SUCCESS != err){
		fprintf(stderr, "checkCudaErrors() Driver API error =  %04d \"%s\" from file <%s>, line %i.\n", err, getCudaDrvErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#endif

// this file is for encoder and encoder manager
/// encoder
HANDLE Encoder::syncMutex;
bool Encoder::syncReset = false;
struct timeval Encoder::syncTv;

int Encoder::CheckDevice(){
	return TRUE;
}
Encoder::Encoder(ENCODER_TYPE type){
	useType = type;
	bitrate = 3000;
	buffer = NULL;
	pBufferInsideGpu = NULL;
	//rtspConf = NULL;
}

int Encoder::encoderPtsSync(int sampleRate){
	struct timeval timev;
	long long us = 0;
	int ret;

	WaitForSingleObject(this->syncMutex, INFINITE);
	if (this->syncReset){
		getTimeOfDay(&syncTv, NULL);
		syncReset = false;
		ReleaseMutex(this->syncMutex);
		return 0;
	}
	getTimeOfDay(&timev, NULL);
	//us = tvdiff_us(&timev, &syncTv);
	ReleaseMutex(this->syncMutex);
	ret = (int)(0.000001 * us * sampleRate);
	return ret > 0 ? ret : 0;
}
