#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "pipeline.h"
#include "encoder.h"
#include "CudaEncoder.h"
#include "../LibCore/TimeTool.h"
#include <Windows.h>

// this file is for encoder and encoder manager
/// encoder
namespace cg{
	Encoder::Encoder(int _id, int height, int width,/*SOURCE_TYPE srcType, */ENCODER_TYPE type,pipeline * _imgPipe, VideoWriter *_writer) 
		:/*useSourceType(srcType),*/ assigned(false), encoderWidth(width), encoderHeight(height), cond(NULL), condMutex(NULL), useEncoderType(type), writer(_writer), sourcePipe(_imgPipe), id(_id), inited(false), pTimer(NULL), encodeTime(0), packTime(0), refIntraMigrationTimer(NULL){
			// create the cond and mutex
			cond = CreateEvent(NULL, FALSE, FALSE, NULL);
			condMutex = CreateMutex(NULL, FALSE, NULL);
			cg::core::infoRecorder->logTrace("[Encoder]: encoder create event:%p, Mutex:%p.\n", cond, condMutex);
			pTimer = new cg::core::PTimer();
	}	

	Encoder::~Encoder(){
		if(cond){
			CloseHandle(cond);
			cond = NULL;
		}
		if(condMutex){
			CloseHandle(condMutex);
			condMutex = NULL;
		}
		if(pTimer){
			delete pTimer;
			pTimer = NULL;
		}

	}

	void Encoder::releaseData(pooldata * data){
		pipeline * pipe = NULL;
		pipe = sourcePipe; 
		if(pipe){
			pipe->release_data(data);
		}
	}

	// load frame from pipeline
	pooldata * Encoder::loadFrame(){
		struct pooldata * data = NULL;
		pipeline * pipe = NULL;

		pipe = sourcePipe;

		data = pipe->load_data();
		if(data == NULL){
			pipe->wait(cond, condMutex);
			if((data = pipe->load_data()) == NULL){
				// failed
#if 0
				cg::core::infoRecorder->logError("[Encoder]: recv unexpected NULL frame.\n");
#endif
				return NULL;
			}
		}
		cg::core::infoRecorder->logTrace("[Encoder]: load frame: %p.\n", data);
		return data;
	}

	void Encoder::registerEvent(){
		if(sourcePipe){
			cg::core::infoRecorder->logTrace("[Encoder]: register event %p to source pipe:%s.\n",cond, sourcePipe->name());
			sourcePipe->client_register(ccg_gettid(), cond);
		}
		else{
			cg::core::infoRecorder->logError("[Encoder]: register event failed, source , pipe:%p.\n",sourcePipe);
		}
	}


	void Encoder::unregisterEvent(){

		sourcePipe->client_unregister(ccg_gettid());
		return;
	}

	void Encoder::onQuit(){
		inited = false;
		sourcePipe = NULL;
		assigned = false;
		encoderWidth = 0, encoderHeight = 0;

		if(cond){
			CloseHandle(cond);
			cond = NULL;
		}
		if(condMutex){
			CloseHandle(condMutex);
			condMutex = NULL;
		}
		writer = NULL;
	}
}