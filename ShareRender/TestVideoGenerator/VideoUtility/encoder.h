#ifndef __ENCODER_H__
#define __ENCODER_H__
//#include <cuda.h>
#include <cuda.h>
#include "helper_cuda_drvapi.h"

#include "videocommon.h"
#include "cthread.h"

#include <Windows.h>
#include "log.h"
#include "pipeline.h"

// this file is for the generous encoder

//int ffio_open_dyn_packet_buf(AVIOContext **, int);

// this will output the proper CUDA error string in the event that a CUDA hsot call returns an error
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

typedef DWORD (WINAPI *EncoderThreadProc)(_In_  LPVOID lpParameter);

#define	IMAGE_SOURCE_CHANNEL_MAX	4    // temp define here
#define ENCODER_POOL_SIZE 4   // encoder pool size 

#if 0
enum
{
	NV_ENC_PRESET_DEFAULT = 0,
	NV_ENC_PRESET_LOW_LATENCY_DEFAULT = 1,
	NV_ENC_PRESET_HP = 2,
	NV_ENC_PRESET_HQ = 3,
	NV_ENC_PRESET_BD = 4,
	NV_ENC_PRESET_LOW_LATENCY_HQ = 5,
	NV_ENC_PRESET_LOW_LATENCY_HP = 6
};
#endif

class Encoder: public CThread{
	friend class EncoderManger;

	/*X264Encoder * x264Encoder;
	CudaEncoder * cudaEncoder;*/
	double bitrate;
	char * buffer; // the buffer for storing the data
	char * pBufferInsideGpu; // pointer to a buffer inside GPU

    bool running;
	bool assigned;
	

	//pipeline * sourcePipe;

	static HANDLE syncMutex;
	static bool syncReset;
	static struct timeval syncTv;
public: 
	pipeline * pipe;   // image source
	int id;
	int encoderWidth, encoderHeight;
	ENCODER_TYPE useType;    // use the x264 encoder or the cuda encoder
	//EncoderConfig * encoderConfig;
	HANDLE sourceNotifier;   // the source notifier, to wait

	inline void registerSourceNotifier(HANDLE h){ sourceNotifier = h; }
	Encoder(ENCODER_TYPE type = X264_ENCODER);
	static int CheckDevice();

	//virtual void setEncoderConfig(EncoderConfig * conf);

	virtual ~Encoder(){

	}
	//virtual int init(void * arg, pipeline * pipe) = NULL;    // initilizing both the encoder
	virtual int startEncoding() = NULL;    // start encoding
	virtual void idle() = NULL;

	virtual void setBitrate(int bitrate){}
	virtual bool setInputBuffer(char * buf){ return true; }
	virtual bool setBufferInsideGpu(char * p){ return true; }


	// thread functions
	virtual BOOL stop(){
		return CThread::stop();
	}
	virtual void run() = 0;
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam) = 0;
	virtual void onThreadStart() = 0;
	virtual void onQuit() = 0;

	inline ENCODER_TYPE getEncoderType(){ return useType; }
	inline bool encoderRunning(){ return running; }
	
	int encoderPtsSync(int sampleRate);
	inline bool isAssigned(){ return this->assigned; }
	
	virtual pipeline * getImageSource(){ return pipe; }
	virtual bool registerImageSource(pipeline * _pipe){ 
		Log::logscreen("[encoder]: register source pipe:%p.\n", pipe);
		this->pipe = _pipe;
		return true; 
	}
	
};

#endif
