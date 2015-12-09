/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NV_VIDEO_ENCODER
#define NV_VIDEO_ENCODER

// This preprocessor definition is required for the use case where we want to use
// GPU device memory input with the CUDA H.264 Encoder.  This define is needed because
// 64-bit CUdevicePtrs are not supported in R260 drivers with NVCUVENC.  With driver versions
// after R265, ?UdevicePtrs are supported with NVCUVENC.  CUDA kernels that want to interop
// with the CUDA H.264 Encoder, this define must also be present for drivers <= R260.
//#define CUDA_FORCE_API_VERSION 3010

#include <stdio.h>
#include <stdlib.h>

#include "encoder.h"

//#include "unknwn.h"
#include <math.h> // log10
#include <nvcuvid.h> // this is for the Cuvideoctxlock
//#include "nvEncoderAPI.h"
#include <NVEncoderAPI.h>
#include <NVEncodeDataTypes.h>
#include <nvcuvid.h>
#include "types.h"
#include <map>

//#include "stopwatch_functions.h"
#include <cuda.h>
#include <cuda_d3d9_interop.h>
#include <cuda_d3d10_interop.h>
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


#include "helper_timer.h"
#include <device_launch_parameters.h>

#include "CudaFilter.h"
#include "inforecoder.h"

#include "GpuMat.hpp"

//#include "stopwatch_functions.h"

#ifndef UINT64_C
#define UINT64_C(val) val##ui64
#endif




//#define FILE_INPUT


using namespace std;

typedef unsigned char  uchar;

extern InfoRecorder * infoRecorder;

void computeFPS(); //compute the fps

enum INPUT_TYPE{
	INPUT_NONE,
	CPUIMAGE,
	GPUIMAGE,
};

enum CUSurfaceFormat{
	SF_UYVY = 0,
	SF_YUY2,
	SF_YV12,
	SF_NV12,
	SF_IYUV,
	SF_BGR,
	SF_GRAY = SF_BGR
};



class CudaEncoder: public Encoder{
private:
	char *				outputName;
	INPUT_TYPE			inputType; // represent what kind of input image is,
	//CPUIMAGE mean the image comes from system memory, and GPUIMAGE means image comes from GPU
	int outputWidth, outputHeight;
	CudaFilter *		cuFilter;
#if 0
	void *				device;
#else
	union DXDevice *	dxDevice;
#endif
	DX_VERSION			dxVersion;
	DXSurface *			sourceSurface;

	//unsigned int Pitch, Height, widthInBytes, elementSizeBytes;
	StopWatchInterface * timer;   // the frame timer for the Encoder

	//////
	NVEncoder			m_pEncoder;
	void             *	m_pSNRData;
	unsigned char    *	m_pVideoFrame;
	unsigned char    *	m_pCharBuf;   // the buffer to store the encoded frame data
	unsigned int		m_nVideoFrameSize;


	// GPU memory
	GpuMat videoFrame_;
	Size frameSize_;

	bool				m_bLastFrame, m_bEncodeDone;
	long				m_lFrameSummation; // this is summation of the frame all frames
	long				m_nFrameCount, m_nLastFrameNumber;

	NVEncoderParams		sEncoderParams,*m_pEncoderParams;
	NVVE_CallbackParams m_NVCB;

	// output file is to record the video
	FILE *				fpOut, *fpConfig;

	LARGE_INTEGER		m_liUserTime0, m_liKernelTime0;
	DWORD				m_dwStartTime;

	double				m_MSE[3];
	// check filter
	bool				checkCuFilter();
	bool				baiscSetup();    // the basic setup when create instance
	pipeline *			cudaPipe;   // the source for cuda encoder
public:
	CudaEncoder();   // the default use of CudaEncoder
	CudaEncoder(int ih, int iw, int oh, int ow, char * outputFileName = NULL); // inith defualt cudaencoder with given intput and output size
	CudaEncoder(NVEncoderParams *pParams, bool bUseDeviceMem = false);
	CudaEncoder(NVEncoderParams *pParams, bool bUseDeviceMem, bool readFromFile);
	~CudaEncoder();

	// General high level initialization and parameter settings
	bool				InitEncoder();
	bool				InitCudaFilter(DX_VERSION ver, void * device);   // create and init the cuda filter
	bool				SetEncodeParameters();

	bool				SetCBFunctions(NVVE_CallbackParams *pCallback, void *pUserData = NULL);
	bool				CreateHWEncoder(NVEncoderParams *pParams);    // nothing to do with pParams
	bool				initGpuMemory();

	// Help functions for NVCUVENC
	bool				GetEncodeParamsConfig(NVEncoderParams *pParams);
	bool				SetCodecType(NVEncoderParams *pParams);
	bool				SetParameters(NVEncoderParams *pParams);
	int					DisplayGPUCaps(int deviceOrdinal, NVEncoderParams *pParams, bool bDisplay);
	int					GetGPUCount(NVEncoderParams *pParams, int *gpuPerf, int *bestGPU);
	void				SetActiveGPU(NVEncoderParams *pParams, int gpuID);
	void				SetGPUOffloadLevel(NVEncoderParams *pParams);

	// These are for setting and getting parameter values
	HRESULT				GetParamValue(DWORD dwParamType, void *pData);
	HRESULT				SetParamValue(DWORD dwParamType, void *pData);

	// Functions to start, stop, and encode frames
	
	size_t				ReadNextFrame(CUdeviceptr dstPtr = NULL);

	void				CopyUYVYorYUY2Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);
	void				CopyYV12orIYUVFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);
	void				CopyNV12Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);
	
	bool				EncodeFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock);

public:
	inline unsigned char *GetVideoFrame(){ return m_pVideoFrame; }
	inline unsigned char *GetCharBuf(){	return m_pCharBuf; }
#ifdef FILE_INPUT
	inline FILE *		fileIn(){ return fpIn; }
#endif
	inline FILE *		fileOut(){	return fpOut; }
	inline FILE *		fileConfig(){	return fpConfig; }
	inline void			resetFrameCount(){ m_nFrameCount = 0; m_nLastFrameNumber = 0; m_bLastFrame = false; m_bEncodeDone = false; m_lFrameSummation = 0;}
	inline long			GetLastFrameNumber(){ return m_nLastFrameNumber; }
	inline long			GetBytesPerFrame(){	return m_nVideoFrameSize; }
	inline void			SetEncodeDone(){ m_bEncodeDone = true; }
	inline bool			IsLastFrameSent(){	return m_bLastFrame; }
	inline bool			IsEncodeDone(){	return m_bEncodeDone; }
	inline DWORD		frameCount(){	return m_nFrameCount;}
	inline void			incFrameCount(){m_nFrameCount++;}
	inline void			setMSE(double *mse){m_MSE[0] = mse[0];m_MSE[1] = mse[1];m_MSE[2] = mse[2];}
	// because frames have a unique #, we keep adding the frame to a running count
	// to check to ensure that every last frame is completed, we will subtract or add
	// to the running count.  If frameSummation == 0, then we have reached the last value.
	inline void			frameSummation(long frame_num){	m_lFrameSummation += frame_num;}
	inline long			getFrameSum(){	return m_lFrameSummation;}

	//// create the timer
	inline bool			createTimer(){	sdkCreateTimer(&timer);}
	inline bool			resetTimer(){	sdkResetTimer(&timer);}
	inline bool			startTimer(){sdkStartTimer(&timer);sdkResetTimer(&timer);}

	inline void			setOutputFileName(char * name){ outputName = _strdup(name); }

	// functions from parent class
	
	virtual int			startEncoding();    // start encoding
	virtual void		idle();

	virtual void		setBitrate(int bitrate);
	virtual bool		setInputBuffer(char * buf);
	virtual bool		setBufferInsideGpu(char * p);

	// thread functions
	virtual BOOL		stop();
	virtual void		run();
	virtual void		onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void		onThreadStart();
	virtual void		onQuit();

	NVEncoderParams *	getEncoderParams(){ return m_pEncoderParams; }
	inline bool			registerDXDevice(DX_VERSION ver, void * device){ this->dxVersion = ver; }//this->device = device; }

	inline void			setInputType(INPUT_TYPE type){ inputType = type; }
	inline INPUT_TYPE	getInputType(){ return inputType; }

	bool				supportInterOp(); // check the device, return true when support interop with D3D9
	inline void			registerCudaSource(pipeline * p){ cudaPipe = p; }
	inline pipeline *	getCudaSource(){ return cudaPipe; }


	inline unsigned char * getCudaVideoFrame(){ return (unsigned char *)dptrVideoFrame; }
	inline int			getCudaVideoFrameSize(){ return m_nVideoFrameSize; }

	// send data functions
	//int sendPacket(int channelId, RTSPContext * rtsp, void * pkt, int64_t encoderPts);
	//inline RTSPContext * getRTSPContext(){ return rtsp; }
private:
	CUcontext			cuContext;
	CUdevice			cuDevice;
	CUvideoctxlock		cuCtxLock;
	CUdeviceptr			dptrVideoFrame;
	unsigned int		Pitch, Height, WidthInBytes, ElementSizeBytes;

	// for rtsp sending 
	//RTSPContext * rtsp;
	struct pooldata *			data;
	HANDLE				condMutex, cond;
	long long			basePts, newPts, pts, ptsSync;
	int rtp_id;


	// the callback
	unsigned char * acquireBitStream(int * pBuffeSize);
	void releaseBitStream(unsigned char * cb, int nBytesInBuffer);
	void onBeginFrame(const NVVE_BeginFrameInfo * pbfi);
	void onEndFrame(const NVVE_EndFrameInfo * pefi);

};

//////////////////////////////new cuda encoder from opencv /////////////////////
class NVEncoderWrapper{
public:
	NVEncoderWrapper() : encoder_(0){
		int err;
		err = NVGetHWEncodeCaps();
		if(err){
			infoRecorder->logError("[NVEncoderWrapper]: GpuNotSupported, no cuda capability present.\n");
		}
		err = NVCreateEncoder(&encoder_);

	}
	~NVEncoderWrapper(){
		if(encoder_){
			NVDestroyEncoder(encoder_);
			//encoder_ = NULL;
		}

	}
	operator NVEncoder() const{
		return encoder_;
	}
private:
	NVEncoder encoder_;
};

class EncoderCallback{
public:
	enum PicType{
		IFRAME = 1,
		PFRAME = 2,
		BFRAME = 3
	};

	virtual ~EncoderCallback(){}
	// ! callback funtion to signal the start of bitstream that is to be encoded
	// ! callback must allocate host buffer for CUDA encoder and return pointer to it and it's size
	virtual uchar * acquireBitStream(int * bufferSize) = 0;
	// ! callback function to signal that the encoded bistream is ready to be written to file
	virtual void releaseBitStream(unsigned char * data, int size) = 0;

	// ! callback funtion to signal that the encoding operation on the frame has started
	virtual void onBeginFrame(int frameNumber, PicType picType) = 0;
	// ! callback funtion to signals that the encoding operation on the frame has finished
	virtual void onEndFrame(int frameNumber, PicType picType) = 0;

};

// the class responsible for deal the get encoded frame and 
class EncoderCallbackNet: public EncoderCallback{
public:
	EncoderCallbackNet();	
	~EncoderCallbackNet();

	unsigned char * acquireBitStream(int * bufferSize);
	void releaseBitStream(unsigned char * data, int size);
	void onBeginFrame(int frameNumber, PicType picType);
	void onEndFrame(int frameNumber, PicType picType);

private:
	bool isKeyFrame_;
	uchar * buf_;     // store the output data
	int bufSize;
};

enum CodecType{
	MPEG1, // not supported now
	MPEG2, // not supported now
	MPEG4, // not supported now
	H264
};

// the new encoder for cuda h264
class CudaEncoderImpl{
public:
	CudaEncoderImpl();
	~CudaEncoderImpl();

	void write(GpuMat frame, bool lastFrame = false);

	NVEncoderParams getEncoderParams() const;

private:
	void initEncoder(double fps);
	void setEncodeParams(const NVEncoderParams & params);
	void initGpuMemory();
	void initCallbacks();
	void createHWEncoder();

	Size frameSize_;
	CodecType codec_;
	CUSurfaceFormat inputFormat_;
	NVVE_SurfaceFormat surfaceFormat_;

	NVEncoderWrapper encoder_;
	GpuMat videoFrame_;
	CUvideoctxlock cuCtxLock_;

	EncoderCallback * callback_;

	// callbacks
	static unsigned char * NVENCAPI HANDLEAcquireBitStream(int * pBufferSize, void * pUserData);
	static void NVENCAPI HandleReleaseBitStream(int nBytesInBuffer, unsigned char * cb, void * pUserData);
	static void NVENCAPI HandleOnBeginFrame(const NVVE_BeginFrameInfo * pbfi, void * pUserData);
	static void NVENCAPI HandleOnEndFrame(const NVVE_EndFrameInfo * pefi, void * pUserData);
	//static 

};

struct ParamStruct{
	CudaEncoder *		encoder;
	void *				userdata;
};

class CudaEncoderFactory{
private:
	static int cudaEncoderCount;
	static CudaEncoderFactory * factory;
	static StopWatchInterface * globalTimer;   // the global timer for time measurement

	map<CudaEncoder *, CudaEncoder *> encoderMap;
	map<CudaEncoder *, NVEncoderParams *> encoderParamsMap;

	CUvideoctxlock cuCtxLock;
	CUcontext cuContext;
	CUdevice cuDevice;
	CUdeviceptr dptrVideoFrame;

	CudaEncoderFactory(){}
	~CudaEncoderFactory(){}
public:
	
	static CudaEncoderFactory * GetCudaEncoderFactory();
	static CudaEncoder * CreateCudaEncoder(const char * configFile, INPUT_TYPE type = GPUIMAGE);
	static bool ReleaseAllEncoder();    /// release all the encoders
	static bool CreateGlobalTimer();
	static bool InitGlobalTimer();
	static bool StartGlobalTimer();
};

#endif // NV_VIDEO_ENCODER