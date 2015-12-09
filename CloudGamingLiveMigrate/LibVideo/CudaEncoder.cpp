//this file is for video encode, as the NVIDIA example


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

// This preprocessor definition is required for the use case where we want to use
// GPU device memory input with the CUDA H.264 Encoder.  This define is needed because
// 64-bit CUdevicePtrs are not supported in R260 drivers with NVCUVENC.  With driver versions
// after R265, ?UdevicePtrs are supported with NVCUVENC.  CUDA kernels that want to interop
// with the CUDA H.264 Encoder, this define must also be present for drivers <= R260.
#define CUDA_FORCE_API_VERSION 3010    // as annotations, use the preprocessor definition for using GPU device memory input
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <cstring>
#include <cassert>

// includes, CUDA
#include <cuda.h>
#include <builtin_types.h>
#include <NVEncodeDataTypes.h>
#include "drvapi_error_string.h"
#include "helper_cuda_drvapi.h"
#include "helper_timer.h"
#include "Pipeline.h"
#include "CThread.h"
#include "Encoder.h"
#include "CudaEncoder.h"

#include "../LibCore/Log.h"
#include "../LibCore/TimeTool.h"
#include "../libCore/Glob.hpp"
//

#ifdef _DEBUG
#define PRINTF(x) printf((x))
#else
#define PRINTF(x)
#endif

#ifndef MAX
#define MAX(a,b) (a > b) ? a : b
#endif

struct ParamStruct param = { NULL };

extern "C"
	void RGB_to_YV12(const GpuMat& src, GpuMat& dst);

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions
#if 0
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
	if (CUDA_SUCCESS != err)
	{
		fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
			err, getCudaDrvErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#endif
// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x10, 8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11, 8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12, 8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13, 8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
	return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions
// end of CUDA Helper Functions

// We have a global pointer to the Video Encoder
//VideoEncoder   *g_pVideoEncoder = NULL;

// Error message handling
inline void CheckNVEncodeError(HRESULT hr_error, const char *NVfunction, const char *error_string)
{
	if (FAILED(hr_error))
	{
		printf("%s(%s) error: 0x%08x\n", NVfunction, error_string, hr_error);
	}
}
#if 0

// NVCUVENC callback function to signal the start of bitstream that is to be encoded
static unsigned char *_stdcall NVDefault_HandleAcquireBitStream(int *pBufferSize, void *pUserdata)
{
	*pBufferSize = 1024 * 1024;
	struct ParamStruct * param = (struct ParamStruct *)pUserdata;
	/// the user data is the encoder pointer
	CudaEncoder * encoder = (CudaEncoder *)param->encoder;
	return encoder->GetCharBuf();
}

//NVCUVENC callback function to signal that the encoded bitstream is ready to be written to file
static void _stdcall NVDefault_HandleReleaseBitStream(int nBytesInBuffer, unsigned char *cb, void *pUserdata)
{

	struct ParamStruct * param = (struct ParamStruct *)pUserdata;

	CudaEncoder * encoder = (CudaEncoder *)param->encoder;
#if 1
	if (encoder && encoder->fileOut())
	{
		fwrite(cb, 1, nBytesInBuffer, encoder->fileOut());
	}
#else
	// write the encoded frame to network
	// create a AVPacket or not using the rtspwritebindata
	// TODO

	if (encoder){
		// sent 
		av_init_packet(&pkt);
		pkt.data = cb;
		pkt.size = nBytesInBuffer;
		pkt.stream_index = 0;
		encoder->sendPacket(0, encoder->getRTSPContext(), &pkt, pkt.pts); // send the packet to the client

		if (pkt.side_data_elems > 0){
			int i;
			for (i = 0; i < pkt.side_data_elems; i++)
				av_free(pkt.side_data[i].data);
			av_freep(&pkt.side_data);
			pkt.side_data_elems = 0;
		}

	}

#endif

	return;
}

//NVCUVENC callback function to signal that the encoding operation on the frame has begun
static void _stdcall NVDefault_HandleOnBeginFrame(const NVVE_BeginFrameInfo *pbfi, void *pUserdata)
{
	/// set the bitrate of the encoder
	return;
}

//NVCUVENC callback function signals that the encoding operation on the frame has ended
static void _stdcall NVDefault_HandleOnEndFrame(const NVVE_EndFrameInfo *pefi, void *pUserdata)
{
	double psnr[3], mse[3] = { 0.0, 0.0, 0.0 };

	struct ParamStruct * param = (struct ParamStruct *)pUserdata;
	CudaEncoder * encoder = (CudaEncoder *)param->encoder;

	if (param->userdata)
	{
		memcpy(psnr, param->userdata, sizeof(psnr));
	}

	mse[0] += psnr[0];
	mse[1] += psnr[1];
	mse[2] += psnr[2];
	encoder->setMSE(mse);

	encoder->frameSummation(-(pefi->nFrameNumber));

	if (encoder->IsLastFrameSent())
	{
		// Check to see if the last frame has been sent
		if (encoder->getFrameSum() == 0)
		{
			printf(">> Encoder has finished encoding last frame\n<< n");
		}
	}
	else
	{
#ifdef _DEBUG
		printf("HandleOnEndFrame (%d), FrameCount (%d), FrameSummation (%d)\n",
			pefi->nFrameNumber, encoder->frameCount(),
			encoder->getFrameSum());
#endif
	}

	return;
}
#endif

StopWatchInterface *frame_timer = 0;
StopWatchInterface *global_timer = 0;

unsigned int g_FrameCount = 0;
unsigned int g_fpsCount = 0;      // FPS count for averaging
unsigned int g_fpsLimit = 16;     // FPS limit for sampling timer;

void computeFPS()
{
	sdkStopTimer(&frame_timer);
	g_FrameCount++;
	g_fpsCount++;

	if (g_fpsCount == g_fpsLimit)
	{
		float ifps = 1.f / (sdkGetAverageTimerValue(&frame_timer) / 1000.f);

		printf("[Frame: %04d, %04.1f fps, frame time: %04.2f (ms) ]\n",
			g_FrameCount, ifps, 1000.f / ifps);

		sdkResetTimer(&frame_timer);
		g_fpsCount = 0;
	}

	sdkStartTimer(&frame_timer);
}

bool ParseEncoderParams(NVEncoderParams * pParams){

	infoRecorder->logTrace("[CudaEncoder]: ParseEncoderParams called.\n");
	pParams->measure_fps = 1;
	pParams->measure_psnr = 0;
	pParams->force_device = 0;
	pParams->iForcedGPU = 0;
	pParams->iUseDeviceMem = 1;

	// by default we want to do motion estimation on the GPU
	pParams->GPUOffloadLevel = NVVE_GPU_OFFLOAD_ALL;
	pParams->iSurfaceFormat = (int)YV12;
	pParams->iPictureType = (int)FRAME_PICTURE;
	// TODO
	// add the cfg file to Params

	char path[MAX_PATH] = { 0 };
	GetCurrentDirectory(MAX_PATH, path);
	infoRecorder->logError("[CudaEncoder]: current directory is: %s.\n", path);
	infoRecorder->logTrace("[CudaEncoder]: current directory is:%s.\n", path);

	strcpy(pParams->configFile, sdkFindFilePath("h264.cfg", path));
	strcpy(pParams->outputFile, "output.264");

	return true;
}

// read the config file is done
bool CudaEncoder::baiscSetup(){
	cuFilter = NULL;
	cond = NULL;
	infoRecorder->logTrace("[CudaEncoder]: basicSetup called.\n");
	HRESULT hr = S_OK;

	m_MSE[0] = 0.0;
	m_MSE[1] = 0.0;
	m_MSE[2] = 0.0;

	infoRecorder->logTrace("[CudaEncoder]: configuration file: '%s'.\n", this->m_pEncoderParams->configFile);

	printf("Configuration  file: <%s>\n", m_pEncoderParams->configFile);
#ifdef FILE_INPUT
	printf("Source  input  file: <%s>\n", m_pEncoderParams->inputFile);
#endif

	infoRecorder->logTrace("[CudaEncoder]: encoded output file: '%s'.\n", this->m_pEncoderParams->outputFile);
	infoRecorder->logTrace("[CudaEncoder]: measurement: %s.\n", "(FPS) frame per second");

	printf("Encoded output file: <%s>\n", m_pEncoderParams->outputFile);
	printf("Measurement: %s\n\n", "(FPS) Frames Per Second");
#ifdef FILE_INPUT
	if ((fpIn = fopen(m_pEncoderParams->inputFile, "rb")) == NULL)
	{
		printf("VideoEncoder() - fopen() error! - The input file \"%s\" could not be opened for reading\n", m_pEncoderParams->inputFile);
		assert(0);
	}

#endif

	if ((fpOut = fopen(m_pEncoderParams->outputFile, "wb")) == NULL)
	{
		infoRecorder->logTrace("[CudaEncoder]: fopen() error - the oiutput file '%s' could not be created for writing.\n", this->m_pEncoderParams->outputFile);
		printf("VideoEncoder() - fopen() error - The output file \"%s\" could not be created for writing\n", m_pEncoderParams->outputFile);
		assert(0);
	}

	// Get the encoding parameters
	if (GetEncodeParamsConfig(m_pEncoderParams) != true)
	{
		infoRecorder->logTrace("[CudaEncoder]: GetEncoderParamsConfig() error.\n");
		printf("\nGetEncodeParamsConfig() error!\n");
		assert(0);
	}
	return true;
}

// create a cuda encoder with default config
CudaEncoder::CudaEncoder(){
	//NVEncoderParams  params = {0};
	ParseEncoderParams(&sEncoderParams);
	m_pEncoderParams = &sEncoderParams;

	m_pSNRData = NULL;
	m_pVideoFrame = NULL;
	m_pCharBuf = NULL;
	m_nLastFrameNumber = 0;
	m_nFrameCount = 0;
	m_lFrameSummation = 0;
	m_bLastFrame = false;
	m_bEncodeDone = false;

	baiscSetup();
	m_pEncoderParams->iUseDeviceMem = 1;
}

CudaEncoder::CudaEncoder(int ih, int iw, int oh, int ow, char * outputFileName):frameSize_(ow, oh){
	//NVEncoderParams  params = {0};
	if(outputFileName)
		infoRecorder->logTrace("[CudaEncoder]: constructor called. output file name:%s.\n", outputFileName);
	else{
		infoRecorder->logTrace("[CudaEncoder]: constructor called. output file not specificed.\n");
	}
	sourceSurface = new DXSurface();
	sourceSurface->d10Surface = NULL;

	ParseEncoderParams(&sEncoderParams);
	m_pEncoderParams = &sEncoderParams;

	m_pSNRData = NULL;
	m_pVideoFrame = NULL;
	m_pCharBuf = NULL;
	m_nLastFrameNumber = 0;
	m_nFrameCount = 0;
	m_lFrameSummation = 0;
	m_bLastFrame = false;
	m_bEncodeDone = false;

	if(outputFileName){
		outputName = _strdup(outputFileName);
		memset(sEncoderParams.outputFile, 0, 256);
		strcpy(sEncoderParams.outputFile, outputFileName);
	}

	baiscSetup();

	// setup the input size and output size
	m_pEncoderParams->iInputSize[0] = iw;
	m_pEncoderParams->iInputSize[1] = ih;
	m_pEncoderParams->iOutputSize[0] = ow;
	m_pEncoderParams->iOutputSize[1] = oh;

	outputWidth = ow;
	outputHeight = oh;

	//frameSize_ = Size(ow, oh);

	m_pEncoderParams->iUseDeviceMem = 1;
	infoRecorder->logTrace("[CudaEncoder]: constructor finished.\n");


}


////// the encoder class
CudaEncoder::CudaEncoder(NVEncoderParams *pParams, bool bUseDeviceMem) : m_pEncoderParams(pParams), m_pSNRData(NULL), m_pVideoFrame(NULL), m_pCharBuf(NULL), m_nLastFrameNumber(0), m_nFrameCount(0), m_lFrameSummation(0), m_bLastFrame(false), m_bEncodeDone(false)
{

	//Encoder(CUDA_ENCODER);
	baiscSetup();

	// If we want to force device memory as input, we can do so
	if (bUseDeviceMem)
	{
		m_pEncoderParams->iUseDeviceMem = 1;
		pParams->iUseDeviceMem = 1;
	}

}


CudaEncoder::~CudaEncoder()
{
	infoRecorder->logTrace("[CudaEncoder]: destructor called.\n");
#if 0
	if (fpIn){
		fclose(fpIn);
	}
#endif

	if (fpOut){
		fclose(fpOut);
	}

	if (fpConfig){
		fclose(fpConfig);
	}

	if (m_pEncoder){
		NVDestroyEncoder(m_pEncoder);
		m_pEncoder = NULL;
	}

	// clear the global pointer
	//g_pVideoEncoder = NULL;

	if (m_pVideoFrame){
		delete[] m_pVideoFrame;
	}

	if (m_pCharBuf){
		delete[] m_pCharBuf;
	}
}

bool CudaEncoder::InitEncoder()
{
	infoRecorder->logTrace("[CudaEncoder]: initEncoder called.\n");
	// Create the Encoder API Interface
	HRESULT hr = NVCreateEncoder((NVEncoder *)&m_pEncoder);
	infoRecorder->logTrace("[CudaEncoder]: NVCreateEncoder <%s>, hr = %08x\n", (FAILED(hr) ? "FAILED!" : "SUCCESS!"), hr);
	printf("VideoEncoder() - NVCreateEncoder <%s>, hr = %08x\n", (FAILED(hr) ? "FAILED!" : "SUCCESS!"), hr);

	if (FAILED(hr))
	{
		return false;
	}

	// Note before we can set the GPU or query the GPU we wish to encode, it is necessary to set the CODECS type
	// This must be set before we can call any GetParamValue, otherwise it will not succeed
	SetCodecType(m_pEncoderParams);

	// Query the GPUs available for encoding
	int gpuPerf = 0, bestGPU = 0;
	GetGPUCount(m_pEncoderParams, &gpuPerf, &bestGPU);

	if (m_pEncoderParams->force_device)
	{
		SetActiveGPU(m_pEncoderParams, m_pEncoderParams->iForcedGPU);
	}
	else
	{
		m_pEncoderParams->iForcedGPU = bestGPU;
		SetActiveGPU(m_pEncoderParams, m_pEncoderParams->iForcedGPU);
	}

	infoRecorder->logTrace("[CudaEncoder]: YUV foramt: %s [%s] (%d-bpp)\n", sSurfaceFormat[m_pEncoderParams->iSurfaceFormat].name,
		sSurfaceFormat[m_pEncoderParams->iSurfaceFormat].yuv_type,
		sSurfaceFormat[m_pEncoderParams->iSurfaceFormat].bpp);

	infoRecorder->logTrace("[CudaEncoder]: frame type:%s.\n",sPictureStructure[m_pEncoderParams->iPictureType]); 
	printf("    YUV Format: %s [%s] (%d-bpp)\n", sSurfaceFormat[m_pEncoderParams->iSurfaceFormat].name,
		sSurfaceFormat[m_pEncoderParams->iSurfaceFormat].yuv_type,
		sSurfaceFormat[m_pEncoderParams->iSurfaceFormat].bpp);
	printf("    Frame Type: %s\n\n", sPictureStructure[m_pEncoderParams->iPictureType]);

	infoRecorder->logTrace("[CudaEncoder]: video input %s memory\n", m_pEncoderParams->iUseDeviceMem ? "GPU Device" : "CPU System");

	printf(" >> Video Input: %s memory\n", m_pEncoderParams->iUseDeviceMem ? "GPU Device" : "CPU System");

	m_pCharBuf = new unsigned char[1024 * 1024];

	infoRecorder->logTrace("[CudaEncoder]: buffer to store encoded frame:0x%p, size:%d.\n", m_pCharBuf, 1024 * 1024);

	//m_pEncoderParams->GPU_count = m_pEncoderParams->GPU_count;
	return true;
}

// init the gpu memory to store the surface image
bool CudaEncoder::initGpuMemory(){
	int err;
	infoRecorder->logTrace("[CudaEncoder]: initGpuMemory, size.height:%d, size.width:%d.\n", frameSize_.height, frameSize_.width);
	// init context
	GpuMat temp(1, 1, CV_8U);

	temp.release();

	static const int bpp[] = 
	{
		16, /// UYVY, 4:2:2
		16, /// YUY2, 4:2:2
		12, /// YV12, 4:2:0
		12, /// NV12, 4:2:0
		12, /// IYUV, 4:2:0
	};

	cuContext = NULL;
	//CUdevice cuDevice;
	cuInit(0);
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&cuContext, CU_CTX_BLOCKING_SYNC, cuDevice);

	if(m_pEncoderParams->iSurfaceFormat == UYVY || m_pEncoderParams->iSurfaceFormat == YUY2){
		videoFrame_.create(frameSize_.height, (frameSize_.width * bpp[m_pEncoderParams->iSurfaceFormat ])/ 8, CV_8UC1);
	}
	else{
		videoFrame_.create((frameSize_.height * bpp[m_pEncoderParams->iSurfaceFormat ]) / 8, frameSize_.width, CV_8UC1);
	}
#if 0
	cuvidCtxLockCreate(&cuCtxLock, cuContext);

	// if we are suing GPU device memory with NVCUVENC, it si necessary to create a
	// CUDA cojntext with a Context Lock cuvidCtxLock. The Context Lock needs to be passed to NVCUVENC

	int iUseDeviceMem = 1;
	err = NVSetParamValue(encoder_, NVVE_DEVICE_MEMORY_INPUT, &iUseDeviceMem);
	assert(err == 0);

	err = NVSetParamValue(encoder_, NVVE_DEVICE_CTX_LOCK, &cuCtxLock);
	assert( err == 0);
#endif

	return true;
}

// before this, update the input and output size
bool CudaEncoder::SetEncodeParameters()
{

	infoRecorder->logTrace("[CudaEncoder]: setEncoderParameters called.\n");
	NVEncoderParams * pParams = &sEncoderParams;
	//  Allocate a little bit more memory, in case we have higher formats to handle
	// the buffer is used when using system memory

	m_nVideoFrameSize = ((pParams->iInputSize[0] * pParams->iInputSize[1] *
		sSurfaceFormat[pParams->iSurfaceFormat].bpp) / 8) * sizeof(char);
	m_pVideoFrame = new unsigned char[m_nVideoFrameSize];

	// Set the GPU Offload Level
	SetGPUOffloadLevel(m_pEncoderParams);

	// Now Set the Encoding Parameters
	bool bRetVal = SetParameters(m_pEncoderParams);

	infoRecorder->logTrace("[CudaEncoder]: setEncoderParameters <%s>, bRetVal = %08x\n", ((bRetVal != TRUE) ? "FAILED!" : "SUCCESS!"), bRetVal);

	printf("VideoEncoder() - SetEncodeParameters <%s>, bRetVal = %08x\n", ((bRetVal != TRUE) ? "FAILED!" : "SUCCESS!"), bRetVal);

	return bRetVal;
}

bool CudaEncoder::SetCBFunctions(NVVE_CallbackParams *pCB, void *pUserData)
{
	infoRecorder->logTrace("[CudaEncoder]: setCBFunctions called. pCB:0x%p, userdata:0x%p.\n", pCB, pUserData);
	if (pCB)
	{
		// Copy NVIDIA callback functions
		m_NVCB = *pCB;
		//Register the callback structure functions
		NVRegisterCB(m_pEncoder, m_NVCB, pUserData); //register the callback structure
	}
	else
	{
#if 0
		// We use the callback functions defined in this class
		memset(&m_NVCB, 0, sizeof(NVVE_CallbackParams));
		m_NVCB.pfnacquirebitstream = NVDefault_HandleAcquireBitStream;
		m_NVCB.pfnonbeginframe = NVDefault_HandleOnBeginFrame;
		m_NVCB.pfnonendframe = NVDefault_HandleOnEndFrame;
		m_NVCB.pfnreleasebitstream = NVDefault_HandleReleaseBitStream;

		//Register the callback structure functions
		// get a parameter structure
		param.encoder = this;
		param.userdata = this->m_pSNRData;
		NVRegisterCB(m_pEncoder, m_NVCB, &param); //register the callback structure
#else
		infoRecorder->logTrace("[CudaEncoder]: NULL NVVE_CallbackParams.\n");
		return false;
#endif
	}

	printf("VideoEncoder() - SetCBFunctions <SUCCESS>\n");
	return true;
}
// the param is useless
bool CudaEncoder::CreateHWEncoder(NVEncoderParams *pParams)
{
	// Create the NVIDIA HW resources for Encoding on NVIDIA hardware
	HRESULT hr = NVCreateHWEncoder(m_pEncoder);
	printf("VideoEncoder() - NVCreateHWEncoder <%s>, hr = %08x\n", (FAILED(hr) ? "FAILED!  Unable to create NVIDIA HW Video Encoder" : "OK!"), hr);

	infoRecorder->logTrace("[CudaEncoder]: CreateHWEncoder called. <%s>, hr = %08x.\n", (FAILED(hr) ? "FALED! Unable to create NVIDIA HW Video Encoder": "OK!"), hr);

	if (FAILED(hr))
	{
		return false;
	}

	unsigned char buf2[10];
	int size;
	hr = NVGetSPSPPS(m_pEncoder, buf2, 10, &size);

	if (FAILED(hr))
	{
		printf("\nNVGetSPSPPS() error getting SPSPPS buffer \n");
		infoRecorder->logTrace("[CudaEncoder]: NVGetSPSPPS error getting SPSPPS buffer.\n");
	}
	else
	{
		printf("VideoEncoder() - NVGetSPSPPS <%s>, hr = %08x\n", (FAILED(hr) ? "FAILED!" : "OK!"), hr);
		infoRecorder->logTrace("[CudaEncoder]: NVGetSPSPPS <%s>, hr = %08x.\n", (FAILED(hr) ? "FAILED!" : "OK!"), hr);
	}

	return (FAILED(hr) ? false : true);
}

// the pipeline stores the duplicated data, so no need to copy

size_t CudaEncoder::ReadNextFrame(CUdeviceptr dstPtr){
	// get the surface from the pipeline
	struct timeval tv;
	struct timespec to;

	infoRecorder->logTrace("[CudaEncoder]: ReadNextFrame called.\n");
	// the source is only the imageSourcePipe
	if(cudaPipe == NULL){
		infoRecorder->logTrace("[CudaEncodedr]: get NULL cuda soruce pipeline .\n");
		return 0;
	}
	data = cudaPipe->load_data();
	if(data == NULL){
		infoRecorder->logTrace("[CudaEncoder]: wait the surface event.\n");
		int err;
		//if ((err = cudaPipe->timedwait(this->sourceNotifier, condMutex, &to)) != 0){
		if((err = cudaPipe->wait(this->sourceNotifier, condMutex))!=0){
			infoRecorder->logError("[CudaEncoder]: image source timed out.\n");
			infoRecorder->logTrace("[CudaEncoder]: surface source timed out.\n");
			return -1;
		}
		data = cudaPipe->load_data();
		if (data == NULL){
			infoRecorder->logTrace("[CudaEncoder]: unexpected NULL frame received ( from '%s', data = %d, buf = %d.\n", cudaPipe->name(), cudaPipe->data_count(), cudaPipe->buf_count());
			return -1;
		}
		else{
			infoRecorder->logTrace("[CudaEncoder]: wait and get a surface.\n");
		}
	}
	VFrame * vf = (VFrame *)data->ptr;
	if(vf == NULL)
	{
		// error
	}

	infoRecorder->logTrace("[CudaEncoder]: read a surface.\n");

	// handle pts
	if (basePts == -1LL){
		basePts = vf->imgPts;
		ptsSync = encoderPtsSync(30);
		newPts = ptsSync;
	}
	else{
		newPts = ptsSync + vf->imgPts - basePts;
	}

	if(vf->type == IMAGE){
		infoRecorder->logTrace("[CudaEncoder]: get a IMAGE source.\n");
		this->inputType = CPUIMAGE;
		// XXX: assume always YUV420P
		ImageFrame * frame = (ImageFrame *)data->ptr;

		// just updata the pointer, the encodeFrame will copy the memory
		memcpy(m_pVideoFrame, frame->imgBuf, frame->getImgBufSize());
		return frame->getImgBufSize();
	}
	else if(vf->type == SURFACE){
		infoRecorder->logTrace("[CudaEncoder]: get a SURFACE source.\n");
		// now we get the surface data
		// we need to use cuda to convert format
		this->inputType = GPUIMAGE;
		SurfaceFrame * sframe = (SurfaceFrame *)data->ptr;
		DX_VERSION dxVersion = sframe->dxVersion;

		if(cuFilter == NULL){
			// 
			cuFilter = new CudaFilter(this->dxDevice, dxVersion);
			cuFilter->initCuda(dxVersion);

			// really need to copy ? yes, copy to mapped surface
			cuFilter->initSurface(sframe->width, sframe->height);
		}

		// TODO : need to copy? yes ,copy to mapped surface, maybe later, change the map operation to map the surface directly.
		if(!cuFilter->updateSurface(sframe->dxSurface)){
			infoRecorder->logTrace("[CudaEncoder]: update surface failed.\n");
			return -1;
		}

		// run the cuda kernel to convert format
		if(dstPtr != NULL){
			infoRecorder->logTrace("[CudaEncoder]: call filter to covert surface format.\n");
			cuvidCtxLock(cuCtxLock, 0);
			//cuFilter->runKernels(dstPtr);
			cuFilter->runKernels(videoFrame_);
			cuvidCtxUnlock(cuCtxLock, 0);
		}
		else{
			// error
			cuvidCtxLock(cuCtxLock, 0);
			cuFilter->runKernels(videoFrame_);
			cuvidCtxUnlock(cuCtxLock, 0);
		}
		// XXX: assume always NV12
		return 0;
	}
	else{
		infoRecorder->logTrace("[CudaEncoder]: get an NUKNOWN source, surface: %p, type:%d, type addr:%p.\n", vf, vf->type, &(vf->type));
		return -1;
	}
}

// UYVY/YUY2 are both 4:2:2 formats (16bpc)
// Luma, U, V are interleaved, chroma is subsampled (w/2,h)
void CudaEncoder::CopyUYVYorYUY2Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{
	// Source is YUVY/YUY2 4:2:2, the YUV data in a packed and interleaved
	// YUV Copy setup
	CUDA_MEMCPY2D stCopyYUV422;
	memset((void *)&stCopyYUV422, 0, sizeof(stCopyYUV422));
	stCopyYUV422.srcXInBytes = 0;
	stCopyYUV422.srcY = 0;
	stCopyYUV422.srcMemoryType = CU_MEMORYTYPE_HOST;
	stCopyYUV422.srcHost = sFrameParams.picBuf;
	stCopyYUV422.srcDevice = 0;
	stCopyYUV422.srcArray = 0;
	stCopyYUV422.srcPitch = sFrameParams.Width * 2;

	stCopyYUV422.dstXInBytes = 0;
	stCopyYUV422.dstY = 0;
	stCopyYUV422.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	stCopyYUV422.dstHost = 0;
	stCopyYUV422.dstDevice = dptr_VideoFrame;
	stCopyYUV422.dstArray = 0;
	stCopyYUV422.dstPitch = m_pEncoderParams->nDeviceMemPitch;

	stCopyYUV422.WidthInBytes = m_pEncoderParams->iInputSize[0] * 2;
	stCopyYUV422.Height = m_pEncoderParams->iInputSize[1];

	// Don't forget we need to lock/unlock between memcopies
	checkCudaErrors(cuvidCtxLock(ctxLock, 0));
	checkCudaErrors(cuMemcpy2D(&stCopyYUV422));     // Now DMA Luma/Chroma
	checkCudaErrors(cuvidCtxUnlock(ctxLock, 0));
}

// YV12/IYUV are both 4:2:0 planar formats (12bpc)
// Luma, U, V chroma planar (12bpc), chroma is subsampled (w/2,h/2)
void CudaEncoder::CopyYV12orIYUVFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{
	// Source is YV12/IYUV, this native format is converted to NV12 format by the video encoder
	// (1) luma copy setup
	CUDA_MEMCPY2D stCopyLuma;
	memset((void *)&stCopyLuma, 0, sizeof(stCopyLuma));
	stCopyLuma.srcXInBytes = 0;
	stCopyLuma.srcY = 0;
	stCopyLuma.srcMemoryType = CU_MEMORYTYPE_HOST;
	stCopyLuma.srcHost = sFrameParams.picBuf;
	stCopyLuma.srcDevice = 0;
	stCopyLuma.srcArray = 0;
	stCopyLuma.srcPitch = sFrameParams.Width;

	stCopyLuma.dstXInBytes = 0;
	stCopyLuma.dstY = 0;
	stCopyLuma.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	stCopyLuma.dstHost = 0;
	stCopyLuma.dstDevice = dptr_VideoFrame;
	stCopyLuma.dstArray = 0;
	stCopyLuma.dstPitch = m_pEncoderParams->nDeviceMemPitch;

	stCopyLuma.WidthInBytes = m_pEncoderParams->iInputSize[0];
	stCopyLuma.Height = m_pEncoderParams->iInputSize[1];

	// (2) chroma copy setup, U/V can be done together
	CUDA_MEMCPY2D stCopyChroma;
	memset((void *)&stCopyChroma, 0, sizeof(stCopyChroma));
	stCopyChroma.srcXInBytes = 0;
	stCopyChroma.srcY = m_pEncoderParams->iInputSize[1] << 1; // U/V chroma offset
	stCopyChroma.srcMemoryType = CU_MEMORYTYPE_HOST;
	stCopyChroma.srcHost = sFrameParams.picBuf;
	stCopyChroma.srcDevice = 0;
	stCopyChroma.srcArray = 0;
	stCopyChroma.srcPitch = sFrameParams.Width >> 1; // chroma is subsampled by 2 (but it has U/V are next to each other)

	stCopyChroma.dstXInBytes = 0;
	stCopyChroma.dstY = m_pEncoderParams->iInputSize[1] << 1; // chroma offset (srcY*srcPitch now points to the chroma planes)
	stCopyChroma.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	stCopyChroma.dstHost = 0;
	stCopyChroma.dstDevice = dptr_VideoFrame;
	stCopyChroma.dstArray = 0;
	stCopyChroma.dstPitch = m_pEncoderParams->nDeviceMemPitch >> 1;

	stCopyChroma.WidthInBytes = m_pEncoderParams->iInputSize[0] >> 1;
	stCopyChroma.Height = m_pEncoderParams->iInputSize[1]; // U/V are sent together

	// Don't forget we need to lock/unlock between memcopies
	checkCudaErrors(cuvidCtxLock(ctxLock, 0));
	checkCudaErrors(cuMemcpy2D(&stCopyLuma));       // Now DMA Luma
	checkCudaErrors(cuMemcpy2D(&stCopyChroma));     // Now DMA Chroma channels (UV side by side)
	checkCudaErrors(cuvidCtxUnlock(ctxLock, 0));
}

// NV12 is 4:2:0 format (12bpc)
// Luma followed by U/V chroma interleaved (12bpc), chroma is subsampled (w/2,h/2)
void CudaEncoder::CopyNV12Frame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{
	// Source is NV12 in pitch linear memory
	// Because we are assume input is NV12 (if we take input in the native format), the encoder handles NV12 as a native format in pitch linear memory
	// Luma/Chroma can be done in a single transfer
	CUDA_MEMCPY2D stCopyNV12;
	memset((void *)&stCopyNV12, 0, sizeof(stCopyNV12));
	stCopyNV12.srcXInBytes = 0;
	stCopyNV12.srcY = 0;
	stCopyNV12.srcMemoryType = CU_MEMORYTYPE_HOST;
	stCopyNV12.srcHost = sFrameParams.picBuf;
	stCopyNV12.srcDevice = 0;
	stCopyNV12.srcArray = 0;
	stCopyNV12.srcPitch = sFrameParams.Width;

	stCopyNV12.dstXInBytes = 0;
	stCopyNV12.dstY = 0;
	stCopyNV12.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	stCopyNV12.dstHost = 0;
	stCopyNV12.dstDevice = dptr_VideoFrame;
	stCopyNV12.dstArray = 0;
	stCopyNV12.dstPitch = m_pEncoderParams->nDeviceMemPitch;

	stCopyNV12.WidthInBytes = m_pEncoderParams->iInputSize[0];
	stCopyNV12.Height = (m_pEncoderParams->iInputSize[1] * 3) >> 1;

	// Don't forget we need to lock/unlock between memcopies
	checkCudaErrors(cuvidCtxLock(ctxLock, 0));
	checkCudaErrors(cuMemcpy2D(&stCopyNV12));    // Now DMA Luma/Chroma
	checkCudaErrors(cuvidCtxUnlock(ctxLock, 0));
}

// If dptr_VideoFrame is != 0, then this is from Device Memory.
// Otherwise we will assume that video is coming from system host memory
bool CudaEncoder::EncodeFrame(NVVE_EncodeFrameParams &sFrameParams, CUdeviceptr dptr_VideoFrame, CUvideoctxlock ctxLock)
{

	infoRecorder->logTrace("[CudaEncoder]: EncodeFrame called.\n");
	// If this is the first frame, we can start timing
	if (m_nFrameCount == 0)
	{
		//start();
	}

	HRESULT hr = S_OK;

	if (m_pEncoderParams->iUseDeviceMem)
	{
		if (inputType == CPUIMAGE){
			// Copies video frame from system memory, and passes it as a System pointer to the API
			switch (m_pEncoderParams->iSurfaceFormat)
			{
			case UYVY: // UYVY (4:2:2)
			case YUY2:  // YUY2 (4:2:2)
				CopyUYVYorYUY2Frame(sFrameParams, dptr_VideoFrame, ctxLock);
				break;

			case YV12: // YV12 (4:2:0), Y V U
			case IYUV: // IYUV (4:2:0), Y U V
				CopyYV12orIYUVFrame(sFrameParams, dptr_VideoFrame, ctxLock);
				break;

			case NV12: // NV12 (4:2:0)
				CopyNV12Frame(sFrameParams, dptr_VideoFrame, ctxLock);
				break;

			default:
				break;
			}
		}
		else if (inputType == GPUIMAGE){
			// no need to copy, just asign the dptr_VideoFrame with gpu addr
			infoRecorder->logTrace("[CudaEncoder]: encoder input type is GPUIMAGE. device ptr:0x%p\n", dptr_VideoFrame);
			//dptr_VideoFrame = NULL;
		}

		sFrameParams.picBuf = NULL;  // Must be set to NULL in order to support device memory input
		//hr = NVEncodeFrame(m_pEncoder, &sFrameParams, 0, (void *)dptr_VideoFrame); //send the video (device memory) to the
		infoRecorder->logTrace("[CudaEncoder]: brefore call NVEncodeFrame.\n");
		hr = NVEncodeFrame(m_pEncoder, &sFrameParams, 0, (void *)videoFrame_.data);
	}
	else
	{
		// Copies video frame from system memory, and passes it as a System pointer to the API
		hr = NVEncodeFrame(m_pEncoder, &sFrameParams, 0, m_pSNRData);
	}

	if(hr == S_OK){
		infoRecorder->logTrace("[CudaEncoder]: encode frame success.\n");
	}else if(hr == E_FAIL){
		infoRecorder->logTrace("[CudaEncoder]: encode frame failed.\n");

	}else if(hr == E_POINTER){
		infoRecorder->logTrace("[CudaEncoder]: encoder handle is invalid.\n");
	}

	if (FAILED(hr))
	{
		printf("VideoEncoder::EncodeFrame() error when encoding frame (%d)\n", m_nFrameCount);
		infoRecorder->logTrace("[CudaEncoder]: encode frame error when encoding frame (%d).\n", m_nFrameCount);


		return false;
	}

	if (sFrameParams.bLast)
	{
		m_bLastFrame = true;
		m_nLastFrameNumber = m_nFrameCount;
	}
	else
	{
		frameSummation(m_nFrameCount);
		m_nFrameCount++;
	}

	return true;
}

HRESULT CudaEncoder::GetParamValue(DWORD dwParamType, void *pData)
{
	HRESULT hr = S_OK;
	hr = NVGetParamValue(m_pEncoder, dwParamType, pData);
	if (hr != S_OK)
	{
		printf("  NVGetParamValue FAIL!: hr = %08x\n", hr);
	}
	return hr;
}

bool CudaEncoder::GetEncodeParamsConfig(NVEncoderParams *pParams)
{
	infoRecorder->logTrace("[CudaEncoder]: GetEncodeParamsConfig called.\n");
	//assert(pParams != NULL);

	//DebugBreak();
	int iAspectRatio = 0;
	if (pParams == NULL){
		infoRecorder->logTrace("[CudaEncoder]: get NULL pParams.\n");
		return false;
	}

	fopen_s(&fpConfig, pParams->configFile, "r");

	if (fpConfig == NULL){
		return false;
	}

	//read the params
	_flushall();
	char cTempArr[250];
	fscanf_s(fpConfig, "%d", &(pParams->iCodecType));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iOutputSize[0]));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iOutputSize[1]));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iInputSize[0]));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iInputSize[1]));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(iAspectRatio));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->Fieldmode));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iP_Interval));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iIDR_Period));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iDynamicGOP));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->RCType));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iAvgBitrate));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iPeakBitrate));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iQP_Level_Intra));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iQP_Level_InterP));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iQP_Level_InterB));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iFrameRate[0]));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iFrameRate[1]));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iDeblockMode));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iProfileLevel));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iForceIntra));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iForceIDR));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iClearStat));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->DIMode));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->Presets));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iDisableCabac));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iNaluFramingType));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iDisableSPSPPS));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	fscanf_s(fpConfig, "%d", &(pParams->iSliceCnt));
	fscanf_s(fpConfig, "%[^'\n']", &cTempArr, 250);

	switch (iAspectRatio){
	case 0:
		{
			pParams->iAspectRatio[0] = 4;
			pParams->iAspectRatio[1] = 3;
		}
		break;

	case 1:
		{
			pParams->iAspectRatio[0] = 16;
			pParams->iAspectRatio[1] = 9;
		}
		break;

	case 2:
		{
			pParams->iAspectRatio[0] = 1;
			pParams->iAspectRatio[1] = 1;
		}
		break;

	default:
		{
			pParams->iAspectRatio[0] = 4;
			pParams->iAspectRatio[1] = 3;
		}
	}

	pParams->iAspectRatio[2] = 0;

	if (fpConfig)
	{
		fclose(fpConfig);
	}

	return true;
}

HRESULT CudaEncoder::SetParamValue(DWORD dwParamType, void *pData)
{
	HRESULT hr = S_OK;
	hr = NVSetParamValue(m_pEncoder, dwParamType, pData);

	//infoRecorder->logTrace("[CudaEncoder]: SetParamValue called.\n");
	if (hr != S_OK)
	{
		infoRecorder->logTrace("  NVSetParamValue: %26s,  hr = %08x ", sNVVE_EncodeParams[dwParamType].name, hr);

		for (int i = 0; i < sNVVE_EncodeParams[dwParamType].params; i++)
		{
			infoRecorder->logTrace(" %8d", *((DWORD *)pData + i));
			infoRecorder->logTrace(", ");
		}

		infoRecorder->logTrace(" FAILED!");
	}
	else
	{
		infoRecorder->logTrace("  NVSetParamValue : %26s = ", sNVVE_EncodeParams[dwParamType].name);

		for (int i = 0; i < sNVVE_EncodeParams[dwParamType].params; i++)
		{
			infoRecorder->logTrace("%8d", *((DWORD *)pData + i));
			infoRecorder->logTrace(", ");
		}
	}

	switch (dwParamType)
	{
	case NVVE_PROFILE_LEVEL:
		infoRecorder->logTrace(" [%s/%s] ",
			sProfileIDX2Char(sProfileName, (*((DWORD *)pData) & 0x00ff)),
			sProfileIDX2Char(sProfileLevel, (*((DWORD *)pData) >> 8) & 0x00ff));
		break;

	case NVVE_FIELD_ENC_MODE:
		infoRecorder->logTrace(" [%s]", sPictureType[*((DWORD *)pData)]);
		break;

	case NVVE_RC_TYPE:
		infoRecorder->logTrace(" [%s]", sNVVE_RateCtrlType[*((DWORD *)pData)]);
		break;

	case NVVE_PRESETS:
		infoRecorder->logTrace(" [%s Profile]", sVideoEncodePresets[*((DWORD *)pData)]);
		break;

	case NVVE_GPU_OFFLOAD_LEVEL:
		switch (*((DWORD *)pData))
		{
		case NVVE_GPU_OFFLOAD_DEFAULT:
			infoRecorder->logTrace(" [%s]", sGPUOffloadLevel[0]);
			break;

		case NVVE_GPU_OFFLOAD_ESTIMATORS:
			infoRecorder->logTrace(" [%s]", sGPUOffloadLevel[1]);
			break;

		case 16:
			infoRecorder->logTrace(" [%s]", sGPUOffloadLevel[2]);
			break;
		}

		break;
	}
	infoRecorder->logTrace("\n");
	//printf("\n");
	return hr;
}

bool CudaEncoder::SetCodecType(NVEncoderParams *pParams)
{
	assert(pParams != NULL);
	infoRecorder->logTrace("[CudaEncoder]: SetCodecType called.\n");
	HRESULT hr = S_OK;
	hr = NVSetCodec(m_pEncoder, pParams->iCodecType);

	if (hr != S_OK){
		infoRecorder->logTrace("\nSetCodecType FAIL\n");
		return false;
	}

	NVSetDefaultParam(m_pEncoder);

	infoRecorder->logTrace("  NVSetCodec ");

	if (pParams->iCodecType == 4){
		infoRecorder->logTrace("<H.264 Video>\n");
	}
	else if (pParams->iCodecType == 5){
		// Support for this codec is being deprecated
		infoRecorder->logTrace("<VC-1 Video> (unsupported)\n");
		return false;
	}
	else{
		infoRecorder->logTrace("Unknown Video Format \"%s\"\n", pParams->iCodecType);
		return false;
	}

	return true;
}


bool CudaEncoder::SetParameters(NVEncoderParams *pParams)
{
	assert(pParams != NULL);
	//infoRecorder->logTrace("[CudaEncoder]: setParameters called.\n");

	HRESULT hr = S_OK;
	hr = SetParamValue(NVVE_OUT_SIZE, &(pParams->iOutputSize));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_IN_SIZE, &(pParams->iInputSize));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_ASPECT_RATIO, &(pParams->iAspectRatio));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_FIELD_ENC_MODE, &(pParams->Fieldmode));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_P_INTERVAL, &(pParams->iP_Interval));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_IDR_PERIOD, &(pParams->iIDR_Period));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_DYNAMIC_GOP, &(pParams->iDynamicGOP));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_RC_TYPE, &(pParams->RCType));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_AVG_BITRATE, &(pParams->iAvgBitrate));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_PEAK_BITRATE, &(pParams->iPeakBitrate));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_QP_LEVEL_INTRA, &(pParams->iQP_Level_Intra));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_QP_LEVEL_INTER_P, &(pParams->iQP_Level_InterP));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_QP_LEVEL_INTER_B, &(pParams->iQP_Level_InterB));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_FRAME_RATE, &(pParams->iFrameRate));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_DEBLOCK_MODE, &(pParams->iDeblockMode));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_PROFILE_LEVEL, &(pParams->iProfileLevel));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_FORCE_INTRA, &(pParams->iForceIntra));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_FORCE_IDR, &(pParams->iForceIDR));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_CLEAR_STAT, &(pParams->iClearStat));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_SET_DEINTERLACE, &(pParams->DIMode));
	if (hr != S_OK){
		return FALSE;
	}

	if (pParams->Presets != -1){
		hr = SetParamValue(NVVE_PRESETS, &(pParams->Presets));
		if (hr != S_OK){
			return FALSE;
		}
	}

	hr = SetParamValue(NVVE_DISABLE_CABAC, &(pParams->iDisableCabac));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_CONFIGURE_NALU_FRAMING_TYPE, &(pParams->iNaluFramingType));
	if (hr != S_OK){
		return FALSE;
	}

	hr = SetParamValue(NVVE_DISABLE_SPS_PPS, &(pParams->iDisableSPSPPS));
	if (hr != S_OK){
		return FALSE;
	}

	printf("\n");
	return true;
}

int CudaEncoder::DisplayGPUCaps(int deviceOrdinal, NVEncoderParams *pParams, bool bDisplay)
{
	NVVE_GPUAttributes GPUAttributes = { 0 };
	HRESULT hr = S_OK;
	int gpuPerformance;

	assert(pParams != NULL);

	GPUAttributes.iGpuOrdinal = deviceOrdinal;
	hr = GetParamValue(NVVE_GET_GPU_ATTRIBUTES, &GPUAttributes);

	if (hr != S_OK)
	{
		printf("  >> NVVE_GET_GPU_ATTRIBUTES error! <<\n\n");
	}

	gpuPerformance = GPUAttributes.iClockRate * GPUAttributes.iMultiProcessorCount;
	gpuPerformance = gpuPerformance * _ConvertSMVer2Cores(GPUAttributes.iMajor, GPUAttributes.iMinor);

	size_t totalGlobalMem;
	CUresult error_id = cuDeviceTotalMem((unsigned int *)&totalGlobalMem, deviceOrdinal);

	if (error_id != CUDA_SUCCESS)
	{
		printf("cuDeviceTotalMem returned %d\n-> %s\n", (int)error_id, getCudaDrvErrorString(error_id));
		return -1;
	}


	if (bDisplay)
	{
		printf("  GPU Device %d (SM %d.%d) : %s\n", GPUAttributes.iGpuOrdinal,
			GPUAttributes.iMajor, GPUAttributes.iMinor,
			GPUAttributes.cName);
		printf("  Total Memory          = %4.0f MBytes\n", ceil((float)totalGlobalMem / 1048576.0f));
		printf("  GPU Clock             = %4.2f MHz\n", (float)GPUAttributes.iClockRate / 1000.f);
		printf("  MultiProcessors/Cores = %d MPs (%d Cores)\n", GPUAttributes.iMultiProcessorCount,
			GPUAttributes.iMultiProcessorCount*_ConvertSMVer2Cores(GPUAttributes.iMajor, GPUAttributes.iMinor));
		printf("  Maximum Offload Mode  = ");

		switch (GPUAttributes.MaxGpuOffloadLevel)
		{
		case NVVE_GPU_OFFLOAD_DEFAULT:
			printf("CPU: PEL Processing Only\n");
			break;

		case NVVE_GPU_OFFLOAD_ESTIMATORS:
			printf("GPU: Motion Estimation & Intra Prediction\n");
			break;

		case NVVE_GPU_OFFLOAD_ALL:
			printf("GPU: Full Offload\n");
			break;
		}

		printf("\n");
	}

	pParams->MaxOffloadLevel = GPUAttributes.MaxGpuOffloadLevel;

	return gpuPerformance;
}

int CudaEncoder::GetGPUCount(NVEncoderParams *pParams, int *gpuPerf, int *bestGPU)
{
	assert(gpuPerf != NULL && bestGPU != NULL && pParams != NULL);

	// Now we can query the GPUs available for encoding
	HRESULT hr = GetParamValue(NVVE_GET_GPU_COUNT, &(pParams->GPU_count));

	if (hr != S_OK)
	{
		printf("  >> NVVE_GET_GPU_COUNT error ! <<\n\n");
	}

	printf("\n[ Detected %d GPU(s) capable of CUDA Accelerated Video Encoding ]\n\n", pParams->GPU_count);
	int temp = 0;

	for (int deviceCount = 0; deviceCount < pParams->GPU_count; deviceCount++)
	{
		temp = DisplayGPUCaps(deviceCount, pParams, !(pParams->force_device));

		if (temp >(*gpuPerf))
		{
			*gpuPerf = temp;
			*bestGPU = deviceCount;
		}
	}

	return (*bestGPU);
}

void CudaEncoder::SetActiveGPU(NVEncoderParams *pParams, int gpuID)
{
	assert(pParams != NULL);
	infoRecorder->logTrace("[CudaEncoder]: setActiveGPU called.\n");
	infoRecorder->logTrace("  >> Setting Active GPU %d for Video Encoding <<\n", gpuID);
	HRESULT hr = SetParamValue(NVVE_FORCE_GPU_SELECTION, &gpuID);

	if (hr != S_OK)
	{
		infoRecorder->logTrace("  >> NVVE_FORCE_GPU_SELECTION Error <<\n\n");
	}

	DisplayGPUCaps(gpuID, pParams, true);
}

void CudaEncoder::SetGPUOffloadLevel(NVEncoderParams *pParams)
{
	assert(pParams != NULL);

	NVVE_GPUOffloadLevel eMaxOffloadLevel = NVVE_GPU_OFFLOAD_DEFAULT;
	HRESULT              hr = GetParamValue(NVVE_GPU_OFFLOAD_LEVEL_MAX, &eMaxOffloadLevel);

	if (hr != S_OK)
	{
		printf("  >> NVVE_GPUOFFLOAD_LEVEL_MAX Error <<\n\n");
	}

	if (pParams->GPUOffloadLevel > eMaxOffloadLevel)
	{
		pParams->GPUOffloadLevel = eMaxOffloadLevel;
		printf("  >> Overriding, setting GPU to: ");

		switch (pParams->GPUOffloadLevel)
		{
		case NVVE_GPU_OFFLOAD_DEFAULT:
			printf("Offload Default (CPU: PEL Processing\n)");
			break;

		case NVVE_GPU_OFFLOAD_ESTIMATORS:
			printf("Offload Motion Estimators\n");
			break;

		case NVVE_GPU_OFFLOAD_ALL:
			printf("Offload Full Encode\n)");
			break;
		}
	}

	hr = SetParamValue(NVVE_GPU_OFFLOAD_LEVEL, &(pParams->GPUOffloadLevel));

	if (hr != S_OK)
	{
		printf("  >> NVVE_GPU_OFFLOAD_LEVEL Error <<\n\n");
	}
}


//////// the encoder factory //////////
CudaEncoderFactory * CudaEncoderFactory::factory;

CudaEncoderFactory * CudaEncoderFactory::GetCudaEncoderFactory(){
	if (factory){
		return factory;
	}
	else{
		//// create a new factory
		return CudaEncoderFactory::GetCudaEncoderFactory();
	}
}

CudaEncoder * CudaEncoderFactory::CreateCudaEncoder(const char * configFile, INPUT_TYPE inputType){
	if (!factory){
		infoRecorder->logError("NULL Cuda Encoder Factory!\n");
		return NULL;
	}
	else{
		//// creat a new cuda encoder and store
		// init the params
		NVEncoderParams sEncoderParams = { 0 };

		sEncoderParams.measure_fps = 1;
		sEncoderParams.measure_psnr = 0;
		sEncoderParams.force_device = 0;    /// if 0, the system will select the best GPU
		sEncoderParams.iForcedGPU = 0;

		sEncoderParams.GPUOffloadLevel = NVVE_GPU_OFFLOAD_ALL;
		sEncoderParams.iSurfaceFormat = (int)NV12;
		sEncoderParams.iPictureType = (int)FRAME_PICTURE;
		sEncoderParams.iSurfaceFormat = NV12;  // UYVY, YUY2, YV12, NV12, IYUV

		sEncoderParams.iUseDeviceMem = 1;

		CudaEncoder * tEncoder = new CudaEncoder(&sEncoderParams);

		tEncoder->setInputType(inputType);  /// default, the image comes from GPU

		return tEncoder;
	}
}

//NVCUVENC callback function to signal that the encoding operation on the frame has begun
static void _stdcall HandleOnBeginFrame(const NVVE_BeginFrameInfo *pbfi, void *pUserdata)
{
	infoRecorder->logTrace("[CudaEncoder]: HnadleOnBeginFrame called, frame number:%d.\n", pbfi->nFrameNumber);
	/// set the bitrate of the encoder
	return;
}


// NVCUVENC callback function to signal the start of bitstream that is to be encoded
static unsigned char * _stdcall HandleAcquireBitStream(int * pBufferSize, void * pUserData){

	infoRecorder->logTrace("[CudaEncoder]: HandelAcquireBitStream called. encoder:0x%p.\n", pUserData);
	CudaEncoder * encoder;
	if (pUserData){
		encoder = (CudaEncoder *)pUserData;
	}
	else{
		// error
		infoRecorder->logError("Cuda encoder: null encoder specified\n");
		return NULL;
	}

	//get the buffer to store the coded frame.

	*pBufferSize = 1024 * 1024;
	return encoder->GetCharBuf();
}
// NVCUVENC callback function to signal that the encoded stream is ready to be written or sent
static void _stdcall HandleReleaseBitStream(int nBytesInBuffer, unsigned char * cb, void * pUserData){
	infoRecorder->logTrace("[CudaEncoder]: HandleReleaseBitStream called.\n");
	CudaEncoder * encoder = NULL;
#if 0
	AVPacket pkt;
#endif
	if (pUserData){
		encoder = (CudaEncoder *)pUserData;
	}
	else{
		// error
		infoRecorder->logTrace("[ERROR]: null encoder specified\n");
		return;
	}
#if 0
	if (encoder){
		// sent 
		av_init_packet(&pkt);
		pkt.data = cb;
		pkt.size = nBytesInBuffer;
		pkt.stream_index = 0;
		encoder->sendPacket(0, encoder->getRTSPContext(), &pkt, pkt.pts); // send the packet to the client

		if (pkt.side_data_elems > 0){
			int i;
			for (i = 0; i < pkt.side_data_elems; i++)
				av_free(pkt.side_data[i].data);
			av_freep(&pkt.side_data);
			pkt.side_data_elems = 0;
		}

	}
#else
	// write to file
	if(encoder->fileOut()){
		infoRecorder->logTrace("[CudaEncoder]: write to output:%d.\n", nBytesInBuffer);
		fwrite(cb, 1, nBytesInBuffer, encoder->fileOut());
	}
	else{
		infoRecorder->logTrace("[CudaEncoder]: file out pointer is NULL.\n");
	}
#endif
	return;
}

// NVCUVENC callback function signals that the encoding operation on the frame has finished
static void _stdcall HandleOnEndFrame(const NVVE_EndFrameInfo * pefi, void *pUserData){
	infoRecorder->logTrace("[CudaEncoder]: HandleOnEndFrame called.\n");
	CudaEncoder * encoder = NULL;
	if (pUserData){
		encoder = (CudaEncoder*)pUserData;

	}
	else{
		//error
		infoRecorder->logTrace("[CudaEncoder]: null encoder specified\n");
		infoRecorder->logError("Cuda encoder: null encoder specified\n");
		return;
	}
	encoder->frameSummation(-(pefi->nFrameNumber));
	if (encoder->IsLastFrameSent()){
		infoRecorder->logTrace("[CudaEncoder]: get last frame.\n");
		// check to see if the last frame has been sent
		if (encoder->getFrameSum() == 0){
			//
		}

		// trigger some event
	}
	else{

	}

	return;

}



BOOL CudaEncoder::stop(){
	printf("\n[VideoEncoder() <Stopped>]\n");
#ifdef FILE_INPUT
	printf("  [input]  = %s\n", m_pEncoderParams->inputFile);
#endif
	printf("  [output] = %s\n", m_pEncoderParams->outputFile);
	return TRUE;
}

///////////////////////////// cudaEncoder::run   //////////////

void CudaEncoder::run(){
	infoRecorder->logTrace("[CudaEncoder]: run called.\n");
	NVVE_EncodeFrameParams efparams;
#if 0
	efparams.Height = sEncoderParams.iOutputSize[1];
	efparams.Width = sEncoderParams.iOutputSize[0];
	
#else
	efparams.Height = frameSize_.height;
	efparams.Width = frameSize_.width;
	efparams.Pitch = (sEncoderParams.nDeviceMemPitch ? sEncoderParams.nDeviceMemPitch : sEncoderParams.iOutputSize[0]);
	//efparams.Pitch = static_cast<int>(videoFrame_.step);
#endif
	
	efparams.PictureStruc = (NVVE_PicStruct)sEncoderParams.iPictureType;
	efparams.SurfFmt = (NVVE_SurfaceFormat)sEncoderParams.iSurfaceFormat;
	efparams.progressiveFrame = (sEncoderParams.iSurfaceFormat == 3) ? 1 : 0;
	efparams.repeatFirstField = 0;
	efparams.topfieldfirst = (sEncoderParams.iSurfaceFormat == 1) ? 1 : 0;

	efparams.progressiveFrame = (sEncoderParams.iSurfaceFormat == NV12) ? 1: 0;
	//efparams.bLast = lastFrame;
	//efparams.picBuf = NULL; /// Must be set to NULL in order to support device memory inpout

	size_t bytes_read = 0;

	if ((inputType == CPUIMAGE || !sEncoderParams.iDeviceMemInput)){
		bytes_read = ReadNextFrame();
		if (bytes_read != -1)
			efparams.picBuf = (unsigned char *)GetVideoFrame(); // get the yuv buffer pointer from file
	}
	else if (inputType == GPUIMAGE){
		efparams.picBuf = NULL; /// set to NULL, use the GPU memory
		ReadNextFrame(dptrVideoFrame);
		infoRecorder->logTrace("[CudaEncoder]: input type is GPU image, ater get next frame.\n");
	}
	else{
		// error
		exit(EXIT_SUCCESS);
	}

	// once we have reached the EOF, we know this is the last frame
	// send this flag to the h.264 encoder so it knows to properly flush the file
	if (IsLastFrameSent()){
		efparams.bLast = false;
	}
	else{
		efparams.bLast = true;
	}

	// if dptrVideoFrame is NULL, then we assume that frames come from system memory, otherwise is comes from GPU memory
	if (EncodeFrame(efparams, dptrVideoFrame, cuCtxLock) == false){
		// error
		infoRecorder->logError("[CudaEncoder]: failed to encoder frame.\n");
		infoRecorder->logTrace("[CudaEncoder]: failed to encode frame, do not know.\n");
	}
	else{
		infoRecorder->logTrace("[CudaEncoder]: success encode frame.\n");
		infoRecorder->logError("[CudaEncoder]: success encode frame.\n");
	}

	computeFPS();
}
void CudaEncoder::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
	infoRecorder->logTrace("[CudaEncoder]: onThreadMsg called.\n");
}
void CudaEncoder::onThreadStart(){
	infoRecorder->logTrace("[CudaEncoder]: onThreadStart called.\n");
	//7NVEncoderParams sEncoderParams = { 0 };
	HRESULT hr = S_OK;

	void * pData = NULL;
	/// register the source pipeline




	// init the mutex
	condMutex = CreateMutex(NULL, FALSE, NULL);
	cond = CreateEvent(NULL, FALSE, FALSE, NULL);

	basePts = -1LL;
	newPts = 0LL;
	pts = -1LL;
	ptsSync = 0LL;

	//if(this->sour
	if(inputType == INPUT_TYPE::CPUIMAGE){
		// find the image source
		if(pipe == NULL){
			infoRecorder->logError("[CudaEncoder]: image source pipeline not registered.\n");
		}
	}else if(inputType == INPUT_TYPE::GPUIMAGE){
		// find the surface source
		if(NULL == pipe){
			infoRecorder->logError("[CudaEncoder]: surface source pipe not registered.\n");
		}
	}else{
		//
		infoRecorder->logError("[CudaEncoder]: unknown source type.\n");

	}

	/// create the timer for the time measurement
	sdkCreateTimer(&frame_timer);
	sdkResetTimer(&frame_timer);

	sdkCreateTimer(&global_timer);
	sdkResetTimer(&global_timer);

	/// this is for GPU device memory input, and support for interop with another CUDA context
	/// The NVIDIA CUDA Encoder will use the CUDA context to be able to pass in shared device memory
	/// convent the surface data to cuda device memory
	if(!sEncoderParams.iUseDeviceMem){
		infoRecorder->logTrace("[CudaEncoder]: do not use device memory.\n");
		sEncoderParams.iUseDeviceMem = 1;
	}
	if (sEncoderParams.iUseDeviceMem){

		//DebugBreak();
		infoRecorder->logTrace("[CudaEncoder]: use device memory.\n");
		HRESULT h = S_OK;
#if 0
		checkCudaErrors(cuInit(0));
		checkCudaErrors(cuDeviceGet(&cuDevice, sEncoderParams.iForcedGPU));
		checkCudaErrors(cuCtxCreate(&cuContext, CU_CTX_BLOCKING_SYNC, cuDevice));
#else

		initGpuMemory();
#endif
		// create the CUDA memory Piched Surface
		if (sEncoderParams.iSurfaceFormat == UYVY || sEncoderParams.iSurfaceFormat == YUY2){
			this->WidthInBytes = (sEncoderParams.iInputSize[0] * sSurfaceFormat[sEncoderParams.iSurfaceFormat].bpp) >> 3; // width
			Height = sEncoderParams.iInputSize[1];
		}
		else{
			WidthInBytes = sEncoderParams.iInputSize[0];
			Height = (unsigned int)(sEncoderParams.iInputSize[1] * sSurfaceFormat[sEncoderParams.iSurfaceFormat].bpp >> 3);
		}

		ElementSizeBytes = 16;
#if 0
#if (CUDA_FORCE_API_VERSION == 3010)
		checkCudaErrors(cuMemAllocPitch(&dptrVideoFrame, &Pitch, WidthInBytes, Height, ElementSizeBytes));
		infoRecorder->logTrace("[CudaEncoder]: after malloc device memroy, ptr:0x%p, pitch:%d, widthInBytes:%d, height:%d, ElementSizeBytes:%d.\n", dptrVideoFrame, Pitch, WidthInBytes, Height, ElementSizeBytes);
#else
		checkCudaErrors(cuMemeAllocPitch(&dptrVideoFrame, (size_t*)&Pitch, WidthInBytes, Height, ElementSizeBytes));
		infoRecorder->logTrace("[CudaEncoder]: after malloc device memroy, ptr:0x%p, pitch:%d, widthInBytes:%d, height:%d, ElementSizeBytes:%d.\n", dptrVideoFrame, Pitch, WidthInBytes, Height, ElementSizeBytes);
#endif  // CUDA_FORCE_API_VERSION

		sEncoderParams.nDeviceMemPitch = Pitch; // copy the device meory pitch(we'll need this for later if we use device memory
#endif
		sEncoderParams.nDeviceMemPitch = videoFrame_.step;
		// pop the cuda context from the stack (this will make the cuda context current)
		// this is needed in order to inherit the cuda contexts created outside of the cuda H.264 encoder
		// cuda H.264 encoder will just inherit the available cuda context
		CUcontext cuContextCurr;
		checkCudaErrors(cuCtxPopCurrent(&cuContextCurr));

		// create the video context lock (used for synchronization)
		checkCudaErrors(cuvidCtxLockCreate(&cuCtxLock, cuContext));

		// if we are using GPU device memory with NVCUVENC, if is necessary to create a 
		// CUDA context with a context lock cuvidCtxLock, the context lock needs to be passed to NVCUVENC

		{
			hr = SetParamValue(NVVE_DEVICE_MEMORY_INPUT, &(sEncoderParams.iUseDeviceMem));
			if (FAILED(hr)){
				// failed
				infoRecorder->logError("[CudaEncoder]: NVVE_DEVICE_MEMORY_INPUT failed.\n");
			}
			hr = SetParamValue(NVVE_DEVICE_CTX_LOCK, &cuCtxLock);

			if (FAILED(hr)){
				// failed
				infoRecorder->logError("[CudaEncoder]: NVVE_DEVICE_CTX_LOCK failed.\n");
			}
		}
	}

	// now provide the callback functions to cuda h.264 encoder
	{
		memset(&m_NVCB, 0, sizeof(NVVE_CallbackParams));
		m_NVCB.pfnacquirebitstream = HandleAcquireBitStream;
		m_NVCB.pfnonbeginframe = HandleOnBeginFrame;
		m_NVCB.pfnonendframe = HandleOnEndFrame;
		m_NVCB.pfnreleasebitstream = HandleReleaseBitStream;
		// TODO
		// form the user data with cuda encoder and the psnr data
#if 0
		if(!SetCBFunctions(&m_NVCB, (void *)this)){
			infoRecorder->logTrace("[CudaEncoder]: setCBFunction failed.\n");
		}
#else
		// set the callback
		infoRecorder->logTrace("[CudaEncoder]: NVRegisterCB called.\n");
		NVRegisterCB(this->m_pEncoder, m_NVCB, this);
#endif
	}

	// now we must create the HW Encoder device
#if 1
	CreateHWEncoder(&sEncoderParams);
#else
	infoRecorder->logTrace("[CudaEncoder]: create HW encoder.\n");
	NVCreateHWEncoder(this->m_pEncoder);
#endif

	// CPU timers needed for performance
	{
		sdkStartTimer(&global_timer);
		sdkResetTimer(&global_timer);

		sdkStartTimer(&frame_timer);
		sdkResetTimer(&frame_timer);
	}
}
void CudaEncoder::onQuit(){
	//Stop();
	infoRecorder->logTrace("[CudaEncoder]: onQuit called.\n");
	sdkStopTimer(&global_timer);
	if (sEncoderParams.iUseDeviceMem){
		checkCudaErrors(cuvidCtxLock(cuCtxLock, 0));
		checkCudaErrors(cuMemFree(dptrVideoFrame));
		checkCudaErrors(cuvidCtxUnlock(cuCtxLock, 0));

		checkCudaErrors(cuvidCtxLockDestroy(cuCtxLock));
		checkCudaErrors(cuCtxDestroy(cuContext));
	}
}

// the function is for initing the cuda

int CudaEncoder::startEncoding(){
	infoRecorder->logTrace("[CudaEncoder]: startEncoding called.\n");
	return this->start();
}
void CudaEncoder::idle(){

}
void CudaEncoder::setBitrate(int bitrate){
	//this->r = bitrate;
}
bool CudaEncoder::setInputBuffer(char * buf){
	this->m_pCharBuf = (unsigned char *)buf;
	return true;
}
bool CudaEncoder::setBufferInsideGpu(char * p){
	return true;
}

bool CudaEncoder::supportInterOp(){
	return true;
}

// create and init the cuda filter
bool CudaEncoder::InitCudaFilter(DX_VERSION ver, void * device){
	infoRecorder->logTrace("[CudaEncoder]: InitCudaFilter called.\n");
	if(this->cuFilter){
		infoRecorder->logTrace("[CudaEncoder]: cuFilter is not NULL.\n");
	}
	cuFilter = new CudaFilter(device, ver);
	if(cuFilter->initCuda(ver)==S_FALSE){
		infoRecorder->logTrace("[CudaEncoder]: register d3d device failed.\n");
		MessageBox(NULL, "register d3d device failed", "ERROR", MB_OK);
	}
	if(!cuFilter->initSurface(outputWidth, outputHeight)){
		infoRecorder->logTrace("[CudaEncoder]: init the cuda file surface failed.\n");
		return false;
	}
	//cuFilter->

	return true;
}



//////////////////////////////// for CudaEncoderImpl ////////////

void CudaEncoderImpl::initGpuMemory(){
	int err;

	// init context
	GpuMat temp(1, 1, CV_8U);

	temp.release();

	static const int bpp[] = 
	{
		16, /// UYVY, 4:2:2
		16, /// YUY2, 4:2:2
		12, /// YV12, 4:2:0
		12, /// NV12, 4:2:0
		12, /// IYUV, 4:2:0
	};

	CUcontext cuContext = NULL;
	CUdevice cuDevice;
	cuInit(0);
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&cuContext, CU_CTX_BLOCKING_SYNC, cuDevice);

	if(surfaceFormat_ == UYVY || surfaceFormat_ == YUY2){
		videoFrame_.create(frameSize_.height, (frameSize_.width * bpp[surfaceFormat_])/ 8, CV_8UC1);
	}
	else{
		videoFrame_.create((frameSize_.height * bpp[surfaceFormat_]) / 8, frameSize_.width, CV_8UC1);
	}

	cuvidCtxLockCreate(&cuCtxLock_, cuContext);

	// if we are suing GPU device memory with NVCUVENC, it si necessary to create a
	// CUDA cojntext with a Context Lock cuvidCtxLock. The Context Lock needs to be passed to NVCUVENC

	int iUseDeviceMem = 1;
	err = NVSetParamValue(encoder_, NVVE_DEVICE_MEMORY_INPUT, &iUseDeviceMem);
	assert(err == 0);

	err = NVSetParamValue(encoder_, NVVE_DEVICE_CTX_LOCK, &cuCtxLock_);
	assert( err == 0);
}

void CudaEncoderImpl::initCallbacks(){
	NVVE_CallbackParams cb;
	memset(&cb, 0, sizeof(NVVE_CallbackParams));

	cb.pfnacquirebitstream	= HandleAcquireBitStream;
	cb.pfnonbeginframe		= HandleOnBeginFrame;
	cb.pfnonendframe		= HandleOnEndFrame;
	cb.pfnreleasebitstream	= HandleReleaseBitStream;
	NVRegisterCB(encoder_, cb, this);

}

void CudaEncoderImpl::createHWEncoder(){
	int err;
	// crewate the NVIDIA HW resource for encoding on NVIDIA hardware
	err = NVCreateHWEncoder(encoder_);
	assert( err == 0);
}

///copy function for CudaEncoderImpl

// UYVY/YUY2 are both 4:2:2 formats (16bpc)
// Luma, U, V are interleved, chroma is subsampled (w/2, h)
void copyUYVYorYUY2Frame(Size frameSize, const GpuMat& src, GpuMat& dst)
    {
        // Source is YUVY/YUY2 4:2:2, the YUV data in a packed and interleaved

        // YUV Copy setup
        CUDA_MEMCPY2D stCopyYUV422;
        memset(&stCopyYUV422, 0, sizeof(CUDA_MEMCPY2D));

        stCopyYUV422.srcXInBytes          = 0;
        stCopyYUV422.srcY                 = 0;
        stCopyYUV422.srcMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyYUV422.srcHost              = 0;
        stCopyYUV422.srcDevice            = (CUdeviceptr) src.data;
        stCopyYUV422.srcArray             = 0;
        stCopyYUV422.srcPitch             = src.step;

        stCopyYUV422.dstXInBytes          = 0;
        stCopyYUV422.dstY                 = 0;
        stCopyYUV422.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyYUV422.dstHost              = 0;
        stCopyYUV422.dstDevice            = (CUdeviceptr) dst.data;
        stCopyYUV422.dstArray             = 0;
        stCopyYUV422.dstPitch             = dst.step;

        stCopyYUV422.WidthInBytes         = frameSize.width * 2;
        stCopyYUV422.Height               = frameSize.height;

        // DMA Luma/Chroma
        cuMemcpy2D(&stCopyYUV422);
    }

    // YV12/IYUV are both 4:2:0 planar formats (12bpc)
    // Luma, U, V chroma planar (12bpc), chroma is subsampled (w/2,h/2)
    void copyYV12orIYUVFrame(Size frameSize, const GpuMat& src, GpuMat& dst)
    {
        // Source is YV12/IYUV, this native format is converted to NV12 format by the video encoder

        // (1) luma copy setup
        CUDA_MEMCPY2D stCopyLuma;
        memset(&stCopyLuma, 0, sizeof(CUDA_MEMCPY2D));

        stCopyLuma.srcXInBytes          = 0;
        stCopyLuma.srcY                 = 0;
        stCopyLuma.srcMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyLuma.srcHost              = 0;
        stCopyLuma.srcDevice            = (CUdeviceptr) src.data;
        stCopyLuma.srcArray             = 0;
        stCopyLuma.srcPitch             = src.step;

        stCopyLuma.dstXInBytes          = 0;
        stCopyLuma.dstY                 = 0;
        stCopyLuma.dstMemoryType        = CU_MEMORYTYPE_DEVICE;
        stCopyLuma.dstHost              = 0;
        stCopyLuma.dstDevice            = (CUdeviceptr) dst.data;
        stCopyLuma.dstArray             = 0;
        stCopyLuma.dstPitch             = dst.step;

        stCopyLuma.WidthInBytes         = frameSize.width;
        stCopyLuma.Height               = frameSize.height;

        // (2) chroma copy setup, U/V can be done together
        CUDA_MEMCPY2D stCopyChroma;
        memset(&stCopyChroma, 0, sizeof(CUDA_MEMCPY2D));

        stCopyChroma.srcXInBytes        = 0;
        stCopyChroma.srcY               = frameSize.height << 1; // U/V chroma offset
        stCopyChroma.srcMemoryType      = CU_MEMORYTYPE_DEVICE;
        stCopyChroma.srcHost            = 0;
        stCopyChroma.srcDevice          = (CUdeviceptr) src.data;
        stCopyChroma.srcArray           = 0;
        stCopyChroma.srcPitch           = src.step >> 1; // chroma is subsampled by 2 (but it has U/V are next to each other)

        stCopyChroma.dstXInBytes        = 0;
        stCopyChroma.dstY               = frameSize.height << 1; // chroma offset (srcY*srcPitch now points to the chroma planes)
        stCopyChroma.dstMemoryType      = CU_MEMORYTYPE_DEVICE;
        stCopyChroma.dstHost            = 0;
        stCopyChroma.dstDevice          = (CUdeviceptr) dst.data;
        stCopyChroma.dstArray           = 0;
        stCopyChroma.dstPitch           = dst.step >> 1;

        stCopyChroma.WidthInBytes       = frameSize.width >> 1;
        stCopyChroma.Height             = frameSize.height; // U/V are sent together

        // DMA Luma
        cuMemcpy2D(&stCopyLuma);

        // DMA Chroma channels (UV side by side)
		cuMemcpy2D(&stCopyChroma);
    }

void copyNV12Frame(Size frameSize, const GpuMat & src, GpuMat& dst){
	// source is NV12 in pitch linear memory
	// because we are assume inoput is NV12 (if we take inout in the native format), the encoder hadles NV12 as a native format in pitch linear memory

	// Luma/Chroma can be done in a single transfer
	CUDA_MEMCPY2D stCopyNV12;
	memset(&stCopyNV12, 0, sizeof(CUDA_MEMCPY2D));

	stCopyNV12.srcXInBytes				= 0;
	stCopyNV12.srcY						= 0;
	stCopyNV12.srcMemoryType			= CU_MEMORYTYPE_DEVICE;
	stCopyNV12.srcHost					= 0;
	stCopyNV12.srcDevice				= (CUdeviceptr) src.data;
	stCopyNV12.srcArray					= 0;
	stCopyNV12.srcPitch					= src.step;

	stCopyNV12.dstXInBytes				= 0;
	stCopyNV12.dstY						= 0;
	stCopyNV12.dstMemoryType			= CU_MEMORYTYPE_DEVICE;
	stCopyNV12.dstHost					= 0;
	stCopyNV12.dstDevice				= (CUdeviceptr) dst.data;
	stCopyNV12.dstArray					= 0;
	stCopyNV12.dstPitch					= dst.step;

	stCopyNV12.WidthInBytes				= frameSize.width;
	stCopyNV12.Height					= (frameSize.height * 3) >> 1;

	// DMA Luma/Chroma
	//checkCudaErrors(cuvidCtxLock(ctxLock, 0));
	cuMemcpy2D(&stCopyNV12);
}

void CudaEncoderImpl::write(GpuMat frame, bool lastFrame){
	if(inputFormat_ == SF_BGR){
		//assert(frame. == frameSize_);
	}

	NVVE_EncodeFrameParams efparams;
	efparams.Width = frameSize_.width;
	efparams.Height = frameSize_.height;
	efparams.Pitch = static_cast<int>(videoFrame_.step);
	efparams.SurfFmt = surfaceFormat_;
	efparams.PictureStruc = FRAME_PICTURE;
	efparams.topfieldfirst = 0;
	efparams.repeatFirstField = 0;
	efparams.progressiveFrame = (surfaceFormat_ == NV12) ? 1: 0;
	efparams.bLast = lastFrame;
	efparams.picBuf = 0;  // must be set to NULL in order to support device memory input

	// Don't forget we need to lock/unlock between memcopies
	checkCudaErrors(cuvidCtxLock(cuCtxLock_, 0));
	if(inputFormat_ == SF_BGR){
		RGB_to_YV12(frame, videoFrame_);
	}
	else{
		switch(surfaceFormat_){
		case UYVY: // UYVY (4:2:2)
		case YUY2: // YUY2 (4:2:2)
			copyUYVYorYUY2Frame(frameSize_, frame, videoFrame_);
			break;
		case YV12: // YV12 (4:2:0), Y V U
		case IYUV: // IYUV (4:2:0), Y U V
			copyYV12orIYUVFrame(frameSize_, frame, videoFrame_);
			break;
		case NV12: // NV12 (4:2:0)
			copyNV12Frame(frameSize_, frame, videoFrame_);
			break;
		}
	}
	checkCudaErrors(cuvidCtxUnlock(cuCtxLock_, 0));
	int err = NVEncodeFrame(encoder_, &efparams, 0, videoFrame_.data);
	assert(err==0);
}

void CudaEncoderImpl::initEncoder(double fps)
{
	int err;

	// Set codec

	static const unsigned long codecs_id[] =
	{
		NV_CODEC_TYPE_MPEG1, NV_CODEC_TYPE_MPEG2, NV_CODEC_TYPE_MPEG4, NV_CODEC_TYPE_H264, NV_CODEC_TYPE_VC1
	};
	err = NVSetCodec(encoder_, codecs_id[codec_]);
	if (err){
		infoRecorder->logTrace("[CudaEncoderImpl]: StsNotImplemented, codec format is not supported.\n");
	}

	// Set default params

	err = NVSetDefaultParam(encoder_);
	assert( err == 0 );

	// Set some common params

	int inputSize[] = { frameSize_.width, frameSize_.height };
	err = NVSetParamValue(encoder_, NVVE_IN_SIZE, &inputSize);
	assert( err == 0 );
	err = NVSetParamValue(encoder_, NVVE_OUT_SIZE, &inputSize);
	assert( err == 0 );

	int aspectRatio[] = { frameSize_.width, frameSize_.height, ASPECT_RATIO_DAR };
	err = NVSetParamValue(encoder_, NVVE_ASPECT_RATIO, &aspectRatio);
	assert( err == 0 );

	// FPS

	int frame_rate = static_cast<int>(fps + 0.5);
	int frame_rate_base = 1;
	while (fabs(static_cast<double>(frame_rate) / frame_rate_base) - fps > 0.001)
	{
		frame_rate_base *= 10;
		frame_rate = static_cast<int>(fps*frame_rate_base + 0.5);
	}
	int FrameRate[] = { frame_rate, frame_rate_base };
	err = NVSetParamValue(encoder_, NVVE_FRAME_RATE, &FrameRate);
	assert( err == 0 );

	// Select device for encoding
	
	int gpuID = 0;
	cudaGetDevice(&gpuID);

	err = NVSetParamValue(encoder_, NVVE_FORCE_GPU_SELECTION, &gpuID);
	assert( err == 0 );
}

void CudaEncoderImpl::setEncodeParams(const NVEncoderParams& params)
{
#if 0
	int err;

	int P_Interval = params.P_Interval;
	err = NVSetParamValue(encoder_, NVVE_P_INTERVAL, &P_Interval);
	assert( err == 0 );

	int IDR_Period = params.IDR_Period;
	err = NVSetParamValue(encoder_, NVVE_IDR_PERIOD, &IDR_Period);
	assert( err == 0 );

	int DynamicGOP = params.DynamicGOP;
	err = NVSetParamValue(encoder_, NVVE_DYNAMIC_GOP, &DynamicGOP);
	assert( err == 0 );

	NVVE_RateCtrlType RCType = static_cast<NVVE_RateCtrlType>(params.RCType);
	err = NVSetParamValue(encoder_, NVVE_RC_TYPE, &RCType);
	assert( err == 0 );

	int AvgBitrate = params.AvgBitrate;
	err = NVSetParamValue(encoder_, NVVE_AVG_BITRATE, &AvgBitrate);
	assert( err == 0 );

	int PeakBitrate = params.PeakBitrate;
	err = NVSetParamValue(encoder_, NVVE_PEAK_BITRATE, &PeakBitrate);
	assert( err == 0 );

	int QP_Level_Intra = params.QP_Level_Intra;
	err = NVSetParamValue(encoder_, NVVE_QP_LEVEL_INTRA, &QP_Level_Intra);
	assert( err == 0 );

	int QP_Level_InterP = params.QP_Level_InterP;
	err = NVSetParamValue(encoder_, NVVE_QP_LEVEL_INTER_P, &QP_Level_InterP);
	assert( err == 0 );

	int QP_Level_InterB = params.QP_Level_InterB;
	err = NVSetParamValue(encoder_, NVVE_QP_LEVEL_INTER_B, &QP_Level_InterB);
	assert( err == 0 );

	int DeblockMode = params.DeblockMode;
	err = NVSetParamValue(encoder_, NVVE_DEBLOCK_MODE, &DeblockMode);
	assert( err == 0 );

	int ProfileLevel = params.ProfileLevel;
	err = NVSetParamValue(encoder_, NVVE_PROFILE_LEVEL, &ProfileLevel);
	assert( err == 0 );

	int ForceIntra = params.ForceIntra;
	err = NVSetParamValue(encoder_, NVVE_FORCE_INTRA, &ForceIntra);
	assert( err == 0 );

	int ForceIDR = params.ForceIDR;
	err = NVSetParamValue(encoder_, NVVE_FORCE_IDR, &ForceIDR);
	assert( err == 0 );

	int ClearStat = params.ClearStat;
	err = NVSetParamValue(encoder_, NVVE_CLEAR_STAT, &ClearStat);
	assert( err == 0 );

	NVVE_DI_MODE DIMode = static_cast<NVVE_DI_MODE>(params.DIMode);
	err = NVSetParamValue(encoder_, NVVE_SET_DEINTERLACE, &DIMode);
	assert( err == 0 );

	if (params.Presets != -1)
	{
		NVVE_PRESETS_TARGET Presets = static_cast<NVVE_PRESETS_TARGET>(params.Presets);
		err = NVSetParamValue(encoder_, NVVE_PRESETS, &Presets);
		assert( err == 0 );
	}

	int DisableCabac = params.DisableCabac;
	err = NVSetParamValue(encoder_, NVVE_DISABLE_CABAC, &DisableCabac);
	assert( err == 0 );

	int NaluFramingType = params.NaluFramingType;
	err = NVSetParamValue(encoder_, NVVE_CONFIGURE_NALU_FRAMING_TYPE, &NaluFramingType);
	assert( err == 0 );

	int DisableSPSPPS = params.DisableSPSPPS;
	err = NVSetParamValue(encoder_, NVVE_DISABLE_SPS_PPS, &DisableSPSPPS);
	assert( err == 0 );
#endif
}

unsigned char * CudaEncoderImpl::HANDLEAcquireBitStream(int * pBufferSize, void * pUserData){
	CudaEncoderImpl * thiz = static_cast<CudaEncoderImpl *>(pUserData);
	return thiz->callback_->acquireBitStream(pBufferSize);
	//return NULL;
}

void CudaEncoderImpl::HandleReleaseBitStream(int nBytesInBuffer, unsigned char * cb, void * pUserData){
	CudaEncoderImpl * thiz = static_cast<CudaEncoderImpl *>(pUserData);
	thiz->callback_->releaseBitStream(cb, nBytesInBuffer);
}

void CudaEncoderImpl::HandleOnBeginFrame(const NVVE_BeginFrameInfo * pbfi, void * pUserData){
	CudaEncoderImpl * thiz = static_cast<CudaEncoderImpl *>(pUserData);
	thiz->callback_->onBeginFrame(pbfi->nFrameNumber, static_cast<EncoderCallback::PicType>(pbfi->nPicType));
}
void CudaEncoderImpl::HandleOnEndFrame(const NVVE_EndFrameInfo * pefi, void * pUserData){
	CudaEncoderImpl * thiz = static_cast<CudaEncoderImpl *>(pUserData);
	thiz->callback_->onEndFrame(pefi->nFrameNumber, static_cast<EncoderCallback::PicType>(pefi->nPicType));
}


/////////////////////EncoderCallbackNet ////////////////
EncoderCallbackNet::EncoderCallbackNet(){

}

EncoderCallbackNet::~EncoderCallbackNet(){

}

unsigned char * EncoderCallbackNet::acquireBitStream(int * bufferSize){
	*bufferSize = static_cast<int>(sizeof(buf_));
	return buf_;
}

void EncoderCallbackNet::releaseBitStream(unsigned char * data, int size){
	// write to file ?
}

void EncoderCallbackNet::onBeginFrame(int _frameNumber, PicType picType){
	(void )_frameNumber;
	isKeyFrame_ = (picType == IFRAME);
}

void EncoderCallbackNet::onEndFrame(int frameNumber, PicType picType){
	(void) frameNumber;
	(void) picType;
}