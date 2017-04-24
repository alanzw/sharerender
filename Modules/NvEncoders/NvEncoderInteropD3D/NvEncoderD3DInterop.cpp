#define WINDOWS_LEAN_AND_MEAN

#pragma push_macro("_WINSOCKAPI_")
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif


#include <string>
//#include <InitGuid.h>

#include "NvEncoderD3DInterop.h"
#include "../common/nvUtils.h"

//only for test
#include "../common/nvFileIO.h"
#include "../../LibCore/InfoRecorder.h"

#pragma comment(lib,"dxva2.lib")
namespace cg{
	namespace nvenc{

		const D3DFORMAT D3DFMT_NV12 = (D3DFORMAT)MAKEFOURCC('N', 'V', '1', '2');
#define BITSTREAM_BUFFER_SIZE 2 * 1024 * 1024

		// for CNvEncoderD3DInteropImpl
		CNvEncoderD3DInteropImpl::CNvEncoderD3DInteropImpl(){
			m_pNvHWEncoder = new CNvHWEncoder();
			m_pD3DEx = NULL;
			m_pD3D9Device = NULL;
			m_pDXVA2VideoProcessor = NULL;
			m_pDXVA2VideoProcessServices = NULL;

			m_uEncodeBufferCount = 0;

			memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));
			memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));

			memset(&m_Brightness, 0, sizeof(m_Brightness));
			memset(&m_Contrast, 0, sizeof(m_Contrast));
			memset(&m_Hue, 0, sizeof(m_Hue));
			memset(&m_Saturation, 0, sizeof(m_Saturation));

			width = 0, height = 0;
			cond = NULL; condMutex = NULL;
			surfaceSource = NULL;
		}

		// create new CNvEncoderD3DInteropImpl object with d3d device, width and height
		CNvEncoderD3DInteropImpl::CNvEncoderD3DInteropImpl(IDirect3DDevice9 * device, int _height, int _width, pipeline * pipe_){
			m_pNvHWEncoder = new CNvHWEncoder();
			m_pD3DEx = NULL;
			m_pD3D9Device = device;
			m_pDXVA2VideoProcessor = NULL;
			m_pDXVA2VideoProcessServices = NULL;

			m_uEncodeBufferCount = 0;
			memset(&m_stEOSOutputBfr, 0, sizeof(m_stEOSOutputBfr));
			memset(&m_stEncodeBuffer, 0, sizeof(m_stEncodeBuffer));

			memset(&m_Brightness, 0, sizeof(m_Brightness));
			memset(&m_Contrast, 0, sizeof(m_Contrast));
			memset(&m_Hue, 0, sizeof(m_Hue));
			memset(&m_Saturation, 0, sizeof(m_Saturation));

			width = _width, height = _height;
			cond = NULL, condMutex = NULL;
			surfaceSource = pipe_;
		}

		CNvEncoderD3DInteropImpl::~CNvEncoderD3DInteropImpl(){
			if (m_pNvHWEncoder)
			{
				delete m_pNvHWEncoder;
				m_pNvHWEncoder = NULL;
			}
		}

		NVENCSTATUS CNvEncoderD3DInteropImpl::InitD3D9(unsigned int deviceID /* = 0 */, IDirect3DDevice9 * device /* = NULL */){
			HRESULT hr = S_OK;
			D3DPRESENT_PARAMETERS d3dpp;
			D3DADAPTER_IDENTIFIER9 adapterId;
			unsigned int iAdapter = NULL;
			// use the given device if not NULL
			if(device == NULL){
				// create new one use deviceID
				Direct3DCreate9Ex(D3D_SDK_VERSION, &m_pD3DEx);
				if (deviceID >= m_pD3DEx->GetAdapterCount())
				{
					PRINTERR("nvEncoder (-deviceID=%d) is not a valid GPU device. Headless video devices will not be detected.  <<\n\n", deviceID);
					return NV_ENC_ERR_INVALID_ENCODERDEVICE;
				}
				hr = m_pD3DEx->GetAdapterIdentifier(deviceID, 0, &adapterId);
				if (hr != S_OK)
				{
					PRINTERR("nvEncoder (-deviceID=%d) is not a valid GPU device. <<\n\n", deviceID);
					return NV_ENC_ERR_INVALID_ENCODERDEVICE;
				}

				// Create the Direct3D9 device and the swap chain. In this example, the swap
				// chain is the same size as the current display mode. The format is RGB-32.
				ZeroMemory(&d3dpp, sizeof(d3dpp));
				d3dpp.Windowed = true;
				d3dpp.BackBufferFormat = D3DFMT_X8R8G8B8;
				d3dpp.BackBufferWidth = 640;
				d3dpp.BackBufferHeight = 480;
				d3dpp.BackBufferCount = 1;
				d3dpp.SwapEffect = D3DSWAPEFFECT_COPY;
				d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
				d3dpp.Flags = D3DPRESENTFLAG_VIDEO;
				DWORD dwBehaviorFlags = D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING;
				hr = m_pD3DEx->CreateDevice(deviceID,
					D3DDEVTYPE_HAL,
					NULL,
					dwBehaviorFlags,
					&d3dpp,
					&m_pD3D9Device);

			}
			else{
				m_pD3D9Device = device;
			}

			// create NV12 convert service
			hr = DXVA2CreateVideoService(m_pD3D9Device, IID_PPV_ARGS(&m_pDXVA2VideoProcessServices));

			DXVA2_VideoDesc vd;
			unsigned int uGuidCount = 0;
			GUID *pGuids = NULL;
			bool bVPGuidAvailable = false;
			hr = m_pDXVA2VideoProcessServices->GetVideoProcessorDeviceGuids(&vd, &uGuidCount, &pGuids);
			for (unsigned int i = 0; i < uGuidCount; i++)
			{
				if (pGuids[i] == DXVA2_VideoProcProgressiveDevice)
				{
					bVPGuidAvailable = true;
					break;
				}
			}
			CoTaskMemFree(pGuids);
			if (!bVPGuidAvailable)
			{
				return NV_ENC_ERR_OUT_OF_MEMORY;
			}

			if (bVPGuidAvailable)
			{
				hr = m_pDXVA2VideoProcessServices->CreateVideoProcessor(DXVA2_VideoProcProgressiveDevice, &vd, D3DFMT_NV12, 0, &m_pDXVA2VideoProcessor);
				hr = m_pDXVA2VideoProcessServices->GetProcAmpRange(DXVA2_VideoProcProgressiveDevice, &vd, D3DFMT_NV12, DXVA2_ProcAmp_Brightness, &m_Brightness);
				hr = m_pDXVA2VideoProcessServices->GetProcAmpRange(DXVA2_VideoProcProgressiveDevice, &vd, D3DFMT_NV12, DXVA2_ProcAmp_Contrast, &m_Contrast);
				hr = m_pDXVA2VideoProcessServices->GetProcAmpRange(DXVA2_VideoProcProgressiveDevice, &vd, D3DFMT_NV12, DXVA2_ProcAmp_Hue, &m_Hue);
				hr = m_pDXVA2VideoProcessServices->GetProcAmpRange(DXVA2_VideoProcProgressiveDevice, &vd, D3DFMT_NV12, DXVA2_ProcAmp_Saturation, &m_Saturation);
			}

			return NV_ENC_SUCCESS;
		}

		// allocate io buffers for D3D
		NVENCSTATUS CNvEncoderD3DInteropImpl::AllocateIOBuffers(unsigned int dwInputWidth, unsigned int dwInputHeight){
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;
			IDirect3DSurface9 *pVPSurfaces[16];

			m_EncodeBufferQueue.Initialize(m_stEncodeBuffer, m_uEncodeBufferCount);
			HRESULT hr = m_pDXVA2VideoProcessServices->CreateSurface(dwInputWidth, dwInputHeight, m_uEncodeBufferCount - 1, D3DFMT_NV12, D3DPOOL_DEFAULT, 0, DXVA2_VideoProcessorRenderTarget, &pVPSurfaces[0], NULL);
			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++)
			{
				nvStatus = m_pNvHWEncoder->NvEncRegisterResource(NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, (void*)pVPSurfaces[i],
					dwInputWidth, dwInputHeight, m_stEncodeBuffer[i].stInputBfr.uNV12Stride, &m_stEncodeBuffer[i].stInputBfr.nvRegisteredResource);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;

				m_stEncodeBuffer[i].stInputBfr.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;
				m_stEncodeBuffer[i].stInputBfr.dwWidth = dwInputWidth;
				m_stEncodeBuffer[i].stInputBfr.dwHeight = dwInputHeight;
				m_stEncodeBuffer[i].stInputBfr.pNV12Surface = pVPSurfaces[i];

				nvStatus = m_pNvHWEncoder->NvEncCreateBitstreamBuffer(BITSTREAM_BUFFER_SIZE, &m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;
				m_stEncodeBuffer[i].stOutputBfr.dwBitstreamBufferSize = BITSTREAM_BUFFER_SIZE;

				nvStatus = m_pNvHWEncoder->NvEncRegisterAsyncEvent(&m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				if (nvStatus != NV_ENC_SUCCESS)
					return nvStatus;
				m_stEncodeBuffer[i].stOutputBfr.bWaitOnEvent = true;
			}

			m_stEOSOutputBfr.bEOSFlag = TRUE;
			nvStatus = m_pNvHWEncoder->NvEncRegisterAsyncEvent(&m_stEOSOutputBfr.hOutputEvent);
			if (nvStatus != NV_ENC_SUCCESS)
				return nvStatus;

			return NV_ENC_SUCCESS;
		}

		// release the IO buffers
		NVENCSTATUS CNvEncoderD3DInteropImpl::ReleaseIOBuffers(){
			for (uint32_t i = 0; i < m_uEncodeBufferCount; i++){
				if (m_stEncodeBuffer[i].stInputBfr.pNV12Surface){
					m_stEncodeBuffer[i].stInputBfr.pNV12Surface->Release();
				}

				m_pNvHWEncoder->NvEncDestroyBitstreamBuffer(m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer);
				m_stEncodeBuffer[i].stOutputBfr.hBitstreamBuffer = NULL;

				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				CloseHandle(m_stEncodeBuffer[i].stOutputBfr.hOutputEvent);
				m_stEncodeBuffer[i].stOutputBfr.hOutputEvent = NULL;
			}

			if (m_stEOSOutputBfr.hOutputEvent){
				m_pNvHWEncoder->NvEncUnregisterAsyncEvent(m_stEOSOutputBfr.hOutputEvent);
				CloseHandle(m_stEOSOutputBfr.hOutputEvent);
				m_stEOSOutputBfr.hOutputEvent  = NULL;
			}

			return NV_ENC_SUCCESS;
		}

		// flush encoder
		NVENCSTATUS CNvEncoderD3DInteropImpl::FlushEncoder(){
			NVENCSTATUS nvStatus = m_pNvHWEncoder->NvEncFlushEncoderQueue(m_stEOSOutputBfr.hOutputEvent);
			if (nvStatus != NV_ENC_SUCCESS)
			{
				assert(0);
				return nvStatus;
			}

			EncodeBuffer *pEncodeBuffer = m_EncodeBufferQueue.GetPending();
			while(pEncodeBuffer)
			{
				m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, writer);
				pEncodeBuffer = m_EncodeBufferQueue.GetPending();
				// UnMap the input buffer after frame is done
				if (pEncodeBuffer && pEncodeBuffer->stInputBfr.hInputSurface)
				{
					nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
					pEncodeBuffer->stInputBfr.hInputSurface = NULL;
				}
			}

			if (WaitForSingleObject(m_stEOSOutputBfr.hOutputEvent, 500) != WAIT_OBJECT_0)
			{
				assert(0);
				nvStatus = NV_ENC_ERR_GENERIC;
			}

			return nvStatus;

		}

		NVENCSTATUS CNvEncoderD3DInteropImpl::Deinitialize(){
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			ReleaseIOBuffers();
			nvStatus = m_pNvHWEncoder->NvEncDestroyEncoder();

			if (m_pDXVA2VideoProcessor)
			{
				m_pDXVA2VideoProcessor->Release();
				m_pDXVA2VideoProcessor = NULL;
			}

			if (m_pDXVA2VideoProcessServices)
			{
				m_pDXVA2VideoProcessServices->Release();
				m_pDXVA2VideoProcessServices = NULL;
			}

			if (m_pD3D9Device)
			{
				m_pD3D9Device->Release();
				m_pD3D9Device = NULL;
			}

			if (m_pD3DEx)
			{
				m_pD3DEx->Release();
				m_pD3DEx = NULL;
			}

			return nvStatus;
		}

		NVENCSTATUS CNvEncoderD3DInteropImpl::ConverRGBToNV12(IDirect3DSurface9 *pSrcRGB, IDirect3DSurface9 * pDstNV12, uint32_t uWidth, uint32_t uHeight){

			DXVA2_VideoProcessBltParams vpblt;
			DXVA2_VideoSample vs;

			RECT srcRect = { 0, 0, uWidth, uHeight };
			RECT dstRect = { 0, 0, uWidth, uHeight };
			// Input
			memset(&vs, 0, sizeof(vs));
			vs.PlanarAlpha.ll = 0x10000;
			vs.SrcSurface = pSrcRGB;
			vs.SrcRect = srcRect;
			vs.DstRect = dstRect;
			vs.SampleFormat.SampleFormat = DXVA2_SampleProgressiveFrame;
			vs.SampleFormat.VideoChromaSubsampling = DXVA2_VideoChromaSubsampling_MPEG2;
			vs.SampleFormat.NominalRange = DXVA2_NominalRange_0_255;
			vs.SampleFormat.VideoTransferMatrix = DXVA2_VideoTransferMatrix_BT601;

			// Output
			memset(&vpblt, 0, sizeof(vpblt));
			vpblt.TargetRect = dstRect;
			vpblt.DestFormat = vs.SampleFormat;
			vpblt.DestFormat.SampleFormat = DXVA2_SampleProgressiveFrame;
			vpblt.Alpha.ll = 0x10000;
			vpblt.TargetFrame = vs.Start;
			vpblt.ProcAmpValues.Brightness = m_Brightness.DefaultValue;
			vpblt.ProcAmpValues.Contrast = m_Contrast.DefaultValue;
			vpblt.ProcAmpValues.Hue = m_Hue.DefaultValue;
			vpblt.ProcAmpValues.Saturation = m_Saturation.DefaultValue;
			vpblt.BackgroundColor.Y = 0x1000;
			vpblt.BackgroundColor.Cb = 0x8000;
			vpblt.BackgroundColor.Cr = 0x8000;
			vpblt.BackgroundColor.Alpha = 0xffff;
			HRESULT hr = m_pDXVA2VideoProcessor->VideoProcessBlt(pDstNV12, &vpblt, &vs, 1, NULL);
			if(FAILED(hr)){
				cg::core::infoRecorder->logError("[CNvEncoderD3DInteropImpl]: video process blt failed with:%d.\n", hr);
			}
			return NV_ENC_SUCCESS;
		}

		// load frame
		struct pooldata * CNvEncoderD3DInteropImpl::loadFrame(){
			pooldata * data = NULL;
			pipeline * pipe = surfaceSource;

			data = pipe->load_data();
			if(data == NULL){
				pipe->wait(cond, condMutex);
				if((data = pipe->load_data()) == NULL){
					// failed
					cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: recv unexpected NULL frame.\n");
					return NULL;
				}
			}
			cg::core::infoRecorder->logError("[CNvEncoderD3DInteropImpl]: get frame:%p.\n", data);

			return data;
		}

		// release the pool data
		void CNvEncoderD3DInteropImpl::releaseData(pooldata * data){
			if(surfaceSource){
				surfaceSource->release_data(data);
			}
		}

		/////////////////// the thread part   //////////////////////
		BOOL CNvEncoderD3DInteropImpl::onThreadStart(){
			// init the thread when started
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			memset(&encodeConfig, 0, sizeof(EncodeConfig));

			encodeConfig.endFrameIdx = INT_MAX;
			encodeConfig.bitrate = 5000000;
			encodeConfig.rcMode = NV_ENC_PARAMS_RC_CONSTQP;
			encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH;
			encodeConfig.deviceType = 0;
			encodeConfig.codec = NV_ENC_H264;
			encodeConfig.fps = 30;
			encodeConfig.qp = 28;
			encodeConfig.presetGUID = NV_ENC_PRESET_DEFAULT_GUID;
			encodeConfig.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

			// fill the size
			encodeConfig.width = (width == 0 ? 800 : width);
			encodeConfig.height = (height == 0 ? 600 : height);

			if(encodeConfig.width == 0 || encodeConfig.height == 0){
				// error 
				cg::core::infoRecorder->logError("[CNvEncoderCudaInteropImpl]: onThreadStart(), encoder config is invalid.\n");
				return FALSE;
			}

			// init d3d
			nvStatus = InitD3D9(0, m_pD3D9Device);
			if(nvStatus!= NV_ENC_SUCCESS){
				return FALSE;   // failed
			}
			nvStatus = m_pNvHWEncoder->Initialize((void*)m_pD3D9Device, NV_ENC_DEVICE_TYPE_DIRECTX);
			if (nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderD3DInteropImpl]: initialize failed with:%d.\n", nvStatus);
				return FALSE;  // failed
			}

			encodeConfig.presetGUID = m_pNvHWEncoder->GetPresetGUID(encodeConfig.encoderPreset, encodeConfig.codec);

			PrintEncodeConfig(encodeConfig, true, false);
			nvStatus = m_pNvHWEncoder->CreateEncoder(&encodeConfig);
			if (nvStatus != NV_ENC_SUCCESS)
				return FALSE; // failed

			m_uEncodeBufferCount = encodeConfig.numB + 4;

			nvStatus = AllocateIOBuffers(encodeConfig.width, encodeConfig.height);
			if (nvStatus != NV_ENC_SUCCESS)
				return FALSE; // failed

			// register the event to source pipeline
			cond = CreateEvent(NULL, FALSE, FALSE, NULL);
			condMutex = CreateMutex(NULL, FALSE, NULL);

			surfaceSource->client_register(ccg_gettid(), cond);

			// SUECCESS
			return TRUE;
		}
		// loop run, the main body
		BOOL CNvEncoderD3DInteropImpl::run(){

			//DebugBreak();

			struct pooldata * data = NULL;
			if(!(data = loadFrame())){
				cg::core::infoRecorder->logError("[CNvEncoderD3DInteropImpl]: load frame from %s pipeline failed.\n", surfaceSource->name());
				return FALSE;
			}

			// load surface
			EncodeBuffer * pEncodeBuffer = NULL;
			NVENCSTATUS nvStatus = NV_ENC_SUCCESS;

			// get the device buffer
			pEncodeBuffer = m_EncodeBufferQueue.GetAvailable();
			if(!pEncodeBuffer){
				pEncodeBuffer = m_EncodeBufferQueue.GetPending();
				m_pNvHWEncoder->ProcessOutput(pEncodeBuffer, writer);
				// unmap the input buffer after frame done
				nvStatus = m_pNvHWEncoder->NvEncUnmapInputResource(pEncodeBuffer->stInputBfr.hInputSurface);
				pEncodeBuffer->stInputBfr.hInputSurface = NULL;
			}

			SurfaceFrame * frame = (SurfaceFrame *)data->ptr;
			IDirect3DSurface9 * pSrcRGB = frame->dxSurface->d9Surface;
			ConverRGBToNV12(pSrcRGB, pEncodeBuffer->stInputBfr.pNV12Surface, encodeConfig.width, encodeConfig.height);

			releaseData(data);
			// to encode
			nvStatus = m_pNvHWEncoder->NvEncMapInputResource(pEncodeBuffer->stInputBfr.nvRegisteredResource, &pEncodeBuffer->stInputBfr.hInputSurface);
			if(nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderD3DInteropImpl: failed to map input buffer %p.\n", pEncodeBuffer->stInputBfr.hInputSurface);
				return FALSE;
			}
			cg::core::infoRecorder->logError("[CNvEncoderD3DInteropImpl]: encode.\n");
			nvStatus = m_pNvHWEncoder->NvEncEncodeFrame(pEncodeBuffer, NULL, encodeConfig.width, encodeConfig.height);
			if(nvStatus != NV_ENC_SUCCESS){
				cg::core::infoRecorder->logError("[CNvEncoderD3DInteropImpl]: encode frame failed with:%d.\n", nvStatus);
			}
			return TRUE;
		}
		// thread msg, now , nothing to do
		void CNvEncoderD3DInteropImpl::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
			// do nothing
		}
		// on quit, release the resources
		void CNvEncoderD3DInteropImpl::onQuit(){
			FlushEncoder();

			// de-initialize
			Deinitialize();
		}
	}
}