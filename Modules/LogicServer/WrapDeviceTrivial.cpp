#include "CommandServerSet.h"
#include "WrapDirect3d9.h"
#include "WrapDirect3ddevice9.h"
#include "WrapDirect3dvertexbuffer9.h"
#include "WrapDirect3dindexbuffer9.h"
#include "WrapDirect3dvertexdeclaration9.h"
#include "WrapDirect3dvertexshader9.h"
#include "WrapDirect3dpixelshader9.h"
#include "WrapDirect3dtexture9.h"
#include "WrapDirect3dstateblock9.h"
#include "WrapDirect3dcubetexture9.h"
#include "WrapDirect3dswapchain9.h"
#include "WrapDirect3dsurface9.h"
#include "WrapDirect3dvolumetexture9.h"
#include "../LibCore/Opcode.h"

#include "../libCore/CmdHelper.h"
#include "../libCore/InfoRecorder.h"
#include "../VideoGen/generator.h"
#include "KeyboardHook.h"
#include "../LibCore/TimeTool.h"
#include "../LibCore/BmpFormat.h"


#include "../LibInput/Controller.h"


#include "GameClient.h"

#include <MMSystem.h>

//#define SCRATCH_MEMO
#define DELAY_TO_DRAW


bool tex_send[4024] = {0};
double time_total = 0.0f;
int frame_all_count = 0;

StateBlockRecorder * stateRecorder = NULL;

WrapperDirect3DDevice9::WrapperDirect3DDevice9(IDirect3DDevice9* device, int _id): m_device(device), IdentifierBase(_id) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("I am in WrapperDirect3DDevice9\n");
#endif
	m_list.AddMember(device, this);

	cur_decl_ = NULL;
	cur_ib_ = NULL;
	is_even_frame_ = 1;

	creationFlag = 0;
	updateFlag = 0;
	stable = true;
	deviceHelper = NULL;

	pPresentTimer = new cg::core::PTimer();
	//pPresentTimer->Start();

}
WrapperDirect3DDevice9::~WrapperDirect3DDevice9(){
	if(this->pPresentParameters){
		free(pPresentParameters);
		pPresentParameters = NULL;
	}
	if(deviceHelper){
		delete deviceHelper;
		deviceHelper = NULL;
	}
}
WrapperDirect3DDevice9* WrapperDirect3DDevice9::GetWrapperDevice9(IDirect3DDevice9* ptr) {
	WrapperDirect3DDevice9* ret = (WrapperDirect3DDevice9*)( m_list.GetDataPtr(ptr) );
#ifdef ENBALE_DEVICE_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DDevice9::GetWrapperDevice9(), ret is NULL\n");
	}
#endif
	return ret;
}

STDMETHODIMP WrapperDirect3DDevice9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	HRESULT hr = m_device->QueryInterface(riid, ppvObj);
	*ppvObj = this;
#ifdef ENBALE_DEVICE_LOG
	if(hr == E_NOINTERFACE){
		infoRecorder->logTrace("WrapperDirect3DDevice9::QueryInterface() failed!, No Interface, RIID:%d\n",riid);
	}
	infoRecorder->logTrace("WrapperDirect3DDevice9::QueryInterface(), base_device=%d, this=%d, riid: %d, ret:%d\n", m_device, this ,riid, hr);
#endif
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DDevice9::AddRef(THIS) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::AddRef() called\n");
#endif
	refCount++;
	ULONG hr = m_device->AddRef();
	infoRecorder->logTrace("[WrapperDirect3DDevice9]: addref, m_device ref:%d, ref count:%d.\n", hr, refCount);
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DDevice9::Release(THIS) {
	ULONG hr = m_device->Release();
#ifdef ENBALE_DEVICE_LOG
#ifdef LOG_REF_COUNT
	infoRecorder->logTrace("WrapperDirect3DDevice9::Release, ref:%d.\n", hr);
#endif  // LOG_REF_COUNT1
#endif
	refCount--;
#ifdef ENABLE_DEVICE_LOG
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DDevice9]: m_device ref:%d, ref count:%d.\n", refCount, hr);
	}
#endif
	return hr;
}

/*** IDirect3DDevice9 methods ***/
STDMETHODIMP WrapperDirect3DDevice9::TestCooperativeLevel(THIS){
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::TestCooperativeLevel() called\n");
#endif
	return m_device->TestCooperativeLevel();
}

STDMETHODIMP_(UINT) WrapperDirect3DDevice9::GetAvailableTextureMem(THIS) { return m_device->GetAvailableTextureMem(); }

STDMETHODIMP WrapperDirect3DDevice9::EvictManagedResources(THIS) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::EvictManagedResources() TODO\n");
#endif
	return m_device->EvictManagedResources();
}

STDMETHODIMP WrapperDirect3DDevice9::GetDirect3D(THIS_ IDirect3D9** ppD3D9) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetDirect3D() called\n");
#endif
	LPDIRECT3D9 base_d3d9;
	HRESULT hr = m_device->GetDirect3D(&base_d3d9);

	*ppD3D9 = WrapperDirect3D9::GetWrapperD3D9(base_d3d9);
	return hr;
}
STDMETHODIMP WrapperDirect3DDevice9::GetDeviceCaps(THIS_ D3DCAPS9* pCaps) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetDeviceCaps() called\n");
#endif
	HRESULT hr = m_device->GetDeviceCaps(pCaps);

	infoRecorder->logTrace("WrapperDirect3DDevice9::GetDeviceCaps() called, max height: %d, max width:%d\n", pCaps->MaxTextureHeight, pCaps->MaxTextureWidth);
	return hr;
}
STDMETHODIMP WrapperDirect3DDevice9::GetDisplayMode(THIS_ UINT iSwapChain,D3DDISPLAYMODE* pMode) { 
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetDisplayMode() called\n");
#endif
	return m_device->GetDisplayMode(iSwapChain, pMode); }
STDMETHODIMP WrapperDirect3DDevice9::GetCreationParameters(THIS_ D3DDEVICE_CREATION_PARAMETERS *pParameters) { 
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetCreationParameters() TODO\n");
#endif
	return m_device->GetCreationParameters(pParameters); }

STDMETHODIMP WrapperDirect3DDevice9::SetCursorProperties(THIS_ UINT XHotSpot,UINT YHotSpot,IDirect3DSurface9* pCursorBitmap) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetCursorProperties() TODO\n");
#endif
	return m_device->SetCursorProperties(XHotSpot, YHotSpot, pCursorBitmap);
}

STDMETHODIMP_(void) WrapperDirect3DDevice9::SetCursorPosition(THIS_ int X,int Y,DWORD Flags) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetCursorPosition() TODO\n");
#endif
	return m_device->SetCursorPosition(X, Y, Flags);
}

STDMETHODIMP_(BOOL) WrapperDirect3DDevice9::ShowCursor(THIS_ BOOL bShow) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::ShowCursor() TODO\n");
#endif
	return m_device->ShowCursor(bShow);
}

STDMETHODIMP WrapperDirect3DDevice9::CreateAdditionalSwapChain(THIS_ D3DPRESENT_PARAMETERS* pPresentationParameters,IDirect3DSwapChain9** pSwapChain) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateAdditionalSwapChain() TODO\n");
#endif

	return m_device->CreateAdditionalSwapChain(pPresentationParameters, pSwapChain);
}

// this function is to create a new swap chain ? (yes, confirmed)
STDMETHODIMP WrapperDirect3DDevice9::GetSwapChain(THIS_ UINT iSwapChain,IDirect3DSwapChain9** pSwapChain) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetSwapChain(), ");
#endif
	IDirect3DSwapChain9* base_chain = NULL;
	HRESULT hr = m_device->GetSwapChain(iSwapChain, &base_chain);

	WrapperDirect3DSwapChain9* chain = WrapperDirect3DSwapChain9::GetWrapperSwapChain9(base_chain);

	if(chain == NULL) {
		// send command
		csSet->beginCommand(GetSwapChain_Opcode, id);
		csSet->writeInt(WrapperDirect3DSwapChain9::ins_count);
		csSet->writeUInt(iSwapChain);
		csSet->endCommand();
		csSet->setCreation(chain->creationFlag);

		chain = new WrapperDirect3DSwapChain9(base_chain, WrapperDirect3DSwapChain9::ins_count++);
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(chain);
#endif
		// set all the current clients' creation flag
	}else{
		// the swap chain exist, check all clients' creation flag
		csSet->checkObj(dynamic_cast<IdentifierBase *>(chain));
	}
	*pSwapChain = dynamic_cast<IDirect3DSwapChain9 *>(chain);
	return hr;
}

STDMETHODIMP_(UINT) WrapperDirect3DDevice9::GetNumberOfSwapChains(THIS) { return m_device->GetNumberOfSwapChains(); }

// reset is an important function in multi-windowed games, be careful the window handle in D3DPRESENT_PARAMETERS
STDMETHODIMP WrapperDirect3DDevice9::Reset(THIS_ D3DPRESENT_PARAMETERS* pPresentationParameter) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::Reset().\n");
#endif
	infoRecorder->logError("[WrapperDirect3DDevice9]::Reset(), backbuffer(%d x %d).\n", pPresentationParameter->BackBufferWidth, pPresentationParameter->BackBufferHeight);

	csSet->beginCommand(Reset_Opcode, id);
	csSet->writeByteArr((char *)pPresentationParameter, sizeof(D3DPRESENT_PARAMETERS));
	csSet->endCommand();
	// copy to store
	memcpy(this->pPresentParameters, pPresentationParameter, sizeof(D3DPRESENT_PARAMETERS));

	return m_device->Reset(pPresentationParameter);
}


extern int serverInputArrive;
static double last_time = 0.0f;
static float last_frame_time = 0.0f;
static bool init = false;

STDMETHODIMP WrapperDirect3DDevice9::Present(THIS_ CONST RECT* pSourceRect, CONST RECT* pDestRect, HWND hDestWindowOverride, CONST RGNDATA* pDirtyRegion) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::Present(), source %d, dst %d, wind %d, rgbdata %d\n", pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion);
#endif
	bool synSign = false, tm1 = false;
	//DebugBreak();

	keyCmdHelper->lock();
	synSign = keyCmdHelper->isSynSigned();
	tm1 = keyCmdHelper->isF10Pressed();
	keyCmdHelper->unlock();

	unsigned char flag = 0;
	static unsigned char valueTag = 0;
	if(synSign){
		//if (tm && presented > 1){
		DWORD tick_end = GetTickCount();
		DWORD tick_start = keyCmdHelper->getSynSignedTime();

#if 1
		extern BTimer * ctrlTimer;
		int renderCost = 0;
		if(ctrlTimer){
			renderCost = ctrlTimer->Stop();
		}
		else{
			ctrlTimer = new PTimer();
		}
		//infoRecorder->logError("[Device]: render cost: %f.\n", renderCost * 1000.0 / ctrlTimer->getFreq());

		DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();
		if(delayRecorder->isInputArrive()){
			delayRecorder->renderEnd();
			infoRecorder->logError("[Delay]: (system + render): %f %f\n", delayRecorder->getSystemProcessDelay(), delayRecorder->getRenderDelay());
		}

#endif
		//infoRecorder->logError("deal input total use:%d \tqueue time:%d \tsystem to now:%d\n", tick_end - serverInputArrive, tick_start - serverInputArrive, tick_end - tick_start);

		keyCmdHelper->lock();
		keyCmdHelper->setSynSigin(false);
		keyCmdHelper->unlock();

		flag |= 1;
		synSign = 0;
		valueTag++;
	}
	if (tm1){
		keyCmdHelper->lock();
		keyCmdHelper->setF10Pressed(false);
		keyCmdHelper->unlock();
		flag |= 2;
		
	}
	if (flag){
		// send command
		csSet->beginCommand(NULLINSTRUCT_Opcode, id);
		csSet->writeUChar(flag);
		csSet->writeUChar(valueTag);
		csSet->endCommand();
	}

	static unsigned int tags = 0;
	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(Present_Opcode, id);
		csSet->writeInt(0);
		csSet->writeInt(0);
		csSet->writeInt(0);
		csSet->writeInt(0);
		csSet->writeUInt(tags++);
		csSet->endCommand();    
	}
	// here may change the context
	csSet->commit();

	// deal with the initializer
	Initializer * initializer = Initializer::GetInitializer();
	// initializer contains the check for Device
	if(initializer){
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logError("[Present]: push initializer, should only show ONCE.\n");
#endif // ENABLE_DEVICE_LOG
		csSet->checkObj(dynamic_cast<IdentifierBase *>(initializer));
	}else{
		// context switch is done, set device's creation
		csSet->checkObj(dynamic_cast<IdentifierBase *>(this));
	}

	if(GameClient::GetGameClient() && !GameClient::IsInitialized()){
		// to notify the dis manager that GAME_READ
		//GameClient::GetGameClient()->notifyGameReady();
		SetEvent(GameClient::GetGameClient()->getClientEvent());
		GameClient::SetInitialized(true);
		cg::input::CreateClientControl(cmdCtrl->getHwnd());
		RTSPConf * conf = RTSPConf::GetRTSPConf();
		cmdCtrl->setMaxFps(conf->video_fps);

		infoRecorder->logError("[WrapperDirect3DDevice9]: to notifier GAME READY, set fps: %d.\n", cmdCtrl->getMaxFps());
	}
	else{
		infoRecorder->logTrace("[WrapperDirect3DDevice9]: no need to notify GAME READY.\n");
	}

	HRESULT hh = D3D_OK;

	//infoRecorder->logError("[WrapperDirect3dDeivce9]:Present(), key cmd helper before commit, is sending:%s, sending step:%d.\n", keyCmdHelper->isSending() ? "true" : "false", keyCmdHelper->getSendStep());

	keyCmdHelper->commit(cmdCtrl);  // deal the control from keyboard

	//infoRecorder->logError("[WrapperDirect3dDeivce9]:Present(), key cmd helper commit, is sending:%s, sending step:%d.\n", keyCmdHelper->isSending() ? "true" : "false", keyCmdHelper->getSendStep());

	cmdCtrl->commitRender();


#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]: render step:%d.\n", cmdCtrl->getFrameStep());
#endif

	if(cmdCtrl->isRender()){
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("[WrapperDirect3Device9]: present, enable render.\n");
#endif // ENABLE_DEVICE_LOG
		hh = m_device->Present(pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion);

		if(cmdCtrl->enableGenerateVideo() || cmdCtrl->enableRTSPService()){
			if(gGenerator == NULL){
				infoRecorder->logTrace("[WrapperDirect3DDevice9]: present, create generator.\n");
				gGenerator = new cg::VideoGen(cmdCtrl->getHwnd(), (void *)cmdCtrl->getDevice(), DX9, true, cmdCtrl->enableGenerateVideo(), cmdCtrl->enableRTSPService());
				gGenerator->setObject(cmdCtrl->getObjName());
			}
		}

#ifdef SERVER_SAVE_BMP

		if(flag & 1){
			// sync sign, to store the image
			char name[1024] = {0};
			IDirect3DSurface9 * rts = NULL;
			HRESULT hr = m_device->GetRenderTarget(0, &rts);
			sprintf(name, "%s%d.bmp", keyCmdHelper->getPrefix(), valueTag);
			infoRecorder->logError("[D3DDevice]: store image %s.\n", name);
			
			LPD3DXBUFFER bmpBuf = NULL;
			
			if(FAILED(D3DXSaveSurfaceToFileInMemory(&bmpBuf, D3DXIMAGE_FILEFORMAT::D3DXIFF_BMP, rts, NULL, NULL))){
				infoRecorder->logError("[D3DDevice]: save render target to file failed");
			}else{
				BYTE * data = (BYTE *)bmpBuf->GetBufferPointer();
				int size = bmpBuf->GetBufferSize();

				BITMAPFILEHEADER * pHeader = (BITMAPFILEHEADER *)data;

				BITMAPINFOHEADER * pInfo = (BITMAPINFOHEADER *)(data + sizeof(BITMAPFILEHEADER));

				int headerSize = sizeof(BITMAPINFOHEADER) + sizeof(BITMAPINFOHEADER);
				BYTE * rawData = data + pHeader->bfOffBits;

				int width =  pInfo->biWidth;
				int heigh = pInfo->biHeight;
				int orgSize = 0;
				//int elemSize = pInfo->
				if(pInfo->biCompression){
					infoRecorder->logError("[D3DDevice]: BMP is no BI_RGB, complicated.\n");
				}
				if(pInfo->biSize){
					orgSize = pInfo->biSize;
				}
				else{
					infoRecorder->logError("[D3DDevice]: use BI_RGB, set 0.\n");
					orgSize = width * pInfo->biBitCount / 8;
				}
				// get the RGB bmp file size
				int padding = 0;
				int scanlinebytes = width * 3;
				int paddedsize = width * 3 * heigh;
				
				BYTE * buffer = new BYTE[paddedsize];

				for(int j = 0; j < heigh; j++)
				for(int i = 0; i < width; i++){
					*(buffer + width * j * 3 + i * 3 + 0) = *(rawData + width * 4 * j + i * 4 + 0);
					*(buffer + width * j * 3 + i * 3 + 1) = *(rawData + width * 4 * j + i * 4 + 1);
					*(buffer + width * j * 3 + i * 3 + 2) = *(rawData + width * 4 * j + i * 4 + 2);
				}
				SaveBMP(buffer, 800, 600, paddedsize, name);

			}
			// reload and remove the alpha channel

		}
#endif

		if(gGenerator && !gGenerator->isInited()){
			// init the generator
			int imageWidth = 0, imageHeight =0 ;
			IDirect3DSurface9 * rts = NULL;
			D3DSURFACE_DESC sdesc;
			
			// acquire the resolution
			HRESULT hr = m_device->GetRenderTarget(0, &rts);
			if(FAILED(hr)){
#ifdef ENBALE_DEVICE_LOG
				infoRecorder->logError("WrapperDirect3DDevice9::Present, to init the generator failed.\n");
#endif
				return hh;
			}

			if(FAILED(rts->GetDesc(&sdesc))){
#ifdef ENBALE_DEVICE_LOG
				infoRecorder->logError("[D3DDevice]: get render target failed.\n");
#endif
				return hh;
			}
			imageWidth = sdesc.Width;
			imageHeight = sdesc.Height;

			if(rts){
				rts->Release();
				rts = NULL;
			}

			gGenerator->setResolution(imageWidth, imageHeight);
			HANDLE presentEvt = CreateEvent(NULL, FALSE, FALSE, NULL);
			gGenerator->setPresentHandle(presentEvt);
			cg::core::infoRecorder->logTrace("[Present]: encode option:%d\n", cmdCtrl->getEncoderOption());

			switch(cmdCtrl->getEncoderOption()){
			case CUDA_ENCODER:
				gGenerator->setEncoderType(CUDA_ENCODER);
				break;
			case NVENC_ENCODER:
				gGenerator->setEncoderType(NVENC_ENCODER);
				break;
			case ADAPTIVE_NVENC:
				gGenerator->setEncoderType(ADAPTIVE_NVENC);
				break;
			case ADAPTIVE_CUDA:
				gGenerator->setEncoderType(ADAPTIVE_CUDA);
				break;
			case X264_ENCODER:
			default:
				gGenerator->setEncoderType(X264_ENCODER);
				break;
			}
			gGenerator->onThreadStart();
			VideoGen::addMap((IDENTIFIER )cmdCtrl->getTaskId(), gGenerator);
		}

		if(gGenerator){
			gGenerator->setVideoTag(tags);
			SetEvent(gGenerator->getPresentEvent());
			gGenerator->run();
		}
	}else{
		infoRecorder->logTrace("[WrapperDirect3DDevice9]:Present(), not render.\n");
	}

	int frameTime = (float)pPresentTimer->Stop();
	infoRecorder->setRenderStep(cmdCtrl->getFrameStep());
	infoRecorder->onFrameEnd((float)frameTime * 1000.0 / pPresentTimer->getFreq(), true);

	/////////////////////////////////
	//limit it to max_fps
#if 1
	
	if(!init){
		last_time = timeGetTime();
		last_frame_time = last_time;
		init = true;
	}
	static double elapse_time = 0.0;
	static double frame_cnt = 0.0;
	static double fps = 0.0;
	double frame_time = 0.0f;

	timeBeginPeriod(1);
	double cur_time = (float)timeGetTime();
	timeEndPeriod(1);

	elapse_time += (cur_time - last_time);
#if 0
	frame_time = cur_time - last_frame_time;
#else
	frame_time = cur_time - last_time;
#endif
	//infoRecorder->onFrameEnd(frame_time, true);

	last_time = cur_time;
	frame_cnt++;

	//limit it to max_fps
	double to_sleep = 1000.0 / cmdCtrl->getMaxFps() * frame_cnt - elapse_time;
	infoRecorder->logTrace("Present, frame:%f, frame count:%f, to sleep:%f, timer overhead:%f.\n", frame_time, frame_cnt, to_sleep, pPresentTimer->getOverhead() * 1000.0 / pPresentTimer->getFreq());

#if 1
	int sleepTime = (int)to_sleep;
	if (to_sleep > 0 && cmdCtrl->enableRateControl()) {
		//infoRecorder->logError("Present: elapse: %f ms, ave frame time:%f sleep: %f ms.\n", elapse_time, elapse_time / frame_cnt, to_sleep);
		Sleep((DWORD)sleepTime);
	}
#if 0
	timeBeginPeriod(1);
	last_frame_time = (float)timeGetTime();
	timeEndPeriod(1);
#endif

#endif
	if (elapse_time >= 1000.0) {
		fps = frame_cnt * 1000.0 / elapse_time;
		frame_cnt = 0;
		elapse_time = 0;
	}
#endif

	pPresentTimer->Start();
	return hh;
}

// create a new IDirect3DSurface9
STDMETHODIMP WrapperDirect3DDevice9::GetBackBuffer(THIS_ UINT iSwapChain,UINT iBackBuffer,D3DBACKBUFFER_TYPE Type,IDirect3DSurface9** ppBackBuffer) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetBackBuffer() called, iSwapChain:%d, iBackBuffer:%d, Buffer type:%d\n", iSwapChain, iBackBuffer, Type);
#endif
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = m_device->GetBackBuffer(iSwapChain, iBackBuffer, Type, &base_surface);

	if(D3D_OK == hr){
		WrapperDirect3DSurface9* ret = new WrapperDirect3DSurface9(base_surface, WrapperDirect3DSurface9::ins_count++);
		// return the surface
		*ppBackBuffer = dynamic_cast<IDirect3DSurface9*>(ret);

		// send command
		csSet->beginCommand(D3DDeviceGetBackBuffer_Opcode, id);
		csSet->writeInt(ret->getId());
		csSet->writeUInt(iSwapChain);
		csSet->writeUInt(iBackBuffer);
		csSet->writeUInt(Type);
		csSet->endCommand();
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(ret);
#endif
		// set all creation flag for surface
		csSet->setCreation(ret->creationFlag);
		ret->creationCommand = D3DDeviceGetBackBuffer_Opcode;
		ret->iSwapChain = iSwapChain;
		ret->iBackBuffer = iBackBuffer;
		ret->type = Type;

	}else{
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("ERROR! WrapperDirect3DDevice::GetBackBuffer() failed!\n");
#endif
	}
	return hr;
}
STDMETHODIMP WrapperDirect3DDevice9::GetRasterStatus(THIS_ UINT iSwapChain,D3DRASTER_STATUS* pRasterStatus) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetRasterStatus() TODO\n");
#endif
	return m_device->GetRasterStatus(iSwapChain, pRasterStatus);}

STDMETHODIMP WrapperDirect3DDevice9::SetDialogBoxMode(THIS_ BOOL bEnableDialogs) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetDialogBoxMode() TODO\n");
#endif
	return m_device->SetDialogBoxMode(bEnableDialogs);
}

STDMETHODIMP_(void) WrapperDirect3DDevice9::SetGammaRamp(THIS_ UINT iSwapChain,DWORD Flags,CONST D3DGAMMARAMP* pRamp) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetGammaRamp() called\n");
#endif
	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetGammaRamp_Opcode, id);
		csSet->writeUInt(iSwapChain);
		csSet->writeUInt(Flags);
		csSet->writeByteArr((char *)pRamp, sizeof(D3DGAMMARAMP));
		csSet->endCommand();
	}

	return m_device->SetGammaRamp(iSwapChain, Flags, pRamp);
}
STDMETHODIMP_(void) WrapperDirect3DDevice9::GetGammaRamp(THIS_ UINT iSwapChain,D3DGAMMARAMP* pRamp) {return m_device->GetGammaRamp(iSwapChain, pRamp);}

STDMETHODIMP WrapperDirect3DDevice9::CreateTexture(THIS_ UINT Width,UINT Height,UINT Levels,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool,IDirect3DTexture9** ppTexture,HANDLE* pSharedHandle) {
#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logError("WrapperDirect3DDevice9::CreateTexture(), width=%d, height=%d, Usage=%d, Format:%d, Pool:%d, id:%d, device id:%d, levels:%d\n", Width, Height, Usage, Format, Pool, WrapperDirect3DTexture9::ins_count, id, Levels);
#endif

	LPDIRECT3DTEXTURE9 base_tex = NULL;
	HRESULT hr = m_device->CreateTexture(Width, Height, Levels, Usage, Format, Pool, &base_tex, pSharedHandle);	
	WrapperDirect3DTexture9 * wt = NULL;

	if(SUCCEEDED(hr)) {
		// store the texture creation information
		wt = new WrapperDirect3DTexture9(base_tex, WrapperDirect3DTexture9::ins_count++, Levels);
		*ppTexture = dynamic_cast<IDirect3DTexture9*>(wt);
		wt->Format = Format;
		wt->Levels = Levels;
		wt->Height = Height;
		wt->Width = Width;
		wt->Usage = Usage;
		wt->Pool = Pool;
		wt->setDeviceID(id);
		//if(!(Usage & D3DUSAGE_RENDERTARGET))
		wt->texHelper = new TextureHelper(Levels, Format, deviceHelper->isSupportAutoGenTex() && (Usage & D3DUSAGE_AUTOGENMIPMAP));

		// send command
		csSet->beginCommand(CreateTexture_Opcode, id);
		csSet->writeInt(wt->getId());
		csSet->writeUInt(Width);
		csSet->writeUInt(Height);
		csSet->writeUInt(Levels);
		csSet->writeUInt(Usage);
		csSet->writeUInt(Format);
		csSet->writeUInt(Pool);
		csSet->endCommand();
		// set the creation flag for texture
		csSet->setCreation(wt->creationFlag);
		infoRecorder->addCreation();


#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(wt);
#endif // INITIAL_ALL_RESOURCE

#ifdef ENBALE_DEVICE_LOG
		if(Usage & D3DUSAGE_AUTOGENMIPMAP){
			// auto gen mipmap
			infoRecorder->logError("[Device]: id:%d is AUTOGENMIPMAP.\n", wt->getId());
		}
		if(Usage & D3DUSAGE_DEPTHSTENCIL){
			// depth stencil
			infoRecorder->logError("[Device]: id:%d is DEPTHSTENCIL.\n", wt->getId());
		}
		if(Usage & D3DUSAGE_RENDERTARGET){
			// render target
			infoRecorder->logError("[Device]: id:%d is RENDERTARGET.\n", wt->getId());
		}
#endif
	}
	else {
		infoRecorder->logError("WrapperDirect3DDevice9::CreateTexture() failed for %d ", WrapperDirect3DTexture9::ins_count);
		switch(hr){
		case D3DERR_INVALIDCALL:
			infoRecorder->logError(" Error msg: D3DERR_INVALIDCALL.\n");
			break;
		case D3DERR_OUTOFVIDEOMEMORY:
			infoRecorder->logError(" Error msg: D3DERR_OUTOFVIDEOMEMORY.\n");
			break;
		case E_OUTOFMEMORY:

			infoRecorder->logError(" Error msg: E_OUTOFMEMORY.\n");
			break;
		default:
			infoRecorder->logError(" Error msg: UNKNOWN.\n");
			break;
		}
	}
	D3DUSAGE_RENDERTARGET;
	return hr;
}

//TODO
// why volume texture did not send to clients ?
STDMETHODIMP WrapperDirect3DDevice9::CreateVolumeTexture(THIS_ UINT Width,UINT Height,UINT Depth,UINT Levels,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool,IDirect3DVolumeTexture9** ppVolumeTexture,HANDLE* pSharedHandle) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("TODO: WrapperDirect3DDevice9::CreateVolumeTexture() called, Width:%d, Height:%d, Depth:%d, Levels:%d, Usage:%d, Format:%d, Pool:%d\n",Width, Height, Depth, Levels,Usage, Format, Pool);
#endif
	IDirect3DVolumeTexture9 * base_tex = NULL;
	HRESULT hr = m_device->CreateVolumeTexture(Width, Height, Depth, Levels, Usage, Format, Pool, &base_tex, pSharedHandle);
	if(SUCCEEDED(hr)){
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("WrapperDirect3DDevice9::CreateVolumeTexture() succeeded! base_tex:%d id:%d\n", base_tex,WrapperDirect3DVolumeTexture9::ins_count);
#endif

		WrapperDirect3DVolumeTexture9 * w_v_t = new WrapperDirect3DVolumeTexture9(base_tex, WrapperDirect3DVolumeTexture9::ins_count++);
		*ppVolumeTexture = dynamic_cast<IDirect3DVolumeTexture9*>(w_v_t);

#ifdef MULTI_CLIENTS
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logError("[WrapperDirect3DDevice9]: create volume texture TODO.\n");
#endif

		// send the creation command
		w_v_t->width = Width;
		w_v_t->height = Height;
		w_v_t->depth = Depth;
		w_v_t->levels = Levels;
		w_v_t->usage = Usage;
		w_v_t->format = Format;
		w_v_t->d3dpool = Pool;

		// store the parameter

#endif
	}
	else{
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("WrapperDirect3DDevice9::CreateVolumeTexture failed! ret:%d\n", hr);
#endif
	}
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::CreateCubeTexture(THIS_ UINT EdgeLength,UINT Levels,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool,IDirect3DCubeTexture9** ppCubeTexture,HANDLE* pSharedHandle) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateCubeTexture() called\n");
#endif

	IDirect3DCubeTexture9* base_cube_tex = NULL;
	WrapperDirect3DCubeTexture9 * wct = NULL;

	HRESULT hr = m_device->CreateCubeTexture(EdgeLength, Levels, Usage, Format, Pool, &base_cube_tex, pSharedHandle);

	if(SUCCEEDED(hr)) {
		wct = new WrapperDirect3DCubeTexture9(base_cube_tex, WrapperDirect3DCubeTexture9::ins_count++);
		wct->edgeLength = EdgeLength;
		wct->levels = Levels;
		wct->usage = Usage;
		wct->format = Format;
		wct->pool = Pool;

		*ppCubeTexture = dynamic_cast<IDirect3DCubeTexture9*>(wct);

		// send command
		csSet->beginCommand(CreateCubeTexture_Opcode, id);
		csSet->writeInt(wct->getId());
		csSet->writeUInt(EdgeLength);
		csSet->writeUInt(Levels);
		csSet->writeUInt(Usage);
		csSet->writeUInt(Format);
		csSet->writeUInt(Pool);
		csSet->endCommand();
		// set creation flag to all
		csSet->setCreation(wct->creationFlag);
		infoRecorder->addCreation();

		infoRecorder->logError("[WrapperDirect3DDevice9]::CreateCubeTexture(), edge len:%d, levels:%d, usage:%d, format:%d, id:%d.\n", EdgeLength, Levels, Usage, Format, wct->getId());


#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(wct);
#endif  // INITIAL_ALL_RESOURCE
	}
	else {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("WrapperDirect3DDevice9::CreateCubeTexture() failed\n");
#endif
	}

	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::CreateVertexBuffer(THIS_ UINT Length,DWORD Usage,DWORD FVF,D3DPOOL Pool,IDirect3DVertexBuffer9** ppVertexBuffer,HANDLE* pSharedHandle) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateVertextBuffer invoked! Usage:%d, FVF:%d, Pool:%d, ",Usage, FVF, Pool);
#endif
	LPDIRECT3DVERTEXBUFFER9 base_vb = NULL;
	HRESULT hr = m_device->CreateVertexBuffer(Length, Usage, FVF, Pool, &base_vb, pSharedHandle);

	WrapperDirect3DVertexBuffer9 * wvb = NULL;
	if(SUCCEEDED(hr)) {
		wvb = new WrapperDirect3DVertexBuffer9(base_vb, WrapperDirect3DVertexBuffer9::ins_count++, Length);
		wvb->Usage = Usage;
		wvb->FVF = FVF;
		wvb->Length = Length;
		wvb->Pool = Pool;
		if(wvb){
			*ppVertexBuffer = dynamic_cast<IDirect3DVertexBuffer9*>(wvb);
#ifdef ENBALE_DEVICE_LOG
			infoRecorder->logTrace("vb id:%d, size:%d, ", wvb->getId(), wvb->GetLength());
#endif
		}
		else{
#ifdef ENBALE_DEVICE_LOG
			infoRecorder->logTrace("constructor Failed, ");
#endif
		}

		// set device's creation

		// send command
		csSet->beginCommand(CreateVertexBuffer_Opcode, id);
		csSet->writeUInt(wvb->getId());
		csSet->writeUInt(Length);
		csSet->writeUInt(Usage);
		csSet->writeUInt(FVF);
		csSet->writeUInt(Pool);
		csSet->endCommand();
		//set the creation flag
		csSet->setCreation(wvb->creationFlag);
		infoRecorder->addCreation();
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(wvb);
#endif
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace(" end,id:%d\n",wvb->getId());
#endif
	}
	else {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("CreateVertexBuffer Failed\n");
#endif
	}

	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::CreateIndexBuffer(THIS_ UINT Length,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool,IDirect3DIndexBuffer9** ppIndexBuffer,HANDLE* pSharedHandle) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateIndexBuffer invoked! Usage:%d, Format:%d, Pool:%d, ",Usage, Format, Pool);
#endif
	LPDIRECT3DINDEXBUFFER9 base_ib = NULL;
	
	HRESULT hr = m_device->CreateIndexBuffer(Length, Usage, Format, Pool, &base_ib, pSharedHandle);
	WrapperDirect3DIndexBuffer9 * wib = NULL;

	if(SUCCEEDED(hr)) {
		wib = new WrapperDirect3DIndexBuffer9(base_ib, WrapperDirect3DIndexBuffer9::ins_count++, Length);
		wib->Usage = Usage;
		wib->Format = Format;
		wib->Pool = Pool;
		*ppIndexBuffer = dynamic_cast<IDirect3DIndexBuffer9*>(wib);
#ifdef ENBALE_DEVICE_LOG
		if(wib == NULL) {
			infoRecorder->logTrace("ret NULL, ");
		}
		else {

			infoRecorder->logTrace("IndexBuffer id=%d, length=%d, ", ((WrapperDirect3DIndexBuffer9*)*ppIndexBuffer)->getId(), ((WrapperDirect3DIndexBuffer9*)*ppIndexBuffer)->GetLength());
		}
#endif

		// send command
		csSet->beginCommand(CreateIndexBuffer_Opcode, id);
		csSet->writeUInt(wib->getId());
		csSet->writeUInt(Length);
		csSet->writeUInt(Usage);
		csSet->writeUInt(Format);
		csSet->writeUInt(Pool);
		csSet->endCommand();
		infoRecorder->addCreation();
		//set creation flag
		csSet->setCreation(wib->creationFlag);
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(wib);
#endif
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("CreateIndexBuffer done.\n");
#endif
	}
	else {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("CreateIndexBuffer Failed.\n");
#endif
	}

	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::CreateRenderTarget(THIS_ UINT Width, UINT Height, D3DFORMAT Format, D3DMULTISAMPLE_TYPE MultiSample, DWORD MultisampleQuality, BOOL Lockable, IDirect3DSurface9** ppSurface, HANDLE* pSharedHandle) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::CreateRenderTarget() TODO\n");
#endif
	HRESULT hr = D3D_OK;

	IDirect3DSurface9 * ret = NULL;
	hr =  m_device->CreateRenderTarget(Width, Height, Format, MultiSample, MultisampleQuality, Lockable, &ret, pSharedHandle);
#ifdef MULTI_CLIENTS


	if(SUCCEEDED(hr)){
		WrapperDirect3DSurface9 * wt = WrapperDirect3DSurface9::GetWrapperSurface9(ret);
		if(!wt){
			wt = new WrapperDirect3DSurface9(ret, WrapperDirect3DSurface9::ins_count++);
			wt->creationCommand = D3DCreateRenderTarget_Opcode;
			wt->width = Width;
			wt->height = Height;
			wt->format = Format;
			wt->multiSample = MultiSample;
			wt->multisampleQuality = MultisampleQuality;

			csSet->beginCommand(D3DCreateRenderTarget_Opcode, id);
			csSet->writeUInt(wt->getId());
			csSet->writeUInt(Width);
			csSet->writeUInt(Height);
			csSet->writeUInt(Format);
			csSet->writeUInt(MultiSample);
			csSet->writeUInt(MultisampleQuality);
			csSet->writeUInt(Lockable);
			csSet->endCommand();
			infoRecorder->addCreation();

			Initializer::PushObj(wt);
		}
		*ppSurface = dynamic_cast<IDirect3DSurface9 *>(wt);
	}
#endif

	// TODO : is the render target is created by other method?
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::CreateDepthStencilSurface(THIS_ UINT Width,UINT Height,D3DFORMAT Format,D3DMULTISAMPLE_TYPE MultiSample,DWORD MultisampleQuality,BOOL Discard,IDirect3DSurface9** ppSurface,HANDLE* pSharedHandle) {
	/*TODO*/
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::CreateDepthStencilSurface(), ");
#endif
	IDirect3DSurface9* base_surface = NULL;

	HRESULT hr = m_device->CreateDepthStencilSurface(Width, Height, Format, MultiSample, MultisampleQuality, Discard, &base_surface, pSharedHandle);
	if(FAILED(hr)){
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logError("failed with:%d.\n", hr);
#endif
		*ppSurface = NULL;
		return hr;
	}
	WrapperDirect3DSurface9 * ws = NULL;
	ws = new WrapperDirect3DSurface9(base_surface, WrapperDirect3DSurface9::ins_count++);

	// check the device's creation
	//this->checkCreation();

	// send command
	csSet->beginCommand(CreateDepthStencilSurface_Opcode, id);
	csSet->writeUInt(ws->getId());
	csSet->writeUInt(Width);
	csSet->writeUInt(Height);
	csSet->writeUInt(Format);
	csSet->writeUInt(MultiSample);
	csSet->writeUInt(MultisampleQuality);
	csSet->writeUInt(Discard);
	csSet->endCommand();
	infoRecorder->addCreation();
	// store the basic information
	ws->width = Width;
	ws->height = Height;
	ws->format = Format;
	ws->multiSample = MultiSample;
	ws->multisampleQuality = MultisampleQuality;
	ws->discard = Discard;
	ws->creationCommand = CreateDepthStencilSurface_Opcode;
	Initializer::PushObj(ws);
	// set the creation flag
	csSet->setCreation(ws->creationFlag);
	ws->setDeviceID(id);
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("with id:%d.\n", ws->getId());
#endif
	*ppSurface = dynamic_cast<IDirect3DSurface9*>(ws);
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::UpdateSurface(THIS_ IDirect3DSurface9* pSourceSurface,CONST RECT* pSourceRect,IDirect3DSurface9* pDestinationSurface,CONST POINT* pDestPoint) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::UpdateSurface() TODO\n");
#endif
	IDirect3DSurface9* psrc = pDestinationSurface ? ((WrapperDirect3DSurface9*)pSourceSurface)->GetSurface9() : NULL;
	IDirect3DSurface9* pdst = pDestinationSurface ? ((WrapperDirect3DSurface9*)pSourceSurface)->GetSurface9() : NULL;
	return m_device->UpdateSurface(psrc, pSourceRect, pdst, pDestPoint);
}
STDMETHODIMP WrapperDirect3DDevice9::UpdateTexture(THIS_ IDirect3DBaseTexture9* pSourceTexture,IDirect3DBaseTexture9* pDestinationTexture) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::UpdateTexture() TODO\n");
#endif
	return m_device->UpdateTexture(pSourceTexture, pDestinationTexture);
}

STDMETHODIMP WrapperDirect3DDevice9::GetRenderTargetData(THIS_ IDirect3DSurface9* pRenderTarget,IDirect3DSurface9* pDestSurface) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetRenderTargetData() TODO\n");
#endif

#ifdef ENBALE_DEVICE_LOG
	if(pRenderTarget == NULL || pDestSurface == NULL) {
		infoRecorder->logTrace("WrapperDirect3DDevice9::GetRenderTargetData(), surface is NULL\n");
	}
#endif

	WrapperDirect3DSurface9 * src = NULL, * dst = NULL;
	src = (WrapperDirect3DSurface9 *)pRenderTarget;
	dst = (WrapperDirect3DSurface9 *)pDestSurface;
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface::getRenderTargetData called! src id:%d, dst id:%d, src IDirect3DSurface:%d, dst IDirect3DSurface:%d\n", src->getId(), dst->getId(), src->GetSurface9(), dst->GetSurface9());
#endif

	//return m_device->GetRenderTargetData(pRenderTarget, pDestSurface);
	return m_device->GetRenderTargetData(src->GetSurface9(), dst->GetSurface9());
}
STDMETHODIMP WrapperDirect3DDevice9::GetFrontBufferData(THIS_ UINT iSwapChain,IDirect3DSurface9* pDestSurface) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetFrontBufferData() TODO\n");
#endif
	return m_device->GetFrontBufferData(iSwapChain, pDestSurface);
}
STDMETHODIMP WrapperDirect3DDevice9::StretchRect(THIS_ IDirect3DSurface9* pSourceSurface,CONST RECT* pSourceRect,IDirect3DSurface9* pDestSurface,CONST RECT* pDestRect,D3DTEXTUREFILTERTYPE Filter) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::StretchRect() TODO\n");
#endif
	return m_device->StretchRect(pSourceSurface, pSourceRect, pDestSurface, pDestRect, Filter);
}

STDMETHODIMP WrapperDirect3DDevice9::ColorFill(THIS_ IDirect3DSurface9* pSurface,CONST RECT* pRect,D3DCOLOR color) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::ColorFill() TODO\n");
#endif
	return m_device->ColorFill(pSurface, pRect, color);
}

STDMETHODIMP WrapperDirect3DDevice9::CreateOffscreenPlainSurface(THIS_ UINT Width,UINT Height,D3DFORMAT Format,D3DPOOL Pool,IDirect3DSurface9** ppSurface,HANDLE* pSharedHandle) {
	// TODO, if the surface need to be send ???
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logError("WrapperDirect3DDevice9::CreateOffscreenPlainSurface() TODO\n");
#endif
	return m_device->CreateOffscreenPlainSurface(Width, Height, Format, Pool, ppSurface, pSharedHandle);
}

STDMETHODIMP WrapperDirect3DDevice9::SetRenderTarget(THIS_ DWORD RenderTargetIndex,IDirect3DSurface9* pRenderTarget) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetRenderTarget(), ");
#endif
	WrapperDirect3DSurface9 * sur = (WrapperDirect3DSurface9*)pRenderTarget;
	if(pRenderTarget == NULL) {
		// send command
		if(keyCmdHelper->isSending()){

			csSet->beginCommand(SetRenderTarget_Opcode, id);
			csSet->writeUInt(RenderTargetIndex);
			csSet->writeInt(-1);
			csSet->writeInt(-1);
			csSet->writeInt(-1);
			csSet->endCommand();
		}

		HRESULT hh = m_device->SetRenderTarget(RenderTargetIndex, pRenderTarget);
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("NULL target.\n");
#endif
		return hh;
	}
	// check the surface's creation flag

	WrapperDirect3DSurface9 * ws = (WrapperDirect3DSurface9 *)pRenderTarget;
	//ws->checkCreation(this->GetID());
	csSet->checkObj(dynamic_cast<IdentifierBase *>(ws));
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetRenderTarget_Opcode, id);
		csSet->writeUInt(RenderTargetIndex);
		csSet->writeInt(ws->getId());
		csSet->writeInt(sur->GetTexId());
		csSet->writeInt(sur->GetLevel());
		csSet->endCommand();
	}

#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]:SetRenderTarget, surface id:%d, parent texture id: %d, level: %d. (parent tex:%p, id:%d)\n",((WrapperDirect3DSurface9*)pRenderTarget)->getId(),sur->GetTexId(), sur->GetLevel(), ws->getParentTexture(), ws->getParentTexture() ? ws->getParentTexture()->getId(): -1);
#endif

	HRESULT hr = m_device->SetRenderTarget(RenderTargetIndex, ((WrapperDirect3DSurface9*)pRenderTarget)->GetSurface9());
	return hr;
}

// create new surface ?
STDMETHODIMP WrapperDirect3DDevice9::GetRenderTarget(THIS_ DWORD RenderTargetIndex,IDirect3DSurface9** ppRenderTarget) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::GetRenderTarget(), ");
#endif

	HRESULT hr = D3D_OK;
	WrapperDirect3DSurface9 * ws = NULL;
	IDirect3DSurface9 * ret = NULL;

	hr = m_device->GetRenderTarget(RenderTargetIndex, &ret);
	if(SUCCEEDED(hr)){
		ws = WrapperDirect3DSurface9::GetWrapperSurface9(ret);
		if(!ws){
			ws = new WrapperDirect3DSurface9(ret, WrapperDirect3DSurface9::ins_count++);
#ifdef ENBALE_DEVICE_LOG
			infoRecorder->logTrace(" new surface with id:%d\n", ws->getId());
#endif

			// send command
			csSet->beginCommand(D3DDGetRenderTarget_Opcode, id);
			csSet->writeInt(ws->getId());
			csSet->writeUInt(RenderTargetIndex);
			csSet->endCommand();
			// set the surface's creation flag
			csSet->setCreation(ws->creationFlag);
			ws->creationCommand = D3DDGetRenderTarget_Opcode; 
			ws->renderTargetIndex = RenderTargetIndex;
			Initializer::PushObj(ws);

		}else{
			#ifdef ENBALE_DEVICE_LOG
			infoRecorder->logTrace("with id:%d, tex id:%d, level:%d\n", ws->getId(), ws->GetTexId(), ws->GetLevel());
#endif
		}

		(*ppRenderTarget)=dynamic_cast<IDirect3DSurface9*>(ws);

	}else{
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("failed!\n");
#endif
	}
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::SetDepthStencilSurface(THIS_ IDirect3DSurface9* pNewZStencil) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetDepthStencilSurface(), ");
#endif
	HRESULT hh = D3D_OK;
	if(pNewZStencil == NULL) {
		if(keyCmdHelper->isSending()){
			// send command
			csSet->beginCommand(SetDepthStencilSurface_Opcode, id);
			csSet->writeInt(-1);
			csSet->endCommand();
		}

		hh = m_device->SetDepthStencilSurface(NULL);
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("NULL depth stencil surface.\n");
#endif
		return hh;
	}

	WrapperDirect3DSurface9 * ws = (WrapperDirect3DSurface9 *)pNewZStencil;

	if(keyCmdHelper->isSending()){
		csSet->checkObj(dynamic_cast<IdentifierBase *>(ws));
		// send command
		csSet->beginCommand(SetDepthStencilSurface_Opcode, id);
		csSet->writeInt(ws->getId());
		csSet->endCommand();
	}
	// add the flag to all
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("with id:%d.\n", ws->getId());
#endif

	hh = m_device->SetDepthStencilSurface(((WrapperDirect3DSurface9*)pNewZStencil)->GetSurface9());

	return hh;
}

// create new depth stencil surface
STDMETHODIMP WrapperDirect3DDevice9::GetDepthStencilSurface(THIS_ IDirect3DSurface9** ppZStencilSurface) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetDepthStencilSurface()\n");
#endif
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = m_device->GetDepthStencilSurface(&base_surface);

	WrapperDirect3DSurface9* surface = NULL;
	surface = WrapperDirect3DSurface9::GetWrapperSurface9(base_surface);
	if(surface == NULL) {
		surface = new WrapperDirect3DSurface9(base_surface, WrapperDirect3DSurface9::ins_count++);

		// send command
		csSet->beginCommand(GetDepthStencilSurface_Opcode, id);
		csSet->writeInt(surface->getId());
		csSet->endCommand();
		// set creation flags to all
		csSet->setCreation(surface->creationFlag);
		surface->creationCommand = GetDepthStencilSurface_Opcode; 
		Initializer::PushObj(surface);
	}

	*ppZStencilSurface =dynamic_cast<IDirect3DSurface9*>( surface);
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::BeginScene(THIS) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::BeginScene() invoke\n");
#endif
	sceneBegin = true;

	Initializer::EndInitialize();
	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(BeginScene_Opcode, id);
		csSet->endCommand();
	}
	HRESULT h = m_device->BeginScene();

	return h;
}
STDMETHODIMP WrapperDirect3DDevice9::EndScene(THIS) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::EndScene() called\n");
#endif

	sceneBegin = false;

	if(keyCmdHelper->isSending()){
		// send command
		csSet->beginCommand(EndScene_Opcode, id);
		csSet->endCommand();
	}
	HRESULT hh = m_device->EndScene();

	return hh; 
}
STDMETHODIMP WrapperDirect3DDevice9::Clear(THIS_ DWORD Count,CONST D3DRECT* pRects,DWORD Flags,D3DCOLOR Color,float Z,DWORD Stencil) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::Clear()\n");
#endif
	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(Clear_Opcode, id);
		csSet->writeUInt(Count);
		if(pRects == NULL){
			csSet->writeChar(1);
		}
		else{
			csSet->writeChar(0);
			csSet->writeByteArr((char *)pRects, sizeof(D3DRECT));

		}
		csSet->writeUInt(Flags);
		csSet->writeUInt(Color);
		csSet->writeFloat(Z);
		csSet->writeUInt(Stencil);
		csSet->endCommand();
	}

	HRESULT hh = D3D_OK;
	if(cmdCtrl->isRender())
		hh = m_device->Clear(Count, pRects, Flags, Color, Z, Stencil);

	return hh;
}
STDMETHODIMP WrapperDirect3DDevice9::SetTransform(THIS_ D3DTRANSFORMSTATETYPE State,CONST D3DMATRIX* pMatrix) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetTransform()\n");
	if(pMatrix == NULL) {
		infoRecorder->logTrace("pMatrix is NULL! \n");
	}
#endif
	//send the command
	unsigned short *mask = NULL, *sMask = NULL;
	// prepare the data to send if needed.
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetTransform_Opcode, id);
		csSet->writeShort(State);

		mask = (unsigned short *)(csSet->getCurPtr(sizeof(unsigned short)));
		if(mask){
			*mask = 0;
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
					if(fabs(pMatrix->m[i][j]) > eps){
						(*mask) ^= (1 << (i *4 + j));
						csSet->writeFloat(pMatrix->m[i][j]);
					}
				}
			}
		}
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetTransform_Opcode, id);
		stateRecorder->WriteShort(State);

		sMask = (unsigned short *)(stateRecorder->GetCurPtr(sizeof(unsigned short)));
		//if(sMask){
			*sMask = 0;
			for(int i = 0; i < 4; i++){
				for(int j = 0; j < 4; j++){
					if(fabs(pMatrix->m[i][j]) > eps){
						(*sMask) ^= (1 << (i * 4 + j));
						stateRecorder->WriteFloat(pMatrix->m[i][j]);
					}
				}
			}
		//}
		stateRecorder->EndCommand();
	}

	HRESULT hh = m_device->SetTransform(State, pMatrix);

	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetTransform(THIS_ D3DTRANSFORMSTATETYPE State,D3DMATRIX* pMatrix) {return m_device->GetTransform(State, pMatrix);}
STDMETHODIMP WrapperDirect3DDevice9::MultiplyTransform(THIS_ D3DTRANSFORMSTATETYPE Type,CONST D3DMATRIX* pD3DMatrix) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::MultiplyTransform() TODO\n");
#endif
	return m_device->MultiplyTransform(Type, pD3DMatrix);}

STDMETHODIMP WrapperDirect3DDevice9::SetViewport(THIS_ CONST D3DVIEWPORT9* pViewport) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetViewport() called! v.x:%d, v.y:%d, v.Width:%d, v.Height:%d, v.Maxz:%d, v.Minz:%d\n",pViewport->X, pViewport->Y, pViewport->Width, pViewport->Height, pViewport->MaxZ, pViewport->MinZ);
#endif
#ifdef ENBALE_DEVICE_LOG
	if(pViewport == NULL) {
		infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetViewport(), pViewport is NULL\n");
	}
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetViewport_Opcode, id);
		csSet->writeByteArr((char *)pViewport, sizeof(*pViewport));
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(SetViewport_Opcode, id);
		stateRecorder->WriteByteArr((char *)pViewport, sizeof(*pViewport));
		stateRecorder->EndCommand();
	}

	HRESULT hr = m_device->SetViewport(pViewport);
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::GetViewport(THIS_ D3DVIEWPORT9* pViewport) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetViewport() TODO\n");
#endif
	return m_device->GetViewport(pViewport);
}

STDMETHODIMP WrapperDirect3DDevice9::SetMaterial(THIS_ CONST D3DMATERIAL9* pMaterial) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetMaterial()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetMaterial_Opcode, id);
		csSet->writeByteArr((char *)pMaterial, sizeof(D3DMATERIAL9));
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetMaterial_Opcode, id);
		stateRecorder->WriteByteArr((char *)pMaterial, sizeof(D3DMATERIAL9));
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetMaterial(pMaterial);
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetMaterial(THIS_ D3DMATERIAL9* pMaterial) {return m_device->GetMaterial(pMaterial);}

STDMETHODIMP WrapperDirect3DDevice9::SetLight(THIS_ DWORD Index,CONST D3DLIGHT9* pD3DLight9) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetLight\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetLight_Opcode, id);
		csSet->writeUInt(Index);
		csSet->writeByteArr((char*)pD3DLight9, sizeof(D3DLIGHT9));
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetLight_Opcode, id);
		stateRecorder->WriteUInt(Index);
		stateRecorder->WriteByteArr((char *)pD3DLight9, sizeof(D3DLIGHT9));
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetLight(Index, pD3DLight9);
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetLight(THIS_ DWORD Index,D3DLIGHT9* pD3DLight9) {return m_device->GetLight(Index, pD3DLight9);}

STDMETHODIMP WrapperDirect3DDevice9::LightEnable(THIS_ DWORD Index,BOOL Enable) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::LightEnable()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(LightEnable_Opcode, id);
		csSet->writeUInt(Index);
		csSet->writeInt(Enable);
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(LightEnable_Opcode, id);
		stateRecorder->WriteUInt(Index);
		stateRecorder->WriteInt(Enable);
		stateRecorder->EndCommand();
	}

	HRESULT hh = m_device->LightEnable(Index, Enable);
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetLightEnable(THIS_ DWORD Index,BOOL* pEnable) {return m_device->GetLightEnable(Index, pEnable);}

STDMETHODIMP WrapperDirect3DDevice9::SetClipPlane(THIS_ DWORD Index,CONST float* pPlane) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetClipPlane() TODO\n");
#endif
	return m_device->SetClipPlane(Index, pPlane);
}

STDMETHODIMP WrapperDirect3DDevice9::GetClipPlane(THIS_ DWORD Index,float* pPlane) {return m_device->GetClipPlane(Index, pPlane);}


STDMETHODIMP WrapperDirect3DDevice9::SetRenderState(THIS_ D3DRENDERSTATETYPE State,DWORD Value) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetRenderState(), state=%d, value=%d\n", State, Value);
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetRenderState_Opcode, id);
		csSet->writeUInt(State);
		csSet->writeUInt(Value);
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetRenderState_Opcode, id);
		stateRecorder->WriteUInt(State);
		stateRecorder->WriteUInt(Value);
		stateRecorder->EndCommand();
	}
	// for initializer to record the render states before BeginScene
	if(Initializer::InitializerEnabled()){
		Initializer::BeginCommand(SetRenderState_Opcode, id);
		Initializer::WriteUInt(State);
		Initializer::WriteUInt(Value);
		Initializer::EndCommand();
	}
	HRESULT hh = m_device->SetRenderState(State, Value);

	return hh;
}
STDMETHODIMP WrapperDirect3DDevice9::GetRenderState(THIS_ D3DRENDERSTATETYPE State,DWORD* pValue) {return m_device->GetRenderState(State, pValue);}

STDMETHODIMP WrapperDirect3DDevice9::CreateStateBlock(THIS_ D3DSTATEBLOCKTYPE Type,IDirect3DStateBlock9** ppSB) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateStateBlock(), ");
#endif
	IDirect3DStateBlock9* sb_base = NULL;
	HRESULT hr = m_device->CreateStateBlock(Type, &sb_base);
	WrapperDirect3DStateBlock9 * wsb = NULL;

	if(SUCCEEDED(hr)) {
		wsb = new WrapperDirect3DStateBlock9(sb_base, WrapperDirect3DStateBlock9::ins_count ++);
		*ppSB = dynamic_cast<IDirect3DStateBlock9*>(wsb);
	}
	else {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("failed\n");
#endif

		*ppSB = NULL;
		return hr;
	}

	// send command
	csSet->beginCommand(CreateStateBlock_Opcode, id);
	csSet->writeUInt(Type);
	csSet->writeInt(wsb->getId());
	csSet->endCommand();
	// set state block's creation flags to all
	csSet->setCreation(wsb->creationFlag);
	infoRecorder->addCreation();
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("with id:%d.\n", wsb->getId());
#endif
	wsb->type = Type;
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::BeginStateBlock(THIS) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::BeginStateBlock() called, seraial number:%d.\n", WrapperDirect3DStateBlock9::ins_count);
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(BeginStateBlock_Opcode, id);
		csSet->endCommand();

	}

	StateBlockRecorder * recoder = StateBlockRecorder::GetRecorder();
	recoder->StateBlockBegin();
	stateRecorder = recoder;
	// record the command
	recoder->BeginCommand(BeginStateBlock_Opcode, id);
	recoder->EndCommand();

	HRESULT hh = m_device->BeginStateBlock();
	return hh;
}

//TODO
// check whether this function is to use a state block or create a new state block
STDMETHODIMP WrapperDirect3DDevice9::EndStateBlock(THIS_ IDirect3DStateBlock9** ppSB) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::EndStateBlock(), ");
#endif
	WrapperDirect3DStateBlock9 * wsb = NULL;

	IDirect3DStateBlock9* sb_base = NULL;
	HRESULT hr = m_device->EndStateBlock(&sb_base);

	if(SUCCEEDED(hr)) {
		wsb = new WrapperDirect3DStateBlock9(sb_base, WrapperDirect3DStateBlock9::ins_count++);
		wsb->setDeviceID(id);
		wsb->creationCommand = EndStateBlock_Opcode;
		*ppSB = dynamic_cast<IDirect3DStateBlock9*>(wsb);
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(wsb);
#endif
	}
	else {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("failed\n");
#endif
		*ppSB = NULL;
		return hr;
	}
	// send commands
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(EndStateBlock_Opcode, id);
		csSet->writeInt(wsb->getId());
		csSet->endCommand();
	}
	// set state block's creation flag to all
	csSet->setCreation(wsb->creationFlag);
	StateBlock * block = NULL;
	if(stateRecorder){
		stateRecorder->BeginCommand(EndStateBlock_Opcode, id);
		stateRecorder->WriteInt(wsb->getId());
		stateRecorder->EndCommand();
		block = stateRecorder->StateBlockEnd(wsb->getId());
		wsb->stateBlock = block;  // set the block to Wrapper
		stateRecorder = NULL;   // cancel the recording
	}
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("with id:%d.\n", wsb->getId());
#endif
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::SetClipStatus(THIS_ CONST D3DCLIPSTATUS9* pClipStatus) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetClipStatus() TODO\n");
#endif
	return m_device->SetClipStatus(pClipStatus);
}

STDMETHODIMP WrapperDirect3DDevice9::GetClipStatus(THIS_ D3DCLIPSTATUS9* pClipStatus) {return m_device->GetClipStatus(pClipStatus);}

STDMETHODIMP WrapperDirect3DDevice9::GetTexture(THIS_ DWORD Stage,IDirect3DBaseTexture9** ppTexture) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetTexture() TODO\n");
#endif
	/*TODO*/
	/*
	LPDIRECT3DBASETEXTURE9 base_texture = NULL;
	HRESULT hr = m_device->GetTexture(Stage, &base_texture);

	*ppTexture = base_texture;
	*/
	return m_device->GetTexture(Stage, ppTexture);
	//return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::SetTexture(THIS_ DWORD Stage,IDirect3DBaseTexture9* pTexture) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetTexture(), pTexture:0x%p\n", pTexture);
#endif

	if(pTexture == NULL) {
		// send command
		if(keyCmdHelper->isSending()){
			csSet->beginCommand(SetTexture_Opcode, this->id);
			csSet->writeUInt(Stage);
			csSet->writeInt(-1);
			csSet->endCommand();
		}
		if(stateRecorder){
			stateRecorder->BeginCommand(SetTexture_Opcode, id);
			stateRecorder->WriteUInt(Stage);
			stateRecorder->WriteInt(-1);
			stateRecorder->EndCommand();
		}
		return m_device->SetTexture(Stage, pTexture);
	}
	//infoRecorder->logError("[WrapperDirect3DDevice9]: to get the texture type, ptr type: %s.\n", typeid(pTexture).name());

	D3DRESOURCETYPE Type = ((IDirect3DTexture9 *)pTexture)->GetType();

	//infoRecorder->logError("[WrapperDirect3DDevice9]: after get the texture type:%s.\n", Type == D3DRTYPE_TEXTURE ?"Texture" : (Type == D3DRTYPE_CUBETEXTURE ? "CUBE_TEXTURE": "unknown type"));

	if(Type == D3DRTYPE_TEXTURE) {
		WrapperDirect3DTexture9 * wt = (WrapperDirect3DTexture9 *)pTexture;
#ifdef ENABLE_DEVICE_LOG
		infoRecorder->logError("[WrapperDirect3DDevice9]::SetTexture, tex id:%d, stage:%d, type:%s, created: %x.\n",wt->getId(), Stage, "TEXTURE", wt->creationFlag);
#endif

		//TODO
		// check the texture data is sent or not

		if(keyCmdHelper->isSending()){
			csSet->checkObj(dynamic_cast<IdentifierBase *>(wt));
			// send the command
			csSet->beginCommand(SetTexture_Opcode, this->id);
			csSet->writeUInt(Stage);
			csSet->writeInt(wt->getId());

			csSet->endCommand();
		}
		if(stateRecorder){
			stateRecorder->pushDependency(wt);
			stateRecorder->BeginCommand(SetTexture_Opcode, id);
			stateRecorder->WriteUInt(Stage);
			stateRecorder->WriteInt(wt->getId());
			stateRecorder->EndCommand();
		}
		HRESULT hh = m_device->SetTexture(Stage, ((WrapperDirect3DTexture9*)pTexture)->GetTex9());
		return hh;
	}
	else if(Type == D3DRTYPE_CUBETEXTURE) {
		WrapperDirect3DCubeTexture9 * wct = (WrapperDirect3DCubeTexture9 *)pTexture;
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logError("Type is CubeTexture id=%d, TODO\n", id);
		infoRecorder->logError("[WrapperDirect3DDevice9]::SetTexture, tex id:%d, stage:%d, type:%s.\n",wct->getId(), Stage, "CUBE_TEXTURE");
#endif

		// check texture's creation and update
		if(keyCmdHelper->isSending()){
			csSet->checkObj(dynamic_cast<IdentifierBase *>(wct));
			// send command
			csSet->beginCommand(SetCubeTexture_Opcode, id);
			csSet->writeUInt(Stage);
			csSet->writeInt(wct->getId());
			csSet->endCommand();
		}
		if(stateRecorder){
			stateRecorder->pushDependency(wct);
			stateRecorder->BeginCommand(SetTexture_Opcode, id);
			stateRecorder->WriteUInt(Stage);
			stateRecorder->WriteInt(wct->getId());
			stateRecorder->EndCommand();
		}
		HRESULT hh = m_device->SetTexture(Stage, ((WrapperDirect3DCubeTexture9*)pTexture)->GetCubeTex9());
		return hh;
	}
	else if(Type == D3DRTYPE_VOLUMETEXTURE){
		infoRecorder->logError("TODO, use volume texture.\n");
		WrapperDirect3DVolumeTexture9 * wvt = (WrapperDirect3DVolumeTexture9 *)pTexture;
		infoRecorder->logError("[WrapperDirect3DDevice9]::SetTexture, tex id:%d, stage:%d, type:%s.\n", wvt->getId(), Stage, "VOLUME_TEXTURE");
		csSet->checkObj(dynamic_cast<IdentifierBase *>(wvt));
		// send command

		// recorde state
		if(stateRecorder){
			stateRecorder->pushDependency(wvt);
			stateRecorder->BeginCommand(SetTexture_Opcode, id);
			stateRecorder->WriteUInt(Stage);
			stateRecorder->WriteInt(wvt->getId());
			stateRecorder->EndCommand();
		}

		return m_device->SetTexture(Stage, wvt->GetVolumeTex9());
	}
	else {
		infoRecorder->logError("TODO, Type is unknown\n");
		return m_device->SetTexture(Stage, pTexture);
	}
}

STDMETHODIMP WrapperDirect3DDevice9::GetTextureStageState(THIS_ DWORD Stage,D3DTEXTURESTAGESTATETYPE Type,DWORD* pValue) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetTextureStageState() TODO\n");
#endif
	return m_device->GetTextureStageState(Stage, Type, pValue);
}

STDMETHODIMP WrapperDirect3DDevice9::SetTextureStageState(THIS_ DWORD Stage,D3DTEXTURESTAGESTATETYPE Type,DWORD Value) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetTextureStageState()\n");
#endif
	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetTextureStageState_Opcode, id);
		csSet->writeUInt(Stage);
		csSet->writeUInt(Type);
		csSet->writeUInt(Value);
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(SetTextureStageState_Opcode, id);
		stateRecorder->WriteUInt(Stage);
		stateRecorder->WriteUInt(Type);
		stateRecorder->WriteUInt(Value);
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetTextureStageState(Stage, Type, Value);

	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetSamplerState(THIS_ DWORD Sampler,D3DSAMPLERSTATETYPE Type,DWORD* pValue) {return m_device->GetSamplerState(Sampler, Type, pValue);}

STDMETHODIMP WrapperDirect3DDevice9::SetSamplerState(THIS_ DWORD Sampler,D3DSAMPLERSTATETYPE Type,DWORD Value) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetSamplerState() invoke\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetSamplerState_Opcode, id);
		csSet->writeUInt(Sampler);
		csSet->writeUChar(Type);
		csSet->writeUInt(Value);
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(SetSamplerState_Opcode, id);
		stateRecorder->WriteUInt(Sampler);
		stateRecorder->WriterUChar(Type);
		stateRecorder->WriteUInt(Value);
		stateRecorder->EndCommand();
	}

	HRESULT hh = m_device->SetSamplerState(Sampler, Type, Value);

	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::ValidateDevice(THIS_ DWORD* pNumPasses) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::ValidateDevice() TODO\n");
#endif
	return m_device->ValidateDevice(pNumPasses);
}

STDMETHODIMP WrapperDirect3DDevice9::SetPaletteEntries(THIS_ UINT PaletteNumber,CONST PALETTEENTRY* pEntries) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetPaletteEntries() TODO\n");
#endif
	return m_device->SetPaletteEntries(PaletteNumber, pEntries);
}

STDMETHODIMP WrapperDirect3DDevice9::GetPaletteEntries(THIS_ UINT PaletteNumber,PALETTEENTRY* pEntries) {return m_device->GetPaletteEntries(PaletteNumber, pEntries);}
STDMETHODIMP WrapperDirect3DDevice9::SetCurrentTexturePalette(THIS_ UINT PaletteNumber) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetCurrentTexturePalette() TODO\n");
#endif
	return m_device->SetCurrentTexturePalette(PaletteNumber);
}
STDMETHODIMP WrapperDirect3DDevice9::GetCurrentTexturePalette(THIS_ UINT *PaletteNumber) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetCurrentTexturePalette() TODO\n");
#endif
	return m_device->GetCurrentTexturePalette(PaletteNumber);
}

STDMETHODIMP WrapperDirect3DDevice9::SetScissorRect(THIS_ CONST RECT* pRect) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]::SetScissorRect() called, left :%d, top :%d, buttom :%d, right :%d\n",pRect->left, pRect->top, pRect->bottom, pRect->right);
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(D3DDSetScissorRect_Opcode, id);
		csSet->writeInt(pRect->left);
		csSet->writeInt(pRect->right);
		csSet->writeInt(pRect->top);
		csSet->writeInt(pRect->bottom);
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(D3DDSetScissorRect_Opcode, id);
		stateRecorder->WriteInt(pRect->left);
		stateRecorder->WriteInt(pRect->right);
		stateRecorder->WriteInt(pRect->top);
		stateRecorder->WriteInt(pRect->bottom);
		stateRecorder->EndCommand();
	}
	return m_device->SetScissorRect(pRect);
}

STDMETHODIMP WrapperDirect3DDevice9::GetScissorRect(THIS_ RECT* pRect) {return m_device->GetScissorRect(pRect);}

STDMETHODIMP WrapperDirect3DDevice9::SetSoftwareVertexProcessing(THIS_ BOOL bSoftware) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetSoftwareVertexProcessing()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetSoftwareVertexProcessing_Opcode, id);
		csSet->writeInt(bSoftware);
		csSet->endCommand();
	}
	HRESULT hh = m_device->SetSoftwareVertexProcessing(bSoftware);
	return hh;
}

STDMETHODIMP_(BOOL) WrapperDirect3DDevice9::GetSoftwareVertexProcessing(THIS) {return m_device->GetSoftwareVertexProcessing();}

STDMETHODIMP WrapperDirect3DDevice9::SetNPatchMode(THIS_ float nSegments) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetNPatchMode()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetNPatchMode_Opcode, id);
		csSet->writeFloat(nSegments);
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetNPatchMode_Opcode, id);
		stateRecorder->WriteFloat(nSegments);
		stateRecorder->EndCommand();
	}

	


	HRESULT hh = m_device->SetNPatchMode(nSegments);

	return hh;
}

STDMETHODIMP_(float) WrapperDirect3DDevice9::GetNPatchMode(THIS) {return m_device->GetNPatchMode();}

/*
place for DrawPrimitive functions
*/

STDMETHODIMP WrapperDirect3DDevice9::ProcessVertices(THIS_ UINT SrcStartIndex,UINT DestIndex,UINT VertexCount,IDirect3DVertexBuffer9* pDestBuffer,IDirect3DVertexDeclaration9* pVertexDecl,DWORD Flags) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::ProcessVertices() TODO\n");
#endif
	return m_device->ProcessVertices(SrcStartIndex, DestIndex, VertexCount, pDestBuffer, pVertexDecl, Flags);
}

/*
place for CreateVertexDeclaration
*/

STDMETHODIMP WrapperDirect3DDevice9::GetVertexDeclaration(THIS_ IDirect3DVertexDeclaration9** ppDecl) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetVertexDeclaration() TODO\n");
#endif
	return m_device->GetVertexDeclaration(ppDecl);
}

STDMETHODIMP WrapperDirect3DDevice9::SetFVF(THIS_ DWORD FVF) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetFVF()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetFVF_Opcode, id);
		csSet->writeUInt(FVF);
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetFVF_Opcode, id);
		stateRecorder->WriteUInt(FVF);
		stateRecorder->EndCommand();
	}
	return m_device->SetFVF(FVF);
}
STDMETHODIMP WrapperDirect3DDevice9::GetFVF(THIS_ DWORD* pFVF) {return m_device->GetFVF(pFVF);}

STDMETHODIMP WrapperDirect3DDevice9::CreateVertexShader(THIS_ CONST DWORD* pFunction,IDirect3DVertexShader9** ppShader) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateVertexShader(), ");
#endif

	int cnt = 0;
	while(true) {
		if(pFunction[cnt++] == D3DVS_END()) break;
	}

	LPDIRECT3DVERTEXSHADER9 base_vs = NULL;
	HRESULT hr = m_device->CreateVertexShader(pFunction, &base_vs);
	WrapperDirect3DVertexShader9 * wvs = NULL;

	if(SUCCEEDED(hr)) {
		wvs = new WrapperDirect3DVertexShader9(base_vs, WrapperDirect3DVertexShader9::ins_count++);
		*ppShader = dynamic_cast<IDirect3DVertexShader9*>(wvs);
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(wvs);
#endif
	}
	else {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("failed\n");
#endif
		*ppShader = NULL;
		return hr;
	}

	// send command
	csSet->beginCommand(CreateVertexShader_Opcode, id);
	csSet->writeInt(wvs->getId());
	csSet->writeInt(cnt);
	csSet->writeByteArr((char *)pFunction, sizeof(DWORD) *(cnt));
	csSet->endCommand();

	csSet->setCreation(wvs->creationFlag);
	infoRecorder->addCreation();

	//TODO :duplicate the shader data
	wvs->shaderLen = sizeof(DWORD) * cnt;
	wvs->funCount = cnt;
	wvs->shaderData = (char *)malloc(wvs->shaderLen);
	memcpy(wvs->shaderData, pFunction, wvs->shaderLen);
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("id:%d, shader size:%d, function count:%d.\n",wvs->getId(), wvs->shaderLen, wvs->funCount);
#endif

	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::SetVertexShader(THIS_ IDirect3DVertexShader9* pShader) {

#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetVertexShader(), ");
#endif
	HRESULT hh;
	if(pShader == NULL) {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("pShader is NULL\n");
#endif

		// send command
		if(keyCmdHelper->isSending()){
			csSet->beginCommand(SetVertexShader_Opcode, id);
			csSet->writeInt(-1);
			csSet->endCommand();
		}
		if(stateRecorder){
			stateRecorder->BeginCommand(SetVertexShader_Opcode, id);
			stateRecorder->WriteInt(-1);
			stateRecorder->EndCommand();
		}
		hh = m_device->SetVertexShader(NULL);

		return hh;
	}
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("id=%d\n", ((WrapperDirect3DVertexShader9*)pShader)->getId() );
#endif

	// check vertex shader's creation
	WrapperDirect3DVertexShader9 * wvs = (WrapperDirect3DVertexShader9 *)pShader;
	csSet->checkObj(dynamic_cast<IdentifierBase *>(wvs));
	// send
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetVertexShader_Opcode, id);
		csSet->writeInt(((WrapperDirect3DVertexShader9 *)pShader)->getId());
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->pushDependency(dynamic_cast<IdentifierBase *>(wvs));
		stateRecorder->BeginCommand(SetVertexShader_Opcode, id);
		stateRecorder->WriteInt(wvs->getId());
		stateRecorder->EndCommand();
	}
	hh = m_device->SetVertexShader(((WrapperDirect3DVertexShader9*)pShader)->GetVS9());
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetVertexShader(THIS_ IDirect3DVertexShader9** ppShader) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetVertexShader()\n");
#endif
	LPDIRECT3DVERTEXSHADER9 base_vertex_shader = NULL;
	HRESULT hr = m_device->GetVertexShader(&base_vertex_shader);

	*ppShader = WrapperDirect3DVertexShader9::GetWrapperVertexShader(base_vertex_shader);
	return hr;
}

float vs_data[10000];

STDMETHODIMP WrapperDirect3DDevice9::SetVertexShaderConstantF(THIS_ UINT StartRegister,CONST float* pConstantData,UINT Vector4fCount) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetVertexShaderConstantF(), size:%d\n", Vector4fCount * 16);
	if(Vector4fCount * 16 > 10000){
		infoRecorder->logError("[WrfapperDirect3DDevice9]:SetVetexShaderConstantF(), size:%d > 10000, error.\n", Vector4fCount * 16);
	}
#endif

	memcpy((char*)vs_data, (char*)pConstantData, Vector4fCount * 16);

	UINT i = 0;
	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetVertexShaderConstantF_Opcode, id);
		csSet->writeUShort(StartRegister);
		csSet->writeUShort(Vector4fCount);

		for(i = 0; i< Vector4fCount /2; ++i){
			csSet->writeVec(SetVertexShaderConstantF_Opcode, vs_data + ( i * 8), 32);
		}
		if(Vector4fCount & 1){
			csSet->writeVec(SetVertexShaderConstantF_Opcode, vs_data + ( i * 8), 16);
		}
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetVertexShaderConstantF_Opcode, id);
		stateRecorder->WriteUShort(StartRegister);
		stateRecorder->WriteUShort(Vector4fCount);
		for(i = 0; i < Vector4fCount / 2; ++i){
			stateRecorder->WriterVec(SetVertexShaderConstantF_Opcode, vs_data + (i * 8), 32);
		}
		if(Vector4fCount & 1){
			stateRecorder->WriterVec(SetVertexShaderConstantF_Opcode, vs_data + (i * 8), 16);
		}
		stateRecorder->EndCommand();
	}

	HRESULT hh = m_device->SetVertexShaderConstantF(StartRegister, pConstantData, Vector4fCount);
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetVertexShaderConstantF(THIS_ UINT StartRegister,float* pConstantData,UINT Vector4fCount) {return m_device->GetVertexShaderConstantF(StartRegister, pConstantData, Vector4fCount);}

STDMETHODIMP WrapperDirect3DDevice9::SetVertexShaderConstantI(THIS_ UINT StartRegister,CONST int* pConstantData,UINT Vector4iCount) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetVertexShaderConstantI()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetVertexShaderConstantI_Opcode, id);
		csSet->writeUInt(StartRegister);
		csSet->writeUInt(Vector4iCount);
		csSet->writeByteArr((char *)pConstantData, sizeof(int) * (Vector4iCount * 4));
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(SetVertexShaderConstantI_Opcode, id);
		stateRecorder->WriteUInt(StartRegister);
		stateRecorder->WriteUInt(Vector4iCount);
		stateRecorder->WriteByteArr((char *)pConstantData, sizeof(int) * (Vector4iCount * 4));
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetVertexShaderConstantI(StartRegister, pConstantData, Vector4iCount);

	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetVertexShaderConstantI(THIS_ UINT StartRegister,int* pConstantData,UINT Vector4iCount) {return m_device->GetVertexShaderConstantI(StartRegister, pConstantData, Vector4iCount);}

STDMETHODIMP WrapperDirect3DDevice9::SetVertexShaderConstantB(THIS_ UINT StartRegister,CONST BOOL* pConstantData,UINT  BoolCount) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetVertexShaderConstantB()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetVertexShaderConstantB_Opcode, id);
		csSet->writeUInt(StartRegister);
		csSet->writeUInt(BoolCount);
		csSet->writeByteArr((char *)pConstantData, sizeof(BOOL) * BoolCount);
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(SetVertexShaderConstantB_Opcode, id);
		stateRecorder->WriteUInt(StartRegister);
		stateRecorder->WriteUInt(BoolCount);
		stateRecorder->WriteByteArr((char *)pConstantData, sizeof(BOOL) * BoolCount);
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetVertexShaderConstantB(StartRegister, pConstantData, BoolCount);

	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetVertexShaderConstantB(THIS_ UINT StartRegister,BOOL* pConstantData,UINT BoolCount) {return m_device->GetVertexShaderConstantB(StartRegister, pConstantData, BoolCount);}

/*
place for SetVertexDeclaration
*/

/*
place for SetStreamSource
*/

STDMETHODIMP WrapperDirect3DDevice9::GetStreamSource(THIS_ UINT StreamNumber,IDirect3DVertexBuffer9** ppStreamData,UINT* pOffsetInBytes,UINT* pStride) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetStreamSource()\n");
#endif

	LPDIRECT3DVERTEXBUFFER9 base_vertex_buffer = NULL;
	HRESULT hr = m_device->GetStreamSource(StreamNumber, &base_vertex_buffer, pOffsetInBytes, pStride);

	*ppStreamData = WrapperDirect3DVertexBuffer9::GetWrapperVertexBuffer9(base_vertex_buffer);
	return hr;
}

// change the pipeline state
STDMETHODIMP WrapperDirect3DDevice9::SetStreamSourceFreq(THIS_ UINT StreamNumber,UINT Setting) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetStreamSourceFreq(), stream number:%d, Setting:%d, TODO\n", StreamNumber, Setting);
#endif

	if(keyCmdHelper->isSending()){
		csSet->beginCommand(D3DDSetStreamSourceFreq_Opcode, id);
		csSet->writeUInt(StreamNumber);
		csSet->writeUInt(Setting);
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(D3DDSetStreamSourceFreq_Opcode, id);
		stateRecorder->WriteUInt(StreamNumber);
		stateRecorder->WriteUInt(Setting);
		stateRecorder->EndCommand();
	}

	return m_device->SetStreamSourceFreq(StreamNumber, Setting);
}

STDMETHODIMP WrapperDirect3DDevice9::GetStreamSourceFreq(THIS_ UINT StreamNumber,UINT* pSetting) {return m_device->GetStreamSourceFreq(StreamNumber, pSetting);}

/*
place for SetIndices
*/

STDMETHODIMP WrapperDirect3DDevice9::GetIndices(THIS_ IDirect3DIndexBuffer9** ppIndexData) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetIndices() TODO\n");
#endif

	LPDIRECT3DINDEXBUFFER9 base_indexed_buffer = NULL;
	HRESULT hr =  m_device->GetIndices(&base_indexed_buffer);

	*ppIndexData = WrapperDirect3DIndexBuffer9::GetWrapperIndexedBuffer9(base_indexed_buffer);

	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::CreatePixelShader(THIS_ CONST DWORD* pFunction,IDirect3DPixelShader9** ppShader) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreatePixelShader(), ");
#endif

	int cnt = 0;
	while(true) {
		if(pFunction[cnt++] == D3DPS_END()) break;
		//cnt++;
	}
	//WrapperDirect3DPixelShader9::ins_count++;
	LPDIRECT3DPIXELSHADER9 base_ps = NULL;

	HRESULT hr = m_device->CreatePixelShader(pFunction, &base_ps);

	WrapperDirect3DPixelShader9 * wps = NULL;
	if(SUCCEEDED(hr)) {
		wps = new WrapperDirect3DPixelShader9(base_ps, WrapperDirect3DPixelShader9::ins_count ++);
		*ppShader = dynamic_cast<IDirect3DPixelShader9*>(wps);
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(wps);
#endif
	}
	else {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("failed\n");
#endif
		*ppShader = NULL;
		return hr;
	}

	// send
	csSet->beginCommand(CreatePixelShader_Opcode, id);
	csSet->writeInt(wps->getId());
	csSet->writeInt(cnt);
	csSet->writeByteArr((char *)pFunction, sizeof(DWORD) *(cnt));
	csSet->endCommand();

	csSet->setCreation(wps->creationFlag);
	infoRecorder->addCreation();

	// duplicate the pixel shader data
	wps->funcCount = cnt;
	wps->shaderSize = sizeof(DWORD) * cnt;
	wps->pFunc = (char *)malloc(wps->shaderSize);
	memcpy(wps->pFunc, pFunction, wps->shaderSize);
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("shader id:%d, size:%d, function count:%d.\n", wps->getId(), wps->shaderSize, wps->funcCount);
#endif

	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::SetPixelShader(THIS_ IDirect3DPixelShader9* pShader) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetPixelShader(), ");
#endif

	if(pShader == NULL) {
#ifdef ENBALE_DEVICE_LOG
		infoRecorder->logTrace("pShader is NULL, I got it\n");
#endif
		// send command
		if(keyCmdHelper->isSending()){
			csSet->beginCommand(SetPixelShader_Opcode, id);
			csSet->writeInt(-1);
			csSet->endCommand();
		}
		if(stateRecorder){
			stateRecorder->BeginCommand(SetPixelShader_Opcode, id);
			stateRecorder->WriteInt(-1);
			stateRecorder->EndCommand();
		}
		return m_device->SetPixelShader(pShader);
	}

	// check the creation of pixel shader

	WrapperDirect3DPixelShader9 * wps = (WrapperDirect3DPixelShader9 *)pShader;
	//wps->checkCreation(id);
	csSet->checkObj(dynamic_cast<IdentifierBase *>(wps));
	// send command

	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetPixelShader_Opcode, id);
		csSet->writeInt(((WrapperDirect3DPixelShader9*)pShader)->getId());
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->pushDependency(wps);
		stateRecorder->BeginCommand(SetPixelShader_Opcode, id);
		stateRecorder->WriteInt(((WrapperDirect3DPixelShader9 *)pShader)->getId());
		stateRecorder->EndCommand();
	}
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("shader id:%d.\n", wps->getId());
#endif
	return m_device->SetPixelShader(((WrapperDirect3DPixelShader9*)pShader)->GetPS9());
}

STDMETHODIMP WrapperDirect3DDevice9::GetPixelShader(THIS_ IDirect3DPixelShader9** ppShader) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::GetPixelShader() TODO\n");
#endif
	LPDIRECT3DPIXELSHADER9 base_pixel_shader = NULL;
	HRESULT hr = m_device->GetPixelShader(&base_pixel_shader);

	*ppShader = WrapperDirect3DPixelShader9::GetWrapperPixelShader(base_pixel_shader);
	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::SetPixelShaderConstantF(THIS_ UINT StartRegister,CONST float* pConstantData,UINT Vector4fCount) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetPixelShaderConstantF(), size:%d\n", Vector4fCount * 16);
	if(Vector4fCount * 16 > 10000){
		infoRecorder->logError("[WrapperDirect3DDevice9]:SetPixelShaderConstantF(), size:%d > 10000, error.\n", Vector4fCount * 16);
	}
#endif
	// TODO ,store the pixel shader data
	memcpy((char*)vs_data, (char*)pConstantData, Vector4fCount * 16);

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetPixelShaderConstantF_Opcode, id);
		csSet->writeUInt(StartRegister);
		csSet->writeUInt(Vector4fCount);

		for(UINT i = 0; i< Vector4fCount; i++){
			csSet->writeVec(SetPixelShaderConstantF_Opcode, vs_data + ( i * 4));
		}
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(SetPixelShaderConstantF_Opcode, id);
		stateRecorder->WriteUInt(StartRegister);
		stateRecorder->WriteUInt(Vector4fCount);
		for(UINT i = 0; i < Vector4fCount; i++){
			stateRecorder->WriterVec(SetPixelShaderConstantF_Opcode, vs_data + (i * 4) , 16);
		}
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetPixelShaderConstantF(StartRegister, pConstantData, Vector4fCount);
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetPixelShaderConstantF(THIS_ UINT StartRegister,float* pConstantData,UINT Vector4fCount) {return m_device->GetPixelShaderConstantF(StartRegister, pConstantData, Vector4fCount);}

STDMETHODIMP WrapperDirect3DDevice9::SetPixelShaderConstantI(THIS_ UINT StartRegister,CONST int* pConstantData,UINT Vector4iCount) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetPixelShaderConstantI()\n");
#endif

	// send command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetPixelShaderConstantI_Opcode, id);
		csSet->writeUInt(StartRegister);
		csSet->writeUInt(Vector4iCount);
		csSet->writeByteArr((char *)pConstantData, sizeof(int) * (Vector4iCount * 4));
		csSet->endCommand();
	}

	if(stateRecorder){
		stateRecorder->BeginCommand(SetPixelShaderConstantI_Opcode, id);
		stateRecorder->WriteUInt(StartRegister);
		stateRecorder->WriteUInt(Vector4iCount);
		stateRecorder->WriteByteArr((char *)pConstantData, sizeof(int) * (Vector4iCount * 4));
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetPixelShaderConstantI(StartRegister, pConstantData, Vector4iCount);

	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetPixelShaderConstantI(THIS_ UINT StartRegister,int* pConstantData,UINT Vector4iCount) {return m_device->GetPixelShaderConstantI(StartRegister, pConstantData, Vector4iCount);}

STDMETHODIMP WrapperDirect3DDevice9::SetPixelShaderConstantB(THIS_ UINT StartRegister,CONST BOOL* pConstantData,UINT  BoolCount) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetPixelShaderConstantB()\n");
#endif

	// send  command
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SetPixelShaderConstantB_Opcode, id);
		csSet->writeUInt(StartRegister);
		csSet->writeUInt(BoolCount);
		csSet->writeByteArr((char *)pConstantData, sizeof(BOOL) * BoolCount); 
		csSet->endCommand();
	}
	if(stateRecorder){
		stateRecorder->BeginCommand(SetPixelShaderConstantB_Opcode, id);
		stateRecorder->WriteUInt(StartRegister);
		stateRecorder->WriteUInt(BoolCount);
		stateRecorder->WriteByteArr((char *)pConstantData, sizeof(BOOL) * BoolCount);
		stateRecorder->EndCommand();
	}
	HRESULT hh = m_device->SetPixelShaderConstantB(StartRegister, pConstantData, BoolCount);

	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::GetPixelShaderConstantB(THIS_ UINT StartRegister,BOOL* pConstantData,UINT BoolCount) {return m_device->GetPixelShaderConstantB(StartRegister, pConstantData, BoolCount);}

STDMETHODIMP WrapperDirect3DDevice9::DrawRectPatch(THIS_ UINT Handle,CONST float* pNumSegs,CONST D3DRECTPATCH_INFO* pRectPatchInfo) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::DrawRectPatch() TODO\n");
#endif
	return m_device->DrawRectPatch(Handle, pNumSegs, pRectPatchInfo);
}

STDMETHODIMP WrapperDirect3DDevice9::DrawTriPatch(THIS_ UINT Handle,CONST float* pNumSegs,CONST D3DTRIPATCH_INFO* pTriPatchInfo) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::DrawTriPatch() TODO\n");
#endif
	return m_device->DrawTriPatch(Handle, pNumSegs, pTriPatchInfo);
}

STDMETHODIMP WrapperDirect3DDevice9::DeletePatch(THIS_ UINT Handle) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::DeletePatch() TODO\n");
#endif
	return m_device->DeletePatch(Handle);
}

STDMETHODIMP WrapperDirect3DDevice9::CreateQuery(THIS_ D3DQUERYTYPE Type,IDirect3DQuery9** ppQuery) {
#ifdef ENBALE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateQuery() TODO\n");
#endif
	return m_device->CreateQuery(Type, ppQuery);
}

//TODO : I want to make the network send operation separate from render thread, use another thread to compress and send