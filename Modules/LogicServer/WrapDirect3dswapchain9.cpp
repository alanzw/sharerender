#include "WrapDirect3dswapchain9.h"
#include "WrapDirect3dsurface9.h"
#include "CommandServerSet.h"
#include "KeyboardHook.h"
#ifdef MULTI_CLIENTS



extern int deviceId;
int WrapperDirect3DSwapChain9::sendCreation(void *ctx){
#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("[WrapperDirect3DSwapChain9]: send creation.\n");
#endif
	ContextAndCache * c =(ContextAndCache *)ctx;

	c->beginCommand(GetSwapChain_Opcode, deviceId);
	c->write_int(getId());
	c->write_uint(iSwapChain);
	c->endCommand();

	return 0;
}
int WrapperDirect3DSwapChain9::checkCreation(void *ctx){
#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("[WrapperDirect3DSwapChain9]: call check creation.\n");
#endif
	int ret =0; 
	ContextAndCache * cc = (ContextAndCache *)ctx;
	if(!cc->isCreated(creationFlag)){
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
		ret = 1;
	}

	return ret;
}

int WrapperDirect3DSwapChain9::checkUpdate(void *ctx){
	int ret = 0;
#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("[WrapperDirect3DSwapChain9]: check update, TODO.\n");
#endif
	return ret;
}

int WrapperDirect3DSwapChain9::sendUpdate(void *ctx){
	int ret = 0;
#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("[WrapperDirect3DSwapChain9]: send update, TODO.\n");
#endif
	return ret;
}

#endif


WrapperDirect3DSwapChain9::WrapperDirect3DSwapChain9(IDirect3DSwapChain9* ptr, int _id): m_chain(ptr), IdentifierBase(_id) {
#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::WrapperDirect3DSwapChain9() called\n");
#endif
	m_list.AddMember(ptr, this);

	creationFlag = 0;
	updateFlag = 0x8fffffff;
	stable = true;
}

WrapperDirect3DSwapChain9* WrapperDirect3DSwapChain9::GetWrapperSwapChain9(IDirect3DSwapChain9* ptr) {
#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetWrapperSwapChain9() called\n");
#endif
	WrapperDirect3DSwapChain9* ret = (WrapperDirect3DSwapChain9*)( m_list.GetDataPtr(ptr) );
#ifdef ENABLE_SWAP_CHAIN_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetWrapperSwapChain9(), ret is NULL, creating a new one\n");
		//ret = new WrapperDirect3DSwapChain9(ptr, WrapperDirect3DSwapChain9::ins_count++);
	}
#endif

	return ret;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DSwapChain9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::QueryInterface() called\n");
#endif
	HRESULT hr = m_chain->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DSwapChain9::AddRef(THIS) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::AddRef() called\n");
#endif
	refCount++;
	return m_chain->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3DSwapChain9::Release(THIS) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::Release() called\n");
#endif
	ULONG hr = m_chain->Release();
#ifdef LOG_REF_COUNT
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logError("WrapperDirect3DSwapChain9::Release(), ref:%d.\n", hr);
#endif
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DSwapChain9]: m_chain ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
}

/*** IDirect3DSwapChain9 methods ***/
STDMETHODIMP WrapperDirect3DSwapChain9::Present(THIS_ CONST RECT* pSourceRect,CONST RECT* pDestRect,HWND hDestWindowOverride,CONST RGNDATA* pDirtyRegion,DWORD dwFlags) {
#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::Present() called\n");
#endif

#ifndef MULTI_CLIENTS
	cs.begin_command(SwapChainPresent_Opcode, id);
	cs.end_command();
#else
	if(keyCmdHelper->isSending()){
		csSet->beginCommand(SwapChainPresent_Opcode, id);
		csSet->endCommand();
	}
	csSet->commit();
	Initializer * initializer = Initializer::GetInitializer();
	if(initializer){
		csSet->checkObj(dynamic_cast<IdentifierBase *>(initializer));
	}

	infoRecorder->onFrameEnd();
	keyCmdHelper->commit(cmdCtrl);
	cmdCtrl->commitRender();

#endif

	/////////////////////////////////////////////////////////////////////
	//record the fps of the game
#if 1
	static float _last_time = (float)timeGetTime();
	static float _elapse_time = 0.0;
	static float _frame_cnt = 0.0;
	static float _fps = 0.0;

	double frame_time = 0.0f;
	float cur_time = (float)timeGetTime();
	_elapse_time += (cur_time - _last_time);
	frame_time = cur_time - _last_time;

	_last_time = cur_time;
	_frame_cnt++;

	double to_sleep = 1000.0 / 30 * _frame_cnt - _elapse_time;
	if(to_sleep > 0){
		Sleep((DWORD)to_sleep);
	}

#endif
	/////////////////////////////////////////////////////////////////////
	HRESULT hr = D3D_OK;
	if(cmdCtrl->isRender()){
		hr = m_chain->Present(pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion, dwFlags);
	}
	return hr;
}

STDMETHODIMP WrapperDirect3DSwapChain9::GetFrontBufferData(THIS_ IDirect3DSurface9* pDestSurface) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetFrontBufferData() TODO\n");
#endif
	return m_chain->GetFrontBufferData(pDestSurface);
}

STDMETHODIMP WrapperDirect3DSwapChain9::GetBackBuffer(THIS_ UINT iBackBuffer,D3DBACKBUFFER_TYPE Type,IDirect3DSurface9** ppBackBuffer) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetBackBuffer() called\n");
#endif
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = m_chain->GetBackBuffer(iBackBuffer, Type, &base_surface);

	WrapperDirect3DSurface9* surface = WrapperDirect3DSurface9::GetWrapperSurface9(base_surface);
	if(surface == NULL) {
		// create new wrapper surface to hold the surface
		surface = new WrapperDirect3DSurface9(base_surface, WrapperDirect3DSurface9::ins_count++);
#ifndef MULTI_CLIENTS
		cs.begin_command(SwapChainGetBackBuffer_Opcode, id);
		cs.write_int(surface->GetID());
		cs.write_uint(iBackBuffer);
		cs.write_uint(Type);
		cs.end_command();
#else
		
		// create a new surface ?
		csSet->beginCommand(SwapChainGetBackBuffer_Opcode, id);
		csSet->writeInt(surface->getId());
		csSet->writeUInt(iBackBuffer);
		csSet->writeUInt(Type);
		csSet->endCommand();
#endif
		csSet->setCreation(surface->creationFlag);
		surface->creationCommand = SwapChainGetBackBuffer_Opcode;
		surface->iBackBuffer = iBackBuffer;
		surface->type = Type;
		surface->iSwapChain = iSwapChain;
	}

	*ppBackBuffer = surface;

	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetBackBuffer(), ppBackBuffer=%d\n", *ppBackBuffer);
#endif
	return hr;
}

STDMETHODIMP WrapperDirect3DSwapChain9::GetRasterStatus(THIS_ D3DRASTER_STATUS* pRasterStatus) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetRasterStatus() called\n");
#endif
	return m_chain->GetRasterStatus(pRasterStatus);
}

STDMETHODIMP WrapperDirect3DSwapChain9::GetDisplayMode(THIS_ D3DDISPLAYMODE* pMode) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetDisplayMode() called\n");
#endif
	return m_chain->GetDisplayMode(pMode);
}

STDMETHODIMP WrapperDirect3DSwapChain9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetDevice() TODO\n");
#endif
	return m_chain->GetDevice(ppDevice);
}

STDMETHODIMP WrapperDirect3DSwapChain9::GetPresentParameters(THIS_ D3DPRESENT_PARAMETERS* pPresentationParameters) {
	#ifdef ENABLE_SWAP_CHAIN_LOG
	infoRecorder->logTrace("WrapperDirect3DSwapChain9::GetPresentParameters() TODO\n");
#endif
	return m_chain->GetPresentParameters(pPresentationParameters);
}


