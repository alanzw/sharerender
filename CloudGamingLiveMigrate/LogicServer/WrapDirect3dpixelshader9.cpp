#include "WrapDirect3dpixelshader9.h"
#include "CommandServerSet.h"
#ifdef MULTI_CLIENTS


int WrapperDirect3DPixelShader9::sendCreation(void * ctx){
#ifdef ENABLE_PIXEL_SHADER_LOG
	infoRecorder->logTrace("[WrapperDirect3DpixelShader9]: send creation.\n");
#endif
	
	ContextAndCache * c = (ContextAndCache *)ctx;

	c->beginCommand(CreatePixelShader_Opcode, getDeviceId());
	c->write_int(getId());
	c->write_int(funcCount);
	c->write_byte_arr((char *)pFunc, sizeof(DWORD) * funcCount);
	c->endCommand();
	return 0;
}

// check the pixel shader's creation flag
int WrapperDirect3DPixelShader9::checkCreation(void * ctx){
	#ifdef ENABLE_PIXEL_SHADER_LOG
	infoRecorder->logTrace("[WrapperDirect3DPixelShader9]: call check creation.\n");
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(!c->isCreated(creationFlag)){
		//
		ret = sendCreation(ctx);
		c->setCreation(creationFlag);
		//csSet->setCreation(creationFlag);
		ret = 1;
	}

	return ret;
}

int WrapperDirect3DPixelShader9::checkUpdate(void *ctx){
	int ret = 0;
	#ifdef ENABLE_PIXEL_SHADER_LOG
	infoRecorder->logTrace("[WrapperDirect3DPixelShader]: check update, TODO.\n");
#endif

	return ret;
}
int WrapperDirect3DPixelShader9::sendUpdate(void *ctx){
	int ret =0;
	#ifdef ENABLE_PIXEL_SHADER_LOG
	infoRecorder->logTrace("[WrapeprDirect3DPixelShader]: send update, TODO.\n");
#endif

	return ret;
}

#endif

WrapperDirect3DPixelShader9::WrapperDirect3DPixelShader9(IDirect3DPixelShader9* ptr, int _id): m_ps(ptr), IdentifierBase(_id) {

	m_list.AddMember(ptr, this);
	creationFlag = 0;
	updateFlag = 0x8fffffff;
	stable = true;
}

IDirect3DPixelShader9* WrapperDirect3DPixelShader9::GetPS9() {
	return m_ps;
}

WrapperDirect3DPixelShader9* WrapperDirect3DPixelShader9::GetWrapperPixelShader(IDirect3DPixelShader9* ptr) {
	WrapperDirect3DPixelShader9* ret = (WrapperDirect3DPixelShader9*)( m_list.GetDataPtr(ptr) );
	#ifdef ENABLE_PIXEL_SHADER_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DPixelShader9::GetWrapperPixelShader(), ret is NULL\n");
	}
#endif
	return ret;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DPixelShader9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	HRESULT hr = m_ps->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DPixelShader9::AddRef(THIS) {
	refCount++;
	return m_ps->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3DPixelShader9::Release(THIS) {
	#ifdef ENABLE_PIXEL_SHADER_LOG
	infoRecorder->logTrace("WrapperDirect3DPixelShader9::Release() called! id:%d\n", this->id);
#endif
	ULONG hr = m_ps->Release();
#ifdef LOG_REF_COUNT
	#ifdef ENABLE_PIXEL_SHADER_LOG
	infoRecorder->logError("WrapperDirect3DPixelShader9::Release(), ref:%d.\n", hr);
#endif
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logTrace("[WrapperDirect3DPixelShader9]: m_ps ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
}

/*** IDirect3DPixelShader9 methods ***/
STDMETHODIMP WrapperDirect3DPixelShader9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
	return m_ps->GetDevice(ppDevice);
}

STDMETHODIMP WrapperDirect3DPixelShader9::GetFunction(THIS_ void* ptr,UINT* pSizeOfData) {
	return m_ps->GetFunction(ptr, pSizeOfData);
}

