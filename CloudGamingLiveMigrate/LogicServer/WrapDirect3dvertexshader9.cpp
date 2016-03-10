
#include "WrapDirect3dvertexshader9.h"
#include "CommandServerSet.h"
#ifdef MULTI_CLIENTS


int WrapperDirect3DVertexShader9::sendCreation(void *ctx){
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexshader9]: send creation.\n");
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;
	c->beginCommand(CreateVertexShader_Opcode, getDeviceId());
	c->write_int(getId());
	c->write_int(funCount);
	c->write_byte_arr((char *)this->shaderData, shaderLen);
	c->endCommand();
	return 0;
}

int WrapperDirect3DVertexShader9::checkCreation(void *ctx){
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexShader9]: check creation.\n");
#endif
	int ret = 0;
	ContextAndCache *cc = (ContextAndCache *)ctx;
	if(!cc->isCreated(creationFlag)){
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
		ret = 1;
	}
	return ret;
}

int WrapperDirect3DVertexShader9::checkUpdate(void *ctx){
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexShader9]: check update, TODO.\n");
#endif
	int ret = 0;
	return ret;
}

int WrapperDirect3DVertexShader9::sendUpdate(void *ctx){
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logTrace("[WrapeprDirect3DVertexshader9]: send update, TODO.\n");
#endif
	int ret = 0;
	return ret;
}
#endif // MULTI_CLIENTS

WrapperDirect3DVertexShader9::WrapperDirect3DVertexShader9(IDirect3DVertexShader9* ptr, int _id): m_vs(ptr), IdentifierBase(_id) {
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexShader9 constructor called!\n");
#endif
	m_list.AddMember(ptr, this);
	creationFlag = 0;
	updateFlag = 0x8fffffff;
	stable = true;
}

IDirect3DVertexShader9* WrapperDirect3DVertexShader9::GetVS9() {
	return m_vs;
}

WrapperDirect3DVertexShader9* WrapperDirect3DVertexShader9::GetWrapperVertexShader(IDirect3DVertexShader9* ptr) {
	WrapperDirect3DVertexShader9* ret = (WrapperDirect3DVertexShader9*)( m_list.GetDataPtr(ptr) );
#ifdef ENABLE_VERTEX_SHADER_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DPixelShader9::GetWrapperPixelShader(), ret is NULL\n");
	}
#endif
	return ret;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DVertexShader9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	HRESULT hr = m_vs->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DVertexShader9::AddRef(THIS) {
	refCount++;
	return m_vs->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3DVertexShader9::Release(THIS) {
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexShader9::Release() called! id:%d\n", id);
#endif
	ULONG hr = m_vs->Release();
#ifdef LOG_REF_COUNT
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logError("WrapperDirect3DVertexShader9::Release(), ref:%d.\n", hr);
#endif
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logTrace("[WrapperDirect3DVertexShader9]: m_vs ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
}

/*** IDirect3DVertexShader9 methods ***/
STDMETHODIMP WrapperDirect3DVertexShader9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
#ifdef ENABLE_VERTEX_SHADER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexShader9::GetDevice called!\n");
#endif
	return m_vs->GetDevice(ppDevice);
}
STDMETHODIMP WrapperDirect3DVertexShader9::GetFunction(THIS_ void* ptr,UINT* pSizeOfData) {
	return m_vs->GetFunction(ptr, pSizeOfData);
}

