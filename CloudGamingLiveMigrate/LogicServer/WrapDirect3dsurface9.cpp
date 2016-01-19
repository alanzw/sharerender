#include "WrapDirect3dsurface9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
#include "CommandServerSet.h"

extern int id;

#ifdef MULTI_CLIENTS
#define ENABLE_SURFACE_LOG

int WrapperDirect3DSurface9::sendCreation(void *ctx){
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("[WrapperDirect3DSurface9]: send creation\n");
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;

	if(creationCommand == D3DDeviceGetBackBuffer_Opcode){
		// from backbuffer
#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logTrace("[WrapperDirect3DSurface9]: back buffer surface.\n");
#endif
		c->beginCommand(D3DDeviceGetBackBuffer_Opcode, getDeviceId());
		c->write_int(getId());
		c->write_uint(iSwapChain);
		c->write_uint(iBackBuffer);
		c->write_uint(type);
		c->endCommand();

	}else if( creationCommand == SwapChainGetBackBuffer_Opcode){
		// from swapchain
#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logTrace("[WrapperDirect3DSurface9}; swap chain back buffer.\n");
#endif
		c->beginCommand(SwapChainGetBackBuffer_Opcode, swapChainId);
		c->write_int(getId());
		c->write_uint(iBackBuffer);
		c->write_uint(type);
		c->endCommand();

	}else if(creationCommand ==  TextureGetSurfaceLevel_Opcode){
#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logTrace("[WrapperDirect3DSurface9]: texture level surface.\n");
#endif

		c->beginCommand(TextureGetSurfaceLevel_Opcode, tex_id);
		c->write_int(tex_id);
		c->write_int(getId());
		c->write_uint(level);
		c->endCommand();
	}else if(creationCommand == CreateDepthStencilSurface_Opcode){
		// created by user
#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logTrace("[WrapperDirect3DSurface9]: depth stencil surface.\n");
#endif
		c->beginCommand(CreateDepthStencilSurface_Opcode, getDeviceId());
		c->write_uint(getId());
		c->write_uint(width);
		c->write_uint(height);
		c->write_uint(format);
		c->write_uint(multiSample);
		c->write_uint(multisampleQuality);
		c->write_uint(discard);
		c->endCommand();

	}else if(creationCommand == D3DDGetRenderTarget_Opcode){
		// the surface is the render target
#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logTrace("[WrapperDirect3DSurface9]: render target surface.\n");
#endif

		c->beginCommand(D3DDGetRenderTarget_Opcode, getDeviceId());
		c->write_int(getId());
		c->write_uint(renderTargetIndex);
		c->endCommand();
	}else if(creationCommand == GetDepthStencilSurface_Opcode){
		// created by get depth stencil surface

#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logTrace("[WrapperDirect3DSurface9]: depth stencil surface.\n");
#endif
		c->beginCommand(GetDepthStencilSurface_Opcode, getDeviceId());
		c->write_int(getId());
		c->endCommand();
	}else if(creationCommand == CubeGetCubeMapSurface_Opcode){
		// 
#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logError("[WrapperDirect3DSurface9]: get cube map surface? ERROR, not implemented.\n");
#endif
	}
	return 0;
}

int WrapperDirect3DSurface9::checkCreation(void *ctx){

#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("[WrapperDirect3DSurface9]: call check creation.\n");
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(!c->isCreated(creationFlag)){
		ret = sendCreation(ctx);
		c->setCreation(creationFlag);
		ret = 1;
	}

	return ret;

}
int WrapperDirect3DSurface9::checkUpdate(void *ctx)
{
	int ret = 0;
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("[WrapperDirect3DSurface9]: check update, TODO.\n");
#endif

	return ret;

}
int WrapperDirect3DSurface9::sendUpdate(void *ctx){
	int ret = 0;
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("[WrapperDirect3DSurface9]: send update, TODO.\n");
#endif

	return ret;
}

#endif

void WrapperDirect3DSurface9::RepalceSurface(IDirect3DSurface9* pnew){
	/*this->m_surface->Release();
	this->m_surface = pnew;*/
}
int WrapperDirect3DSurface9::GetTexId(){
	return this->tex_id;
}
int WrapperDirect3DSurface9::GetLevel(){
	return this->level;
}
void WrapperDirect3DSurface9::SendSurface(){
	//if(isSent == false){
	// send data to client
	D3DLOCKED_RECT rect;
	D3DSURFACE_DESC desc;
	this->LockRect(&rect, 0, 0);
	this->GetDesc(&desc);
	int byte_per_pixel = rect.Pitch / desc.Width;

	LPD3DXBUFFER tbuf = NULL;
	D3DXSaveSurfaceToFileInMemory(&tbuf,D3DXIFF_PNG,this,NULL,NULL);
	int size = tbuf->GetBufferSize();
	
#if 0
	char  fname[50];
	sprintf(fname,"surface\\face_%d.png",this->id);
	D3DXSaveSurfaceToFile(fname, D3DXIFF_PNG,this, NULL, NULL);
#endif
	isSent = true;
}
WrapperDirect3DSurface9::WrapperDirect3DSurface9(IDirect3DSurface9* ptr, int _id): m_surface(ptr), IdentifierBase(_id) {

	this->tex_id = -1;
	this->level = -1;
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9 constructor called! ptr:%d, id:%d, this:%d\n",ptr, id,this);
#endif
	m_list.AddMember(ptr, this);
	this->isSent = false;

	updateFlag = 0x8fffffff;
	creationFlag = 0;

	stable = false;
	wrappterTex9 = NULL;
}

WrapperDirect3DSurface9* WrapperDirect3DSurface9::GetWrapperSurface9(IDirect3DSurface9* ptr) {
	//infoRecorder->logError("WrapperDirect3DSurface9::GetWrapperSurface9(), ptr=%u\n", ptr);
	WrapperDirect3DSurface9* ret = (WrapperDirect3DSurface9*)(m_list.GetDataPtr(ptr));
#ifdef ENABLE_SURFACE_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DSurface9::GetWrapperSurface9(), ret is NULL, IDSurface:%d, ins_count:%d\n",ptr,WrapperDirect3DSurface9::ins_count);
		// add the following line , 2013/7/23 21:12
		//	ret = new WrapperDirect3DSurface9(ptr, WrapperDirect3DSurface9::ins_count++);
	}
#endif
	return ret;
}

IDirect3DSurface9* WrapperDirect3DSurface9::GetSurface9() {
	return this->m_surface;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DSurface9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::QueryInterface() called\n");
#endif
	HRESULT hr = m_surface->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}

STDMETHODIMP_(ULONG) WrapperDirect3DSurface9::AddRef(THIS) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::AddRef() called\n");
#endif
	refCount++;
	ULONG hr = m_surface->AddRef();
	infoRecorder->logError("[WrapperDirect3DSurface9]: %d (tex id:%d, level:%d) add ref, ref:%d, refcount:%d.\n", id, tex_id, level, hr, refCount);

	return hr; 
}
STDMETHODIMP_(ULONG) WrapperDirect3DSurface9::Release(THIS) {
	ULONG hr = m_surface->Release();
#ifdef LOG_REF_COUNT
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::Release(), ref:%d.\n", hr);
#endif
#endif
	refCount--;
	//if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DSurface9]: m_surface id:%d(tex id:%d, level:%d) ref:%d, ref count:%d, creation cmd:%d, tex:%d.\n",id, tex_id, level, refCount, hr, creationCommand, tex_id);
	//}


	if(hr == 0){
		if(!m_list.DeleteMember(m_surface)){
			infoRecorder->logError("[WrapperDirect3DSurface9]: delete %d from m_list failed.\n", id);

		}
	}
	return hr;
}

/*** IDirect3DResource9 methods ***/
STDMETHODIMP WrapperDirect3DSurface9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetDevice() TODO\n");
#endif
	IDirect3DDevice9* base_device = NULL;
	HRESULT hr = m_surface->GetDevice(&base_device);

	WrapperDirect3DDevice9* device = WrapperDirect3DDevice9::GetWrapperDevice9(base_device);
	*ppDevice = device;

#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetDevice(), base_device=%d, device=%d\n", base_device, device);
#endif

	return hr;
}

STDMETHODIMP WrapperDirect3DSurface9::SetPrivateData(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags){
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::SetPrivateData() TODO\n");
#endif
	return m_surface->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

STDMETHODIMP WrapperDirect3DSurface9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetPrivateData() called\n");
#endif
	return m_surface->GetPrivateData(refguid, pData, pSizeOfData);
}

STDMETHODIMP WrapperDirect3DSurface9::FreePrivateData(THIS_ REFGUID refguid) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::FreePrivateData() called\n");
#endif
	return m_surface->FreePrivateData(refguid);
}

STDMETHODIMP_(DWORD) WrapperDirect3DSurface9::SetPriority(THIS_ DWORD PriorityNew) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::SetPriority() TODO\n");
#endif
	return m_surface->SetPriority(PriorityNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DSurface9::GetPriority(THIS) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetPriority() called\n");
#endif
	return m_surface->GetPriority();
}

STDMETHODIMP_(void) WrapperDirect3DSurface9::PreLoad(THIS) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::PreLoad() called\n");
#endif
	return m_surface->PreLoad();
}

STDMETHODIMP_(D3DRESOURCETYPE) WrapperDirect3DSurface9::GetType(THIS) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetType() called\n");
#endif
	return m_surface->GetType();
}

STDMETHODIMP WrapperDirect3DSurface9::GetContainer(THIS_ REFIID riid,void** ppContainer) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetContainer() TODO\n");
#endif
	return m_surface->GetContainer(riid, ppContainer);
}

STDMETHODIMP WrapperDirect3DSurface9::GetDesc(THIS_ D3DSURFACE_DESC *pDesc) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("WrapperDirect3DSurface9::GetDesc() called\n");
#endif
	return m_surface->GetDesc(pDesc);
}

STDMETHODIMP WrapperDirect3DSurface9::LockRect(THIS_ D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("WrapperDirect3DSurface9::LockRect() called\n");
#endif
	HRESULT hr = m_surface->LockRect(pLockedRect, pRect, Flags);

	if(wrappterTex9 && 0 == level){
		// if is a texture's surface
		if(!wrappterTex9->texHelper){
			wrappterTex9->texHelper = new TextureHelper(pLockedRect->Pitch, wrappterTex9->Height, wrappterTex9->Levels > 1? true : false);
			wrappterTex9->texHelper->allocateTextureBuffer();
		}
		// 
		wrappterTex9->texHelper->getRectAddr()->pBits = pLockedRect->pBits;
		pLockedRect->pBits = wrappterTex9->texHelper->getRectAddr()->pBits;
	}

	return hr;
}

STDMETHODIMP WrapperDirect3DSurface9::UnlockRect(THIS) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("WrapperDirect3DSurface9::UnlockRect() called\n");
#endif
	if(level == 0 && wrappterTex9){
		// if is a texture's top surface
		wrappterTex9->texHelper->copyTextureData(wrappterTex9->texHelper->getRectAddr()->pBits);
		wrappterTex9->texHelper->getRectAddr()->pBits = NULL;
	}

	return m_surface->UnlockRect();
}

STDMETHODIMP WrapperDirect3DSurface9::GetDC(THIS_ HDC *phdc) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetDC() called\n");
#endif
	return m_surface->GetDC(phdc);
}

STDMETHODIMP WrapperDirect3DSurface9::ReleaseDC(THIS_ HDC hdc) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::ReleaseDC() called\n");
#endif
	return m_surface->ReleaseDC(hdc);
}
