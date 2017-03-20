#include "WrapDirect3dsurface9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
#include "CommandServerSet.h"
#include "WrapDirect3DTexture9.h"

//extern int id;

#ifdef MULTI_CLIENTS


WrapperDirect3DSurface9::WrapperDirect3DSurface9(const WrapperDirect3DSurface9& sur){
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("[WrapperDirect3DSurface9]: copy constructor for %p, id:%d.\n", this, id);	
#endif
}

void WrapperDirect3DSurface9::setParentTexture(IdentifierBase *parent){
	WrapperDirect3DTexture9 * wtex = (WrapperDirect3DTexture9 *)parent;
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("[WrapperDirect3DSurface9]: set parent texture, surface id:%d, parent texture ptr:%p, id:%d(tex id:%d), level: %d.\n", id, parent, wtex->getId(), tex_id, level);
#endif
	parentTexture = parent;
}

void WrapperDirect3DSurface9::releaseData(){
#if 0
	if(surfaceHelper){
		delete surfaceHelper;
		surfaceHelper = NULL;
	}
#endif
	parentTexture = NULL;
	m_list.DeleteMember(m_surface);
}

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
		c->beginCommand(SwapChainGetBackBuffer_Opcode, iSwapChain);
		c->write_int(getId());
		c->write_uint(iBackBuffer);
		c->write_uint(type);
		c->endCommand();

	}else if(creationCommand ==  TextureGetSurfaceLevel_Opcode){
#ifdef ENABLE_SURFACE_LOG
		infoRecorder->logError("[WrapperDirect3DSurface9]: surface id:%d, created from texture %d level %d surface, parent tex:%p.\n", id, tex_id, level, parentTexture);
		if(parentTexture)
		infoRecorder->logError("[WrapperDirect3DSurface9]: parent texture id:%d creation:%s.\n", parentTexture->getId(), c->isCreated(parentTexture->creationFlag) ? "true": "false");

		else
			infoRecorder->logError("[WrapperDirect3DSurface9]: parent texture is NULL.\n");
#endif
		c->checkObj(parentTexture);
#if 0
		c->beginCommand(TextureGetSurfaceLevel_Opcode, parentTexture->getId());
		c->write_int(parentTexture->getId());
#else
		c->beginCommand(TextureGetSurfaceLevel_Opcode, tex_id);
		c->write_int(tex_id);
#endif
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


int WrapperDirect3DSurface9::GetTexId(){
	return this->tex_id;
}
int WrapperDirect3DSurface9::GetLevel(){
	return this->level;
}

WrapperDirect3DSurface9::WrapperDirect3DSurface9(IDirect3DSurface9* ptr, int _id): m_surface(ptr), IdentifierBase(_id) {

	this->tex_id = -1;
	this->level = -1;
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("WrapperDirect3DSurface9 constructor called! ptr:%d, id:%d, this:%d\n",ptr, id, this);
#endif
	m_list.AddMember(ptr, this);

	updateFlag = 0x8fffffff;
	creationFlag = 0;

	stable = false;
#ifdef USE_WRAPPER_TEXTURE
	wrappterTex9 = NULL;
#else
	surfaceHelper = NULL;
	parentTexture = NULL;
#endif
}

WrapperDirect3DSurface9* WrapperDirect3DSurface9::GetWrapperSurface9(IDirect3DSurface9* ptr) {
	WrapperDirect3DSurface9* ret = (WrapperDirect3DSurface9*)(m_list.GetDataPtr(ptr));
#ifdef ENABLE_SURFACE_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DSurface9::GetWrapperSurface9(), ret is NULL, IDSurface:%d, ins_count:%d\n",ptr,WrapperDirect3DSurface9::ins_count);
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
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("[WrapperDirect3DSurface9]: %d (tex id:%d, level:%d) add ref, ref:%d, refcount:%d.\n", id, tex_id, level, hr, refCount);
	if(hr <= 0){
		infoRecorder->logError("[WrapperDirect3DSurface9]: %d (tex id:%d, level:%d) add ref, ref:%d, refcount:%d.\n", id, tex_id, level, hr, refCount);
	}
	if(hr <= 0){
		m_list.DeleteMember(m_surface);
	}
#endif

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
	
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DSurface9]: m_surface id:%d(tex id:%d, level:%d) ref:%d, ref count:%d, creation cmd:%d, tex:%d.\n",id, tex_id, level, refCount, hr, creationCommand, tex_id);
	}
	csSet->beginCommand(D3DSurfaceRelease_Opcode, id);
	csSet->endCommand();

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
	infoRecorder->logTrace("WrapperDirect3DSurface9::GetDesc() called\n");
#endif
	return m_surface->GetDesc(pDesc);
}

STDMETHODIMP WrapperDirect3DSurface9::LockRect(THIS_ D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logTrace("WrapperDirect3DSurface9::LockRect() id:%d (tex id:%d, level:%d) called, flag:%d.\n",id, tex_id, level, Flags);
#endif

	HRESULT hr = E_FAIL;
	D3DSURFACE_DESC desc;
	hr = m_surface->GetDesc(&desc);
	if(FAILED(hr)){
		infoRecorder->logError("[WrapperDirect3DSurface9]: GetDesc failed for %d.\n", id);
	}
	hr = m_surface->LockRect(pLockedRect, pRect, Flags);

#ifdef ENABLE_SURFACE_LO

	infoRecorder->logError("WrapperDirect3DSurface9::LockRect() id:%d (tex id:%d, level:%d) called, flag:%d, locked pitch:%d, desc(width:%d, height:%d).",id, tex_id, level, Flags, pLockedRect->Pitch, desc.Width, desc.Height);

	if(pRect){
		infoRecorder->logError(" Lock Rect: (%d, %d) -> (%d, %d).\n", pRect->left, pRect->top, pRect->right, pRect->bottom);
	}
	else{
		infoRecorder->logError(" Lock Rect: NULL.\n");
	}
#endif

#ifdef USE_WRAPPER_TEXTURE   // use WrapperTexture
	if(wrappterTex9){
		// if is a texture's surface
		if(wrappterTex9->texHelper->isAutoGenable()){
			// only buffer the top surface
			if(0 == level){
				if(!wrappterTex9->texHelper->isAquired(level)){
					pLockedRect->pBits = wrappterTex9->texHelper->allocateSurfaceBuffer(0, pLockedRect->Pitch, desc.Height);
				}
				else{
					pLockedRect->pBits = wrappterTex9->texHelper->getSurfaceBuffer(level);
				}
			}
		}
		else{
			// buffer each surface
			if(!wrappterTex9->texHelper->isAquired(level)){
				pLockedRect->pBits = wrappterTex9->texHelper->allocateSurfaceBuffer(level, pLockedRect->Pitch, desc.Height);
			}
			else{
				pLockedRect->pBits = wrappterTex9->texHelper->getSurfaceBuffer(level);
			}
		}
		// 
	}
#else   // use SurfaceHelper
	if(FAILED(hr)){
		infoRecorder->logError("[WrapperDirect3DSurface9]:LockRect() failed with:%d.\n", hr);
	}
	if(surfaceHelper){
		surfaceHelper->setLockFlags(Flags);
		if(!(Flags & D3DLOCK_READONLY)){
			surfaceHelper->setRealSurfacePointer(pLockedRect->pBits);
			// if has a surface helper, means that the surface need to be stored
			if(!surfaceHelper->isAquired()){
				pLockedRect->pBits = surfaceHelper->allocateSurfaceBuffer(pLockedRect->Pitch, desc.Height);
			}else{
				pLockedRect->pBits = surfaceHelper->getSurfaceData();
			}
		}
		this->getParentTexture()->updateFlag = 0x8fffffff;
	}
#endif

	return hr;
}

STDMETHODIMP WrapperDirect3DSurface9::UnlockRect(THIS) {
#ifdef ENABLE_SURFACE_LOG
	infoRecorder->logError("WrapperDirect3DSurface9::UnlockRect() id:%d (tex id:%d, level:%d) called\n", id, tex_id, level);
#endif
	//D3DUSAGE_DYNAMIC;
	D3DLOCK_READONLY;
#ifdef USE_WRAPPER_TEXTURE
	if(wrappterTex9){
		if(wrappterTex9->texHelper->isAutoGenable()){
			if(0 == level){
				wrappterTex9->texHelper->copyTextureData(level);
			}
		}
		else{
			wrappterTex9->texHelper->copyTextureData(level);
		}
	}
#else
	// use surface helper
	if(surfaceHelper && !(surfaceHelper->getLockFlags() & D3DLOCK_READONLY)){
		surfaceHelper->copyTextureData();
	}
#endif

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
