#include "WrapDirect3dcubetexture9.h"
#include "WrapDirect3dsurface9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
#include "../LibCore/Log.h"
#include "CommandServerSet.h"

#ifndef MULTI_CLIENTS
#define MULTI_CLIENTS
#endif   // MULTI_CLIENTS

#define ENABLE_CUBE_TEXTURE_LOG

#ifdef MULTI_CLIENTS

int WrapperDirect3DCubeTexture9::sendCreation(void * ctx){
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("[WrapperDirect3DCubeTextrue9]: send the creation for cube tex:%d.\n", id);
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;

	c->beginCommand( CreateCubeTexture_Opcode, getDeviceId());
	c->write_int( this->getId());
	c->write_uint( this->edgeLength);
	c->write_uint( this->levels);
	c->write_uint( this->usage);
	c->write_uint( this->format);
	c->write_uint( this->pool);
	c->endCommand();

	return 0;
}

int WrapperDirect3DCubeTexture9::checkCreation(void * ctx){
	int ret = 0;
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("[WrapperDirect3DCubeTexture9]: call check creation for cube tex:%d.\n", id);
#endif
	ContextAndCache *cc = (ContextAndCache *)ctx;
	if(!cc->isCreated(creationFlag)){
		// did not created
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
		ret = 1;
	}
	return ret;
	
}
// send the texture data when first used
int WrapperDirect3DCubeTexture9::checkUpdate(void * ctx){
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("[WrapperDirect3DCubeTexture9]: check update for cube tex:%d.\n", id);
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(c->isChanged(updateFlag)){
		// send
		ret = sendUpdate(c);
		c->resetChanged(updateFlag);
		ret = 1;
	}
	else{
		infoRecorder->logTrace("[WrapperDirect3DCudaTexture9]: not changed.\n");
	}


	return ret;
}
int WrapperDirect3DCubeTexture9::sendUpdate(void *ctx){
	infoRecorder->logError("[WrapeprDirect3DCubeTexture9]: send update, TODO.\n");
	int ret = 0;

	return ret;
}

#endif

WrapperDirect3DCubeTexture9::WrapperDirect3DCubeTexture9(IDirect3DCubeTexture9* ptr, int _id): m_cube_tex(ptr), IdentifierBase(_id) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::WrapperDirect3DCubeTexture9() called, base_tex:0x%p, WrapperedTex:0x%p\n", ptr, this);

	m_list.AddMember(ptr, this);
	creationFlag = 0;
	updateFlag = 0x8fffffff;
}

IDirect3DCubeTexture9* WrapperDirect3DCubeTexture9::GetCubeTex9() {
	return this->m_cube_tex;
}

WrapperDirect3DCubeTexture9* WrapperDirect3DCubeTexture9::GetWrapperCubeTexture9(IDirect3DCubeTexture9* ptr) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetWrapperCubeTexture9()\n");
	WrapperDirect3DCubeTexture9* ret = (WrapperDirect3DCubeTexture9*)( m_list.GetDataPtr(ptr) );
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetWrapperCubeTexture9(), ret is NULL\n");
	}
	return ret;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DCubeTexture9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::QueryInterface(), ppvObj=%d\n", *ppvObj);
	HRESULT hr = m_cube_tex->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::QueryInterface() end, ppvObj=%d\n", *ppvObj);
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DCubeTexture9::AddRef(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::AddRef() called\n");
	refCount++;
	return m_cube_tex->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3DCubeTexture9::Release(THIS) {
	HRESULT hr = m_cube_tex->Release();
#ifdef LOG_REF_COUNT
	infoRecorder->logError("WrapperDirect3DCubeTexture9::Release(), ref:%d\n", hr);
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DCubTexture9]: tex ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
}

/*** IDirect3DBaseTexture9 methods ***/
STDMETHODIMP WrapperDirect3DCubeTexture9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetDevice() TODO\n");
#endif
	//return m_cube_tex->GetDevice(ppDevice);
	IDirect3DDevice9* base = NULL;
	HRESULT hr = this->m_cube_tex->GetDevice(&base);
	WrapperDirect3DDevice9 * ret = WrapperDirect3DDevice9::GetWrapperDevice9(base);
	if(ret == NULL){
		infoRecorder->logError("WrapperDirect3DCubeTexture9::GetDevice() return NULL\n");
	}
	else{

	}
	//*ppDevice = ret;
	*ppDevice = dynamic_cast<IDirect3DDevice9 *>(base);
	return hr;
}
STDMETHODIMP WrapperDirect3DCubeTexture9::SetPrivateData(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::SetPrivateData() TODO\n");
#endif
	return m_cube_tex->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

STDMETHODIMP WrapperDirect3DCubeTexture9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetPrivateData() TODO\n");
#endif
	return m_cube_tex->GetPrivateData(refguid, pData, pSizeOfData);
}
STDMETHODIMP WrapperDirect3DCubeTexture9::FreePrivateData(THIS_ REFGUID refguid) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::FreePrivateData() TODO\n");
#endif
	return m_cube_tex->FreePrivateData(refguid);
}
STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::SetPriority(THIS_ DWORD PriorityNew) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::SetPriority() TODO\n");
#endif
	return m_cube_tex->SetPriority(PriorityNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::GetPriority(THIS) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetPriority() called\n");
#endif
	return m_cube_tex->GetPriority();
}

STDMETHODIMP_(void) WrapperDirect3DCubeTexture9::PreLoad(THIS) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::PreLoad() TODO\n");
#endif
	return m_cube_tex->PreLoad();
}

STDMETHODIMP_(D3DRESOURCETYPE) WrapperDirect3DCubeTexture9::GetType(THIS) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetType() called\n");
#endif
	return m_cube_tex->GetType();
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::SetLOD(THIS_ DWORD LODNew) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::SetLOD() TODO\n");
#endif
	return m_cube_tex->SetLOD(LODNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::GetLOD(THIS) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::GetLOD() called\n");
#endif
	return m_cube_tex->GetLOD();
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::GetLevelCount(THIS) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::GetLevelCount() TODO\n");
#endif
	return m_cube_tex->GetLevelCount();
}

STDMETHODIMP WrapperDirect3DCubeTexture9::SetAutoGenFilterType(THIS_ D3DTEXTUREFILTERTYPE FilterType) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::SetAutoGenFilterType() TODO\n");
#endif
	return m_cube_tex->SetAutoGenFilterType(FilterType);
}

STDMETHODIMP_(D3DTEXTUREFILTERTYPE) WrapperDirect3DCubeTexture9::GetAutoGenFilterType(THIS) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::GetAutoGenFilterType() called\n");
#endif
	return m_cube_tex->GetAutoGenFilterType();
}

STDMETHODIMP_(void) WrapperDirect3DCubeTexture9::GenerateMipSubLevels(THIS) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::GenerateMipSubLevels() TODO\n");
#endif
	return m_cube_tex->GenerateMipSubLevels();
}

STDMETHODIMP WrapperDirect3DCubeTexture9::GetLevelDesc(THIS_ UINT Level,D3DSURFACE_DESC *pDesc) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::GetLevelDesc() TODO\n");
#endif
	return m_cube_tex->GetLevelDesc(Level, pDesc);
}


// the get surface is to create a new surface
STDMETHODIMP WrapperDirect3DCubeTexture9::GetCubeMapSurface(THIS_ D3DCUBEMAP_FACES FaceType,UINT Level,IDirect3DSurface9** ppCubeMapSurface) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::GetCubeMapSurface(), id:%d, face:%d, level:%d\n", id, FaceType, Level);
#endif
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = m_cube_tex->GetCubeMapSurface(FaceType, Level, &base_surface);

	WrapperDirect3DSurface9* surface = WrapperDirect3DSurface9::GetWrapperSurface9(base_surface);
	if(surface == NULL) {
		surface = new WrapperDirect3DSurface9(base_surface, WrapperDirect3DSurface9::ins_count++);

#ifndef MULTI_CLIENTS
		cs.begin_command(CubeGetCubeMapSurface_Opcode, id);
		cs.write_int(surface->getId());
		cs.write_uint(FaceType);
		cs.write_uint(Level);
		cs.end_command();
#else	// MULTI_CLIENTS
		// check texture creation
		//csSet->checkCreation(this);
		csSet->checkObj(dynamic_cast<IdentifierBase *>(this));
		// second, use the object
		csSet->beginCommand(CubeGetCubeMapSurface_Opcode, id);
		csSet->writeInt(surface->getId());
		csSet->writeUInt(FaceType);
		csSet->writeUInt(Level);
		csSet->endCommand();

		// TODO
		// do not distinguish the new-add client ?
		surface->creationCommand = CubeGetCubeMapSurface_Opcode; 
		surface->SetTexId(id);
		surface->SetLevel(Level);
		surface->SetFaceType(FaceType);
#endif	// MULTI_CLIENTS
		
	}
	*ppCubeMapSurface = dynamic_cast<IDirect3DSurface9 *>(surface);

	return hr;
}

STDMETHODIMP WrapperDirect3DCubeTexture9::LockRect(THIS_ D3DCUBEMAP_FACES FaceType,UINT Level,D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::LockRect(), id:%d, face type:%d, level:%d.\n", id, FaceType, Level);
#endif
	csSet->setChangedToAll(updateFlag);
	return m_cube_tex->LockRect(FaceType, Level, pLockedRect, pRect, Flags);
}

STDMETHODIMP WrapperDirect3DCubeTexture9::UnlockRect(THIS_ D3DCUBEMAP_FACES FaceType,UINT Level) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::UnlockRect(), Level=%d\n", Level);
#endif

	//this->SendTextureData();
	//tex_send[this->id] = true;

	return m_cube_tex->UnlockRect(FaceType, Level);
}

STDMETHODIMP WrapperDirect3DCubeTexture9::AddDirtyRect(THIS_ D3DCUBEMAP_FACES FaceType,CONST RECT* pDirtyRect) {
#ifdef ENABLE_CUBE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DCubeTexture9::AddDirtyRect() TODO\n");
#endif
	
	return m_cube_tex->AddDirtyRect(FaceType, pDirtyRect);
}
