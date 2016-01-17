#include "WrapDirect3dcubetexture9.h"
#include "WrapDirect3dsurface9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
#include "../LibCore/Log.h"
#include "CommandServerSet.h"

#ifndef MULTI_CLIENTS
#define MULTI_CLIENTS
#endif   // MULTI_CLIENTS

#ifdef MULTI_CLIENTS

int WrapperDirect3DCubeTexture9::sendCreation(void * ctx){
	infoRecorder->logTrace("[WrapperDirect3DCubeTextrue9]: send the creation command.\n");
	ContextAndCache * c = (ContextAndCache *)ctx;

	c->beginCommand( CreateCubeTexture_Opcode, getDeviceId());
	c->write_int( this->getId());
	c->write_uint( this->edgeLength);
	c->write_uint( this->levels);
	c->write_uint( this->usage);
	c->write_uint( this->format);
	c->write_uint( this->format);
	c->write_uint( this->pool);
	c->endCommand();

	return 0;
}

int WrapperDirect3DCubeTexture9::checkCreation(void * ctx){
	int ret = 0;
	infoRecorder->logError("[WrapperDirect3DCubeTexture9]: call check creation.\n");
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
	infoRecorder->logTrace("[WrapperDirect3DCubeTexture9]: check update.\n");
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
	infoRecorder->logTrace("[WrapeprDirect3DCubeTexture9]: send update, TODO.\n");
	int ret = 0;

	return ret;
}

#endif

WrapperDirect3DCubeTexture9::WrapperDirect3DCubeTexture9(IDirect3DCubeTexture9* ptr, int _id): m_cube_tex(ptr), IdentifierBase(_id) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::WrapperDirect3DCubeTexture9() called\n");

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
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetDevice() TODO\n");
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
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::SetPrivateData() TODO\n");
	return m_cube_tex->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

STDMETHODIMP WrapperDirect3DCubeTexture9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetPrivateData() TODO\n");
	return m_cube_tex->GetPrivateData(refguid, pData, pSizeOfData);
}
STDMETHODIMP WrapperDirect3DCubeTexture9::FreePrivateData(THIS_ REFGUID refguid) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::FreePrivateData() TODO\n");
	return m_cube_tex->FreePrivateData(refguid);
}
STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::SetPriority(THIS_ DWORD PriorityNew) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::SetPriority() TODO\n");
	return m_cube_tex->SetPriority(PriorityNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::GetPriority(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetPriority() called\n");
	return m_cube_tex->GetPriority();
}

STDMETHODIMP_(void) WrapperDirect3DCubeTexture9::PreLoad(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::PreLoad() TODO\n");
	return m_cube_tex->PreLoad();
}

STDMETHODIMP_(D3DRESOURCETYPE) WrapperDirect3DCubeTexture9::GetType(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetType() called\n");
	return m_cube_tex->GetType();
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::SetLOD(THIS_ DWORD LODNew) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::SetLOD() TODO\n");
	return m_cube_tex->SetLOD(LODNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::GetLOD(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetLOD() called\n");
	return m_cube_tex->GetLOD();
}

STDMETHODIMP_(DWORD) WrapperDirect3DCubeTexture9::GetLevelCount(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetLevelCount() TODO\n");
	return m_cube_tex->GetLevelCount();
}

STDMETHODIMP WrapperDirect3DCubeTexture9::SetAutoGenFilterType(THIS_ D3DTEXTUREFILTERTYPE FilterType) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::SetAutoGenFilterType() TODO\n");
	return m_cube_tex->SetAutoGenFilterType(FilterType);
}

STDMETHODIMP_(D3DTEXTUREFILTERTYPE) WrapperDirect3DCubeTexture9::GetAutoGenFilterType(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetAutoGenFilterType() called\n");
	return m_cube_tex->GetAutoGenFilterType();
}

STDMETHODIMP_(void) WrapperDirect3DCubeTexture9::GenerateMipSubLevels(THIS) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GenerateMipSubLevels() TODO\n");
	return m_cube_tex->GenerateMipSubLevels();
}

STDMETHODIMP WrapperDirect3DCubeTexture9::GetLevelDesc(THIS_ UINT Level,D3DSURFACE_DESC *pDesc) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetLevelDesc() TODO\n");
	return m_cube_tex->GetLevelDesc(Level, pDesc);
}


// the get surface is to create a new surface
STDMETHODIMP WrapperDirect3DCubeTexture9::GetCubeMapSurface(THIS_ D3DCUBEMAP_FACES FaceType,UINT Level,IDirect3DSurface9** ppCubeMapSurface) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::GetCubeMapSurface()\n");
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
#else
		// check texture creation
		csSet->checkCreation(this);
		//csSet->checkObj(this);
		// second, use the object
		csSet->beginCommand(CubeGetCubeMapSurface_Opcode, id);
		csSet->writeInt(surface->getId());
		csSet->writeUInt(FaceType);
		csSet->writeUInt(Level);
		csSet->endCommand();

		// TODO
		// do not distinguish the new-add client ?
		surface->creationCommand = CubeGetCubeMapSurface_Opcode; 
#endif
		
	}

	*ppCubeMapSurface = dynamic_cast<IDirect3DSurface9 *>(surface);

	return hr;
}

STDMETHODIMP WrapperDirect3DCubeTexture9::LockRect(THIS_ D3DCUBEMAP_FACES FaceType,UINT Level,D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags) {
	
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::LockRect()\n");
	csSet->setChangedToAll(updateFlag);
	return m_cube_tex->LockRect(FaceType, Level, pLockedRect, pRect, Flags);
}

STDMETHODIMP WrapperDirect3DCubeTexture9::UnlockRect(THIS_ D3DCUBEMAP_FACES FaceType,UINT Level) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::UnlockRect(), Level=%d\n", Level);

	//this->SendTextureData();
	//tex_send[this->id] = true;

	return m_cube_tex->UnlockRect(FaceType, Level);
}

STDMETHODIMP WrapperDirect3DCubeTexture9::AddDirtyRect(THIS_ D3DCUBEMAP_FACES FaceType,CONST RECT* pDirtyRect) {
	infoRecorder->logTrace("WrapperDirect3DCubeTexture9::AddDirtyRect() TODO\n");
	
	return m_cube_tex->AddDirtyRect(FaceType, pDirtyRect);
}
