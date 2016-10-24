#include "WrapDirect3dvolumetexture9.h"
#include "WrapDirect3ddevice9.h"
#include "WrapDirect3dvolume9.h"
#include "../LibCore/Opcode.h"
#include "CommandServerSet.h"

int WrapperDirect3DVolumeTexture9::ins_count = 0;
HashSet WrapperDirect3DVolumeTexture9::m_list;

#define ENABLE_VOLUME_TEXTURE_LOG


//check the creation flag for volume texture
#ifdef MULTI_CLIENTS
int WrapperDirect3DVolumeTexture9::sendCreation(void *ctx){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DVolumeTexture9]: send creation, TODO.\n");
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;

	//c->beginCommand(Create)

	return 0;
}

int WrapperDirect3DVolumeTexture9::checkCreation(void *ctx){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DVolumeTexture9]: call check creation, TODO.\n");
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

int WrapperDirect3DVolumeTexture9::checkUpdate(void *ctx){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DVolumeTexture9]: check update, TODO.\n");
#endif
	int ret = 0;
	
	return ret;
}
int WrapperDirect3DVolumeTexture9::sendUpdate(void *ctx){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DVolumeTexture9]: send update, TODO.\n");
#endif
	int ret =0;

	return ret;
}

#endif

WrapperDirect3DVolumeTexture9::WrapperDirect3DVolumeTexture9(IDirect3DVolumeTexture9 * ptr, int _id): m_tex(ptr), IdentifierBase(_id){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9 constructor called!\n");
#endif
	m_list.AddMember(ptr,this);
	creationFlag = 0;
	updateFlag = 0x8fffffff;
	stable = true;
}

WrapperDirect3DVolumeTexture9 * WrapperDirect3DVolumeTexture9::GetWrapperTexture9(IDirect3DVolumeTexture9 * ptr){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9:: GetWrapperDirect3DVolumeTexture9 called!\n");
#endif
	WrapperDirect3DVolumeTexture9 * ret = NULL;
	ret = (WrapperDirect3DVolumeTexture9 *)m_list.GetDataPtr(ptr);
	if(ret == NULL){
		infoRecorder->logTrace("ERROR:GetWrapperDirect3DVolumeTexture9 return NULL\n");
	}
	return ret;
}
IDirect3DVolumeTexture9 * WrapperDirect3DVolumeTexture9::GetVolumeTex9(){
	return this->m_tex;
}

/******* IUnknown Mehtods ***********/
STDMETHODIMP WrapperDirect3DVolumeTexture9::QueryInterface(THIS_ REFIID riid, void **ppvObj){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::QueryInterface called!\n");
#endif
	HRESULT hr = this->m_tex->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DVolumeTexture9::AddRef(THIS){

#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::AddRef called!\n");
#endif
	refCount++;
	return this->m_tex->AddRef();
}

STDMETHODIMP_(ULONG) WrapperDirect3DVolumeTexture9::Release(THIS){
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::Release() called! id:%d\n",id);
#endif
	ULONG hr = m_tex->Release();
#ifdef LOG_REF_COUNT
	infoRecorder->logError("WrapperDirect3DVolumeTexture9::Release(), ref:%d.\n", hr);
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DVolumeTexture9]: m_volumeTexture ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
}

/*************** Methods **************/
STDMETHODIMP WrapperDirect3DVolumeTexture9::GetDevice(THIS_ IDirect3DDevice9** ppDevice)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9:: GetDevice called!\n");
#endif
	IDirect3DDevice9 * base_device = NULL;
	HRESULT hr = this->m_tex->GetDevice(ppDevice);
	//WrapperDirect3DVolumeTexture9 * ret = GetWrapperDirect3DVolumeTexture9(base_device);
	
	//*ppDevice = device;
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::SetPrivateData(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9:: SetPrivateData()  TODO!!\n");
#endif
	HRESULT hr = this->m_tex->SetPrivateData(refguid, pData, SizeOfData, Flags);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::GetPrivateData()  TODO!!\n");
#endif
	HRESULT hr = this->m_tex->GetPrivateData(refguid, pData, pSizeOfData);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::FreePrivateData(THIS_ REFGUID refguid)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::FreePrivateData()  TODO!!\n");
#endif
	HRESULT hr = this->m_tex->FreePrivateData(refguid);
	return hr;
}
STDMETHODIMP_(DWORD) WrapperDirect3DVolumeTexture9::SetPriority(THIS_ DWORD PriorityNew)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::SetPriority()  TODO!!\n");
#endif
	DWORD hr = this->m_tex->SetPriority(PriorityNew);
	return hr;
}
STDMETHODIMP_(DWORD) WrapperDirect3DVolumeTexture9::GetPriority(THIS)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9:: GetPriority()  TODO!!\n");
#endif
	DWORD hr = this->m_tex->GetPriority();
	return hr;
}
STDMETHODIMP_(void) WrapperDirect3DVolumeTexture9::PreLoad(THIS)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9:: PreLoad()  TODO!!\n");
#endif
	this->m_tex->PreLoad();

}
STDMETHODIMP_(D3DRESOURCETYPE) WrapperDirect3DVolumeTexture9::GetType(THIS)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9:: GetType()  TODO!!\n");
#endif
	return this->m_tex->GetType();
}
STDMETHODIMP_(DWORD) WrapperDirect3DVolumeTexture9::SetLOD(THIS_ DWORD LODNew)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::SetLOD()  TODO!\n");
#endif
	DWORD hr = this->m_tex->SetLOD(LODNew);
	return hr;
}
STDMETHODIMP_(DWORD) WrapperDirect3DVolumeTexture9::GetLOD(THIS)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::GetLOD()  TODO!\n");
#endif
	DWORD hr = this->m_tex->GetLOD();
	return hr;
}
STDMETHODIMP_(DWORD) WrapperDirect3DVolumeTexture9::GetLevelCount(THIS)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9:: GetLevelCount()  TODO!\n");
#endif
	DWORD hr = this->m_tex->GetLevelCount();
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::SetAutoGenFilterType(THIS_ D3DTEXTUREFILTERTYPE FilterType)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::SetAutoGenFilterType()  TODO!\n");
#endif
	HRESULT hr = this->m_tex->SetAutoGenFilterType(FilterType);
	return hr;
}
STDMETHODIMP_(D3DTEXTUREFILTERTYPE) WrapperDirect3DVolumeTexture9::GetAutoGenFilterType(THIS)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::GetAutoGenFilterType()  TODO!!\n");
#endif
	return this->m_tex->GetAutoGenFilterType();
}
STDMETHODIMP_(void) WrapperDirect3DVolumeTexture9::GenerateMipSubLevels(THIS)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::GenerateMipSubLevels()  TODO!!\n");
#endif
	this->m_tex->GenerateMipSubLevels();
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::GetLevelDesc(THIS_ UINT Level,D3DVOLUME_DESC *pDesc)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::GetLevelDesc()  TODO!!\n");
#endif
	HRESULT hr = this->m_tex->GetLevelDesc(Level, pDesc);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::GetVolumeLevel(THIS_ UINT Level,IDirect3DVolume9** ppVolumeLevel)
{
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::GetVolumeLevel()  TODO!!\n");
	IDirect3DVolume9 * tpVolumeLevel = NULL;
	HRESULT hr = this->m_tex->GetVolumeLevel(Level, &tpVolumeLevel);
	WrapperDirect3DVolume9 *ret = WrapperDirect3DVolume9::GetWrapperDirect3DVolume9(tpVolumeLevel);
	if(ret ==NULL){
		infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::GetVolumeLevel  get NULL!\n");
	}
	*ppVolumeLevel =dynamic_cast<IDirect3DVolume9*>(ret);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::LockBox(THIS_ UINT Level,D3DLOCKED_BOX* pLockedVolume,CONST D3DBOX* pBox,DWORD Flags)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::LockBox()  TODO!!\n");
#endif
	HRESULT hr = this->m_tex->LockBox(Level, pLockedVolume, pBox, Flags);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::UnlockBox(THIS_ UINT Level)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::UnlockBox()  TODO!!\n");
#endif
	HRESULT hr = this->m_tex->UnlockBox(Level);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolumeTexture9::AddDirtyBox(THIS_ CONST D3DBOX* pDirtyBox)
{
#ifdef ENABLE_VOLUME_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DVolumeTexture9::AddDirtyBox()  TODO!!\n");
#endif
	HRESULT hr = this->m_tex->AddDirtyBox(pDirtyBox);
	return hr;
}