#include "WrapDirect3dvolume9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
#include "CommandServerSet.h"
int WrapperDirect3DVolume9::ins_count = 0;
HashSet WrapperDirect3DVolume9::m_list;

#ifdef MULTI_CLIENTS
int WrapperDirect3DVolume9::checkCreation(void *ctx){
	infoRecorder->logTrace("[WrapperDirect3Dvolume9]: check creation, TODO.\n");
	int ret = 0;
	ContextAndCache * cc = (ContextAndCache *)ctx;
	if(!cc->isCreated(creationFlag)){
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
		ret = 1;
	}

	return ret;
}
int WrapperDirect3DVolume9::sendCreation(void * ctx){
	infoRecorder->logTrace("[WrapperDirect3DVolume9]: send creation, TODO.\n");
	int ret = 0;

	ContextAndCache * c = (ContextAndCache *)ctx;

	return ret;
}
int WrapperDirect3DVolume9::checkUpdate(void *ctx){
	infoRecorder->logTrace("[WrapperDirect3DVolume9]: check update, TODO.\n");

	int ret = 0;
	return ret;
}


int WrapperDirect3DVolume9::sendUpdate(void * ctx){
	infoRecorder->logTrace("[WrapperDirect3DVolume9]: send update, TODO.\n");
	int ret =0;

	return ret;
}

#endif //MULTI_CLIENTS


WrapperDirect3DVolume9::WrapperDirect3DVolume9(IDirect3DVolume9 *ptr, int _id):m_d3dvolume(ptr),IdentifierBase(_id){
	infoRecorder->logTrace("WrapperDirect3DVolume9 constructor called!\n");
	m_list.AddMember(ptr,this);
}


WrapperDirect3DVolume9 * WrapperDirect3DVolume9::GetWrapperDirect3DVolume9(IDirect3DVolume9 *ptr){
	infoRecorder->logTrace("WrapperDirect3DVolume9:: GetWrapperDirect3DVolume9 called!\n");
	WrapperDirect3DVolume9 * ret = NULL;
	ret = (WrapperDirect3DVolume9 *)m_list.GetDataPtr(ptr);
	if(ret == NULL){
		infoRecorder->logTrace("ERROR:WrapperDirect3DVolume9:: GetWrapperDirect3DVolume9 return NULL!\n");
	}
	return ret;
}
IDirect3DVolume9 * WrapperDirect3DVolume9::GetIDirect3DVolume9(){
	return this->m_d3dvolume;
}

/********************** IUnknown Methods *************/
STDMETHODIMP WrapperDirect3DVolume9::QueryInterface(THIS_ REFIID riid, void **ppvObj){
	infoRecorder->logTrace("WrapperDirect3DVolume9::QueryInterface called!\n");
	HRESULT hr =this->m_d3dvolume->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DVolume9::AddRef(THIS){
	infoRecorder->logTrace("WrapperDirect3DVolume9::AddRef called!\n");
	refCount++;
	return this->m_d3dvolume->AddRef();
}

STDMETHODIMP_(ULONG) WrapperDirect3DVolume9::Release(THIS){
	infoRecorder->logTrace("WrapperDirect3DVolume9::Release called!\n");
	ULONG hr = m_d3dvolume->Release();
#ifdef LOG_REF_COUNT
	infoRecorder->logError("WrapperDirect3DVolume9::Release(), ref:%d.\n", hr);
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DVolume9]: m_volume ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
}

/*************** Methods *************************/
STDMETHODIMP WrapperDirect3DVolume9::GetDevice(THIS_ IDirect3DDevice9** ppDevice)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::GetDevice  called!\n");
	IDirect3DDevice9* base_device = NULL;
	HRESULT hr =this->m_d3dvolume->GetDevice(&base_device);
	WrapperDirect3DDevice9* ret =  WrapperDirect3DDevice9::GetWrapperDevice9(base_device);
	*ppDevice = ret;
	return hr;
}
STDMETHODIMP WrapperDirect3DVolume9::SetPrivateData(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::SetPrivateData  called!\n");
	HRESULT hr =this->m_d3dvolume->SetPrivateData(refguid, pData, SizeOfData, Flags);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolume9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::GetPrivateData  called!\n");
	HRESULT hr =this->m_d3dvolume->GetPrivateData(refguid, pData, pSizeOfData);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolume9::FreePrivateData(THIS_ REFGUID refguid)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::FreePrivateData  called!\n");
	HRESULT hr =this->m_d3dvolume->FreePrivateData(refguid);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolume9::GetContainer(THIS_ REFIID riid,void** ppContainer)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::GetContainer  called!\n");
	HRESULT hr =this->m_d3dvolume->GetContainer(riid, ppContainer);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolume9::GetDesc(THIS_ D3DVOLUME_DESC *pDesc)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::GetDesc  called!\n");
	HRESULT hr =this->m_d3dvolume->GetDesc(pDesc);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolume9::LockBox(THIS_ D3DLOCKED_BOX * pLockedVolume,CONST D3DBOX* pBox,DWORD Flags)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::LockBox  called!\n");
	HRESULT hr =this->m_d3dvolume->LockBox(pLockedVolume, pBox, Flags);
	return hr;
}
STDMETHODIMP WrapperDirect3DVolume9::UnlockBox(THIS)
{
	infoRecorder->logTrace("WrapperDirect3DVolume9::UnlockBox  called!\n");
	HRESULT hr =this->m_d3dvolume->UnlockBox();
	return hr;
}
