#include "CommandServerSet.h"
#include "WrapDirect3d9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"

#include "../VideoGen/generator.h"
#include "../VideoUtility/rtspconf.h"
#include "../libCore/CmdHelper.h"

#include <cuda_d3d9_interop.h>



//#define ENABLE_DIRECT3D_LOG

IDirect3DDevice9 * cur_d3ddevice = NULL;
D3DCAPS9 d3d_caps;
bool GetDeviceCapsCalled = false;

int deviceId = 0;
extern cg::VideoGen * gGenerator;

void __D3DErr(HRESULT hr){
	char * msg = NULL;
	switch(hr){
	case D3DERR_WRONGTEXTUREFORMAT :
		msg = _strdup("D3DERR_WRONGTEXTUREFORMAT");
		break;
	case D3DERR_UNSUPPORTEDCOLOROPERATION :
		msg = _strdup("D3DERR_UNSUPPORTEDCOLOROPERATION");
		break;
	case D3DERR_UNSUPPORTEDCOLORARG :
		msg = _strdup("D3DERR_UNSUPPORTEDCOLORARG");
		break;
	case D3DERR_UNSUPPORTEDALPHAOPERATION:
		msg = _strdup("D3DERR_UNSUPPORTEDALPHAOPERATION");
		break;
	case D3DERR_UNSUPPORTEDALPHAARG:
		msg = _strdup("D3DERR_UNSUPPORTEDALPHAARG");
		break;
	case D3DERR_TOOMANYOPERATIONS:
		msg = _strdup("D3DERR_TOOMANYOPERATIONS");
		break;
	case D3DERR_CONFLICTINGTEXTUREFILTER:
		msg = _strdup("D3DERR_CONFLICTINGTEXTUREFILTER");
		break;
	case D3DERR_UNSUPPORTEDFACTORVALUE:
		msg = _strdup("D3DERR_UNSUPPORTEDFACTORVALUE");
		break;
	case D3DERR_CONFLICTINGRENDERSTATE :
		msg = _strdup("D3DERR_CONFLICTINGRENDERSTATE");
		break;
	case D3DERR_UNSUPPORTEDTEXTUREFILTER :
		msg = _strdup("D3DERR_UNSUPPORTEDTEXTUREFILTER");
		break;
	case D3DERR_CONFLICTINGTEXTUREPALETTE:
		msg = _strdup("D3DERR_CONFLICTINGTEXTUREPALETTE");
		break;
	case D3DERR_DRIVERINTERNALERROR :
		msg = _strdup("D3DERR_DRIVERINTERNALERROR");
		break;
	case D3DERR_NOTFOUND:
		msg = _strdup("D3DERR_NOTFOUND");
		break;
	case D3DERR_MOREDATA:
		msg = _strdup("D3DERR_MOREDATA");
		break;
	case D3DERR_DEVICELOST:
		msg = _strdup("D3DERR_DEVICELOST");
		break;
	case D3DERR_DEVICENOTRESET :
		msg = _strdup("D3DERR_DEVICENOTRESET");
		break;
	case D3DERR_NOTAVAILABLE :
		msg = _strdup("D3DERR_NOTAVAILABLE");
		break;
	case D3DERR_OUTOFVIDEOMEMORY :
		msg = _strdup("D3DERR_OUTOFVIDEOMEMORY");
		break;
	case D3DERR_INVALIDDEVICE  :
		msg = _strdup("D3DERR_INVALIDDEVICE");
		break;
	case D3DERR_INVALIDCALL :
		msg = _strdup("D3DERR_INVALIDCALL");
		break;
	case D3DERR_DRIVERINVALIDCALL:
		msg = _strdup("D3DERR_DRIVERINVALIDCALL");
		break;
	case D3DERR_WASSTILLDRAWING :
		msg = _strdup("D3DERR_WASSTILLDRAWING");
		break;
	case D3DOK_NOAUTOGEN:
		msg = _strdup("D3DOK_NOAUTOGEN");
		break;
	default:
		msg = _strdup("NO_INFOR_FOR_ERROR");
		break;
	}
	infoRecorder->logError("[Global]: error, %d(%s).\n", hr, msg);
}


WrapperDirect3D9::WrapperDirect3D9(IDirect3D9* ptr, int _id): m_d3d(ptr), IdentifierBase(_id) {}


WrapperDirect3D9* WrapperDirect3D9::GetWrapperD3D9(IDirect3D9* ptr) {
	WrapperDirect3D9* ret = (WrapperDirect3D9*)( m_list.GetDataPtr((PVOID)ptr) );
	if(ret == NULL) {
#ifdef ENABLE_DIRECT3D_LOG
		infoRecorder->logTrace("WrapperDirect3D9::GetWrapperD3D9(), ret is NULL ins_count:%d\n",ins_count);
#endif
		ret = new WrapperDirect3D9(ptr, ins_count++);
		m_list.AddMember(ptr, ret);

		// test the AutoGen capabilities
		//HRESULT hr = ptr->CheckDeviceFormat()


	}
	return ret;
}

STDMETHODIMP WrapperDirect3D9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::QueryInterface() called\n");
#endif
	HRESULT hr = m_d3d->QueryInterface(riid,ppvObj);
	*ppvObj = this;
	return hr;
}

STDMETHODIMP_(ULONG) WrapperDirect3D9::AddRef(THIS) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::AddRef() called\n");
#endif
	refCount++;
	return m_d3d->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3D9::Release(THIS) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::Release() called\n");
#endif
	refCount--;
	ULONG hr = m_d3d->Release();
#ifdef ENABLE_DIRECT3D_LOG
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3D]: m_device ref:%d, ref count:%d.\n", refCount, hr);
	}
#endif
	return hr;
}

/*** IDirect3D9 methods ***/
STDMETHODIMP WrapperDirect3D9::RegisterSoftwareDevice(THIS_ void* pInitializeFunction) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::RegisterSoftwareDevice() TODO!\n");
#endif
	return m_d3d->RegisterSoftwareDevice(pInitializeFunction);
}
STDMETHODIMP_(UINT) WrapperDirect3D9::GetAdapterCount(THIS) {
	UINT ret = m_d3d->GetAdapterCount();
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::GetAdapterCount() TODO! ret:%d\n",ret);
#endif
	return ret;
}
STDMETHODIMP WrapperDirect3D9::GetAdapterIdentifier(THIS_ UINT Adapter,DWORD Flags,D3DADAPTER_IDENTIFIER9* pIdentifier) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::GetAdapterIdentifier() TODO! Adapter:%d, Flags:%d\n", Adapter, Flags);
#endif
	return m_d3d->GetAdapterIdentifier(Adapter, Flags, pIdentifier);
}
STDMETHODIMP_(UINT) WrapperDirect3D9::GetAdapterModeCount(THIS_ UINT Adapter,D3DFORMAT Format) {
	UINT ret= m_d3d->GetAdapterModeCount(Adapter, Format);
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::GetAdapterModeCount() TODO! Adapter:%d, Format:%d, ret:%d\n", Adapter, Format, ret);
#endif
	return ret;
}
STDMETHODIMP WrapperDirect3D9::EnumAdapterModes(THIS_ UINT Adapter,D3DFORMAT Format,UINT Mode,D3DDISPLAYMODE* pMode) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::EnumAdapterModes() TODO! Adapter:%d, Format:%d, Mode:%d\n", Adapter, Format, Mode);
#endif
	return m_d3d->EnumAdapterModes(Adapter, Format, Mode, pMode);
}
STDMETHODIMP WrapperDirect3D9::GetAdapterDisplayMode(THIS_ UINT Adapter,D3DDISPLAYMODE* pMode) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::GetAdapterDisplayMode() TODO! Adapter:%d\n", Adapter);
#endif
	return m_d3d->GetAdapterDisplayMode(Adapter, pMode);
}
STDMETHODIMP WrapperDirect3D9::CheckDeviceType(THIS_ UINT Adapter,D3DDEVTYPE DevType,D3DFORMAT AdapterFormat,D3DFORMAT BackBufferFormat,BOOL bWindowed) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::CheckDeviceType() TODO! Adapter:%d, DeviceType:%d, Adapter Format:%d, BackBuffer Format:%d, windowed:%d\n",Adapter, DevType, AdapterFormat, BackBufferFormat, bWindowed);
#endif
	return m_d3d->CheckDeviceType(Adapter, DevType, AdapterFormat, BackBufferFormat, bWindowed);
}
STDMETHODIMP WrapperDirect3D9::CheckDeviceFormat(THIS_ UINT Adapter,D3DDEVTYPE DeviceType,D3DFORMAT AdapterFormat,DWORD Usage,D3DRESOURCETYPE RType,D3DFORMAT CheckFormat) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::CheckDeviceFormat() called! Adapter:%d, Device Type:%d, Adapter Format:%d, Usage:%d, Resource Type:%d, Check Format:%d\n", Adapter, DeviceType, AdapterFormat, Usage, RType, CheckFormat);
#endif
	//infoRecorder->logTrace("WrapperDirect3D9::CheckDeviceFormat() called\n");
	return m_d3d->CheckDeviceFormat(Adapter, DeviceType, AdapterFormat, Usage, RType, CheckFormat);
}
STDMETHODIMP WrapperDirect3D9::CheckDeviceMultiSampleType(THIS_ UINT Adapter,D3DDEVTYPE DeviceType,D3DFORMAT SurfaceFormat,BOOL Windowed,D3DMULTISAMPLE_TYPE MultiSampleType,DWORD* pQualityLevels) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::CheckDeviceMultiSampleType() TODO! Adapter:%d, Device Type:%d, Surface Format:%d, windowed:%d, MultiSample Type:%d\n", Adapter, DeviceType, SurfaceFormat, Windowed, MultiSampleType);
#endif
	return m_d3d->CheckDeviceMultiSampleType(Adapter, DeviceType, SurfaceFormat, Windowed, MultiSampleType, pQualityLevels);
}
STDMETHODIMP WrapperDirect3D9::CheckDepthStencilMatch(THIS_ UINT Adapter,D3DDEVTYPE DeviceType,D3DFORMAT AdapterFormat,D3DFORMAT RenderTargetFormat,D3DFORMAT DepthStencilFormat) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::CheckDepthStencilMatch() TODO!\n");
#endif
	return m_d3d->CheckDepthStencilMatch(Adapter, DeviceType, AdapterFormat, RenderTargetFormat, DepthStencilFormat);
}
STDMETHODIMP WrapperDirect3D9::CheckDeviceFormatConversion(THIS_ UINT Adapter,D3DDEVTYPE DeviceType,D3DFORMAT SourceFormat,D3DFORMAT TargetFormat) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::CheckDeviceFormatConversion() TODO! Adapter:%d, Device Type:%d, Source Format:%d, Target Format:%d\n", Adapter, DeviceType, SourceFormat, TargetFormat);
#endif
	return m_d3d->CheckDeviceFormatConversion(Adapter, DeviceType, SourceFormat, TargetFormat);
}

STDMETHODIMP WrapperDirect3D9::GetDeviceCaps(THIS_ UINT Adapter,D3DDEVTYPE DeviceType,D3DCAPS9* pCaps) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::GetDeviceCaps called! adapetor:%d, device type:%d\n", Adapter, DeviceType);
#endif
	// limit the device capabilities
	HRESULT hr = m_d3d->GetDeviceCaps(Adapter, DeviceType, pCaps);
#if 0
	pCaps->MaxTextureHeight = pCaps->MaxTextureHeight > 1024 ? 1024 : pCaps->MaxTextureHeight;
	pCaps->MaxTextureWidth = pCaps->MaxTextureWidth > 1024 ? 1024 : pCaps->MaxTextureWidth;
#endif
	infoRecorder->logError("[WrapperDirect3d]: max texture with:%d, height:%d.\n", pCaps->MaxTextureWidth, pCaps->MaxTextureHeight);
	return hr;//m_d3d->GetDeviceCaps(Adapter, DeviceType, pCaps);
}
STDMETHODIMP_(HMONITOR) WrapperDirect3D9::GetAdapterMonitor(THIS_ UINT Adapter) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::GetAdapterMonitor() TODO! Adapter:%d\n", Adapter);
#endif
	return m_d3d->GetAdapterMonitor(Adapter);
}

STDMETHODIMP WrapperDirect3D9::CreateDevice(THIS_ UINT Adapter,D3DDEVTYPE DeviceType,HWND hFocusWindow,DWORD BehaviorFlags,D3DPRESENT_PARAMETERS* pPresentationParameters,IDirect3DDevice9** ppReturnedDeviceInterface) {
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("WrapperDirect3D9::CreateDevice() called\n");
	infoRecorder->logTrace("Create Device start\n");
	infoRecorder->logTrace("create device parameters: Adapter %d, DeviceType %d, bf %d dsf %d, backformat %d, backbuffer width:%d, backbuffer height:%d\n", Adapter, DeviceType, BehaviorFlags | D3DCREATE_MULTITHREADED, pPresentationParameters->AutoDepthStencilFormat, pPresentationParameters->BackBufferFormat,pPresentationParameters->BackBufferWidth, pPresentationParameters->BackBufferHeight);
#endif

	IDirect3DDevice9* base_device = NULL;
	/*
	pPresentationParameters->Windowed = true;
	if(pPresentationParameters->BackBufferHeight < 600)
	pPresentationParameters->BackBufferHeight = 600;
	if(pPresentationParameters->BackBufferWidth < 800)
	pPresentationParameters->BackBufferWidth = 800;
	*/
	HRESULT hr = m_d3d->CreateDevice(Adapter, DeviceType, hFocusWindow, BehaviorFlags | D3DCREATE_MULTITHREADED, pPresentationParameters, &base_device);
	cur_d3ddevice = base_device;

	WrapperDirect3DDevice9 * wdd = NULL;
	if(SUCCEEDED(hr)) {
		//MessageBox(NULL, "success", NULL, MB_OK);
		wdd = new WrapperDirect3DDevice9(base_device, WrapperDirect3DDevice9::ins_count++);
		*ppReturnedDeviceInterface = dynamic_cast<IDirect3DDevice9*>(wdd);
		deviceId = wdd->getId();

		// create the texture helper
		wdd->deviceHelper = new DeviceHelper();
		wdd->deviceHelper->checkSupportForAutoGenMipmap(base_device);

#ifdef ENABLE_DIRECT3D_LOG
		infoRecorder->logError("WrapperDirect3D9::CreateDevice(), base_device=%p, device=%p\n", base_device, *ppReturnedDeviceInterface);
#endif
	}
	else {
#ifdef ENABLE_DIRECT3D_LOG
		infoRecorder->logError("Create Device Failed with:%d\n", hr);
#endif
		__D3DErr(hr);
	}

	cudaError_t err;
	if((err =  cudaD3D9SetDirect3DDevice(base_device)) != cudaSuccess){
#ifdef ENABLE_DIRECT3D_LOG
		infoRecorder->logError("[Direct3D9]: cudaD3D9SetDirectDevice failed with:%d.\n", err);
#endif
	}

#ifndef MULTI_CLIENTS
	cs.begin_command(CreateDevice_Opcode, 0);
	cs.write_int(wdd->GetID());
	cs.write_uint(Adapter);
	cs.write_uint(DeviceType);
	cs.write_uint(BehaviorFlags);
	cs.write_byte_arr((char*)(pPresentationParameters), sizeof(D3DPRESENT_PARAMETERS));
	cs.end_command();
#else
	//HWND h = pPresentationParameters->

	csSet->beginCommand(CreateDevice_Opcode, 0);
	csSet->writeInt(wdd->getId());
	csSet->writeUInt(Adapter);
	csSet->writeUInt(DeviceType);
	csSet->writeUInt(BehaviorFlags | D3DCREATE_MULTITHREADED);
	csSet->writeByteArr((char *)(pPresentationParameters), sizeof(D3DPRESENT_PARAMETERS));
	csSet->endCommand();

	csSet->setCreation(wdd->creationFlag);
	Initializer::BeginInitalize();

	// save the creation parameters
	wdd->adapter = Adapter;
	wdd->deviceType = DeviceType;
	wdd->behaviorFlags = BehaviorFlags | D3DCREATE_MULTITHREADED;
	wdd->pPresentParameters = (D3DPRESENT_PARAMETERS *)malloc(sizeof(D3DPRESENT_PARAMETERS));
	memcpy(wdd->pPresentParameters, pPresentationParameters, sizeof(D3DPRESENT_PARAMETERS));

	Initializer::PushObj(dynamic_cast<IdentifierBase *>(wdd));


#endif
#ifdef ENABLE_DIRECT3D_LOG
	infoRecorder->logTrace("Create Device End. With Device :%d ADDR:%d\n",WrapperDirect3DDevice9::ins_count,base_device);
#endif

	// get the window handle
	HWND hd = pPresentationParameters->hDeviceWindow;
	RECT winRect;
	BOOL ret = GetWindowRect(hd, &winRect);

#ifdef ENABLE_DIRECT3D_LOG
	if (ret == TRUE){
		infoRecorder->logError("CreateDevice: window %p rect:(%d, %d) - (%d, %d)\n", hd, winRect.left, winRect.top, winRect.right, winRect.bottom);

	}
	else{
		infoRecorder->logError("CreateDevice: failed to get the window size, window:%p.\n", hd);
	}
#endif

#ifndef ENABLE_HOT_PLUG
	// create the video context for video streaming.
	VideoItem * item = new VideoItem();
	item->presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);  // create the present event for video generator
	item->windowHandle = hd;
	item->device = base_device;
	item->winWidth = pPresentationParameters->BackBufferWidth;
	item->winHeight = pPresentationParameters->BackBufferHeight;

	VideoContext * vctx = VideoContext::GetContext();
	// TODO, correct the identifier for the item.
	vctx->addMap(id, item);
#else
	cmdCtrl->setDevice(base_device);
	cmdCtrl->setHwnd(hd);

#endif 

	return hr;
}
