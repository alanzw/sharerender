//#include "ccg_win32.h"
#include "ccg_win32.h"
#include "ccg_config.h"

#include <string.h>
#include <time.h>
#include <map>
#include "rtspconf.h"
#include "rtspcontext.h"
#include "pipeline.h"
#include "encoder.h"

#include "FilterRGB2YUV.h"
#include <queue>
#include "disnetwork.h"
#include "distributorforrender.h"
#include "log.h"
#include "renderchannel.h"

#define X(a) { #a, (ApiFunc*)&fptable.##a }


#if 0
void RenderChannel::initFptable(){
	fptable.DCreateWindow = &RenderChannel::FakeDCreateWindow;
	fptable.DDirectCreate = &RenderChannel::FakeDDirectCreate;

	fptable.CreateDevice = &RenderChannel::FakedCreateDevice;
	fptable.BeginScene = &RenderChannel::FakedBeginScene;
	fptable.EndScene = &RenderChannel::FakedEndScene;
	fptable.Clear = &RenderChannel::FakedClear;
	fptable.Present = &RenderChannel::FakedPresent;

	fptable.SetTransform = &RenderChannel::FakedSetTransform;
	fptable.SetRenderState = &RenderChannel::FakedSetRenderState;
	fptable.SetStreamSource = &RenderChannel::FakedSetStreamSource;
	fptable.SetFVF = &RenderChannel::FakedSetFVF;
	fptable.DrawPrimitive = &RenderChannel::FakedDrawPrimitive;
	fptable.DrawIndexedPrimitive = &RenderChannel::FakedDrawIndexedPrimitive;

	fptable.CreateVertexBuffer = &RenderChannel::FakedCreateVertexBuffer;
	fptable.VertexBufferLock = &RenderChannel::FakedVertexBufferLock;
	fptable.VertexBufferUnlock = &RenderChannel::FakedVertexBufferUnlock;

	fptable.SetIndices = &RenderChannel::FakedSetIndices;
	fptable.CreateIndexBuffer = &RenderChannel::FakedCreateIndexBuffer;
	fptable.IndexBufferLock = &RenderChannel::FakedIndexBufferLock;
	fptable.IndexBufferUnlock = &RenderChannel::FakedIndexBufferUnlock;
	fptable.SetSamplerState = &RenderChannel::FakedSetSamplerState;

	fptable.CreateVertexDeclaration = &RenderChannel::FakedCreateVertexDeclaration;
	fptable.SetVertexDeclaration = &RenderChannel::FakedSetVertexDeclaration;
	fptable.SetSoftwareVertexProcessing = &RenderChannel::FakedSetSoftwareVertexProcessing;
	fptable.SetLight = &RenderChannel::FakedSetLight;
	fptable.LightEnable = &RenderChannel::FakedLightEnable;

	fptable.CreateVertexShader = &RenderChannel::FakedCreateVertexShader;
	fptable.SetVertexShader = &RenderChannel::FakedSetVertexShader;
	fptable.SetVertexShaderConstantF = &RenderChannel::FakedSetVertexShaderConstantF;
	fptable.CreatePixelShader = &RenderChannel::FakedCreatePixelShader;
	fptable.SetPixelShader = &RenderChannel::FakedSetPixelShader;
	fptable.SetPixelShaderConstantF = &RenderChannel::FakedSetPixelShaderConstantF;

	fptable.DrawPrimitiveUP = &RenderChannel::FakedDrawPrimitiveUP;
	fptable.DrawIndexedPrimitiveUP = &RenderChannel::FakedDrawIndexedPrimitiveUP;
	fptable.SetVertexShaderConstantI = &RenderChannel::FakedSetVertexShaderConstantI;
	fptable.SetVertexShaderConstantB = &RenderChannel::FakedSetVertexShaderConstantB;
	fptable.SetPixelShaderConstantI = &RenderChannel::FakedSetPixelShaderConstantI;
	fptable.SetPixelShaderConstantB = &RenderChannel::FakedSetPixelShaderConstantB;

	fptable.Reset = &RenderChannel::FakedReset;
	fptable.SetMaterial = &RenderChannel::FakedSetMaterial;
	fptable.CreateTexture = &RenderChannel::FakedCreateTexture;
	fptable.SetTexture = &RenderChannel::FakedSetTexture;
	fptable.SetTextureStageState = &RenderChannel::FakedSetTextureStageState;

	fptable.TransmitTextureData = &RenderChannel::FakedTransmitTextureData;

	fptable.CreateStateBlock = &RenderChannel::FakedCreateStateBlock;
	fptable.BeginStateBlock = &RenderChannel::FakedBeginStateBlock;
	fptable.EndStateBlock = &RenderChannel::FakedEndStateBlock;

	fptable.StateBlockCapture = &RenderChannel::FakedStateBlockCapture;
	fptable.StateBlockApply = &RenderChannel::FakedStateBlockApply;

	fptable.DeviceAddRef = &RenderChannel::FakedDeviceAddRef;
	fptable.DeviceRelease = &RenderChannel::FakedDeviceRelease;

	fptable.SetViewport = &RenderChannel::FakedSetViewport;
	fptable.SetNPatchMode = &RenderChannel::FakedSetNPatchMode;

	fptable.CreateCubeTexture = &RenderChannel::FakedCreateCubeTexture;
	fptable.SetCubeTexture = &RenderChannel::FakedSetCubeTexture;

	fptable.GetSwapChain = &RenderChannel::FakedGetSwapChain;
	fptable.SwapChainPresent = &RenderChannel::FakedSwapChainPresent;
	fptable.SetAutoGenFilterType = &RenderChannel::FakedSetAutoGenFilterType;
	fptable.GenerateMipSubLevels = &RenderChannel::FakedGenerateMipSubLevels;

	fptable.SetRenderTarget = &RenderChannel::FakedSetRenderTarget;
	fptable.SetDepthStencilSurface = &RenderChannel::FakedSetDepthStencilSurface;
	fptable.TextureGetSurfaceLevel = &RenderChannel::FakedTextureGetSurfaceLevel;
	fptable.SwapChainGetBackBuffer = &RenderChannel::FakedSwapChainGetBackBuffer;
	fptable.GetDepthStencilSurface = &RenderChannel::FakedGetDepthStencilSurface;
	fptable.CreateDepthStencilSurface = &RenderChannel::FakedCreateDepthStencilSurface;
	fptable.CubeGetCubeMapSurface = &RenderChannel::FakedCubeGetCubeMapSurface;

	/*
	// direct input concered functions
	fptable.DIConfigureDevice= FakeDIConfigureDevice;
	fptable.DICreateDevice = FakeDICreateDevice;
	fptable.DIDAcquire = FakeDIDAcquire;
	fptable.DIDBuildActionMap = FakeDIDBuildActionMap;
	fptable.DIDCreateEffect = FakeDIDCreateEffect;
	fptable.DIDCreateEffectObjects = FakeDIDCreateEffectObjects;
	fptable.DIDEnumEffects = FakeDIDEnumEffects;
	fptable.DIDEscape = FakeDIDEscape;
	fptable.DIDGetCapabilities = FakeDIDGetCapabilities;
	fptable.DIDGetDeviceData = FakeDIDGetDeviceData;
	fptable.DIDGetDeviceInfo = FakeDIDGetDeviceInfo;
	fptable.DIDGetDeviceState = FakeDIDGetDeviceState;
	fptable.DIDRunControlPanel = FakeDIDRunControlPanel;
	fptable.DIDSetActionMap = FakeDIDSetActionMap;
	fptable.DIDCooperativeLevel = FakeDIDCooperativeLevel;
	fptable.DIDSetDataFormat = FakeDIDSetDataFormat;
	fptable.DIDUnacquire = FakeDIDUnacquire;


	fptable.DIDirectInputCreate = FakeDIDirectInputCreate;
	fptable.DIDAddRef = FakeDIDAddRef;
	fptable.DIDRelease = FakeDIDRelease;
	fptable.DIDSetProperty = FakeDIDSetProperty;
	*/

	fptable.TransmitSurface = &RenderChannel::FakeTransmitSurface;

	fptable.D3DDeviceGetBackBuffer = &RenderChannel::FakeD3DDeviceGetBackBuffer;
	fptable.D3DGetDeviceCaps = &RenderChannel::FakeD3DGetDeviceCaps;
	fptable.D3DDGetRenderTarget = &RenderChannel::FakeD3DDGetRenderTarget;
	fptable.D3DDSetScissorRect = &RenderChannel::FakeD3DDSetScissorRect;

	fptable.SetVertexBufferFormat = &RenderChannel::FakedSetVertexBufferFormat;
	fptable.SetDecimateResult = &RenderChannel::FakedSetDecimateResult;

	fptable.SetGammaRamp = &RenderChannel::FakedSetGammaRamp;
	fptable.NullInstruct = &RenderChannel::FakeNullInstruct;
}

void RenderChannel::mapFunctions(){
	funcs[CreateDevice_Opcode] = X(CreateDevice);
	funcs[BeginScene_Opcode] = X(BeginScene);
	funcs[EndScene_Opcode] = X(EndScene);
	funcs[Clear_Opcode] = X(Clear);
	funcs[Present_Opcode] = X(Present);

	funcs[SetTransform_Opcode] = X(SetTransform);
	funcs[SetRenderState_Opcode] = X(SetRenderState);
	funcs[SetStreamSource_Opcode] = X(SetStreamSource);
	funcs[SetFVF_Opcode] = X(SetFVF);
	funcs[DrawPrimitive_Opcode] = X(DrawPrimitive);
	funcs[DrawIndexedPrimitive_Opcode] = X(DrawIndexedPrimitive);

	funcs[CreateVertexBuffer_Opcode] = X(CreateVertexBuffer);
	funcs[VertexBufferLock_Opcode] = X(VertexBufferLock);
	funcs[VertexBufferUnlock_Opcode] = X(VertexBufferUnlock);

	funcs[SetIndices_Opcode] = X(SetIndices);
	funcs[CreateIndexBuffer_Opcode] = X(CreateIndexBuffer);
	funcs[IndexBufferLock_Opcode] = X(IndexBufferLock);
	funcs[IndexBufferUnlock_Opcode] = X(IndexBufferUnlock);
	funcs[SetSamplerState_Opcode] = X(SetSamplerState);

	funcs[CreateVertexDeclaration_Opcode] = X(CreateVertexDeclaration);
	funcs[SetVertexDeclaration_Opcode] = X(SetVertexDeclaration);
	funcs[SetSoftwareVertexProcessing_Opcode] = X(SetSoftwareVertexProcessing);
	funcs[SetLight_Opcode] = X(SetLight);
	funcs[LightEnable_Opcode] = X(LightEnable);

	funcs[CreateVertexShader_Opcode] = X(CreateVertexShader);
	funcs[SetVertexShader_Opcode] = X(SetVertexShader);
	funcs[SetVertexShaderConstantF_Opcode] = X(SetVertexShaderConstantF);
	funcs[CreatePixelShader_Opcode] = X(CreatePixelShader);
	funcs[SetPixelShader_Opcode] = X(SetPixelShader);
	funcs[SetPixelShaderConstantF_Opcode] = X(SetPixelShaderConstantF);

	funcs[DrawPrimitiveUP_Opcode] = X(DrawPrimitiveUP);
	funcs[DrawIndexedPrimitiveUP_Opcode] = X(DrawIndexedPrimitiveUP);
	funcs[SetVertexShaderConstantI_Opcode] = X(SetVertexShaderConstantI);
	funcs[SetVertexShaderConstantB_Opcode] = X(SetVertexShaderConstantB);
	funcs[SetPixelShaderConstantI_Opcode] = X(SetPixelShaderConstantI);
	funcs[SetPixelShaderConstantB_Opcode] = X(SetPixelShaderConstantB);

	funcs[Reset_Opcode] = X(Reset);
	funcs[SetMaterial_Opcode] = X(SetMaterial);
	funcs[CreateTexture_Opcode] = X(CreateTexture);
	funcs[SetTexture_Opcode] = X(SetTexture);
	funcs[SetTextureStageState_Opcode] = X(SetTextureStageState);

	funcs[TransmitTextureData_Opcode] = X(TransmitTextureData);

	funcs[CreateStateBlock_Opcode] = X(CreateStateBlock);
	funcs[BeginStateBlock_Opcode] = X(BeginStateBlock);
	funcs[EndStateBlock_Opcode] = X(EndStateBlock);

	funcs[StateBlockCapture_Opcode] = X(StateBlockCapture);
	funcs[StateBlockApply_Opcode] = X(StateBlockApply);

	funcs[DeviceAddRef_Opcode] = X(DeviceAddRef);
	funcs[DeviceRelease_Opcode] = X(DeviceRelease);

	funcs[SetViewport_Opcode] = X(SetViewport);
	funcs[SetNPatchMode_Opcode] = X(SetNPatchMode);

	funcs[CreateCubeTexture_Opcode] = X(CreateCubeTexture);
	funcs[SetCubeTexture_Opcode] = X(SetCubeTexture);

	funcs[GetSwapChain_Opcode] = X(GetSwapChain);
	funcs[SwapChainPresent_Opcode] = X(SwapChainPresent);
	funcs[TextureSetAutoGenFilterType_Opcode] = X(SetAutoGenFilterType);
	funcs[TextureGenerateMipSubLevels_Opcode] = X(GenerateMipSubLevels);

	funcs[SetRenderTarget_Opcode] = X(SetRenderTarget);
	funcs[SetDepthStencilSurface_Opcode] = X(SetDepthStencilSurface);
	funcs[TextureGetSurfaceLevel_Opcode] = X(TextureGetSurfaceLevel);
	funcs[SwapChainGetBackBuffer_Opcode] = X(SwapChainGetBackBuffer);
	funcs[GetDepthStencilSurface_Opcode] = X(GetDepthStencilSurface);
	funcs[CreateDepthStencilSurface_Opcode] = X(CreateDepthStencilSurface);
	funcs[CubeGetCubeMapSurface_Opcode] = X(CubeGetCubeMapSurface);

	// add by alan 2013/1/5
	funcs[DIConfigureDevices_Opcode] = X(DIConfigureDevice);
	funcs[DICreateDevice_Opcode] = X(DICreateDevice);
	funcs[DIGetDeviceStatus_Opcode] = X(DIGetDeviceStatus);
	funcs[DIRunControlPanel_Opcode] = X(DIRunContorlPanel);
	funcs[DIDAcquire_Opcode] = X(DIDAcquire);
	funcs[DIDBuildActionMap_Opcode] = X(DIDBuildActionMap);
	funcs[DIDCreateEffect_Opcode] = X(DIDCreateEffect);
	funcs[DIDEnumCreateEffectObjects_Opcode] = X(DIDCreateEffectObjects);
	funcs[DIDEnumEffects_Opcode] = X(DIDEnumEffects);
	funcs[DIDEscape_Opcode] = X(DIDEscape);
	funcs[DIDGetCapabilities_Opcode] = X(DIDGetCapabilities);
	funcs[DIDGetDeviceData_Opcode] = X(DIDGetDeviceData);
	funcs[DIDGetDeviceInfo_Opcode] = X(DIDGetDeviceInfo);
	funcs[DIDGetDeviceState_Opcode] = X(DIDGetDeviceState);
	funcs[DIDRunControlPanel_Opcode] = X(DIDRunControlPanel);
	funcs[DIDSetActionMap_Opcode] = X(DIDSetActionMap);
	funcs[DIDSetCooperativeLevel_Opcode] = X(DIDCooperativeLevel);
	funcs[DIDSetDataFormat_Opcode] = X(DIDSetDataFormat);
	funcs[DIDUnacquire_Opcode] = X(DIDUnacquire);
	funcs[CreateWindow_Opcode] = X(DCreateWindow);
	funcs[DirectCreate_Opcode] = X(DDirectCreate);
	funcs[DirectInputCreate_Opcode] = X(DIDirectInputCreate);
	funcs[DIDAddRef_Opcode] = X(DIDAddRef);
	funcs[DIDRelease_Opcode] = X(DIDRelease);
	funcs[DIDSetProperty_Opcode] = X(DIDSetProperty);
	funcs[TransmitSurfaceData_Opcode] = X(TransmitSurface);

	funcs[D3DDeviceGetBackBuffer_Opcode] = X(D3DDeviceGetBackBuffer);
	funcs[D3DGetDeviceCaps_Opcode] = X(D3DGetDeviceCaps);
	funcs[D3DDGetRenderTarget_Opcode] = X(D3DDGetRenderTarget);
	funcs[D3DDSetScissorRect_Opcode] = X(D3DDSetScissorRect);

	funcs[SetVertexBufferFormat_Opcode] = X(SetVertexBufferFormat);
	funcs[SetDecimateResult_Opcode] = X(SetDecimateResult);

	funcs[SetGammaRamp_Opcode] = X(SetGammaRamp);
	funcs[NULLINSTRUCT_Opcode] = X(NullInstruct);
}

HRESULT RenderChannel::FakeDCreateWindow() {
	Log::log("FakedCreateWindow() called\n");
	TCHAR szAppName[] = TEXT("HelloWin");
	TCHAR szClassName[] = TEXT("HelloWinClass");

	DWORD dwExStyle = cc->read_uint();
	DWORD dwStyle = cc->read_uint();
	int x = cc->read_int();
	int y = cc->read_int();
	int nWidth = cc->read_int();
	int nHeight = cc->read_int();

	static bool window_created = false;

	if (!window_created)
		initWindow(nWidth, nHeight, dwExStyle, dwStyle);
	window_created = true;
	return D3D_OK;
}

HRESULT RenderChannel::FakeDDirectCreate(){
	Log::log("server Direct3DCreate9 called\n");

	Log::log("server Direct3DCreate9 gD3d = %p.\n", gD3d);

	gD3d = NULL;
	gD3d = Direct3DCreate9(D3D_SDK_VERSION);
	if (gD3d)
		return D3D_OK;
	else{
		Log::log("Direct3DCreate9 failed!\n");
		return D3D_OK;
	}
}

HRESULT RenderChannel::FakedCreateDevice() {
	//printf("FakedCreateDevice called\n");
	Log::log("FakedCreateDevice called\n");

	return clientInit();
}

HRESULT RenderChannel::FakedBeginScene() {
	//printf("FakedBeginScene called\n");
	Log::log("FakedBeginScene called\n");

	//getDevice(obj_id);
	getDevice(obj_id);
	return curDevice->BeginScene();
}

HRESULT RenderChannel::FakedEndScene() {
	//printf("FakedEndScene called\n");
	Log::log("FakedEndScene called\n");
	getDevice(obj_id);
	return curDevice->EndScene();
}

HRESULT RenderChannel::FakedClear() {
	//printf("Faked Clear called\n");
	DWORD count = cc->read_uint();
	D3DRECT pRects;

	bool is_null = cc->read_char();
	if (!is_null) {
		cc->read_byte_arr((char*)(&pRects), sizeof(pRects));
	}

	DWORD Flags = cc->read_uint();
	D3DCOLOR color = cc->read_uint();
	float Z = cc->read_float();
	DWORD stencil = cc->read_uint();

	Log::log("Faked Clear called(), Color=0x%08x\n", color);

	getDevice(obj_id);
	if (!is_null)
		return curDevice->Clear(count, &pRects, Flags, color, Z, stencil);
	else
		return curDevice->Clear(count, NULL, Flags, color, Z, stencil);
}

HRESULT RenderChannel::FakedPresent() {
	Log::log("Faked Present called\n");

	static float last_present = 0;
	float now_present = timeGetTime();

	//Log::slog("FakedPresent(), present gap=%.4f\n", now_present - last_present);
	gap = now_present - last_present;
	last_present = now_present;

	float t1 = timeGetTime();

	const RECT* pSourceRect = (RECT*)(cc->read_int());
	const RECT* pDestRect = (RECT*)(cc->read_int());
	HWND hDestWindowOverride = (HWND)(cc->read_int());
	const RGNDATA* pDirtyRegion = (RGNDATA*)(cc->read_int());

	//return curDevice->Present(pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion);
	getDevice(obj_id);

	assert(curDevice);
	Log::log("Fakepresent: beafore call present.......\n");

	WaitForSingleObject(presentMutex, INFINITE);
	HRESULT hr = curDevice->Present(NULL, NULL, NULL, NULL);
	SetEvent(this->presentEvent);
	ReleaseMutex(presentMutex);

	float t2 = timeGetTime();
	Log::log("FakedPresent(), del_time=%.4f\n", t2 - t1);

	//
#if 0
	if (presentEvent){
		Log::log("[present]: set the present event.\n");
		SetEvent(presentEvent);
	}
	else{
		Log::log("FakePresent(), presentEvent is NULL.\n");
	}
#else
	if (ch){
		Log::log("FakePreset(), do the capture for channel.\n");
		ch->doCapture(curDevice);
	}
#endif

	return hr;
}

HRESULT RenderChannel::FakedSetTransform() {
	//D3DTRANSFORMSTATETYPE State,CONST D3DMATRIX* pMatrix
	Log::log("FakedSetTransform called\n");

	short st = cc->read_short();
	D3DTRANSFORMSTATETYPE state = (D3DTRANSFORMSTATETYPE)st;
	unsigned short mask = cc->read_ushort();

	D3DMATRIX mat;
	int out_len = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (mask & (1 << (i * 4 + j))) {
				mat.m[i][j] = cc->read_float();
			}
			else
				mat.m[i][j] = 0.0f;
		}
	}
	getDevice(obj_id);

	return curDevice->SetTransform(state, &mat);
}

HRESULT RenderChannel::FakedSetRenderState() {
	Log::log("FakedSetRenderState called\n");
	//D3DRENDERSTATETYPE State,DWORD Value

	D3DRENDERSTATETYPE State = (D3DRENDERSTATETYPE)(cc->read_uint());
	DWORD Value = cc->read_uint();

	getDevice(obj_id);
	return curDevice->SetRenderState(State, Value);
}

HRESULT RenderChannel::FakedSetStreamSource() {
	Log::log("FakedSetStreamSource called\n");
	//UINT StreamNumber,UINT vb_id,UINT OffsetInBytes,UINT Stride

	int StreamNumber, vb_id, OffestInBytes, Stride;


	StreamNumber = cc->read_uint();
	vb_id = cc->read_int();
	OffestInBytes = cc->read_uint();
	Stride = cc->read_uint();

	//Log::slog("FakedSetStreamSource(), hit_cnt=%d, set_cnt=%d, ratio=%.3f\n", sss_cache->hit_cnt, sss_cache->set_cnt, sss_cache->hit_cnt * 1.0 / (sss_cache->hit_cnt + sss_cache->set_cnt));
	getDevice(obj_id);

	if (vb_id == -1) {
		return curDevice->SetStreamSource(StreamNumber, NULL, OffestInBytes, Stride);
	}
	else{
		Log::log("FakedSetStreamSource: StreamNum:%d, vb_id:%d, Offset:%d, Stride:%d\n", StreamNumber, vb_id, OffestInBytes, Stride);
	}

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(vb_list[vb_id]);
	IDirect3DVertexBuffer9 * vb = NULL;
	if (svb == NULL){
		Log::log("FakedSetStreamSource: vertexbuffer NULL!\n");
	}
	else{
		vb = svb->GetVB();
		svb->stride = Stride;
	}

	return curDevice->SetStreamSource(StreamNumber, vb, OffestInBytes, Stride);
}

HRESULT RenderChannel::FakedSetFVF() {
	//DWORD FVF
	Log::log("FakedSetFVF called\n");

	DWORD FVF = cc->read_uint();

	getDevice(obj_id);
	return curDevice->SetFVF(FVF);
}

HRESULT RenderChannel::FakedDrawPrimitive() {
	Log::log("FakedDrawPrimitive called\n");
	//D3DPRIMITIVETYPE PrimitiveType,UINT StartVertex,UINT PrimitiveCount

	char type = cc->read_char();
	UINT StartVertex = 0, PrimitiveCount = 0;
	D3DPRIMITIVETYPE PrimitiveType = (D3DPRIMITIVETYPE)type;
	StartVertex = cc->read_uint();
	PrimitiveCount = cc->read_uint();


	getDevice(obj_id);
	return curDevice->DrawPrimitive(PrimitiveType, StartVertex, PrimitiveCount);
}

HRESULT RenderChannel::FakedDrawIndexedPrimitive() {
	Log::log("FakedDrawIndexedPrimitive called\n");
	//D3DPRIMITIVETYPE Type,INT BaseVertexIndex,UINT MinVertexIndex,UINT NumVertices,UINT startIndex,UINT primCount
	getDevice(obj_id);

	D3DPRIMITIVETYPE Type = (D3DPRIMITIVETYPE)(cc->read_char());
	int BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount;
	BaseVertexIndex = cc->read_int();
	MinVertexIndex = cc->read_int();
	NumVertices = cc->read_int();
	startIndex = cc->read_int();
	primCount = cc->read_int();

	Log::log("FakedDrawIndexedPrimitive(), BaseVertexIndex=%d, MinVertexIndex=%d, NumVertices=%d, startIndex=%d, primCount=%d\n", BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);

	return curDevice->DrawIndexedPrimitive(Type, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);
}

HRESULT RenderChannel::FakedCreateVertexBuffer() {
	Log::log("FakedCreateVertexBuffer called\n");

	UINT id = cc->read_uint();
	UINT Length = cc->read_uint();
	DWORD Usage = cc->read_uint();
	DWORD FVF = cc->read_uint();
	D3DPOOL Pool = (D3DPOOL)(cc->read_uint());

	LPDIRECT3DVERTEXBUFFER9 vb = NULL;

	Log::log("FakedCreateVertexBuffer before create! Length:%d, Usage:%x, FVF:%x, Pool:%d, id:%d\n", Length, Usage, FVF, Pool, id);
	getDevice(obj_id);
	HRESULT hr = curDevice->CreateVertexBuffer(Length, Usage, FVF, Pool, &vb, NULL);
	Log::log("FakedCreateVertexBuffer created. \n");
	vb_list[id] = new ClientVertexBuffer9(vb, Length);

	Log::log("FakedCreateVertexBuffer End. id:%d\n", id);
	return hr;
}

HRESULT RenderChannel::FakedVertexBufferLock() {
	Log::log("FakedVertexBufferLock called\n");

	int id = obj_id;

	UINT OffestToLock = cc->read_uint();
	UINT SizeToLock = cc->read_uint();
	DWORD Flags = cc->read_uint();

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(vb_list[id]);
	Log::log("FakedVertexBufferLock id:%d\n", id);
	return svb->Lock(OffestToLock, SizeToLock, Flags);
}

HRESULT RenderChannel::FakedVertexBufferUnlock() {
	Log::log("FakedVertexBufferUnlock called\n");

	int id = obj_id;

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(vb_list[id]);
	if (svb == NULL){
		Log::log("FakedVertexBufferUnlock is NULL id:%d\n", id);
	}
	else{
		Log::log("FakedVertexBufferUnlock id:%d\n", id);
	}

	return svb->Unlock(cc);
}

HRESULT RenderChannel::FakedSetIndices() {
	Log::log("FakedSetIndices called!!\n");

	int ib_id;

	ib_id = cc->read_short();

	getDevice(obj_id);
	IDirect3DIndexBuffer9 * ib = NULL;

	Log::log("FakedSetIndices called, ib_id=%d\n", ib_id);

	if (ib_id == -1) return curDevice->SetIndices(NULL);

	ClientIndexBuffer9* sib = NULL;
	sib = (ClientIndexBuffer9*)(ib_list[ib_id]);

	if (sib == NULL) {
		Log::log("FakedSetIndices, sib is NULL\n");
	}

	ib = NULL;
	if (sib == NULL){
		Log::log("FakedSetIndices, sib is NULL\n");
	}
	else{
		ib = sib->GetIB();
	}

	return curDevice->SetIndices(ib);
}

HRESULT RenderChannel::FakedCreateIndexBuffer() {
	Log::log("FakedCreateIndexBuffer called\n");
	//UINT Length,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool,IDirect3DIndexBuffer9** ppIndexBuffer,HANDLE* pSharedHandle

	UINT id = cc->read_uint();
	UINT Length = cc->read_uint();
	DWORD Usage = cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(cc->read_uint());
	D3DPOOL Pool = (D3DPOOL)(cc->read_uint());

	LPDIRECT3DINDEXBUFFER9 ib = NULL;

	getDevice(obj_id);
	HRESULT hr = curDevice->CreateIndexBuffer(Length, Usage, Format, Pool, &ib, NULL);

	ib_list[id] = new ClientIndexBuffer9(ib, Length);
	Log::log("FakedCreateIndexBuffer End. id:%d\n", id);
	return hr;
}

HRESULT RenderChannel::FakedIndexBufferLock() {
	Log::log("FakedIndexBufferLock called\n");

	int id = obj_id;
	UINT OffestToLock = cc->read_uint();
	UINT SizeToLock = cc->read_uint();
	DWORD Flags = cc->read_uint();

	Log::log("FakedIndexBufferLock called,id:%d, OffestToLock=%d, SizeToLock=%d, Flags=%d\n", id, OffestToLock, SizeToLock, Flags);

	ClientIndexBuffer9* sib = NULL;
	sib = (ClientIndexBuffer9*)(ib_list[id]);
	return sib->Lock(OffestToLock, SizeToLock, Flags);

}

HRESULT RenderChannel::FakedIndexBufferUnlock() {
	Log::log("FakedIndexBufferUnlock called\n");

	int id = obj_id;

	ClientIndexBuffer9* sib = NULL;
	sib = (ClientIndexBuffer9*)(ib_list[id]);
	Log::log("FakedIndexBufferUnlock IB id:%d\n", id);
	if (sib == NULL){
		Log::log("FakedIndexBufferUnlock ClientIndexBuffer is NULL id=%d\n", id);
		return S_OK;
	}

	return sib->Unlock(cc);
}

HRESULT RenderChannel::FakedSetSamplerState() {
	Log::log("FakedSetSamplerState called\n");
	//DWORD Sampler,D3DSAMPLERSTATETYPE Type,DWORD Value

	int Sampler, Value;
	Sampler = cc->read_int();
	char type = cc->read_char();
	D3DSAMPLERSTATETYPE Type = (D3DSAMPLERSTATETYPE)type;
	Value = cc->read_int();

	getDevice(obj_id);
	return curDevice->SetSamplerState(Sampler, Type, Value);
}

HRESULT RenderChannel::FakedCreateVertexDeclaration() {

	int id = cc->read_int();
	int cnt = cc->read_int();
	D3DVERTEXELEMENT9 dt[100];
	cc->read_byte_arr((char*)dt, cnt * sizeof(D3DVERTEXELEMENT9));

	Log::log("FakedCreateVertexDeclaration(), id=%d, cnt=%d\n", id, cnt);

	LPDIRECT3DVERTEXDECLARATION9 vd = NULL;

	getDevice(obj_id);
	HRESULT hr = curDevice->CreateVertexDeclaration(dt, &vd);

	vd_list[id] = vd;

	if (SUCCEEDED(hr)) {
		Log::log("FakedCreateVertexDeclaration() succeeded\n");
	}
	else {
		Log::log("FakedCreateVertexDeclaration() failed\n");
	}

	return hr;

	//return D3D_OK;
}

HRESULT RenderChannel::FakedSetVertexDeclaration() {

	short id = cc->read_short();

	Log::log("FakedSetVertexDeclaration(), id=%d\n", id);

	if (id == -1) return curDevice->SetVertexDeclaration(NULL);

	LPDIRECT3DVERTEXDECLARATION9 vd = NULL;
	vd = (LPDIRECT3DVERTEXDECLARATION9)(vd_list[id]);

	getDevice(obj_id);
	HRESULT hr = curDevice->SetVertexDeclaration(vd);
	if (SUCCEEDED(hr)) {
		Log::log("FakedSetVertexDeclaration() succeeded\n");
	}
	else {
		Log::log("FakedSetVertexDeclaration() failed\n");
	}
	return hr;

	//return curDevice->SetVertexDeclaration(_decl);
}

HRESULT RenderChannel::FakedSetSoftwareVertexProcessing() {
	BOOL bSoftware = cc->read_int();

	Log::log("FakedSetSoftwareVertexProcessing(), bSoftware=%d\n", bSoftware);

	getDevice(obj_id);
	return curDevice->SetSoftwareVertexProcessing(bSoftware);
}

HRESULT RenderChannel::FakedSetLight() {
	Log::log("FakedSetLight called\n");
	//DWORD Index,CONST D3DLIGHT9* pD3DLight9

	DWORD Index = cc->read_uint();
	D3DLIGHT9 light;
	cc->read_byte_arr((char*)(&light), sizeof(D3DLIGHT9));

	getDevice(obj_id);
	return curDevice->SetLight(Index, &light);
}

HRESULT RenderChannel::FakedLightEnable() {
	Log::log("FakedLightEnable called\n");
	//DWORD Index,BOOL Enable

	DWORD Index = cc->read_uint();
	BOOL Enable = cc->read_int();

	getDevice(obj_id);
	return curDevice->LightEnable(Index, Enable);
}

HRESULT RenderChannel::FakedCreateVertexShader() {
	//DWORD* pFunction,IDirect3DVertexShader9** ppShader
	Log::log("FakedCreateVertexShader called\n");

	int id = cc->read_int();
	int cnt = cc->read_int();
	DWORD* ptr = new DWORD[cnt];
	cc->read_byte_arr((char*)ptr, cnt * sizeof(DWORD));

	LPDIRECT3DVERTEXSHADER9 base_vs = NULL;

	Log::log("FakedCreateVertexShader(), id=%d, cnt=%d\n", id, cnt);

	getDevice(obj_id);
	HRESULT hr = curDevice->CreateVertexShader(ptr, &base_vs);

	vs_list[id] = base_vs;

	if (SUCCEEDED(hr)) {
		Log::log("FakedCreateVertexShader() succeeded\n");
	}
	else {
		Log::log("FakedCreateVertexShader() failed\n");
	}

	delete[] ptr;
	ptr = NULL;

	return hr;
}

HRESULT RenderChannel::FakedSetVertexShader() {
	Log::log("FakedSetVertexShader called\n");

	int id = cc->read_int();

	Log::log("FakedSetVertexShader(), id=%d\n", id);

	if (id == -1) return curDevice->SetVertexShader(NULL);

	LPDIRECT3DVERTEXSHADER9 base_vs = NULL;
	base_vs = (LPDIRECT3DVERTEXSHADER9)(vs_list[id]);

	Log::log("FakedSetVertexShader(), vs=%d\n", vs_list[id]);

	getDevice(obj_id);
	return curDevice->SetVertexShader(base_vs);
}

HRESULT RenderChannel::FakedSetVertexShaderConstantF() {
	//UINT StartRegister,CONST float* pConstantData,UINT Vector4fCount
	Log::log("FakedSetVertexShaderConstantF called\n");


	UINT StartRegister = cc->read_ushort();
	UINT Vector4fcount = cc->read_ushort();

	int i;
	for (i = 0; i < Vector4fcount / 2; ++i) {
		cc->read_vec(vs_data + (i * 8), 32);
	}

	if (Vector4fcount & 1) {
		cc->read_vec(vs_data + (i * 8), 16);
	}

	getDevice(obj_id);
	//HRESULT hr = curDevice->SetVertexShaderConstantF(StartRegister, (float*)(cc->get_cur_ptr((Vector4fcount * 4) * sizeof(float))), Vector4fcount);
	HRESULT hr = curDevice->SetVertexShaderConstantF(StartRegister, vs_data, Vector4fcount);

	return hr;
}

HRESULT RenderChannel::FakedCreatePixelShader() {
	Log::log("FakedCreatePixelShader called\n");


	int id = cc->read_int();
	int cnt = cc->read_int();
	DWORD* ptr = new DWORD[cnt];
	cc->read_byte_arr((char*)ptr, cnt * sizeof(DWORD));

	LPDIRECT3DPIXELSHADER9 base_ps = NULL;

	Log::log("FakedCreatePixelShader(), id=%d, cnt=%d\n", id, cnt);

	getDevice(obj_id);
	HRESULT hr = curDevice->CreatePixelShader(ptr, &base_ps);

	ps_list[id] = base_ps;

	if (SUCCEEDED(hr)) {
		Log::log("FakedCreatePixelShader() succeeded\n");
	}
	else {
		Log::log("FakedCreatePixelShader() failed\n");
	}

	delete[] ptr;
	ptr = NULL;

	return hr;
}

HRESULT RenderChannel::FakedSetPixelShader() {
	Log::log("FakedSetPixelShader called\n");

	int id = cc->read_int();

	if (id == -1) return curDevice->SetPixelShader(NULL);

	LPDIRECT3DPIXELSHADER9 base_ps = NULL;
	base_ps = (LPDIRECT3DPIXELSHADER9)(ps_list[id]);

	getDevice(obj_id);
	return curDevice->SetPixelShader(base_ps);
}

HRESULT RenderChannel::FakedSetPixelShaderConstantF() {
	Log::log("FakedSetPixelShaderConstantF called\n");

	UINT StartRegister = cc->read_uint();
	UINT Vector4fcount = cc->read_uint();

	for (int i = 0; i < Vector4fcount; ++i) {
		cc->read_vec(vs_data + (i * 4));
	}

	getDevice(obj_id);
	//HRESULT hr = curDevice->SetPixelShaderConstantF(StartRegister, (float*)(cc->get_cur_ptr((Vector4fcount * 4) * sizeof(float))), Vector4fcount);
	HRESULT hr = curDevice->SetPixelShaderConstantF(StartRegister, vs_data, Vector4fcount);

	return hr;
}

HRESULT RenderChannel::FakedDrawPrimitiveUP() {
	Log::log("FakedDrawPrimitiveUP called\n");
	//D3DPRIMITIVETYPE PrimitiveType,UINT PrimitiveCount,CONST void* pVertexStreamZeroData,UINT VertexStreamZeroStride

	/*
	D3DPRIMITIVETYPE PrimitiveType = (D3DPRIMITIVETYPE)(cc->read_uint());
	UINT PrimitiveCount = cc->read_uint();
	UINT VertexCount = cc->read_uint();
	UINT VertexStreamZeroStride = cc->read_uint();
	*/

	cc->read_vec((float*)arr);
	D3DPRIMITIVETYPE PrimitiveType = (D3DPRIMITIVETYPE)arr[0];
	UINT PrimitiveCount = arr[1];
	UINT VertexCount = arr[2];
	UINT VertexStreamZeroStride = arr[3];

	cc->read_vec((float*)up_buf, VertexCount * VertexStreamZeroStride);

	getDevice(obj_id);

	//return curDevice->DrawPrimitiveUP(PrimitiveType, PrimitiveCount, (void*)(cc->get_cur_ptr(VertexCount * VertexStreamZeroStride)), VertexStreamZeroStride);

	return curDevice->DrawPrimitiveUP(PrimitiveType, PrimitiveCount, (void*)up_buf, VertexStreamZeroStride);
}

HRESULT RenderChannel::FakedDrawIndexedPrimitiveUP() {
	Log::log("FakedDrawIndexedPrimitiveUP called\n");
	//D3DPRIMITIVETYPE PrimitiveType,UINT MinVertexIndex,UINT NumVertices,UINT PrimitiveCount,CONST void* pIndexData,D3DFORMAT IndexDataFormat,CONST void* pVertexStreamZeroData,UINT VertexStreamZeroStride

	D3DPRIMITIVETYPE PrimitiveType = (D3DPRIMITIVETYPE)(cc->read_uint());
	UINT MinVertexIndex = cc->read_uint();
	UINT NumVertices = cc->read_uint();
	UINT PrimitiveCount = cc->read_uint();
	D3DFORMAT IndexDataFormat = (D3DFORMAT)(cc->read_uint());
	UINT VertexStreamZeroStride = cc->read_uint();

	int indexSize = NumVertices * 2;
	if (IndexDataFormat == D3DFMT_INDEX32) indexSize = NumVertices * 4;
	char* indexData = new char[indexSize];
	cc->read_byte_arr(indexData, indexSize);

	getDevice(obj_id);
	HRESULT hr = curDevice->DrawIndexedPrimitiveUP(PrimitiveType, MinVertexIndex, NumVertices, PrimitiveCount, (void*)indexData, IndexDataFormat, (void*)(cc->get_cur_ptr(NumVertices * VertexStreamZeroStride)), VertexStreamZeroStride);

	delete[] indexData;
	indexData = NULL;

	return hr;
}

HRESULT RenderChannel::FakedSetVertexShaderConstantI() {
	Log::log("FakedSetVertexShaderConstantI called\n");
	//UINT StartRegister,CONST int* pConstantData,UINT Vector4iCount


	UINT StartRegister = cc->read_uint();
	UINT Vector4icount = cc->read_uint();
	int* ptr = new int[Vector4icount * 4];
	cc->read_byte_arr((char*)ptr, (Vector4icount * 4) * sizeof(int));

	getDevice(obj_id);
	HRESULT hr = curDevice->SetVertexShaderConstantI(StartRegister, ptr, Vector4icount);

	delete[] ptr;
	ptr = NULL;

	return hr;
}

HRESULT RenderChannel::FakedSetVertexShaderConstantB() {
	Log::log("FakedSetVertexShaderConstantB called\n");
	//UINT StartRegister,CONST BOOL* pConstantData,UINT  BoolCount

	UINT StartRegister = cc->read_uint();
	UINT BoolCount = cc->read_uint();

	getDevice(obj_id);

	return curDevice->SetVertexShaderConstantB(StartRegister, (BOOL*)(cc->get_cur_ptr(sizeof(BOOL)* BoolCount)), BoolCount);
}

HRESULT RenderChannel::FakedSetPixelShaderConstantI() {
	Log::log("FakedSetPixelShaderConstantI called\n");
	//UINT StartRegister,CONST int* pConstantData,UINT Vector4iCount

	UINT StartRegister = cc->read_uint();
	UINT Vector4iCount = cc->read_uint();

	getDevice(obj_id);


	return curDevice->SetPixelShaderConstantI(StartRegister, (int*)(cc->get_cur_ptr(sizeof(int)* (Vector4iCount * 4))), Vector4iCount);
}

HRESULT RenderChannel::FakedSetPixelShaderConstantB() {
	Log::log("FakedSetPixelShaderConstantB called\n");
	//UINT StartRegister,CONST BOOL* pConstantData,UINT  BoolCount

	UINT StartRegister = cc->read_uint();
	UINT BoolCount = cc->read_uint();

	getDevice(obj_id);

	return curDevice->SetPixelShaderConstantB(StartRegister, (BOOL*)(cc->get_cur_ptr(sizeof(BOOL)* BoolCount)), BoolCount);
}

HRESULT RenderChannel::FakedReset() {
	Log::log("FakedReset called\n");


	D3DPRESENT_PARAMETERS t_d3dpp;
	cc->read_byte_arr((char*)(&t_d3dpp), sizeof(t_d3dpp));

	getDevice(obj_id);
	return curDevice->Reset(&d3dpp);
}

HRESULT RenderChannel::FakedSetMaterial() {
	Log::log("FakedSetMaterial called\n");

	D3DMATERIAL9 material;
	cc->read_byte_arr((char*)(&material), sizeof(D3DMATERIAL9));

	getDevice(obj_id);
	return curDevice->SetMaterial(&material);
}

BOOL RenderChannel::IsTextureFormatOk(D3DFORMAT TextureFormat, D3DFORMAT AdapterFormat)
{

	HRESULT hr = gD3d->CheckDeviceFormat(D3DADAPTER_DEFAULT,
		D3DDEVTYPE_HAL,
		AdapterFormat,
		0,
		D3DRTYPE_TEXTURE,
		TextureFormat);

	return SUCCEEDED(hr);
}

HRESULT RenderChannel::FakedCreateTexture()
{
	//UINT Width,UINT Height,UINT Levels,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool,IDirect3DTexture9** ppTexture,HANDLE* pSharedHandle
	Log::log("FakedCreateTexture called\n");

	int id = cc->read_int();
	UINT Width = cc->read_uint();
	UINT Height = cc->read_uint();
	UINT Levels = cc->read_uint();
	DWORD Usage = cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(cc->read_uint());
	D3DPOOL Pool = (D3DPOOL)(cc->read_uint());

	LPDIRECT3DTEXTURE9 base_tex = NULL;

	getDevice(obj_id);

	if (IsTextureFormatOk(Format, displayMode.Format)){
		// format is ok
	}
	else{
		// form at is not ok.
		switch (Format){
		case D3DFMT_MULTI2_ARGB8:
			Log::log("Invalid Format:%d D3DFMT_MULTI2_ARGB8\n", Format);
			break;
		case D3DFMT_G8R8_G8B8:
			Log::log("Invalid Format:%d D3DFMT_G8R8_G8B8\n", Format);
			break;
		case D3DFMT_R8G8_B8G8:
			Log::log("Invalid Format:%d D3DFMT_R8G8_B8G8\n", Format);
			break;
		case D3DFMT_UYVY:
			Log::log("Invalid Format:%d D3DFMT_UYVY\n", Format);
			break;
		case D3DFMT_YUY2:
			Log::log("Invalid Format:%d D3DFMT_YUY2\n", Format);
			break;
		case D3DFMT_DXT1:
			Log::log("Invalid Format:%d D3DFMT_DXT1\n", Format);
			break;
		case D3DFMT_DXT2:
			Log::log("Invalid Format:%d D3DFMT_DXT2\n", Format);
			break;
		case D3DFMT_DXT3:
			Log::log("Invalid Format:%d D3DFMT_DXT3\n", Format);
			break;
		case D3DFMT_DXT4:
			Log::log("Invalid Format:%d D3DFMT_DXT4\n", Format);
			break;

		}
		Log::log("Invalid Format:%d \n", Format);
		if (Format == 1515474505)
			Format = D3DFMT_D32;
	}
	HRESULT hr = curDevice->CreateTexture(Width, Height, Levels, Usage, Format, Pool, &base_tex, NULL);

	switch (hr){
	case D3D_OK:
		Log::log("FakeCreateTexture(), return D3D_OK\n");
		break;
	case D3DERR_INVALIDCALL:
		Log::log("FakeCreateTexture(), return D3DERR_INVALIDCALL, height:%d, width:%d, level:%d, usage:%d, format:%d, pool:%d\n", Height, Width, Levels, Usage, Format, Pool);
		break;
	case D3DERR_OUTOFVIDEOMEMORY:
		Log::log("FakeCreateTexture(), return D3DERR_OUTOFVIDEOMEMORY\n");
		break;
	case E_OUTOFMEMORY:
		Log::log("FakeCreateTexture(), return E_OUTOFMEMORY\n");
		break;
	default:
		break;
	}
	if (base_tex == NULL) {
		Log::log("FakedCreateTexture(), CreateTexture failed, id:%d\n", id);
	}
	else{
		Log::log("FakedCreateTexture created, id:%d\n", id);
	}
	tex_list[id] = new ClientTexture9(base_tex);

	return hr;
}

HRESULT RenderChannel::FakedSetTexture() {
	//DWORD Stage,IDirect3DBaseTexture9* pTexture

	int Stage, tex_id;

	Stage = cc->read_uint();
	tex_id = cc->read_int();

	getDevice(obj_id);
	if (tex_id == -1) {
		return curDevice->SetTexture(Stage, NULL);
	}

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(tex_list[tex_id]);

	if (tex == NULL) {
		Log::log("FakedSetTexture(), Texture is NULL, tex id:%d\n", tex_id);
	}


	return curDevice->SetTexture(Stage, tex->GetTex9());
}

HRESULT RenderChannel::FakedSetTextureStageState() {
	//DWORD Stage,D3DTEXTURESTAGESTATETYPE Type,DWORD Value
	Log::log("FakedSetTextureStageState called\n");

	DWORD Stage = cc->read_uint();
	D3DTEXTURESTAGESTATETYPE Type = (D3DTEXTURESTAGESTATETYPE)(cc->read_uint());
	DWORD Value = cc->read_uint();

	getDevice(obj_id);
	return curDevice->SetTextureStageState(Stage, Type, Value);
}

HRESULT RenderChannel::FakedTransmitTextureData() {
	Log::log("FakedTransmitTextureData called\n");

#ifndef LOCAL_IMG

	int id = obj_id;
	//int levelCount = cc->read_int();
	int level = cc->read_int();

#ifdef MEM_FILE_TEX
	int Pitch = cc->read_uint();
#endif

	int size = cc->read_int();

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(tex_list[id]);

	Log::log("server TransmitTextureData get texture id:%d tex:%d\n", id, tex);
#ifndef MEM_FILE_TEX
	char  fname[50];
	sprintf(fname, "surface\\face_%d_leve_%d.png", id, level);
	LPDIRECT3DSURFACE9 des = NULL;
	tex->GetTex9()->GetSurfaceLevel(level, &des);
	//D3DXLoadSurfaceFromFile(des,NULL,NULL,fname,NULL,D3DX_FILTER_LINEAR,0xff000000,NULL);
	D3DXLoadSurfaceFromFileInMemory(des, NULL, NULL, cc->get_cur_ptr(size), size, NULL, D3DX_FILTER_LINEAR, 0xff000000, NULL);
	return D3D_OK;
#else
	return tex->FillData(level, Pitch, size, (void*)cc->get_cur_ptr(size));

#endif
#else

	int id = obj_id;
	int level = cc->read_int();

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(tex_list[id]);

	Log::log("client TransmitTextureData get texture id:%d tex:%d\n", id, tex);

	char  fname[50];
	sprintf(fname, "surface\\face_%d_leve_%d.png", id, level);
	LPDIRECT3DSURFACE9 des = NULL;
	tex->GetTex9()->GetSurfaceLevel(level, &des);
	D3DXLoadSurfaceFromFile(des, NULL, NULL, fname, NULL, D3DX_FILTER_LINEAR, 0xff000000, NULL);
	return D3D_OK;//tex->FillData(level, Pitch, size, (void*)cur_ptr);
#endif
}

HRESULT RenderChannel::FakedCreateStateBlock() {
	Log::log("FakedCreateStateBlock() called\n");

	D3DSTATEBLOCKTYPE Type = (D3DSTATEBLOCKTYPE)(cc->read_uint());
	int id = cc->read_int();

	IDirect3DStateBlock9* base_sb = NULL;

	getDevice(obj_id);
	HRESULT hr = curDevice->CreateStateBlock(Type, &base_sb);

	sb_list[id] = new ClientStateBlock9(base_sb);
	return hr;
}

HRESULT RenderChannel::FakedBeginStateBlock() {
	Log::log("FakedBeginStateBlock() called\n");

	getDevice(obj_id);
	return curDevice->BeginStateBlock();
}

HRESULT RenderChannel::FakedEndStateBlock() {
	Log::log("FakedEndStateBlock() called\n");

	int id = cc->read_int();
	IDirect3DStateBlock9* base_sb = NULL;

	getDevice(obj_id);
	HRESULT hr = curDevice->EndStateBlock(&base_sb);
	sb_list[id] = new ClientStateBlock9(base_sb);

	return hr;
}

HRESULT RenderChannel::FakedStateBlockCapture() {
	Log::log("FakedStateBlockCapture() called\n");
	int id = obj_id;
	ClientStateBlock9* sb = NULL;
	sb = (ClientStateBlock9*)(sb_list[id]);
	return sb->Capture();
}

HRESULT RenderChannel::FakedStateBlockApply() {
	Log::log("FakedStateBlockApply() called\n");
	int id = obj_id;
	ClientStateBlock9* sb = NULL;
	sb = (ClientStateBlock9*)(sb_list[id]);
	return sb->Apply();
}

HRESULT RenderChannel::FakedDeviceAddRef() {
	Log::log("FakedDeviceAddRef() called\n");

	getDevice(obj_id);
	return D3D_OK;
	return curDevice->AddRef();
}

HRESULT RenderChannel::FakedDeviceRelease() {
	Log::log("FakedDeviceRelease() called\n");

	getDevice(obj_id);
	return D3D_OK;
	return curDevice->Release();
}

HRESULT RenderChannel::FakedSetViewport() {
	Log::log("FakedSetViewport() called\n");

	getDevice(obj_id);
	D3DVIEWPORT9 viewport;
	cc->read_byte_arr((char*)(&viewport), sizeof(viewport));
	Log::log("FakeSetViewport(), viewport v.x:%d, v.y:%d, y.width:%d, v.Height:%d, v.Maxz:%f, v.Minz:%f\n", viewport.X, viewport.Y, viewport.Width, viewport.Height, viewport.MaxZ, viewport.MinZ);
	return curDevice->SetViewport(&viewport);
}

HRESULT RenderChannel::FakedSetNPatchMode() {
	Log::log("FakedSetNPatchMode() called\n");

	getDevice(obj_id);
	float nSegments = cc->read_float();
	return curDevice->SetNPatchMode(nSegments);
}

HRESULT RenderChannel::FakedCreateCubeTexture() {
	//int id, UINT EdgeLength,UINT Levels,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool
	Log::log("FakedCreateCubeTexture() called\n");

	getDevice(obj_id);

	int id = cc->read_int();
	UINT EdgeLength = cc->read_uint();
	UINT Levels = cc->read_uint();
	DWORD Usage = cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(cc->read_uint());
	D3DPOOL Pool = (D3DPOOL)(cc->read_uint());

	IDirect3DCubeTexture9* base_cube_tex = NULL;
	HRESULT hr = curDevice->CreateCubeTexture(EdgeLength, Levels, Usage, Format, Pool, &base_cube_tex, NULL);

	if (SUCCEEDED(hr)) {
		ctex_list[id] = new ClientCubeTexture9(base_cube_tex);
	}
	else {
		Log::log("FakedCreateCubeTexture() failed\n");
	}
	return hr;
}

HRESULT RenderChannel::FakedSetCubeTexture() {
	Log::log("FakedSetCubeTexture() called\n");

	getDevice(obj_id);

	DWORD Stage = cc->read_uint();
	int id = cc->read_int();

	ClientCubeTexture9* cube_tex = NULL;
	cube_tex = (ClientCubeTexture9*)(ctex_list[id]);

	if (cube_tex == NULL) {
		Log::log("FakedSetCubeTexture(), cube_tex is NULL\n");
	}

	return curDevice->SetTexture(Stage, cube_tex->GetCubeTex9());
}

HRESULT RenderChannel::FakedGetSwapChain() {
	Log::log("FakedGetSwapChain() called\n");

	getDevice(obj_id);

	int id = cc->read_int();
	UINT iSwapChain = cc->read_uint();

	IDirect3DSwapChain9* base_chain = NULL;
	HRESULT hr = curDevice->GetSwapChain(iSwapChain, &base_chain);

	if (base_chain == NULL) {
		Log::log("FakedGetSwapChain(), base_chain is NULL\n");
	}

	ClientSwapChain9* swap_chain = NULL;
	swap_chain = (ClientSwapChain9*)(chain_list[id]);

	if (swap_chain == NULL) {
		swap_chain = new ClientSwapChain9(base_chain);
		chain_list[id] = swap_chain;
	}

	return hr;
}

HRESULT RenderChannel::FakedSwapChainPresent() {
	Log::log("FakedSwapChainPresent() called\n");

	//RECT* pSourceRect,CONST RECT* pDestRect,CONST RGNDATA* pDirtyRegion,DWORD dwFlags

	/*
	RECT SourceRect, DestRect;
	RGNDATA DirtyRegion;


	int id = obj_id;

	ClientSwapChain9* swap_chain = NULL;
	swap_chain = (ClientSwapChain9*)(chain_list[id]);

	if(swap_chain == NULL) {
	Log::log("FakedSwapChainPresent(), swap_chain is NULL\n");
	}

	extern HWND hWnd;
	*/
	//擦，这里有点问题啊，mark先!
	return curDevice->Present(NULL, NULL, NULL, NULL);
	//return swap_chain->Present(pSourceRect, pDestRect, hWnd, pDirtyRegion, dwFlags);
}

HRESULT RenderChannel::FakedSetAutoGenFilterType() {
	Log::log("FakedSetAutoGenFilterType() called\n");

	int id = obj_id;
	D3DTEXTUREFILTERTYPE FilterType = (D3DTEXTUREFILTERTYPE)(cc->read_uint());

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(tex_list[id]);

	if (tex == NULL) {
		Log::log("FakedSetAutoGenFilterType(), tex is NULL\n");
	}

	return tex->SetAutoGenFilterType(FilterType);
}

void RenderChannel::FakedGenerateMipSubLevels() {
	Log::log("FakedGenerateMipSubLevels() called\n");

	int id = obj_id;

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(tex_list[id]);

	return tex->GenerateMipSubLevels();
}

HRESULT RenderChannel::FakedSetRenderTarget() {
	Log::log("FakedSetRenderTarget() called\n");
	//DWORD RenderTargetIndex,IDirect3DSurface9* pRenderTarget

	getDevice(obj_id);
	DWORD RenderTargetIndex = cc->read_uint();
	int sfid = cc->read_int();
	int tex_id = cc->read_int();
	int level = cc->read_int();

	if (sfid == -1) return curDevice->SetRenderTarget(RenderTargetIndex, NULL);

	ClientSurface9* surface = (ClientSurface9*)(surface_list[sfid]);
	ClientTexture9* texture = NULL;
	if (tex_id != -1 && tex_id < 10000)
		texture = (ClientTexture9*)(tex_list[tex_id]);
	if (texture != NULL){
		IDirect3DSurface9 * real_sur = NULL;
		texture->GetTex9()->GetSurfaceLevel(level, &real_sur);
		surface->ReplaceSurface(real_sur);
		/*char  fname[50];
		sprintf(fname,"surface\\Target%d_NOtex_%d.png",sfid,tex_id);
		D3DXSaveSurfaceToFile(fname,D3DXIFF_PNG,surface->GetSurface9(), NULL, NULL);*/
		return curDevice->SetRenderTarget(RenderTargetIndex, real_sur);
		//MessageBoxA(NULL,"surface from tex!","WARNING",MB_OK);
	}
	else if (surface){
		/*char  fname[50];
		sprintf(fname,"surface\\Target%d_tex_%d.png",sfid,tex_id);
		D3DXSaveSurfaceToFile(fname,D3DXIFF_PNG,surface->GetSurface9(), NULL, NULL);*/
		Log::log("surface:%d,texture id:%d, surface id:%d, level:%d\n", surface, tex_id, sfid, level);
		return curDevice->SetRenderTarget(RenderTargetIndex, surface->GetSurface9());
		//MessageBoxA(NULL,"surface NO!","WARNING",MB_OK);
	}
	else{
		Log::log("surface is NULL, texture id: %d, surface id: %d, level:%d\n", tex_id, sfid, level);
		return curDevice->SetRenderTarget(RenderTargetIndex, NULL);
	}
}

HRESULT RenderChannel::FakedSetDepthStencilSurface() {
	Log::log("FakedSetDepthStencilSurface() called\n");


	getDevice(obj_id);
	int sfid = cc->read_int();
	Log::log("surface id:%d\n", sfid);
	if (sfid == -1) return curDevice->SetDepthStencilSurface(NULL);

	ClientSurface9* surface = (ClientSurface9*)(surface_list[sfid]);

	return curDevice->SetDepthStencilSurface(surface->GetSurface9());
}

HRESULT RenderChannel::FakedTextureGetSurfaceLevel() {
	Log::log("FakedTextureGetSurfaceLevel() called\n");

	int id = obj_id;
	int t_id = cc->read_int();
	int sfid = cc->read_int();
	UINT Level = cc->read_int();

	ClientTexture9* tex = (ClientTexture9*)(tex_list[t_id]);
	if (!tex){
		Log::log("client texture null, id:%d\n", t_id);
	}

	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = tex->GetSurfaceLevel(Level, &base_surface);

	surface_list[sfid] = new ClientSurface9(base_surface);

	return hr;
}

HRESULT RenderChannel::FakedSwapChainGetBackBuffer() {
	//UINT iBackBuffer,D3DBACKBUFFER_TYPE Type
	Log::log("FakedSwapChainGetBackBuffer() called\n");

	int chain_id = obj_id;
	int surface_id = cc->read_int();
	UINT iBackBuffer = cc->read_uint();
	D3DBACKBUFFER_TYPE Type = (D3DBACKBUFFER_TYPE)(cc->read_uint());

	ClientSwapChain9* chain = (ClientSwapChain9*)(chain_list[chain_id]);

	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = chain->GetSwapChain9()->GetBackBuffer(iBackBuffer, Type, &base_surface);

	surface_list[surface_id] = new ClientSurface9(base_surface);

	return hr;
}

HRESULT RenderChannel::FakedGetDepthStencilSurface() {
	Log::log("FakedGetDepthStencilSurface() called\n");

	getDevice(obj_id);
	int sfid = cc->read_int();
	Log::log("surface id:%d\n", sfid);
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = curDevice->GetDepthStencilSurface(&base_surface);
	if (hr == D3D_OK){
		Log::log("GetDepthStencilSurface is OK!\n");
		surface_list[sfid] = new ClientSurface9(base_surface);
	}
	else{
		Log::log("ERROR! GetDepthStencilSurface failed!\n");

	}

	return hr;
}

HRESULT RenderChannel::FakedCreateDepthStencilSurface() {
	Log::log("FakedCreateDepthStencilSurface() called\n");
	//UINT Width,UINT Height,D3DFORMAT Format,D3DMULTISAMPLE_TYPE MultiSample,DWORD MultisampleQuality,BOOL Discard,IDirect3DSurface9** ppSurface,HANDLE* pSharedHandle

	getDevice(obj_id);
	int id = cc->read_int();
	UINT Width = cc->read_uint();
	UINT Height = cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(cc->read_uint());
	D3DMULTISAMPLE_TYPE MultiSample = (D3DMULTISAMPLE_TYPE)(cc->read_uint());
	DWORD MultisampleQuality = cc->read_uint();
	BOOL Discard = cc->read_int();

	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = curDevice->CreateDepthStencilSurface(Width, Height, Format, MultiSample, MultisampleQuality, Discard, &base_surface, NULL);

	surface_list[id] = new ClientSurface9(base_surface);

	return hr;
}

HRESULT RenderChannel::FakedCubeGetCubeMapSurface() {
	Log::log("FakedCubeGetCubeMapSurface() called\n");
	//D3DCUBEMAP_FACES FaceType,UINT Level

	int cube_id = obj_id;
	int surface_id = cc->read_int();
	D3DCUBEMAP_FACES FaceType = (D3DCUBEMAP_FACES)(cc->read_uint());
	UINT Level = cc->read_uint();

	ClientCubeTexture9* cube = (ClientCubeTexture9*)(ctex_list[cube_id]);
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = cube->GetCubeTex9()->GetCubeMapSurface(FaceType, Level, &base_surface);

	surface_list[surface_id] = new ClientSurface9(base_surface);

	return hr;
}

HRESULT RenderChannel::FakeTransmitSurface(){
	Log::log("FakeTransmitSurface() called!\n");

	int id = obj_id;
	int size = cc->read_int();
	LPDIRECT3DSURFACE9 surface = NULL;
	//curDevice->CreateDepthStencilSurface(
	//D3DXLoadSurfaceFromFileInMemory(

	return D3D_OK;
}
//newly added
HRESULT RenderChannel::FakeD3DDeviceGetBackBuffer(){
	Log::log("FakeDeviceGetBackBuffer() called!\n");
	int id = obj_id;   // device id

	int surface_id = cc->read_int();
	UINT iSwapChain = cc->read_uint();
	UINT iBackBuffer = cc->read_uint();
	UINT Type = cc->read_uint();
	D3DBACKBUFFER_TYPE type = (D3DBACKBUFFER_TYPE)Type;
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = curDevice->GetBackBuffer(iSwapChain, iBackBuffer, type, &base_surface);

	if (hr == D3D_OK){
		surface_list[surface_id] = new ClientSurface9(base_surface);
	}
	else{
		Log::log("ERROR! D3DDeviceGetBackBuffer() failed!\n");
	}
	return hr;
}

HRESULT RenderChannel::FakeD3DGetDeviceCaps(){
	HRESULT hr;
	Log::log("FakeD3DGetDeviceCaps called!\n");
	D3DDEVTYPE type = (D3DDEVTYPE)cc->read_int();
	D3DCAPS9 d3d_caps;
	hr = gD3d->GetDeviceCaps(D3DADAPTER_DEFAULT, type, &d3d_caps);
	if (SUCCEEDED(hr)){
		// send back the parameters of device
		//cc->send_raw_buffer((char *)&d3d_caps, sizeof(D3DCAPS9));
		//cc->write_byte_arr((char*) &d3d_caps, sizeof(D3DCAPS9));
	}
	else{
		Log::log("FakeD3DGetDeviceCaps failed!\n");
}


	return hr;
}
HRESULT RenderChannel::FakeD3DDGetRenderTarget(){
	HRESULT hr;
	Log::log("FakeD3DDGetRenderTarget() called!\n");
	int sid = -1;
	sid = cc->read_int();
	DWORD RenderTargetIndex = (DWORD)cc->read_uint();
	IDirect3DSurface9 * target = NULL;
	hr = curDevice->GetRenderTarget(RenderTargetIndex, &target);

	if (hr == D3D_OK){
		surface_list[sid] = new ClientSurface9(target);
	}
	else{
		Log::log("ERROR! D3DDeviceGetBackBuffer() failed!\n");
	}


	return hr;
}
HRESULT RenderChannel::FakeD3DDSetScissorRect(){
	HRESULT hr;
	Log::log("FakeD3DDSetScissorRect() calle!\n");
	int left = cc->read_int();
	int right = cc->read_int();
	int top = cc->read_int();
	int bottom = cc->read_int();

	getDevice(obj_id);

	RECT re;
	re.left = left;
	re.right = right;
	re.bottom = bottom;
	re.top = top;
	hr = curDevice->SetScissorRect(&re);

	return hr;
}

HRESULT RenderChannel::FakedSetVertexBufferFormat() {
	Log::log("FakedSetVertexBufferFormat() called\n");

	int id = obj_id;

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(vb_list[id]);
	if (svb == NULL){
		Log::log("FakedSetVertexBufferFormat is NULL id:%d\n", id);
	}
	else{
		Log::log("FakedSetVertexBufferFormat id:%d\n", id);
}

	return svb->SetVertexBufferFormat(cc);
}

HRESULT RenderChannel::FakedSetDecimateResult() {
	Log::log("FakedSetDecimateResult() called\n");

	int id = obj_id;

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(vb_list[id]);
	if (svb == NULL){
		Log::log("FakedSetDecimateResult is NULL id:%d\n", id);
	}
	else{
		Log::log("FakedSetDecimateResult id:%d\n", id);
	}

	return svb->SetDecimateResult(cc);
}

HRESULT RenderChannel::FakedSetGammaRamp() {
	Log::log("FakedSetGammaRamp() called\n");

	UINT iSwapChain = cc->read_uint();
	DWORD Flags = cc->read_uint();
	D3DGAMMARAMP pRamp;
	cc->read_byte_arr((char*)&pRamp, sizeof(D3DGAMMARAMP));

	getDevice(obj_id);

	curDevice->SetGammaRamp(iSwapChain, Flags, &pRamp);
	return D3D_OK;
}

#ifdef OLD
extern int inputTickEnd;
extern int inputTickStart;



HRESULT RenderChannel::FakeNullInstruct(){
	SYSTEMTIME sys, now;
	GetLocalTime(&now);
	inputTickEnd = GetTickCount();
	Log::slog("%d\n", inputTickEnd - inputTickStart);
	return D3D_OK;
}

#else
#if 0
extern int inputTickEnd;
extern int inputTickStart;
extern bool fromserver;
extern SYSTEMTIME start_sys;
extern time_t end_t;
extern time_t start_t;
extern bool synPress;
extern CRITICAL_SECTION syn_sec;
#endif


// the null function will synchronize the frame index
HRESULT RenderChannel::FakeNullInstruct(){
	SYSTEMTIME sys, now;
	char flag = cc->read_char();
#if 0
	//GetLocalTime(&now);
	GetSystemTime(&now);
	time(&end_t);
	inputTickEnd = GetTickCount();
	if (flag & 1){
		//Log::slog("%d\n", inputTickEnd - inputTickStart);
		bool syn = false;
		EnterCriticalSection(&syn_sec);
		syn = synPress;
		LeaveCriticalSection(&syn_sec);
		if (syn){
			Log::slog("%d\t%4f\n", (now.wHour - start_sys.wHour) * 60 * 60 * 1000 +
				(now.wMinute - start_sys.wMinute) * 1000 * 60 +
				1000 * (now.wSecond - start_sys.wSecond) +
				(now.wMilliseconds - start_sys.wMilliseconds), gap);//, inputTickEnd - inputTickStart);
			EnterCriticalSection(&syn_sec);
			synPress = false;
			LeaveCriticalSection(&syn_sec);

		}
		else{
			//Log::slog("invalid interval! ");
			/*
			Log::slog("%d\n", (now.wHour - start_sys.wHour) * 60 * 60 * 1000 +
			(now.wMinute - start_sys.wMinute) * 1000 * 60 +
			1000 *(now.wSecond - start_sys.wSecond) +
			(now.wMilliseconds - start_sys.wMilliseconds));
			*/
		}


	}
	if (flag & 2){
		// capture the screen
		Log::slog("f10 from server\n");
		INPUT in;
		memset(&in, 0, sizeof(INPUT));
		in.type = INPUT_KEYBOARD;
		in.ki.wVk = VK_F10;

		//in.ki.dwFlags |= KEYEVENTF_KEYUP;

		in.ki.wScan = MapVirtualKey(in.ki.wVk, MAPVK_VK_TO_VSC);
		SendInput(1, &in, sizeof(INPUT));

		in.ki.dwFlags |= KEYEVENTF_KEYUP;
		//in.ki.dwFlags |= KEYEVENTF_KEYDOWN;
		SendInput(1, &in, sizeof(INPUT));

		fromserver = true;
	}
#endif

	int frameIndex = cc->read_int();

	return D3D_OK;
}

#endif
#endif
