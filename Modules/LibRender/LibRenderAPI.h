#ifndef __CLIENT_API__
#define __CLIENT_API__

#include <d3d9.h>
#ifndef Max_Obj_Cnt
#define Max_Obj_Cnt 20010
#endif

class RenderChannel;
struct FuncJumpTable {
	char * name;
	void(**func)(RenderChannel * rch);
};

struct fptype {
	HRESULT(*CreateDevice)(RenderChannel * rch);
	HRESULT(*BeginScene)(RenderChannel * rch);
	HRESULT(*EndScene)(RenderChannel * rch);
	HRESULT(*Clear)(RenderChannel * rch);
	HRESULT(*Present)(RenderChannel * rch);

	HRESULT(*SetTransform)(RenderChannel * rch);
	HRESULT(*SetRenderState)(RenderChannel * rch);
	HRESULT(*SetStreamSource)(RenderChannel * rch);
	HRESULT(*SetFVF)(RenderChannel * rch);
	HRESULT(*DrawPrimitive)(RenderChannel * rch);

	HRESULT(*DrawIndexedPrimitive)(RenderChannel * rch);
	HRESULT(*CreateVertexBuffer)(RenderChannel * rch);
	HRESULT(*VertexBufferLock)(RenderChannel * rch);
	HRESULT(*VertexBufferUnlock)(RenderChannel * rch);
	HRESULT(*SetIndices)(RenderChannel * rch);

	HRESULT(*CreateIndexBuffer)(RenderChannel * rch);
	HRESULT(*IndexBufferLock)(RenderChannel * rch);
	HRESULT(*IndexBufferUnlock)(RenderChannel * rch);
	HRESULT(*SetSamplerState)(RenderChannel * rch);
	HRESULT(*CreateVertexDeclaration)(RenderChannel * rch);

	HRESULT(*SetVertexDeclaration)(RenderChannel * rch);
	HRESULT(*SetSoftwareVertexProcessing)(RenderChannel * rch);
	HRESULT(*SetLight)(RenderChannel * rch);
	HRESULT(*LightEnable)(RenderChannel * rch);
	HRESULT(*CreateVertexShader)(RenderChannel * rch);

	HRESULT(*SetVertexShader)(RenderChannel * rch);
	HRESULT(*SetVertexShaderConstantF)(RenderChannel * rch);
	HRESULT(*CreatePixelShader)(RenderChannel * rch);
	HRESULT(*SetPixelShader)(RenderChannel * rch);
	HRESULT(*SetPixelShaderConstantF)(RenderChannel * rch);

	HRESULT(*DrawPrimitiveUP)(RenderChannel * rch);
	HRESULT(*DrawIndexedPrimitiveUP)(RenderChannel * rch);
	HRESULT(*SetVertexShaderConstantI)(RenderChannel * rch);
	HRESULT(*SetVertexShaderConstantB)(RenderChannel * rch);
	HRESULT(*SetPixelShaderConstantI)(RenderChannel * rch);

	HRESULT(*SetPixelShaderConstantB)(RenderChannel * rch);
	HRESULT(*Reset)(RenderChannel * rch);
	HRESULT(*SetMaterial)(RenderChannel * rch);
	HRESULT(*CreateTexture)(RenderChannel * rch);
	HRESULT(*SetTexture)(RenderChannel * rch);

	HRESULT(*SetTextureStageState)(RenderChannel * rch);
	HRESULT(*TransmitTextureData)(RenderChannel * rch);
	HRESULT(*CreateStateBlock)(RenderChannel * rch);
	HRESULT(*BeginStateBlock)(RenderChannel * rch);
	HRESULT(*EndStateBlock)(RenderChannel * rch);

	HRESULT(*StateBlockCapture)(RenderChannel * rch);
	HRESULT(*StateBlockApply)(RenderChannel * rch);
	HRESULT(*DeviceAddRef)(RenderChannel * rch);
	HRESULT(*DeviceRelease)(RenderChannel * rch);
	HRESULT(*SetViewport)(RenderChannel * rch);

	HRESULT(*SetNPatchMode)(RenderChannel * rch);
	HRESULT(*CreateCubeTexture)(RenderChannel * rch);
	HRESULT(*SetCubeTexture)(RenderChannel * rch);
	HRESULT(*GetSwapChain)(RenderChannel * rch);
	HRESULT(*SwapChainPresent)(RenderChannel * rch);

	HRESULT(*SetAutoGenFilterType)(RenderChannel * rch);
	void(*GenerateMipSubLevels)(RenderChannel * rch);
	HRESULT(*SetRenderTarget)(RenderChannel * rch);
	HRESULT(*SetDepthStencilSurface)(RenderChannel * rch);
	HRESULT(*TextureGetSurfaceLevel)(RenderChannel * rch);

	HRESULT(*SwapChainGetBackBuffer)(RenderChannel * rch);
	HRESULT(*GetDepthStencilSurface)(RenderChannel * rch);
	HRESULT(*CreateDepthStencilSurface)(RenderChannel * rch);
	HRESULT(*CubeGetCubeMapSurface)(RenderChannel * rch);
	// add by alan 2013/1/5
	HRESULT(*DIConfigureDevice)(RenderChannel * rch);

	HRESULT(*DICreateDevice)(RenderChannel * rch);
	HRESULT(*DIGetDeviceStatus)(RenderChannel * rch);
	HRESULT(*DIRunContorlPanel)(RenderChannel * rch);
	HRESULT(*DIDAcquire)(RenderChannel * rch);
	HRESULT(*DIDBuildActionMap)(RenderChannel * rch);

	HRESULT(*DIDCreateEffect)(RenderChannel * rch);
	HRESULT(*DIDCreateEffectObjects)(RenderChannel * rch);
	HRESULT(*DIDEnumEffects)(RenderChannel * rch);
	HRESULT(*DIDEscape)(RenderChannel * rch);
	HRESULT(*DIDGetCapabilities)(RenderChannel * rch);

	HRESULT(*DIDGetDeviceData)(RenderChannel * rch);
	HRESULT(*DIDGetDeviceInfo)(RenderChannel * rch);
	HRESULT(*DIDGetDeviceState)(RenderChannel * rch);
	HRESULT(*DIDRunControlPanel)(RenderChannel * rch);
	HRESULT(*DIDSetActionMap)(RenderChannel * rch);

	HRESULT(*DIDCooperativeLevel)(RenderChannel * rch);
	HRESULT(*DIDSetDataFormat)(RenderChannel * rch);
	HRESULT(*DIDUnacquire)(RenderChannel * rch);
	HRESULT(*DCreateWindow)(RenderChannel * rch);
	HRESULT(*DDirectCreate)(RenderChannel * rch);

	HRESULT(*DIDirectInputCreate)(RenderChannel * rch);
	HRESULT(*DIDAddRef)(RenderChannel * rch);
	HRESULT(*DIDRelease)(RenderChannel * rch);
	HRESULT(*DIDSetProperty)(RenderChannel * rch);
	HRESULT(*TransmitSurface)(RenderChannel * rch);

	HRESULT(*D3DDeviceGetBackBuffer)(RenderChannel * rch);
	HRESULT(*D3DGetDeviceCaps)(RenderChannel * rch);
	HRESULT(*D3DDGetRenderTarget)(RenderChannel * rch);
	HRESULT(*D3DDSetScissorRect)(RenderChannel * rch);
	HRESULT(*SetVertexBufferFormat)(RenderChannel * rch);

	HRESULT(*SetDecimateResult)(RenderChannel * rch);
	HRESULT(*SetGammaRamp)(RenderChannel * rch);
	HRESULT(*NullInstruct)(RenderChannel * rch);
	HRESULT(*D3DCreateRenderTarget)(RenderChannel * rch);
	HRESULT(*D3DDSetStreamSourceFreq)(RenderChannel *rch);
	HRESULT(*D3DSurfaceRelease)(RenderChannel *rch);
};

extern fptype fptable;

typedef void(*ApiFunc)(RenderChannel * rch);

extern FuncJumpTable funcs[];

HRESULT FakedCreateDevice(RenderChannel * rch);
HRESULT FakedBeginScene(RenderChannel * rch);
HRESULT FakedEndScene(RenderChannel * rch);
HRESULT FakedClear(RenderChannel * rch);
HRESULT FakedPresent(RenderChannel * rch);

HRESULT FakedSetTransform(RenderChannel * rch);
HRESULT FakedSetRenderState(RenderChannel * rch);
HRESULT FakedSetStreamSource(RenderChannel * rch);
HRESULT FakedSetFVF(RenderChannel * rch);
HRESULT FakedDrawPrimitive(RenderChannel * rch);
HRESULT FakedDrawIndexedPrimitive(RenderChannel * rch);

HRESULT FakedCreateVertexBuffer(RenderChannel * rch);
HRESULT FakedVertexBufferLock(RenderChannel * rch);
HRESULT FakedVertexBufferUnlock(RenderChannel * rch);

HRESULT FakedSetIndices(RenderChannel * rch);
HRESULT FakedCreateIndexBuffer(RenderChannel * rch);
HRESULT FakedIndexBufferLock(RenderChannel * rch);
HRESULT FakedIndexBufferUnlock(RenderChannel * rch);
HRESULT FakedSetSamplerState(RenderChannel * rch);

HRESULT FakedCreateVertexDeclaration(RenderChannel * rch);
HRESULT FakedSetVertexDeclaration(RenderChannel * rch);
HRESULT FakedSetSoftwareVertexProcessing(RenderChannel * rch);
HRESULT FakedSetLight(RenderChannel * rch);
HRESULT FakedLightEnable(RenderChannel * rch);

HRESULT FakedCreateVertexShader(RenderChannel * rch);
HRESULT FakedSetVertexShader(RenderChannel * rch);
HRESULT FakedSetVertexShaderConstantF(RenderChannel * rch);
HRESULT FakedCreatePixelShader(RenderChannel * rch);
HRESULT FakedSetPixelShader(RenderChannel * rch);
HRESULT FakedSetPixelShaderConstantF(RenderChannel * rch);

HRESULT FakedDrawPrimitiveUP(RenderChannel * rch);
HRESULT FakedDrawIndexedPrimitiveUP(RenderChannel * rch);
HRESULT FakedSetVertexShaderConstantI(RenderChannel * rch);
HRESULT FakedSetVertexShaderConstantB(RenderChannel * rch);
HRESULT FakedSetPixelShaderConstantI(RenderChannel * rch);
HRESULT FakedSetPixelShaderConstantB(RenderChannel * rch);

HRESULT FakedReset(RenderChannel * rch);
HRESULT FakedSetMaterial(RenderChannel * rch);
HRESULT FakedCreateTexture(RenderChannel * rch);
HRESULT FakedSetTexture(RenderChannel * rch);
HRESULT FakedSetTextureStageState(RenderChannel * rch);

HRESULT FakedTransmitTextureData(RenderChannel * rch);

HRESULT FakedCreateStateBlock(RenderChannel * rch);
HRESULT FakedBeginStateBlock(RenderChannel * rch);
HRESULT FakedEndStateBlock(RenderChannel * rch);

HRESULT FakedStateBlockCapture(RenderChannel * rch);
HRESULT FakedStateBlockApply(RenderChannel * rch);

HRESULT FakedDeviceAddRef(RenderChannel * rch);
HRESULT FakedDeviceRelease(RenderChannel * rch);

HRESULT FakedSetViewport(RenderChannel * rch);
HRESULT FakedSetNPatchMode(RenderChannel * rch);

HRESULT FakedCreateCubeTexture(RenderChannel * rch);
HRESULT FakedSetCubeTexture(RenderChannel * rch);

HRESULT FakedGetSwapChain(RenderChannel * rch);
HRESULT FakedSwapChainPresent(RenderChannel * rch);
HRESULT FakedSetAutoGenFilterType(RenderChannel * rch);
void FakedGenerateMipSubLevels(RenderChannel * rch);

HRESULT FakedSetRenderTarget(RenderChannel * rch);
HRESULT FakedSetDepthStencilSurface(RenderChannel * rch);
HRESULT FakedTextureGetSurfaceLevel(RenderChannel * rch);
HRESULT FakedSwapChainGetBackBuffer(RenderChannel * rch);
HRESULT FakedGetDepthStencilSurface(RenderChannel * rch);
HRESULT FakedCreateDepthStencilSurface(RenderChannel * rch);
HRESULT FakedCubeGetCubeMapSurface(RenderChannel * rch);

// add by alan , dinput api , 2013/1/4
HRESULT FakeDIConfigureDevice(RenderChannel * rch);
HRESULT FakeDICreateDevice(RenderChannel * rch);
HRESULT FakeDIGetDeviceStatus(RenderChannel * rch);
HRESULT FakeDIRunContorlPanel(RenderChannel * rch);

HRESULT FakeDIDAcquire(RenderChannel * rch);
HRESULT FakeDIDBuildActionMap(RenderChannel * rch);
HRESULT FakeDIDCreateEffect(RenderChannel * rch);
HRESULT FakeDIDCreateEffectObjects(RenderChannel * rch);
HRESULT FakeDIDEnumEffects(RenderChannel * rch);
HRESULT FakeDIDEscape(RenderChannel * rch);
HRESULT FakeDIDGetCapabilities(RenderChannel * rch);
HRESULT FakeDIDGetDeviceData(RenderChannel * rch);
HRESULT FakeDIDGetDeviceInfo(RenderChannel * rch);
HRESULT FakeDIDGetDeviceState(RenderChannel * rch);
HRESULT FakeDIDRunControlPanel(RenderChannel * rch);
HRESULT FakeDIDSetActionMap(RenderChannel * rch);
HRESULT FakeDIDCooperativeLevel(RenderChannel * rch);
HRESULT FakeDIDSetDataFormat(RenderChannel * rch);
HRESULT FakeDIDUnacquire(RenderChannel * rch);

HRESULT FakeDCreateWindow(RenderChannel * rch);
HRESULT FakeDDirectCreate(RenderChannel * rch);
HRESULT FakeDIDirectInputCreate(RenderChannel * rch);
HRESULT FakeDIDAddRef(RenderChannel * rch);
HRESULT FakeDIDRelease(RenderChannel * rch);
HRESULT FakeDIDSetProperty(RenderChannel * rch);
HRESULT FakeTransmitSurface(RenderChannel * rch);

HRESULT FakeD3DDeviceGetBackBuffer(RenderChannel * rch);
HRESULT FakeD3DGetDeviceCaps(RenderChannel * rch);
HRESULT FakeD3DDGetRenderTarget(RenderChannel * rch);
HRESULT FakeD3DDSetScissorRect(RenderChannel * rch);

HRESULT FakedSetVertexBufferFormat(RenderChannel * rch);
HRESULT FakedSetDecimateResult(RenderChannel * rch);

HRESULT FakedSetGammaRamp(RenderChannel * rch);
HRESULT FakeNullInstruct(RenderChannel * rch);
HRESULT FakedD3DCreateRenderTarget(RenderChannel * rch);

HRESULT FakedD3DDSetStreamSourceFreq(RenderChannel * rch);

HRESULT FakedD3DSurfaceRelease(RenderChannel *rch);

#endif