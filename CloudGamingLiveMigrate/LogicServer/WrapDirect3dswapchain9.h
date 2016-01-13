#ifndef __WRAP_DIRECT3DSWAPCHAIN9__
#define __WRAP_DIRECT3DSWAPCHAIN9__

#include "GameServer.h"
#include "../LibCore/Opcode.h"

class WrapperDirect3DSwapChain9 : public IDirect3DSwapChain9 
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DSwapChain9* m_chain;
public:
#ifdef MULTI_CLIENTS
	virtual int checkCreation(void *ctx);
	virtual int sendCreation(void *ctx);
	virtual int checkUpdate(void *ctx);
	virtual int sendUpdate(void *ctx);
	UINT iSwapChain;
	short deivceId;
#endif

	static HashSet m_list;
	static int ins_count;

	WrapperDirect3DSwapChain9(IDirect3DSwapChain9* ptr, int _id);
	static WrapperDirect3DSwapChain9* GetWrapperSwapChain9(IDirect3DSwapChain9* ptr);

public:
	/*** IUnknown methods ***/
	COM_METHOD(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObj);
	COM_METHOD(ULONG,AddRef)(THIS);
	COM_METHOD(ULONG,Release)(THIS);

	/*** IDirect3DSwapChain9 methods ***/
	COM_METHOD(HRESULT,Present)(THIS_ CONST RECT* pSourceRect,CONST RECT* pDestRect,HWND hDestWindowOverride,CONST RGNDATA* pDirtyRegion,DWORD dwFlags);
	COM_METHOD(HRESULT,GetFrontBufferData)(THIS_ IDirect3DSurface9* pDestSurface);
	COM_METHOD(HRESULT,GetBackBuffer)(THIS_ UINT iBackBuffer,D3DBACKBUFFER_TYPE Type,IDirect3DSurface9** ppBackBuffer);
	COM_METHOD(HRESULT,GetRasterStatus)(THIS_ D3DRASTER_STATUS* pRasterStatus);
	COM_METHOD(HRESULT,GetDisplayMode)(THIS_ D3DDISPLAYMODE* pMode);
	COM_METHOD(HRESULT,GetDevice)(THIS_ IDirect3DDevice9** ppDevice);
	COM_METHOD(HRESULT,GetPresentParameters)(THIS_ D3DPRESENT_PARAMETERS* pPresentationParameters);
};

#endif
