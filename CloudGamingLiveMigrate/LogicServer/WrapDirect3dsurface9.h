#ifndef __WRAP_DIRECT3DSURFACE9__
#define __WRAP_DIRECT3DSURFACE9__

#include "GameServer.h"
#include "WrapDirect3dtexture9.h"

class WrapperDirect3DSurface9 : public IDirect3DSurface9 
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DSurface9* m_surface;
	bool isSent;
	// add 17:00
	int tex_id;
	int level;
	WrapperDirect3DTexture9 * wrappterTex9;
public:
#ifdef MULTI_CLIENTS
	//map<SOCKET, bool> created; // indicate whether the surface is exist in client, false for initializing
	// store the surface basic information for depth stencil surface
	UINT width, height;
	D3DFORMAT format;
	D3DMULTISAMPLE_TYPE multiSample;
	DWORD multisampleQuality;
	BOOL discard;
	HANDLE * sharedHandle;
	D3DSURFACE_DESC desc;

	int iSwapChain;
	int iBackBuffer;
	D3DBACKBUFFER_TYPE type;

	static HashSet m_list;
	// used by swap chain
	int swapChainId;

	// used by render target
	int renderTargetIndex;
	int creationCommand;

	virtual int checkCreation(void *ctx);
	virtual int sendCreation(void *ctx);
	virtual int checkUpdate(void *ctx);
	virtual int sendUpdate(void *ctx);

#endif
	void RepalceSurface(IDirect3DSurface9* pnew);
	int GetTexId();
	int GetLevel();
	inline void SetTexId(int tex){
		tex_id = tex;
	}
	inline void SetLevel(int _level){
		level= _level;
	}
	inline void setTex9(WrapperDirect3DTexture9 * _tex){ wrappterTex9 = _tex; }
	static int ins_count;
	WrapperDirect3DSurface9(IDirect3DSurface9* ptr, int id);
	static WrapperDirect3DSurface9* GetWrapperSurface9(IDirect3DSurface9* ptr);
	IDirect3DSurface9* GetSurface9();

	void SendSurface();

	/*** IUnknown methods ***/
	COM_METHOD(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObj);
	COM_METHOD(ULONG,AddRef)(THIS);
	COM_METHOD(ULONG,Release)(THIS);

	/*** IDirect3DResource9 methods ***/
	COM_METHOD(HRESULT,GetDevice)(THIS_ IDirect3DDevice9** ppDevice);
	COM_METHOD(HRESULT,SetPrivateData)(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags);
	COM_METHOD(HRESULT,GetPrivateData)(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData);
	COM_METHOD(HRESULT,FreePrivateData)(THIS_ REFGUID refguid);
	COM_METHOD(DWORD, SetPriority)(THIS_ DWORD PriorityNew);
	COM_METHOD(DWORD, GetPriority)(THIS);
	COM_METHOD(void, PreLoad)(THIS);
	COM_METHOD(D3DRESOURCETYPE, GetType)(THIS);
	COM_METHOD(HRESULT,GetContainer)(THIS_ REFIID riid,void** ppContainer);
	COM_METHOD(HRESULT,GetDesc)(THIS_ D3DSURFACE_DESC *pDesc);
	COM_METHOD(HRESULT,LockRect)(THIS_ D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags);
	COM_METHOD(HRESULT,UnlockRect)(THIS);
	COM_METHOD(HRESULT,GetDC)(THIS_ HDC *phdc);
	COM_METHOD(HRESULT,ReleaseDC)(THIS_ HDC hdc);
};

#endif