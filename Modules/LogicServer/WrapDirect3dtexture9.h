#ifndef __WRAP_DIRECT3DTEXTURE9__
#define __WRAP_DIRECT3DTEXTURE9__

#include "GameServer.h"
#include "CommandServerSet.h"
#include "TextureHelper.h"
#include "WrapDirect3dsurface9.h"

class WrapperDirect3DTexture9: public IDirect3DTexture9
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DTexture9*	m_tex;
	WrapperDirect3DTexture9(const WrapperDirect3DTexture9 &tex);
	WrapperDirect3DSurface9 ** surfaceArray;
public:

	D3DFORMAT			Format;
	D3DPOOL				Pool;
	UINT				Height, Width;
	UINT				Levels;
	DWORD				Usage;
	DWORD				Filter;
	DWORD				MipFilter;
	D3DCOLOR			ColorKey;

	TextureHelper *		texHelper;
	int					bufferSize;
	static int			totalBuffer, maxBufferSize;

	static HashSet		m_list;
	static HashSet		m_side_list;    // the list store's the id to surface map
	static int			ins_count;

#ifdef MULTI_CLIENTS
	virtual int			checkCreation(void *ctx);
	virtual int			sendCreation(void * ctx);
	virtual int			checkUpdate(void * ctx);
	virtual int			sendUpdate(void *ctx);
#endif // MULTI_CLIENTS
	UINT				getUID(int tex_id, char level);
	void				getTexIdAndLevel(UINT uid, int &id, short &level);

	IDirect3DTexture9*	GetTex9();
	HRESULT				SendTextureData();
	HRESULT				SendTextureData(ContextAndCache * ctx);
	int					getBufferSize();

	WrapperDirect3DTexture9(IDirect3DTexture9* ptr, int _id, int levels);
	static WrapperDirect3DTexture9* GetWrapperTexture9(IDirect3DTexture9* ptr);

	/*** IUnknown methods ***/
	COM_METHOD(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObj);
	COM_METHOD(ULONG,AddRef)(THIS);
	COM_METHOD(ULONG,Release)(THIS);

	/*** IDirect3DBaseTexture9 methods ***/
	COM_METHOD(HRESULT,GetDevice)(THIS_ IDirect3DDevice9** ppDevice);
	COM_METHOD(HRESULT,SetPrivateData)(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags);
	COM_METHOD(HRESULT,GetPrivateData)(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData);
	COM_METHOD(HRESULT,FreePrivateData)(THIS_ REFGUID refguid);
	COM_METHOD(DWORD, SetPriority)(THIS_ DWORD PriorityNew);
	COM_METHOD(DWORD, GetPriority)(THIS);
	COM_METHOD(void, PreLoad)(THIS);
	COM_METHOD(D3DRESOURCETYPE, GetType)(THIS);
	COM_METHOD(DWORD, SetLOD)(THIS_ DWORD LODNew);
	COM_METHOD(DWORD, GetLOD)(THIS);
	COM_METHOD(DWORD, GetLevelCount)(THIS);
	COM_METHOD(HRESULT,SetAutoGenFilterType)(THIS_ D3DTEXTUREFILTERTYPE FilterType);
	COM_METHOD(D3DTEXTUREFILTERTYPE, GetAutoGenFilterType)(THIS);
	COM_METHOD(void, GenerateMipSubLevels)(THIS);
	COM_METHOD(HRESULT,GetLevelDesc)(THIS_ UINT Level,D3DSURFACE_DESC *pDesc);
	COM_METHOD(HRESULT,GetSurfaceLevel)(THIS_ UINT Level,IDirect3DSurface9** ppSurfaceLevel);
	COM_METHOD(HRESULT,LockRect)(THIS_ UINT Level,D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags);
	COM_METHOD(HRESULT,UnlockRect)(THIS_ UINT Level);
	COM_METHOD(HRESULT,AddDirtyRect)(THIS_ CONST RECT* pDirtyRect);
};

#endif
