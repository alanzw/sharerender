#ifndef __WRAP_DIRECT3DSURFACE9__
#define __WRAP_DIRECT3DSURFACE9__

#include "GameServer.h"

//#define USE_WRAPPER_TEXTURE    // default use SurfaceHelper, not TextureHelper

#ifdef USE_WRAPPER_TEXTURE
#include "WrapDirect3dtexture9.h"
#else
#include "TextureHelper.h"

#endif

class WrapperDirect3DSurface9 : public IDirect3DSurface9 
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DSurface9*			m_surface;
	int							tex_id;
	int							level;
	unsigned int				faceType;
#ifdef USE_WRAPPER_TEXTURE
	WrapperDirect3DTexture9 *	wrappterTex9;
#else
	SurfaceHelper *				surfaceHelper;
	IdentifierBase *			parentTexture;
	WrapperDirect3DSurface9(const WrapperDirect3DSurface9 &sur);
#endif
public:
#ifdef MULTI_CLIENTS
	// store the surface basic information for depth stencil surface
	UINT						width, height;
	D3DFORMAT					format;
	D3DMULTISAMPLE_TYPE			multiSample;
	DWORD						multisampleQuality;
	BOOL						discard;
	HANDLE *					sharedHandle;
	D3DSURFACE_DESC				desc;

	int							iSwapChain;
	int							iBackBuffer;
	D3DBACKBUFFER_TYPE			type;

	static HashSet				m_list;

	// used by render target
	int							renderTargetIndex;
	int							creationCommand;

	virtual int					checkCreation(void *ctx);
	virtual int					sendCreation(void *ctx);
	virtual int					checkUpdate(void *ctx);
	virtual int					sendUpdate(void *ctx);

#endif
	int							GetTexId();
	int							GetLevel();
	inline unsigned	int			GetFaceType(){ return faceType; }
	inline void					SetTexId(int tex){ tex_id = tex; }
	inline void					SetLevel(int _level){ level= _level; }
	inline void					SetFaceType(unsigned int face){ faceType = face; }

	void						setParentTexture(IdentifierBase *parent);//{ parentTexture = parent;}
	void						releaseData();
	inline IdentifierBase *		getParentTexture(){ return parentTexture; }
#ifdef USE_WRAPPER_TEXTURE
	inline void					setTex9(WrapperDirect3DTexture9 * _tex){ wrappterTex9 = _tex; }
#else
	inline void					setSurfaceHelper(SurfaceHelper * _helper){ surfaceHelper = _helper; }
#endif

	static int					ins_count;
	WrapperDirect3DSurface9(IDirect3DSurface9* ptr, int id);
	static WrapperDirect3DSurface9* GetWrapperSurface9(IDirect3DSurface9* ptr);
	IDirect3DSurface9* GetSurface9();

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