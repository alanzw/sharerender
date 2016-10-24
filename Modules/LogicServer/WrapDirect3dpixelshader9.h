#ifndef __WRAP_DIRECT3DPIXELSHADER9__
#define __WRAP_DIRECT3DPIXELSHADER9__

#include "GameServer.h"

class WrapperDirect3DPixelShader9: public IDirect3DPixelShader9 
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DPixelShader9* m_ps;
public:
	static HashSet m_list;
#ifdef MULTI_CLIENTS

	int funcCount;   // store the functioin count
	char * pFunc;    // stores the detailed shader functions
	short shaderSize;   // stores the shader size

	virtual int checkCreation(void *ctx);
	virtual int sendCreation(void * ctx);
	virtual int checkUpdate(void *ctx);
	virtual int sendUpdate(void *ctx);
#endif

	static int ins_count;
	WrapperDirect3DPixelShader9(IDirect3DPixelShader9* ptr, int _id);
	IDirect3DPixelShader9* GetPS9();
	static WrapperDirect3DPixelShader9* GetWrapperPixelShader(IDirect3DPixelShader9* ptr);

	/*** IUnknown methods ***/
	COM_METHOD(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObj);
	COM_METHOD(ULONG,AddRef)(THIS);
	COM_METHOD(ULONG,Release)(THIS);

	/*** IDirect3DPixelShader9 methods ***/
	COM_METHOD(HRESULT,GetDevice)(THIS_ IDirect3DDevice9** ppDevice);
	COM_METHOD(HRESULT,GetFunction)(THIS_ void*,UINT* pSizeOfData);

};

#endif