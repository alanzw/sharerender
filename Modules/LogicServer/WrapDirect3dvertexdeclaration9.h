#ifndef __WRAP_DIRECT3DVERTEXDECLARATION9__
#define __WRAP_DIRECT3DVERTEXDECLARATION9__

#include "GameServer.h"

class WrapperDirect3DVertexDeclaration9: public IDirect3DVertexDeclaration9 
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DVertexDeclaration9* m_vd;
public:
	UINT numElements;
	D3DVERTEXELEMENT9 * pDecl;
	//char * pDecl;
	short declSize;
	static HashSet m_list;

#ifdef MULTI_CLIENTS
	//TODO
	
	virtual int checkCreation(void *ctx);
	virtual int sendCreation(void *ctx);
	virtual int checkUpdate(void *ctx);
	virtual int sendUpdate(void *ctx);
	void print();

#endif

	static int ins_count;
	WrapperDirect3DVertexDeclaration9(IDirect3DVertexDeclaration9* ptr, int _id);
	IDirect3DVertexDeclaration9* GetVD9();
	WrapperDirect3DVertexDeclaration9 *GetWrapperVertexDeclaration9(IDirect3DVertexDeclaration9 * ptr);

public:
	/*** IUnknown methods ***/
	COM_METHOD(HRESULT,QueryInterface)(THIS_ REFIID riid, void** ppvObj);
	COM_METHOD(ULONG,AddRef)(THIS);
	COM_METHOD(ULONG,Release)(THIS);

	/*** IDirect3DVertexDeclaration9 methods ***/
	COM_METHOD(HRESULT,GetDevice)(THIS_ IDirect3DDevice9** ppDevice);
	COM_METHOD(HRESULT,GetDeclaration)(THIS_ D3DVERTEXELEMENT9* pElement,UINT* pNumElements);
};

#endif
