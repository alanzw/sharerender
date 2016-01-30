#ifndef __WRAP_DIRECT3DVERTEXBUFFER9__
#define __WRAP_DIRECT3DVERTEXBUFFER9__

#include "GameServer.h"
#include "CommandServerSet.h"

#define USE_MEM_VERTEX_BUFFER

class YMesh;

class WrapperDirect3DVertexBuffer9: public IDirect3DVertexBuffer9 
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DVertexBuffer9* m_vb;
	
public:
	
	char* cache_buffer; // cache for vertex buffer, release at destruction
	char* ram_buffer;   // each context has a buffer

	bool isFirst;
	bool readed_;
	int max_vb;

	DWORD Usage;
	DWORD FVF;
	D3DPOOL Pool;
	UINT Length;
	BufferLockData m_LockData;   

	IDirect3DVertexDeclaration9 * decl;

	// used in SetStreamSource
	short streamNumber;
	unsigned int offsetInBytes;

	bool isFormated();
	bool getNormalTangentOffsetAndSize();

#ifdef MULTI_CLIENTS
	virtual int checkCreation(void * ctx);
	virtual int sendCreation(void *ctx);
	virtual int checkUpdate(void * ctx);
	virtual int sendUpdate(void * ctx);
#endif

	static HashSet m_list;
	static int ins_count;
	WrapperDirect3DVertexBuffer9(IDirect3DVertexBuffer9* ptr, int _id, int _length);
	WrapperDirect3DVertexBuffer9(IDirect3DVertexBuffer9* ptr, int _id);
	LPDIRECT3DVERTEXBUFFER9 GetVB9();
	char * get_cache_buffer(){
		return cache_buffer;
	}
	int GetLength();

	static WrapperDirect3DVertexBuffer9* GetWrapperVertexBuffer9(IDirect3DVertexBuffer9* ptr);

#if 0
	void read_data_from_buffer(char** ptr, int offest, int size);
	void write_data_to_buffer(char* ptr, int offest, int size);
#endif

	int PrepareVertexBuffer(ContextAndCache *ctx);
	int UpdateVertexBuffer(ContextAndCache *ctx);

public:
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

	COM_METHOD(HRESULT,Lock)(THIS_ UINT OffsetToLock,UINT SizeToLock,void** ppbData,DWORD Flags);
	COM_METHOD(HRESULT,Unlock)(THIS);
	COM_METHOD(HRESULT,GetDesc)(THIS_ D3DVERTEXBUFFER_DESC *pDesc);
};

#endif
