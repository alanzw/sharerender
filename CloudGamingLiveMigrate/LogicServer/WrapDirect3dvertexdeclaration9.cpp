#include "CommandServerSet.h"
#include "WrapDirect3dvertexdeclaration9.h"

#ifdef MULTI_CLIENTS


#ifdef ENABLE_VERTEX_DECLARATION_LOG
char * TypeToString(BYTE type){
	switch(type){
	case D3DDECLTYPE_FLOAT1:
		return _strdup("D3DDECLTYPE_FLOAT1");
	case D3DDECLTYPE_FLOAT2:
		return _strdup("D3DDECLTYPE_FLOAT2");
	case D3DDECLTYPE_FLOAT3:
		return _strdup("D3DDECLTYPE_FLOAT3");
	case D3DDECLTYPE_FLOAT4:
		return _strdup("D3DDECLTYPE_FLOAT4");
	case D3DDECLTYPE_D3DCOLOR:
		return _strdup("D3DDECLTYPE_D3DCOLOR");
	case D3DDECLTYPE_UBYTE4:
		return _strdup("D3DDECLTYPE_UBYTE4");
	case D3DDECLTYPE_SHORT2:
		return _strdup("D3DDECLTYPE_SHORT2");
	case D3DDECLTYPE_SHORT4:
		return _strdup("D3DDECLTYPE_SHORT4");
	case D3DDECLTYPE_UBYTE4N:
		return _strdup("D3DDECLTYPE_UBYTE4N");

	case D3DDECLTYPE_SHORT2N:
		return _strdup("D3DDECLTYPE_SHORT2N");
	case D3DDECLTYPE_SHORT4N:
		return _strdup("D3DDECLTYPE_SHORT4N");
	case D3DDECLTYPE_USHORT2N:
		return _strdup("D3DDECLTYPE_USHORT2N");
	case D3DDECLTYPE_USHORT4N:
		return _strdup("D3DDECLTYPE_USHORT4N");

	case D3DDECLTYPE_UDEC3:
		return _strdup("D3DDECLTYPE_UDEC3");
	case D3DDECLTYPE_DEC3N:
		return _strdup("D3DDECLTYPE_DEC3N");

	case D3DDECLTYPE_FLOAT16_2:
		return _strdup("D3DDECLTYPE_FLOAT16_2");
	case D3DDECLTYPE_FLOAT16_4:
		return _strdup("D3DDECLTYPE_FLOAT16_4");
	case D3DDECLTYPE_UNUSED:
		return _strdup("D3DDECLTYPE_UNUSED");
	}
}
char * MethodToString(BYTE method){
	switch(method){
	case D3DDECLMETHOD_DEFAULT:
		return _strdup("D3DDECLMETHOD_DEFAULT");
	case D3DDECLMETHOD_PARTIALU:
		return _strdup("D3DDECLMETHOD_PARTIALU");
	case D3DDECLMETHOD_PARTIALV:
		return _strdup("D3DDECLMETHOD_PARTIALV:");
	case D3DDECLMETHOD_CROSSUV:
		return _strdup("D3DDECLMETHOD_CROSSUV");
	case D3DDECLMETHOD_UV:
		return _strdup("D3DDECLMETHOD_UV");
	case D3DDECLMETHOD_LOOKUP:
		return _strdup("D3DDECLMETHOD_LOOKUP");
	case D3DDECLMETHOD_LOOKUPPRESAMPLED:
		return _strdup("D3DDECLMETHOD_LOOKUPPRESAMPLED");
	}
}
char * UsageToString(BYTE usage){
	switch(usage){
	case D3DDECLUSAGE_POSITION:
		return _strdup("D3DDECLUSAGE_POSITION");
	case D3DDECLUSAGE_BLENDWEIGHT:
		return _strdup("D3DDECLUSAGE_BLENDWEIGHT");
	case D3DDECLUSAGE_BLENDINDICES:
		return _strdup("D3DDECLUSAGE_BLENDINDICES");
	case D3DDECLUSAGE_NORMAL:
		return _strdup("D3DDECLUSAGE_NORMAL");
	case D3DDECLUSAGE_PSIZE:
		return _strdup("D3DDECLUSAGE_PSIZE");
	case D3DDECLUSAGE_TEXCOORD:
		return _strdup("D3DDECLUSAGE_TEXCOORD");
	case D3DDECLUSAGE_TANGENT:
		return _strdup("D3DDECLUSAGE_TANGENT");
	case D3DDECLUSAGE_BINORMAL:
		return _strdup("D3DDECLUSAGE_BINORMAL");
	case D3DDECLUSAGE_TESSFACTOR:
		return _strdup("D3DDECLUSAGE_TESSFACTOR");
	case D3DDECLUSAGE_POSITIONT:
		return _strdup("D3DDECLUSAGE_POSITIONT:");
	case D3DDECLUSAGE_COLOR:
		return _strdup("D3DDECLUSAGE_COLOR");
	case D3DDECLUSAGE_FOG:
		return _strdup("D3DDECLUSAGE_FOG");
	case D3DDECLUSAGE_DEPTH:
		return _strdup("D3DDECLUSAGE_DEPTH");
	case D3DDECLUSAGE_SAMPLE:
		return _strdup("D3DDECLUSAGE_SAMPLE");
	}
}
#endif

void WrapperDirect3DVertexDeclaration9::print(){
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexDeclaration]:id:%d, count:%d.\n", id, numElements);
	char * strType = NULL;
	char * strMethod = NULL;
	char * strUsage = NULL;
	for(int i = 0; i< numElements; i++){
		strType = TypeToString(pDecl[i].Type);
		strMethod = MethodToString(pDecl[i].Method);
		strUsage = UsageToString(pDecl[i].Usage);
		infoRecorder->logTrace("stream:%d offset:%d type:%s method:%s usage:%s usage index:%d\n", pDecl[i].Stream, pDecl[i].Offset, strType, strMethod, strUsage, pDecl[i].UsageIndex);
		free(strType);
		free(strMethod);
		free(strUsage);
	}
#endif
}

int WrapperDirect3DVertexDeclaration9::sendCreation(void *ctx){
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexDeclaration]: send Creation.\n");
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;

	c->beginCommand(CreateVertexDeclaration_Opcode, getDeviceId());
	c->write_int(getId());
	c->write_int(this->numElements);
	c->write_byte_arr((char *)this->pDecl, (this->numElements) * sizeof(D3DVERTEXELEMENT9));
	c->endCommand();
	return 0;
}

int WrapperDirect3DVertexDeclaration9::checkCreation(void *ctx){
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexDeclaration9]: call check creation, id:%d.\n", id);
#endif
	ContextAndCache * cc = (ContextAndCache *)ctx;
	int ret= 0;
	if(!cc->isCreated(creationFlag)){
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
		ret = 1;
	}

	print();

	return ret;

}
int WrapperDirect3DVertexDeclaration9::checkUpdate(void *ctx){
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexDeclaration9]: check update, TODO.\n");
#endif
	int ret = 0;

	return ret;
}
int WrapperDirect3DVertexDeclaration9::sendUpdate(void *ctx){
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logTrace("[WrapeprDirect3DVertexDeclaration9]: send update, TODO.\n");
#endif
	int ret = 0;

	return ret;
}

#endif

WrapperDirect3DVertexDeclaration9::WrapperDirect3DVertexDeclaration9(IDirect3DVertexDeclaration9* ptr, int _id): m_vd(ptr), IdentifierBase(_id) {
	creationFlag = 0;
	//updateFlag = 0x8fffffff;
	updateFlag = 0;
	stable = true;
	m_list.AddMember(ptr, this);
}

IDirect3DVertexDeclaration9* WrapperDirect3DVertexDeclaration9::GetVD9() {
	return m_vd;
}
WrapperDirect3DVertexDeclaration9 * WrapperDirect3DVertexDeclaration9::GetWrapperVertexDeclaration9(IDirect3DVertexDeclaration9 * ptr){
	WrapperDirect3DVertexDeclaration9 * ret = (WrapperDirect3DVertexDeclaration9*)(m_list.GetDataPtr(ptr));
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	if(NULL == ret){
		infoRecorder->logError("WrapperDirect3DVertexDeclaration9::GetWrapperVertexDeclaration() ret is NULL.\n");
	}
#endif
	return ret;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DVertexDeclaration9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	HRESULT hr = m_vd->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}

STDMETHODIMP_(ULONG) WrapperDirect3DVertexDeclaration9::AddRef(THIS) {
	refCount++;
	return m_vd->AddRef();
}

STDMETHODIMP_(ULONG) WrapperDirect3DVertexDeclaration9::Release(THIS) {
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexDeclaration9::Release() called! id:%d\n", this->id);
#endif
	ULONG hr = m_vd->Release();
#ifdef LOG_REF_COUNT
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logError("WrapperDirect3DVetexDeclaration9::Release(), ref:%d.\n", hr);
#endif
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logTrace("[WrapperDirect3DVertexDeclaration9]: m_vd ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
}

/*** IDirect3DVertexDeclaration9 methods ***/
STDMETHODIMP WrapperDirect3DVertexDeclaration9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
#ifdef ENABLE_VERTEX_DECLARATION_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexDeclaration9::GetDevice called!\n");
#endif
	return m_vd->GetDevice(ppDevice);
}

STDMETHODIMP WrapperDirect3DVertexDeclaration9::GetDeclaration(THIS_ D3DVERTEXELEMENT9* pElement,UINT* pNumElements) {
	return m_vd->GetDeclaration(pElement, pNumElements);
}