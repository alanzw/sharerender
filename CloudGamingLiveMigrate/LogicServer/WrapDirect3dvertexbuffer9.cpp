#include "CommandServerSet.h"
#include "WrapDirect3dvertexbuffer9.h"
#include "../LibCore/Opcode.h"
//#define FLOAT_COMPRESS
#define INT_COMPRESS
#define COMPRESS_TO_DWORD


#define ENABLE_VERTEX_BUFFER_LOG


#ifdef MULTI_CLIENTS
// check the creation flag for each client
int WrapperDirect3DVertexBuffer9::sendCreation(void * ctx){
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: send creation for %d.\n", id);
#endif

	ContextAndCache * c = (ContextAndCache *)ctx;

	c->beginCommand(CreateVertexBuffer_Opcode, getDeviceId());
	c->write_uint(getId());
	c->write_uint(Length);
	c->write_uint(Usage);
	c->write_uint(FVF);
	c->write_uint(Pool);
	c->endCommand();

	if(pTimer){
		pTimer->Start();
	}
	int ret = PrepareVertexBuffer(c);
	if(pTimer){
		unsigned int interval =  pTimer->Stop();
		infoRecorder->logError("[WrpaperDirect3DVertexBuffer9]: prepare vertex buffer use: %f ms.\n", interval * 1000.0 / pTimer->getFreq());
	}

	return ret;
}
int WrapperDirect3DVertexBuffer9::checkCreation(void *ctx){
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: check creation for %d.\n", id);
#endif
	int ret= 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(!c->isCreated(creationFlag)){
		// not created, send the creation command.
		ret = sendCreation(ctx);
		// change the creation flag
		if(ret)
			c->setCreation(creationFlag);
	}
	return ret;
}
int WrapperDirect3DVertexBuffer9::checkUpdate(void * ctx){
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: check update for %d.\n", id);
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(c->isChanged(updateFlag)){
		ret = sendUpdate(ctx);
		if(ret)
			c->resetChanged(updateFlag);
	}else{
#ifdef ENABLE_VERTEX_BUFFER_LOG
		// no changed
		infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: unchanged.\n");
#endif
	}
	return ret;
}
int WrapperDirect3DVertexBuffer9::sendUpdate(void * ctx){
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: send update for %d.\n", id);
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	//PrepareVertexBuffer(c);
	if(pTimer) pTimer->Start();
	ret = UpdateVertexBuffer(c);
	if(pTimer){
		unsigned int interval = pTimer->Stop();
		infoRecorder->logError("[WrapperDirect3DVertexBuffer9]: update vertex buffer %d use %f ms.\n", id, interval * 1000.0 /pTimer->getFreq());
	}
	return ret;
}

#endif

WrapperDirect3DVertexBuffer9::WrapperDirect3DVertexBuffer9(IDirect3DVertexBuffer9* ptr, int _id, int _length): m_vb(ptr), IdentifierBase(_id), Length(_length) {
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9 constructor called, id=%d, length=%d\n", _id, _length);
#endif

	cache_buffer = new char[_length];
	memset(cache_buffer, 0, _length);
	ram_buffer = new char[_length];
	memset(ram_buffer, 0, _length);
	m_LockData.pRAMBuffer = ram_buffer;
	isFirst = true;
	decl = NULL;

	m_list.AddMember(ptr, this);

	max_vb = 0;

	creationFlag = 0;
	//updateFlag = 0x8fffffff;
	updateFlag = 0;
	stable = false;
}

WrapperDirect3DVertexBuffer9::WrapperDirect3DVertexBuffer9(IDirect3DVertexBuffer9* ptr, int _id): m_vb(ptr), IdentifierBase(_id), Length(0) {
	//cache_buffer = new char[_length];
	isFirst = true;
	decl = NULL;

	m_list.AddMember(ptr, this);
	max_vb = 0;
	creationFlag= 0;
	m_LockData.pRAMBuffer = NULL;
	//updateFlag = 0x8fffffff;
	updateFlag = 0;
	stable = false;
}


LPDIRECT3DVERTEXBUFFER9 WrapperDirect3DVertexBuffer9::GetVB9() {
	return m_vb;
}

int WrapperDirect3DVertexBuffer9::GetLength() {
	return this->Length;
}

WrapperDirect3DVertexBuffer9* WrapperDirect3DVertexBuffer9::GetWrapperVertexBuffer9(IDirect3DVertexBuffer9* ptr) {
	WrapperDirect3DVertexBuffer9* ret = (WrapperDirect3DVertexBuffer9*)( m_list.GetDataPtr( ptr ) );
#ifdef ENABLE_VERTEX_BUFFER_LOG
	if(ret == NULL) {
		infoRecorder->logError("WrapperDirect3DVertexBuffer9::GetWrapperVertexBuffer9(), ret is NULL\n");
	}
#endif
	return ret;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DVertexBuffer9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {

	HRESULT hr = m_vb->QueryInterface(riid, ppvObj);
	*ppvObj = this;
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::QueryInterface() called, base_vb=%d, this=%d, ppvObj=%d\n", m_vb, this, *ppvObj);
#endif
	return hr;
};
STDMETHODIMP_(ULONG) WrapperDirect3DVertexBuffer9::AddRef(THIS) {
	refCount++;
	return m_vb->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3DVertexBuffer9::Release(THIS) {
	ULONG hr = m_vb->Release();
#ifdef LOG_REF_COUNT
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::Release(), ref:%d.\n", hr);
#endif
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DVertexBuffer9]: m_vb id:%d ref:%d, ref count:%d.\n",id, refCount, hr);
		//m_list.DeleteMember(m_vb);
	}
	return hr;
}

/*** IDirect3DResource9 methods ***/
STDMETHODIMP WrapperDirect3DVertexBuffer9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::GetDevice() TODO\n");
#endif
	return m_vb->GetDevice(ppDevice);
}
STDMETHODIMP WrapperDirect3DVertexBuffer9::SetPrivateData(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags) {
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::SetPrivateData() TODO\n");
#endif
	return m_vb->SetPrivateData(refguid, pData, SizeOfData, Flags);
}
STDMETHODIMP WrapperDirect3DVertexBuffer9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData) {
	return m_vb->GetPrivateData(refguid, pData, pSizeOfData);
}
STDMETHODIMP WrapperDirect3DVertexBuffer9::FreePrivateData(THIS_ REFGUID refguid) {
	return m_vb->FreePrivateData(refguid);
}
STDMETHODIMP_(DWORD) WrapperDirect3DVertexBuffer9::SetPriority(THIS_ DWORD PriorityNew) {
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::SetPriority() TODO\n");
#endif
	return m_vb->SetPriority(PriorityNew);
}
STDMETHODIMP_(DWORD) WrapperDirect3DVertexBuffer9::GetPriority(THIS) {
	return m_vb->GetPriority();
}
STDMETHODIMP_(void) WrapperDirect3DVertexBuffer9::PreLoad(THIS) {
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::PreLoad() TODO\n");
#endif
	return m_vb->PreLoad();
}
STDMETHODIMP_(D3DRESOURCETYPE) WrapperDirect3DVertexBuffer9::GetType(THIS) {
	return m_vb->GetType();
}

STDMETHODIMP WrapperDirect3DVertexBuffer9::Lock(THIS_ UINT OffsetToLock,UINT SizeToLock,void** ppbData,DWORD Flags) {
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::Lock(), id=%d, length=%d, offest=%d, size_to_lock=%d, flag=%d\n",this->id, Length, OffsetToLock, SizeToLock, Flags);
#endif
#ifndef BUFFER_UNLOCK_UPDATE
	void * tmp = NULL;
	HRESULT hr = m_vb->Lock(OffsetToLock, SizeToLock, &tmp, Flags);
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::Lock() end\n");
	m_LockData.OffsetToLock = OffsetToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;

	if(SizeToLock == 0) m_LockData.SizeToLock = Length - OffsetToLock;

	*ppbData = tmp;

#ifdef MULTI_CLIENTS
	// changed to all context, like 0xffff?
	csSet->setChangedToAll(updateFlag);
#endif
	readed_ = false;
#else   // BUFFER_UNLOCK_UPDATE

#ifndef USE_MEM_VERTEX_BUFFER
	void * tmp = NULL;
	// lock the whole buffer
	HRESULT hr = m_vb->Lock(OffsetToLock, SizeToLock, &tmp, Flags);

	m_LockData.OffsetToLock = OffsetToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;
	if(SizeToLock == 0) m_LockData.SizeToLock = Length - OffsetToLock;

	*ppbData = (void *)(((char *)tmp) + OffsetToLock);
	m_LockData.pVideoBuffer = tmp;   // to the start of the entire buffer

#ifdef MULTI_CLIENTS
	csSet->setChangedToAll(updateFlag);
#endif // MULTI_CLIENTS
	readed_ = false;
	return hr;
#else  // USE_MEM_VERTEX_BUFFER
	
	// store the lock information
	m_LockData.OffsetToLock = OffsetToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;

	if(SizeToLock == 0) 
		m_LockData.SizeToLock = Length - OffsetToLock;

	*ppbData = (void*)(((char *)ram_buffer)+OffsetToLock);

	// lock the video mem as well
	HRESULT hr = m_vb->Lock(OffsetToLock, SizeToLock, &(m_LockData.pVideoBuffer), Flags);
#ifdef MULTI_CLIENTS
	
	//updateFlag = 0x8fffffff;
#endif // MULTI_CLIENTS
	return hr;

#endif  // USE_MEM_VERTEX_BUFFER
#endif // BUFFER_UNLOCK_UPDATE

}

STDMETHODIMP WrapperDirect3DVertexBuffer9::Unlock(THIS) {
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::Unlock(), id:%d, UnlockSize=%d Bytes, total len:%d, start:%d.\n", this->id,m_LockData.SizeToLock, Length, m_LockData.OffsetToLock);

#endif  //ENABLE_VERTEX_BUFFER_LOG
	if(isFirst){
		//memset(cache_buffer, 0, Length);
		isFirst = false;
	}
	// update the vertex buffer
#ifdef BUFFER_UNLOCK_UPDATE

	infoRecorder->logError("WrapperDirect3DVertexBuffer9::Unlock(), id:%d, UnlockSize=%d Bytes, total len:%d, start:%d.\n", this->id,m_LockData.SizeToLock, Length, m_LockData.OffsetToLock);
	if(pTimer){
		pTimer->Start();
	}
	// the buffer is updated, read data to ram_buffer
	int last = 0, cnt = 0, c_len = 0, size = 0, base = 0;
#ifndef USE_MEM_VERTEX_BUFFER
	// copy from video buffer
	memcpy(ram_buffer + m_LockData.OffsetToLock, (char *)m_LockData.pVideoBuffer , m_LockData.SizeToLock);

#else   // USE_MEM_VERTEX_BUFFER
	// copy to video buffer
	memcpy(m_LockData.pVideoBuffer, (char *)m_LockData.pRAMBuffer + m_LockData.OffsetToLock, m_LockData.SizeToLock);

	csSet->checkObj(this);
	//csSet->setChangedToAll(updateFlag);
	updateFlag = 0x8fffffff;

#endif  // USE_MEM_VERTEX_BUFFER
	base = m_LockData.OffsetToLock;

	csSet->beginCommand(VertexBufferUnlock_Opcode, id);
	csSet->writeUInt(m_LockData.OffsetToLock);
	csSet->writeUInt(m_LockData.SizeToLock);
	csSet->writeUInt(m_LockData.Flags);
	csSet->writeInt(CACHE_MODE_DIFF);

	for(unsigned int i = 0; i< m_LockData.SizeToLock; ++i){
		if(cache_buffer[base + i] ^ *((char *)(m_LockData.pRAMBuffer) + m_LockData.OffsetToLock + i) ){
			int d = i - last;
			csSet->writeInt(d);
			last = i;
			csSet->writeChar(*((char *)(m_LockData.pRAMBuffer) + m_LockData.OffsetToLock + i));
			cnt++;
			cache_buffer[base + i] = *((char *)(m_LockData.pRAMBuffer) + m_LockData.OffsetToLock + i);
		}
	}
	int neg = ( 1 << 28 ) - 1;
	csSet->writeInt(neg);

	c_len = csSet->getCommandLength();

	if(c_len > m_LockData.SizeToLock){
#ifdef ENABLE_VERTEX_BUFFER_LOG
		infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: Unlock(), change too much, cancel command, id=%d.\n", id);
#endif
		c_len = m_LockData.SizeToLock;

		csSet->cancelCommand();
		csSet->beginCommand(VertexBufferUnlock_Opcode, id);
		csSet->writeUInt(m_LockData.OffsetToLock);
		csSet->writeUInt(m_LockData.SizeToLock);
		csSet->writeUInt(m_LockData.Flags);

		csSet->writeInt(CACHE_MODE_COPY);

		csSet->writeByteArr((char *)m_LockData.pRAMBuffer, c_len);
		csSet->endCommand();

		if(c_len > max_vb){
			max_vb = c_len;
#ifdef ENABLE_VERTEX_BUFFER_LOG
			infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: max vb:%d.\n, id:%d.\n", c_len, id);
#endif
		}
	}
	else{
		if(cnt > 0){
#ifdef ENABLE_VERTEX_BUFFER_LOG
			infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: unlock changed vertex count:%d, c_len:%d.\n", cnt, c_len);
#endif
			csSet->endCommand();
		}
		else{
#ifdef ENABLE_VERTEX_BUFFER_LOG
			infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: unlock not changed, c_len:%d, cnt:%d.\n", c_len, cnt);
#endif
			csSet->cancelCommand();
		}
	}

	csSet->resetChanged(updateFlag);
	if(pTimer){
		unsigned int interval = pTimer->Stop();
		infoRecorder->logError("[WrapperDirect3DVertexBuffer]: %d unlock use time: %f.\n", id, interval * 1000.0 / pTimer->getFreq());
	}
#endif   // BUFFER_UNLOCK_UPDATE
	return m_vb->Unlock();
}


STDMETHODIMP WrapperDirect3DVertexBuffer9::GetDesc(THIS_ D3DVERTEXBUFFER_DESC *pDesc) {
	//infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::GetDesc() TODO\n");
	return m_vb->GetDesc(pDesc);
}
// prepare the vertex buffer
int WrapperDirect3DVertexBuffer9::PrepareVertexBuffer(ContextAndCache *ctx){
	infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: Prepare vertex buffer for %d.\n", id);
	ctx->beginCommand(VertexBufferUnlock_Opcode, getId());
	ctx->write_uint(0);
	ctx->write_uint(Length);
	ctx->write_uint(D3DLOCK_NOSYSLOCK);

	ctx->write_int(CACHE_MODE_COPY);
	if(isFirst){
		ctx->write_byte_arr((char *)ram_buffer, Length);
		memcpy(cache_buffer, ram_buffer, Length);
		ctx->resetChanged(updateFlag);
	}
	else{
		ctx->write_byte_arr((char *)cache_buffer, Length); 
	}
	ctx->endCommand();

	return Length;
}

// update the vertex buffer with byte comparasion

int WrapperDirect3DVertexBuffer9::UpdateVertexBuffer(ContextAndCache * ctx){
#ifdef ENABLE_VERTEX_BUFFER_LOG
	infoRecorder->logTrace("[WrapperDirect3DVertexBuffer9]: update the vertex buffer.\n");
#endif
	infoRecorder->logError("[WrapperDirect3DVertexBuffer9]: update the vertex buffer %d, that means data chang3ed after creation but unlock didt not happen.\n", id);
	if(isFirst){
		//memset(cache_buffer, 0, Length);
		//isFirst = false;
		infoRecorder->logError("[WrapperDirect3DVertexBuffer9]: is first is true ? ERROR.\n");
	}

	int last = 0, cnt = 0, c_len = 0, size = 0;
	int base = m_LockData.OffsetToLock;

	ctx->beginCommand(VertexBufferUnlock_Opcode, getId());
	ctx->write_uint(m_LockData.OffsetToLock);
	ctx->write_uint(m_LockData.SizeToLock);
	ctx->write_uint(m_LockData.Flags);
	ctx->write_int(CACHE_MODE_DIFF);

	for(int i = 0; i< m_LockData.SizeToLock; ++i){
		if(cache_buffer[base + i] ^ *((char *)(m_LockData.pRAMBuffer) + base + i) ){
			int d = i - last;
			last = i;
			ctx->write_int(d);
			ctx->write_char(*((char *)(m_LockData.pRAMBuffer)+ base + i));
			cnt++;
			cache_buffer[base + i] = *((char *)(m_LockData.pRAMBuffer)+base + i);
		}
	}
	int neg = ( 1 << 28 ) - 1;

	ctx->write_int(neg);

	// get current command length
	c_len = ctx->getCommandLength();

	if(c_len > m_LockData.SizeToLock){
		c_len = m_LockData.SizeToLock;

		ctx->cancelCommand();
		ctx->beginCommand(VertexBufferUnlock_Opcode, getId());
		ctx->write_uint(m_LockData.OffsetToLock);
		ctx->write_uint(m_LockData.SizeToLock);
		ctx->write_uint(m_LockData.Flags);

		ctx->write_int(CACHE_MODE_COPY);
		ctx->write_byte_arr((char *)m_LockData.pRAMBuffer, c_len);
		ctx->endCommand();

		if(c_len > max_vb){
			max_vb = c_len;
		}
	}
	else{
		if(cnt > 0){
			ctx->endCommand();
		}
		else{
			ctx->cancelCommand();
		}
	}

	return cnt > 0;
}