#include "CommandServerSet.h"
#include "WrapDirect3dindexbuffer9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
static int max_ib = 0;


#ifdef MULTI_CLIENTS
//#define ENABLE_INDEX_LOG

int WrapperDirect3DIndexBuffer9::sendCreation(void *ctx){
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("[WrapperDirect3DIndexBuffer9]: send the creation command for %d.\n", id);
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;

	c->beginCommand(CreateIndexBuffer_Opcode, getDeviceId());
	c->write_uint(getId());
	c->write_uint(length);
	c->write_uint(Usage);
	c->write_uint(Format);
	c->write_uint(Pool);
	c->endCommand();
	// change the creation flag
	c->setCreation(creationFlag);


	if(pTimer) pTimer->Start();
	PrepareIndexBuffer(c);
	if(pTimer){
		unsigned interval = pTimer->Stop();
		infoRecorder->logError("[WrapeprDirect3DIndexBuffer9]: prepare index buffer %d use %f ms.\n", id, 1000.0 * interval / pTimer->getFreq());
	}

	return 0;
}

int WrapperDirect3DIndexBuffer9::checkCreation(void * ctx){
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("[WrapperDirect3DIndexBuffer9]: call check creation for %d.\n", id);
#endif
	int ret = 0;

	ContextAndCache * c = (ContextAndCache *)ctx;
	if(!c->isCreated(creationFlag)){
		ret = sendCreation(ctx);
		c->setCreation(creationFlag);
		ret = 1;
	}
	return ret;
}

int WrapperDirect3DIndexBuffer9::sendUpdate(void * ctx){
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("[WrapperDirect3DIndexedBuffer9]: send update for %d.\n", id);
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	//sendData(getDeviceId());
	if(pTimer) pTimer->Start();
	UpdateIndexBuffer(c);
	if(pTimer){
		unsigned int interval = pTimer->Stop();
		infoRecorder->logError("[WrapperDirect3DeIndexBuffer9]: update index buffer %d use %f ms\n", id, 1000.0 * interval / pTimer->getFreq());
	}

	return 0;
}

int WrapperDirect3DIndexBuffer9::checkUpdate(void * ctx){
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("[WrapperDirect3DindexBuffer9]: check update for %d.\n", id);
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(c->isChanged(updateFlag)){
		ret = sendUpdate(ctx);
		// tag unchanged
		c->resetChanged(updateFlag);
		ret = 1;
	}
	else{
		// unchanged
#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("[WrapperDirect3DIndexBuffer9]: unchanged.\n");
#endif
	}
	return ret;
}

#endif


WrapperDirect3DIndexBuffer9::WrapperDirect3DIndexBuffer9(IDirect3DIndexBuffer9* ptr, int _id, int _length): m_ib(ptr), IdentifierBase(_id), length(_length) {
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9 constructor, size:%d\n", length);
#endif
	cache_buffer = new char[_length];
	memset(cache_buffer, 0, _length);
#ifndef MULTI_CLIENTS

	ram_buffer = new char[_length];
#else
	ram_buffer = new char[_length];
	m_LockData.pRAMBuffer = ram_buffer;
	//ram_buffer = NULL;
#endif
	isFirst = true;
	m_list.AddMember(ptr, this);
	creationFlag = 0;
	updateFlag = 0;
	//updateFlag = 0x8fffffff;
	stable = false;
}

WrapperDirect3DIndexBuffer9::WrapperDirect3DIndexBuffer9(IDirect3DIndexBuffer9* ptr, int _id): m_ib(ptr), IdentifierBase(_id), length(0) {
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9 constructor, size:0\n");
#endif
	isFirst = true;
	cache_buffer = NULL;
	ram_buffer = NULL;
	m_list.AddMember(ptr, this);
	creationFlag = 0;
	//updateFlag = 0x8fffffff;
	updateFlag = 0;
	stable = false;
}

IDirect3DIndexBuffer9* WrapperDirect3DIndexBuffer9::GetIB9() {
	return m_ib;
}

int WrapperDirect3DIndexBuffer9::GetLength() {
	return length;
}

WrapperDirect3DIndexBuffer9* WrapperDirect3DIndexBuffer9::GetWrapperIndexedBuffer9(IDirect3DIndexBuffer9* base_indexed_buffer) {
	WrapperDirect3DIndexBuffer9* ret = (WrapperDirect3DIndexBuffer9*)( m_list.GetDataPtr( (PVOID)base_indexed_buffer ) );

#ifdef ENABLE_INDEX_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::GetWrapperIndexedBuffer9() ret is NULL\n");
	}
#endif
	return ret;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DIndexBuffer9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	//infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::QueryInterface() called, base_index=%d, this=%d, ppvObj=%d\n", m_ib, this, *ppvObj);
	HRESULT hr = m_ib->QueryInterface(riid, ppvObj);
	*ppvObj  = this;
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::QueryInterface() end, base_index=%d, this=%d, ppvObj=%d\n", m_ib, this, *ppvObj);
#endif
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DIndexBuffer9::AddRef(THIS) { refCount++; return m_ib->AddRef(); }
STDMETHODIMP_(ULONG) WrapperDirect3DIndexBuffer9::Release(THIS) { 
	ULONG hr = m_ib->Release();
#ifdef LOG_REF_COUNT
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::Release(), ref:%d.\n", hr);
#endif
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DIndexBuffer9]: m_ib ref:%d, ref count:%d.\n", refCount, hr);
		//m_list.DeleteMember(m_ib);
	}
	return hr; 
}

/*** IDirect3DResource9 methods ***/
STDMETHODIMP WrapperDirect3DIndexBuffer9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) { 
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::GetDevice called!\n");
#endif
	IDirect3DDevice9* base = NULL;
	HRESULT hr = this->m_ib->GetDevice(&base);
	WrapperDirect3DDevice9 * ret = WrapperDirect3DDevice9::GetWrapperDevice9(base);
#ifdef ENABLE_INDEX_LOG
	if(ret == NULL){
		infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::GetDevice return NULL!\n");
	}
#endif
	*ppDevice = ret;
	return hr;
}

STDMETHODIMP WrapperDirect3DIndexBuffer9::SetPrivateData(THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags) {
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::SetPrivateData() TODO\n");
#endif
	return m_ib->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

STDMETHODIMP WrapperDirect3DIndexBuffer9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData) { return m_ib->GetPrivateData(refguid, pData, pSizeOfData); }
STDMETHODIMP WrapperDirect3DIndexBuffer9::FreePrivateData(THIS_ REFGUID refguid) { return m_ib->FreePrivateData(refguid); }

STDMETHODIMP_(DWORD) WrapperDirect3DIndexBuffer9::SetPriority(THIS_ DWORD PriorityNew) {
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::SetPriority() TODO\n");
#endif
	return m_ib->SetPriority(PriorityNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DIndexBuffer9::GetPriority(THIS) { return m_ib->GetPriority(); }

STDMETHODIMP_(void) WrapperDirect3DIndexBuffer9::PreLoad(THIS) {
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::PreLoad() TODO\n");
#endif
	return m_ib->PreLoad();
}

STDMETHODIMP_(D3DRESOURCETYPE) WrapperDirect3DIndexBuffer9::GetType(THIS) { return m_ib->GetType(); }

STDMETHODIMP WrapperDirect3DIndexBuffer9::Lock(THIS_ UINT OffsetToLock,UINT SizeToLock,void** ppbData,DWORD Flags) {
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::Lock(),id:%d, OffestToLock=%d, SizeToLock=%d, Flags=%d, size:%d\n",this->id, OffsetToLock, SizeToLock, Flags,this->length);
#endif
#ifndef BUFFER_UNLOCK_UPDATE
	void * tmp = NULL;

	HRESULT hr = m_ib->Lock(OffsetToLock, SizeToLock, &tmp, Flags);

	m_LockData.OffsetToLock = OffsetToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;
	if(SizeToLock == 0) 
		m_LockData.SizeToLock = length - OffsetToLock;

	m_LockData.pRAMBuffer = tmp;
	*ppbData = tmp;
	// set changed flag to all
	csSet->setChangedToAll(updateFlag);
#else   // BUFFER_UNLOCK_UPDATE

#ifndef USE_MEM_INDEX_BUFFER
	void * tmp = NULL;
	// lock the whole buffer
	HRESULT hr = m_ib->Lock(0, 0, &tmp, Flags);
	m_LockData.OffsetToLock = OffsetToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;
	if(SizeToLock == 0) m_LockData.SizeToLock = this->length - OffsetToLock;
	*ppbData = (void *)(((char *)tmp) + OffsetToLock);

	m_LockData.pVideoBuffer = tmp;
#ifdef MULTI_CLIENTS
	csSet->setChangedToAll(updateFlag);
#endif // MULTI_CLIENTS

#else  // USE_MEM_INDEX_BUFFER
	if(!ram_buffer){
		ram_buffer = (char*)malloc(sizeof(char)*GetLength());
		m_LockData.pRAMBuffer = ram_buffer;
	}
	// store the lock information
	m_LockData.OffsetToLock = OffsetToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;
	if(0 == SizeToLock){
		m_LockData.SizeToLock = length - OffsetToLock;
	}
	*ppbData = (void*)(((char *)ram_buffer) + OffsetToLock);

	// lock the video mem as well
	HRESULT hr = m_ib->Lock(OffsetToLock, SizeToLock, &(m_LockData.pVideoBuffer), Flags);
#ifdef MULTI_CLIENTS
	csSet->setChangedToAll(updateFlag);
#endif // MULTI_CLIENTS

#endif // USE_MEM_INDEX_BUFFER

#endif  // BUFFER_UNLOCK_UPDATE


	return hr;
}

STDMETHODIMP WrapperDirect3DIndexBuffer9::Unlock(THIS) {
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::Unlock(),id:%d, UnlockSize=%d Bytes, total len:%d, start:%d.\n",id, m_LockData.SizeToLock, length, m_LockData.OffsetToLock);
#endif
	if(isFirst) {
		memset(cache_buffer, 0, length);
		isFirst = false;
	}
	if(pTimer){
		pTimer->Start();
	}
#ifdef BUFFER_UNLOCK_UPDATE
	int last = 0, cnt = 0, c_len = 0, size = 0, base = 0;
	// copy from video buffer
#ifndef USE_MEM_INDEX_BUFFER
	memcpy(ram_buffer, (char *)m_LockData.pVideoBuffer + m_LockData.OffsetToLock, m_LockData.SizeToLock);
#else  // USE_MEM_INDEX_BUFFER
	memcpy(m_LockData.pVideoBuffer, (char *)m_LockData.pRAMBuffer + m_LockData.OffsetToLock, m_LockData.SizeToLock);
#endif  // USE_MEM_INDEX_BUFFER

	base = m_LockData.OffsetToLock;

	//csSet->checkCreation(dynamic_cast<IdentifierBase *>(this));
	csSet->checkObj(dynamic_cast<IdentifierBase *>(this));

	csSet->beginCommand(IndexBufferUnlock_Opcode, id);
	csSet->writeUInt(m_LockData.OffsetToLock);
	csSet->writeUInt(m_LockData.SizeToLock);
	csSet->writeUInt(m_LockData.Flags);
	csSet->writeInt(CACHE_MODE_DIFF);

	int stride = (Format == D3DFMT_INDEX16) ? 2 : 4;
	csSet->writeChar(stride);

	for(int i=0; i<m_LockData.SizeToLock; ++i) {
		if( cache_buffer[base + i] ^ *((char*)(m_LockData.pRAMBuffer) + i) ) {
			int d = i - last;
			csSet->writeInt(d);
			last = i;
			csSet->writeChar( *((char*)(m_LockData.pRAMBuffer) + i) );
			cnt++;
			cache_buffer[base + i] = *((char*)(m_LockData.pRAMBuffer) + i);
		}
	}
	int neg = (1<<28)-1;
	csSet->writeInt(neg);

	c_len = csSet->getCommandLength();

	if(c_len > m_LockData.SizeToLock) {

		c_len = m_LockData.SizeToLock;
		csSet->cancelCommand();
		csSet->beginCommand(IndexBufferUnlock_Opcode, getId());
		csSet->writeUInt(m_LockData.OffsetToLock);
		csSet->writeUInt(m_LockData.SizeToLock);
		csSet->writeUInt(m_LockData.Flags);

		csSet->writeInt(CACHE_MODE_COPY);
		csSet->writeChar(stride);
		csSet->writeByteArr((char *)m_LockData.pRAMBuffer, c_len);
		csSet->endCommand();

		if(c_len > max_ib){
			max_ib = c_len;
		}
	}
	else {
		if(cnt > 0) {
			csSet->endCommand();
		}
		else{
			csSet->cancelCommand();
		}
	}
	csSet->resetChanged(updateFlag);
	if(pTimer){
		unsigned int interval = pTimer->Stop();
		infoRecorder->logError("[WrapperDirect3DIndexBuffer9]: %d unlock use time:%f ms.\n", id, interval * 1000.0 / pTimer->getFreq());
	}
#endif  // BUFFER_UNLOCK_UPDATE

	return m_ib->Unlock();
}


STDMETHODIMP WrapperDirect3DIndexBuffer9::GetDesc(THIS_ D3DINDEXBUFFER_DESC *pDesc) { return m_ib->GetDesc(pDesc); }

// called when to update an index buffer
int WrapperDirect3DIndexBuffer9::PrepareIndexBuffer(ContextAndCache *ctx){

	infoRecorder->logTrace("[WrapperDirect3DIndexBuffer]: Prepare index buffer for %d.\n", id);

	int stride = (Format == D3DFMT_INDEX16) ? 2 : 4;
	// all comes to CACHE_MODE_COPY
	ctx->beginCommand(IndexBufferUnlock_Opcode, getId());
	ctx->write_uint(0);
	ctx->write_uint(length);
	ctx->write_uint(D3DLOCK_NOSYSLOCK);
	ctx->write_int(CACHE_MODE_COPY);
	ctx->write_char(stride);

	if(isFirst){
		// need to copy all ram_buffer, the update flag should be reset
		ctx->write_byte_arr(ram_buffer, length);
		ctx->resetChanged(updateFlag);
	}
	else{
		// the cache_buffer stores the last updated data, no change to update flag
		ctx->write_byte_arr(cache_buffer, length);
	}
	//ctx->write_byte_arr((char *)m_LockData.pRAMBuffer, c_len);
	ctx->endCommand();
	return length;
}

// called when to create an index buffer
int WrapperDirect3DIndexBuffer9::UpdateIndexBuffer(ContextAndCache * ctx) {
	infoRecorder->logTrace("[WrapperDirect3DIndexBuffer9]: call UpdateIndexBuffer for %d, that means data changed after creation and unlock did not happened.\n", id);
	if(isFirst) {
		//memset(cache_buffer, 0, length);
		//isFirst = false;
		infoRecorder->logError("[WrapperDirect3DIndexBuffer9]: is first is true ? ERROR.\n");
	}

	int last = 0, cnt = 0, c_len = 0, size = 0;
	int base = m_LockData.OffsetToLock;

	ctx->beginCommand(IndexBufferUnlock_Opcode, getId());
	ctx->write_uint(m_LockData.OffsetToLock);
	ctx->write_uint(m_LockData.SizeToLock);
	ctx->write_uint(m_LockData.Flags);
	ctx->write_int(CACHE_MODE_DIFF);

	int stride = (Format == D3DFMT_INDEX16) ? 2 : 4;

	ctx->write_char(stride);

	for(int i=0; i<m_LockData.SizeToLock; ++i) {
		if( cache_buffer[base + i] ^ *((char*)(m_LockData.pRAMBuffer) + i) ) {
			int d = i - last;
			last = i;

			ctx->write_int(d);
			ctx->write_char(*((char *)(m_LockData.pRAMBuffer) + i));

			cnt++;
			cache_buffer[base + i] = *((char*)(m_LockData.pRAMBuffer) + i);
		}
	}
	int neg = (1<<28)-1;
	ctx->write_int(neg);
	c_len = ctx->getCommandLength();

	if(c_len > m_LockData.SizeToLock) {
		c_len = m_LockData.SizeToLock;

		ctx->cancelCommand();
		ctx->beginCommand(IndexBufferUnlock_Opcode, getId());
		ctx->write_uint(m_LockData.OffsetToLock);
		ctx->write_uint(m_LockData.SizeToLock);
		ctx->write_uint(m_LockData.Flags);

		ctx->write_int(CACHE_MODE_COPY);
		ctx->write_char(stride);
		ctx->write_byte_arr((char *)m_LockData.pRAMBuffer, c_len);
		ctx->endCommand();
		if(c_len > max_ib){
			max_ib = c_len;
		}
	}
	else {

		if(cnt > 0) {
			ctx->endCommand();
		}
		else{
			ctx->cancelCommand();
		}
	}
	return (cnt > 0);
}
