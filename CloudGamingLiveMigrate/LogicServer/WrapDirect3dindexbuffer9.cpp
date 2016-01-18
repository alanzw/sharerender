#include "CommandServerSet.h"
#include "WrapDirect3dindexbuffer9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
static int max_ib = 0;


#ifdef MULTI_CLIENTS
//#define ENABLE_INDEX_LOG

int WrapperDirect3DIndexBuffer9::sendCreation(void *ctx){
#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("[WrapperDirect3DIndexBuffer9]: send the creation command.\n");
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

	return 0;
}

int WrapperDirect3DIndexBuffer9::checkCreation(void * ctx){
	#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("[WrapperDirect3DIndexBuffer9]: call check creation.\n");
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
	infoRecorder->logTrace("[WrapperDirect3DIndexedBuffer9]: send update.\n");
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	//sendData(getDeviceId());
	PrepareIndexBuffer(c);
	
	return 0;
}

int WrapperDirect3DIndexBuffer9::checkUpdate(void * ctx){
	#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("[WrapperDirect3DindexBuffer9]: check update.\n");
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
	updateFlag = 0x8fffffff;
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
	updateFlag = 0x8fffffff;
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
	if(SizeToLock == 0) m_LockData.SizeToLock = length - OffsetToLock;
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

#ifdef BUFFER_UNLOCK_UPDATE
	int last = 0, cnt = 0, c_len = 0, size = 0, base = 0;
	// copy from video buffer
#ifndef USE_MEM_INDEX_BUFFER
	memcpy(ram_buffer, (char *)m_LockData.pVideoBuffer + m_LockData.OffsetToLock, m_LockData.SizeToLock);
#else  // USE_MEM_INDEX_BUFFER
	memcpy(m_LockData.pVideoBuffer, (char *)m_LockData.pRAMBuffer + m_LockData.OffsetToLock, m_LockData.SizeToLock);
#endif  // USE_MEM_INDEX_BUFFER

	base = m_LockData.OffsetToLock;

	csSet->checkCreation(dynamic_cast<IdentifierBase *>(this));

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
			//infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::PrepareVertexBuffer, last=%d, d=%d\n", last, d);
		}
	}
	int neg = (1<<28)-1;
	csSet->writeInt(neg);

	//获取当前这条指令的长度
	c_len = csSet->getCommandLength();

	if(c_len > m_LockData.SizeToLock) {
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::Unlock(), change too much cancel command, id=%d, c_len:%d, total len:%d\n", id, c_len, length);
#endif

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
			#ifdef ENABLE_INDEX_LOG
			infoRecorder->logTrace("max ib:%d\n", max_ib);
#endif
		}
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
#endif
	}
	else {

		if(cnt > 0) {
			#ifdef ENABLE_INDEX_LOG
			infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
#endif
			csSet->endCommand();
		}
		else
			csSet->cancelCommand();
	}
	csSet->resetChanged(updateFlag);
#endif  // BUFFER_UNLOCK_UPDATE

	return m_ib->Unlock();
}

void WrapperDirect3DIndexBuffer9::read_data_from_buffer(char** ptr, int offest, int size) {
	if(size == 0) size = length - offest;
	if(!ram_buffer) {
		ram_buffer = (char*)(malloc(this->length));
		m_LockData.pRAMBuffer = ram_buffer;
	}

	void* p = NULL;
	Lock(offest, size, &p, 2048);
	memcpy(ram_buffer, p, size);
	Unlock();

	*ptr = ram_buffer;
}

void WrapperDirect3DIndexBuffer9::write_data_to_buffer(char* ptr, int offest, int size) {
	if(size == 0) size = length - offest;
	void* p = NULL;
	Lock(offest, size, &p, 2048);
	memcpy((char*)p, ptr, size);
	Unlock();
}

STDMETHODIMP WrapperDirect3DIndexBuffer9::GetDesc(THIS_ D3DINDEXBUFFER_DESC *pDesc) { return m_ib->GetDesc(pDesc); }

int WrapperDirect3DIndexBuffer9::PrepareIndexBuffer() {
	
#ifndef MULTI_CLIENTS
	if(isFirst) {
		memset(cache_buffer, 0, length);
		isFirst = false;
	}
	//TODO : check flags for each clients

	if(!changed){
		infoRecorder->logTrace("WrapperDirect3DIndexBuffer9 not changed!\n");
		return 1;
	}
	
	double tick_s = 0.0, tick_e = 0.0, tick_a  = 0.0;
	tick_s = GetTickCount();

	char* p = NULL;
	read_data_from_buffer(&p, 0, 0);

	int last = 0, cnt = 0, c_len = 0, size = 0;
	int base = m_LockData.OffsetToLock;
	tick_e = GetTickCount();

	cs.begin_command(IndexBufferUnlock_Opcode, GetID());
	cs.write_uint(m_LockData.OffsetToLock);
	cs.write_uint(m_LockData.SizeToLock);
	cs.write_uint(m_LockData.Flags);

	cs.write_int(CACHE_MODE_DIFF);
	
	int stride = (Format == D3DFMT_INDEX16) ? 2 : 4;
	cs.write_char(stride);
	
	for(int i=0; i<m_LockData.SizeToLock; ++i) {
		if( cache_buffer[base + i] ^ *((char*)(m_LockData.pRAMBuffer) + i) ) {
			int d = i - last;
			cs.write_int(d);
			last = i;
			cs.write_char( *((char*)(m_LockData.pRAMBuffer) + i) );
			cnt++;
			cache_buffer[base + i] = *((char*)(m_LockData.pRAMBuffer) + i);
			//infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::PrepareVertexBuffer, last=%d, d=%d\n", last, d);
		}
	}
	int neg = (1<<28)-1;
	cs.write_int(neg);
	tick_a = GetTickCount();
	//获取当前这条指令的长度
	c_len = cs.get_command_length();


	infoRecorder->logTrace("\tLock index buffer:%f, cache time:%f\n", tick_e - tick_s, tick_a- tick_s);


	if(c_len > m_LockData.SizeToLock) {
		infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::PrepareIndexBuffer(), cancel command, id=%d\n", id);

		c_len = m_LockData.SizeToLock;
		infoRecorder->logTrace("offest=%d, c_len=%d\n", m_LockData.OffsetToLock, c_len);

		cs.cancel_command();
		cs.begin_command(IndexBufferUnlock_Opcode, GetID());
		cs.write_uint(m_LockData.OffsetToLock);
		cs.write_uint(m_LockData.SizeToLock);
		cs.write_uint(m_LockData.Flags);

		cs.write_int(CACHE_MODE_COPY);
		cs.write_char(stride);
		cs.write_byte_arr((char*)(m_LockData.pRAMBuffer), c_len);
		cs.end_command();


		//infoRecorder->logError("ib: id:%d, size:%d\n", this->id, this->length);
		if(c_len > max_ib){
			max_ib = c_len;
			infoRecorder->logTrace("max ib:%d\n", max_ib);
		}
		infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
	}
	else {


		if(cnt > 0) {
			infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
			cs.end_command();
		}
		else cs.cancel_command();


	}
	
	this->changed = false;
	return (cnt > 0);
#else
	if(isFirst) {
		memset(cache_buffer, 0, length);
		isFirst = false;
	}
	// check flags for each clients
	
	double tick_s = 0.0, tick_e = 0.0, tick_a  = 0.0;
	tick_s = GetTickCount();

	char* p = NULL;
	read_data_from_buffer(&p, 0, 0);

	int last = 0, cnt = 0, c_len = 0, size = 0;
	int base = m_LockData.OffsetToLock;
	tick_e = GetTickCount();

	csSet->beginCommand(IndexBufferUnlock_Opcode, getId());
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
			//infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::PrepareVertexBuffer, last=%d, d=%d\n", last, d);
		}
	}
	int neg = (1<<28)-1;
	csSet->writeInt(neg);
	tick_a = GetTickCount();
	//获取当前这条指令的长度
	c_len = csSet->getCommandLength();

	#ifdef ENABLE_INDEX_LOG
	infoRecorder->logTrace("\tLock index buffer:%f, cache time:%f\n", tick_e - tick_s, tick_a- tick_s);
#endif

	if(c_len > m_LockData.SizeToLock) {
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::Unlock(), cancel command, id=%d\n", id);
#endif

		c_len = m_LockData.SizeToLock;
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("offest=%d, c_len=%d\n", m_LockData.OffsetToLock, c_len);
#endif

		csSet->cancelCommand();
		csSet->beginCommand(IndexBufferUnlock_Opcode, getId());
		csSet->writeUInt(m_LockData.OffsetToLock);
		csSet->writeUInt(m_LockData.SizeToLock);
		csSet->writeUInt(m_LockData.Flags);

		csSet->writeInt(CACHE_MODE_COPY);
		csSet->writeChar(stride);
		csSet->writeByteArr((char *)m_LockData.pRAMBuffer, c_len);
		csSet->endCommand();


		//infoRecorder->logError("ib: id:%d, size:%d\n", this->id, this->length);
		if(c_len > max_ib){
			max_ib = c_len;
			#ifdef ENABLE_INDEX_LOG
			infoRecorder->logTrace("max ib:%d\n", max_ib);
#endif
		}
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
#endif
	}
	else {

		if(cnt > 0) {
			#ifdef ENABLE_INDEX_LOG
			infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
#endif
			csSet->endCommand();
		}
		else
			csSet->cancelCommand();
	}
	
	csSet->setChanged(updateFlag);
	return (cnt > 0);
#endif
}

int WrapperDirect3DIndexBuffer9::PrepareIndexBuffer(ContextAndCache * ctx) {

	if(isFirst) {
		memset(cache_buffer, 0, length);
		isFirst = false;
	}
	// check flags for each clients

	#ifdef ENABLE_INDEX_LOG
	double tick_s = 0.0, tick_e = 0.0, tick_a  = 0.0;
	tick_s = GetTickCount();
#endif

	char* p = NULL;
	read_data_from_buffer(&p, 0, 0);

	int last = 0, cnt = 0, c_len = 0, size = 0;
	int base = m_LockData.OffsetToLock;
	#ifdef ENABLE_INDEX_LOG
	tick_e = GetTickCount();
#endif
#if 0
	csSet->beginCommand(IndexBufferUnlock_Opcode, getId());
	csSet->writeUInt(m_LockData.OffsetToLock);
	csSet->writeUInt(m_LockData.SizeToLock);
	csSet->writeUInt(m_LockData.Flags);
	csSet->writeInt(CACHE_MODE_DIFF);
#endif

	ctx->beginCommand(IndexBufferUnlock_Opcode, getId());
	ctx->write_uint(m_LockData.OffsetToLock);
	ctx->write_uint(m_LockData.SizeToLock);
	ctx->write_uint(m_LockData.Flags);
	ctx->write_int(CACHE_MODE_DIFF);

	int stride = (Format == D3DFMT_INDEX16) ? 2 : 4;
#if 0
	csSet->writeChar(stride);
#else
	ctx->write_char(stride);
#endif

	for(int i=0; i<m_LockData.SizeToLock; ++i) {
		if( cache_buffer[base + i] ^ *((char*)(m_LockData.pRAMBuffer) + i) ) {
			int d = i - last;
			last = i;

#if 0
			csSet->writeInt(d);
			csSet->writeChar( *((char*)(m_LockData.pRAMBuffer) + i) );
#else
			ctx->write_int(d);
			ctx->write_char(*((char *)(m_LockData.pRAMBuffer) + i));
#endif
			
			cnt++;
			cache_buffer[base + i] = *((char*)(m_LockData.pRAMBuffer) + i);
			//infoRecorder->logTrace("WrapperDirect3DVertexBuffer9::PrepareVertexBuffer, last=%d, d=%d\n", last, d);
		}
	}
	int neg = (1<<28)-1;

#if 0
	csSet->writeInt(neg);
#else
	ctx->write_int(neg);
#endif
	c_len = ctx->getCommandLength();
#ifdef ENABLE_INDEX_LOG
	tick_a = GetTickCount();
	//获取当前这条指令的长度
	//c_len = csSet->getCommandLength();
	infoRecorder->logTrace("\tLock index buffer:%f, cache time:%f\n", tick_e - tick_s, tick_a- tick_s);

#endif

	if(c_len > m_LockData.SizeToLock) {
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("WrapperDirect3DIndexBuffer9::PrepareIndexBuffer(), cancel command, id=%d\n", id);
#endif

		c_len = m_LockData.SizeToLock;
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("offest=%d, c_len=%d\n", m_LockData.OffsetToLock, c_len);
#endif
#if 0
		csSet->cancelCommand();
		csSet->beginCommand(IndexBufferUnlock_Opcode, getId());
		csSet->writeUInt(m_LockData.OffsetToLock);
		csSet->writeUInt(m_LockData.SizeToLock);
		csSet->writeUInt(m_LockData.Flags);

		csSet->writeInt(CACHE_MODE_COPY);
		csSet->writeChar(stride);
		csSet->writeByteArr((char *)m_LockData.pRAMBuffer, c_len);
		csSet->endCommand();
#else
		ctx->cancelCommand();
		ctx->beginCommand(IndexBufferUnlock_Opcode, getId());
		ctx->write_uint(m_LockData.OffsetToLock);
		ctx->write_uint(m_LockData.SizeToLock);
		ctx->write_uint(m_LockData.Flags);

		ctx->write_int(CACHE_MODE_COPY);
		ctx->write_char(stride);
		ctx->write_byte_arr((char *)m_LockData.pRAMBuffer, c_len);
		ctx->endCommand();
#endif


		//infoRecorder->logError("ib: id:%d, size:%d\n", this->id, this->length);
		if(c_len > max_ib){
			max_ib = c_len;
			#ifdef ENABLE_INDEX_LOG
			infoRecorder->logTrace("max ib:%d\n", max_ib);
#endif
		}
		#ifdef ENABLE_INDEX_LOG
		infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
#endif
	}
	else {

		if(cnt > 0) {
			#ifdef ENABLE_INDEX_LOG
			infoRecorder->logTrace("ib  id:%d, changed %d, org:%d\n",this->id, c_len,this->length);
#endif
#if 0 
			csSet->endCommand();
#else
			ctx->endCommand();
#endif
		}
		else{
#if 0
			csSet->cancelCommand();
#else
			ctx->cancelCommand();
#endif
		}
	}
	return (cnt > 0);
}
