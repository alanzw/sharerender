#include <WinSock2.h>
#include "LibRenderIndexbuffer9.h"
#include <assert.h>
#include <stdlib.h>

using namespace cg;
using namespace cg::core;

ClientIndexBuffer9::ClientIndexBuffer9(IDirect3DIndexBuffer9* ptr, int _length): m_ib(ptr), length(_length), isFirst(true) {
	//cg::core::infoRecorder->logTrace("ClientIndexBuffer9 constructor called! id:%d\n", 
}

HRESULT ClientIndexBuffer9::Lock(UINT OffestToLock, UINT SizeToLock, DWORD Flags) {
	m_LockData.OffsetToLock = OffestToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;

	if(SizeToLock == 0) m_LockData.SizeToLock = length - OffestToLock;

	return S_OK;
}

HRESULT ClientIndexBuffer9::Unlock(cg::core::CommandClient * cc) {

	m_LockData.OffsetToLock = cc->read_uint();
	m_LockData.SizeToLock = cc->read_uint();
	m_LockData.Flags = cc->read_uint();
	cg::core::infoRecorder->logTrace("ClientIndexBuffer9::Unlock(), OffestToLock=%d, SizeToLock=%d Bytes, Flags=%d\n", m_LockData.OffsetToLock, m_LockData.SizeToLock, m_LockData.Flags);

	UpdateIndexBuffer(cc);

	return D3D_OK;
}

IDirect3DIndexBuffer9* ClientIndexBuffer9::GetIB() {
	return m_ib;
}

int ClientIndexBuffer9::GetLength() {
	return length;
}

void ClientIndexBuffer9::UpdateIndexBuffer(cg::core::CommandClient * cc) {
	void* ib_ptr;

	if(isFirst) {
		//m_ib->Lock(0, 0, (void**)&ib_ptr, m_LockData.Flags);
		HRESULT  hh =m_ib->Lock(0, 0, (void**)&ib_ptr, D3DLOCK_NOSYSLOCK);
		if(hh == D3DERR_INVALIDCALL){
			cg::core::infoRecorder->logTrace("ClientIndexBuffer9:: lock index buffer failed!\n");
		}
		memset(ib_ptr, 0, length);
		m_ib->Unlock();
		isFirst = false;
	}

	m_ib->Lock(m_LockData.OffsetToLock, m_LockData.SizeToLock, (void**)&ib_ptr,D3DLOCK_NOSYSLOCK);// m_LockData.Flags);

	int md = cc->read_int();
	int stride = cc->read_char();

	cg::core::infoRecorder->logTrace("IndexBufferUnlock(), mode=%d, sizetolock=%d\n", md, m_LockData.SizeToLock);

	if(md == CACHE_MODE_COPY) {
		cg::core::infoRecorder->logTrace("index buffer, cache copy, md=%d, length=%d\n", md, length);
		

#if 0
		char* ptr = cc->get_cur_ptr();
		for(int i=0; i+stride * 3<=length; i+=stride * 3) {

			int a, b, c;
			if(stride) {
				a = *( (unsigned short*)(ptr) );
				b = *( (unsigned short*)(ptr + 2) );
				c = *( (unsigned short*)(ptr + 4) );
			}
			else {
				a = *( (unsigned int*)(ptr) );
				b = *( (unsigned int*)(ptr + 4) );
				c = *( (unsigned int*)(ptr + 8) );
			}
			ptr += 3 * stride;
			cg::core::infoRecorder->logTrace("a=%d b=%d c=%d\n", a, b, c);
		}
#endif
		
		//memcpy((char*)ib_ptr, cur_ptr, m_LockData.SizeToLock);
		cc->read_byte_arr((char*)ib_ptr, m_LockData.SizeToLock);

		m_ib->Unlock();
		return;
	}

	int last = 0;
	int d = 0, size = 0;
	cg::core::infoRecorder->logTrace("index buffer, cache diff.\n");

#if defined(USE_CHAR_COMPRESS)
	while(true) {
		d = cc->read_int();
		if(d == (1<<28)-1) break;
		last += d;
		*((char*)(ib_ptr) + last) = cc->read_char();
	}
	
#elif defined(USE_SHORT_COMPRESS)
	while(true) {
		d = cc->read_int();
		if(d == (1<<28)-1) break;
		last += d;
		*((SHORT *)(ib_ptr) + last) = cc->read_ushort();
	}

#elif defined(USE_INT_COMPRESS)
	while(true) {
		d = cc->read_int();
		if(d == (1<<28)-1) break;
		last += d;
		*((UINT*)(ib_ptr) + last) = cc->read_uint();
	}
#endif

	HRESULT hr = m_ib->Unlock();
}
