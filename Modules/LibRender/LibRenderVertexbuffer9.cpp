#include <WinSock2.h>
#include "LibRenderVertexbuffer9.h"
#include <assert.h>
#include <stdlib.h>
#include "../LibCore/FloatHelper.h"
#include "../LibCore/InfoRecorder.h"

#define INT_COMPRESS
//#define FLOAT_COMPRESS
#define COMPRESS_TO_DWORD
// #define COMPRESS_TO_BYTE
using namespace cg;
using namespace cg::core;

float mx, my, mz;

ClientVertexBuffer9::ClientVertexBuffer9(IDirect3DVertexBuffer9* ptr, int _length): m_vb(ptr), length(_length), isFirst(true) {
	cg::core::infoRecorder->logTrace("clientVertexBuffer constructor entered, len:%d\n", _length);
#if 0
	this->vb_data = (char *)malloc(sizeof(char ) * _length);
	if(!this->vb_data){
		cg::core::infoRecorder->logTrace("clientVertexBuffer constructor malloc vb_data failed!\n");
	}

#endif
	vertex_count_ = 0;

	remain_vertices_.clear();
}

HRESULT ClientVertexBuffer9::Lock(UINT OffestToLock, UINT SizeToLock, DWORD Flags) {
	m_LockData.OffsetToLock = OffestToLock;
	m_LockData.SizeToLock = SizeToLock;
	m_LockData.Flags = Flags;
	if(SizeToLock == 0) m_LockData.SizeToLock = length - OffestToLock;

	return S_OK;
}

HRESULT ClientVertexBuffer9::Unlock(CommandClient * cc) {
	
	m_LockData.OffsetToLock = cc->read_uint();
	m_LockData.SizeToLock = cc->read_uint();
	m_LockData.Flags = cc->read_uint();
	

	cg::core::infoRecorder->logTrace("ClientVertexBuffer9::Unlock(), OffestToLock=%d, SizeToLock=%d Bytes, Flags=%d\n", m_LockData.OffsetToLock, m_LockData.SizeToLock, m_LockData.Flags);
	UpdateVertexBuffer(cc);

	return D3D_OK;
}

IDirect3DVertexBuffer9* ClientVertexBuffer9::GetVB() {
	return m_vb;
}

int ClientVertexBuffer9::GetLength() {
	return length;
}

float maxX = 0.0, maxY = 0.0, maxZ=  0.0;
void ClientVertexBuffer9::UpdateVertexBuffer(CommandClient * cc) {
	cg::core::infoRecorder->logTrace("ClientVertexBuffer9::UpdateVertexBuffer() called\n");

	char* vb_ptr;
	if(isFirst) {
		m_vb->Lock(0, 0, (void**)&vb_ptr, m_LockData.Flags);
		memset(vb_ptr, 0, length);
		m_vb->Unlock();
		isFirst = false;
	}

	int mode = cc->read_int();
	
	HRESULT hr = m_vb->Lock(m_LockData.OffsetToLock, m_LockData.SizeToLock, (void**)&vb_ptr, m_LockData.Flags);
	//HRESULT hr = m_vb->Lock(0, 0, (void**)&vb_ptr, D3DLOCK_NOOVERWRITE);//m_LockData.Flags);

	if(SUCCEEDED(hr)) {
		cg::core::infoRecorder->logTrace("lock success, ptr=%d, ", vb_ptr);
	}
	else {
		cg::core::infoRecorder->logTrace("lock failed, ");
	}

	if(mode == CACHE_MODE_COPY){
		cg::core::infoRecorder->logTrace("vertexbuffer,cache copy, update size:%d.\n", m_LockData.SizeToLock);
		cc->read_byte_arr(vb_ptr, m_LockData.SizeToLock);

		m_vb->Unlock();
		return;
	}
	cg::core::infoRecorder->logTrace("vertexbuffer, cache diff.\n");
	// diff
	int last = 0, d = 0, size = 0;
#if defined(USE_CHAR_COMPRESS)
	while(true){
		d = cc->read_int();
		if(d == (1 << 28) -1) break;
		last+=d;
		*((char*)(vb_ptr) + last) = cc->read_char();
	}
#elif defined(USE_SHORT_COMPRESS)
	while(true){
		d = cc->read_int();
		if(d == (1 << 28) -1) break;
		last+=d;
		*((unsigned short*)(vb_ptr) + last) = cc->read_ushort();
	}

#elif defined(USE_INT_COMPRESS)
	while(true){
		d = cc->read_int();
		if(d == (1 << 28) -1) break;
		last+=d;
		*((UINT*)(vb_ptr) + last) = cc->read_uint();
	}

#endif
	m_vb->Unlock();
}

// decode function when enable vertex compression
#if 0
void decode_position(char* pos, int offest, int size, CommandClient * cc) {
#ifndef USE_FLOAT_COMPRESS
	
	float x = cc->read_float();
	float y = cc->read_float();
	float z = cc->read_float();
	
	
	*( (float*)(pos + offest) ) = x;
	*( (float*)(pos + offest + 4) ) = y;
	*( (float*)(pos + offest + 8) ) = z;
	
	
#else

	DWORD val = cc->read_uint();
	decompress_pos_to_buffer(val, (float*)(pos + offest), mx, my, mz);
#endif

	//cg::core::infoRecorder->logError("position: x=%f, y=%f, z=%f\n", x, y, z);
}

void decode_normal(char* pos, int offest, int size, CommandClient * cc) {
#ifndef USE_FLOAT_COMPRESS
	float nx = cc->read_float();
	float ny = cc->read_float();
	float nz = cc->read_float();

	*( (float*)(pos + offest) ) = nx;
	*( (float*)(pos + offest + 4) ) = ny;
	*( (float*)(pos + offest + 8) ) = nz;
#else

	DWORD val = cc->read_uint();
	decompress_normal_to_buffer(val, (float*)(pos + offest));
#endif

	//cg::core::infoRecorder->logError("normal: nx=%f, ny=%f, nz=%f\n", nx, ny, nz);
}

void decode_tex(char* pos, int offest, int size, CommandClient * cc) {
	cc->read_vec((float*)(pos + offest), size);
}

void decode_tangent(char* pos, int offest, int size, CommandClient * cc) {
#ifndef USE_FLOAT_COMPRESS
	float tx = cc->read_float();
	float ty = cc->read_float();
	float tz = cc->read_float();
	float ta = cc->read_float();

	*( (float*)(pos + offest) ) = tx;
	*( (float*)(pos + offest + 4) ) = ty;
	*( (float*)(pos + offest + 8) ) = tz;
	*( (float*)(pos + offest + 12) ) = ta;
#else

	DWORD val = cc->read_uint();
	decompress_tangent_to_buffer(val, (float*)(pos + offest));
#endif
}

void decode_color(char* pos, int offest, int size, CommandClient * cc) {
	cc->read_byte_arr(pos + offest, size);
}

void decode_weight(char* pos, int offest, int size, CommandClient * cc) {

	float tx = cc->read_float();
	float ty = cc->read_float();
	float tz = cc->read_float();
	float ta = cc->read_float();

	*( (float*)(pos + offest) ) = tx;
	*( (float*)(pos + offest + 4) ) = ty;
	*( (float*)(pos + offest + 8) ) = tz;
	*( (float*)(pos + offest + 12) ) = ta;
}

void decode_indice(char* pos, int offest, int size, CommandClient * cc) {
	unsigned char tx = cc->read_uchar();
	unsigned char ty = cc->read_uchar();
	unsigned char tz = cc->read_uchar();
	unsigned char ta = cc->read_uchar();

	*( (unsigned char*)(pos + offest) ) = tx;
	*( (unsigned char*)(pos + offest + 1) ) = ty;
	*( (unsigned char*)(pos + offest + 2) ) = tz;
	*( (unsigned char*)(pos + offest + 3) ) = ta;
}

#endif