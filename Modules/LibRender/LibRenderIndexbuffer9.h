#ifndef __CLIENT_INDEXBUFFER9__
#define __CLIENT_INDEXBUFFER9__

#include <d3d9.h>
#include "../LibCore/CommandClient.h"
class ClientIndexBuffer9 {
private:
	IDirect3DIndexBuffer9* m_ib;
	
	int length;
	bool isFirst;
public:
	BufferLockData m_LockData;
	void UpdateIndexBuffer(cg::core::CommandClient * cc);
	ClientIndexBuffer9(IDirect3DIndexBuffer9* ptr, int _length);
	HRESULT Lock(UINT OffestToLock, UINT SizeToLock, DWORD Flags);
	HRESULT Unlock(cg::core::CommandClient * cc);

	IDirect3DIndexBuffer9* GetIB();
	int GetLength();
};

#endif