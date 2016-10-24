#ifndef __CLIENT_STATEBLOCK9__
#define __CLIENT_STATEBLOCK9__

#include <d3d9.h>


class ClientStateBlock9 {
private:
	IDirect3DStateBlock9* m_sb;
public:
	ClientStateBlock9(IDirect3DStateBlock9* ptr);
	HRESULT Capture();
	HRESULT Apply();


	bool isValid(){ return m_sb != NULL ? true : false; }
};

#endif