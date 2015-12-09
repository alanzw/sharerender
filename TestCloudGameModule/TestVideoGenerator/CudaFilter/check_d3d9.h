#ifndef __CHECK_D3D9_H__
#define __CHECK_D3D9_H__

#include <d3d9.h>

#define CHECK_D3D_ERROR(hr) (_checkD3DError(hr, __FILE__, __LINE__))

void _checkD3DError(HRESULT hr, const char * file, const int line);

#endif