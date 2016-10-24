#ifndef __ENCODERTHREAD_H__
#define __ENCODERTHREAD_H__
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <d3d9.h>

// do the include work
void onDeviceCreation(IDirect3DDevice9 * device, int height, int width, HWND hwnd);
void onPresent();

#endif