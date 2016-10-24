#ifndef __ENCODERTHREAD_H__
#define __ENCODERTHREAD_H__

#include <d3d9.h>

// do the include work
void onDeviceCreation(IDirect3DDevice9 * device, int height, int width, HWND hwnd);
void onPresent();

#endif