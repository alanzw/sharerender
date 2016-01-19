#pragma once
#include <d3d9.h>
#include "LibRenderCubetexture9.h"
#include "LibRenderTexture9.h"


#ifndef USE_TEXTURE_GENERATOR
#define USE_TEXTURE_GENERATOR
#endif

class AutoGenTextureGenerator{
	IDirect3DDevice9 *pDevice;  // the device for the channel

public:
	AutoGenTextureGenerator(IDirect3DDevice9 *_device);

	ClientTexture9 * generateTexture();
	ClientCubeTexture9 *generateCubeTexture();
	
};