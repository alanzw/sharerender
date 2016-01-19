#pragma once

// this is for Texture, CubeTexture and VolumeTexture, get and store the top level surface data and auto generate the sub mipmap in render proxy
#include <d3d9.h>
#include <d3dx9.h>

#ifndef USE_TEXTURE_HELPER
#define USE_TEXTURE_HELPER
#endif


class TextureHelper{
private:
	unsigned char *textureData; // stores the top level surface data in raw format
	unsigned char *ptr;			// tmp ptr for solving texture in huge size
	int pitch, height;
	bool autoGenable;			// whether the texture can use auto gen

	bool aquired;				// indicate the top level mipmap is got or not
	D3DLOCKED_RECT rect;
	static int BufferedTextureSize;
public:
	inline unsigned char * getTextureData(){ return textureData; }
	inline bool isAutoGenable(){ return autoGenable; }
	inline int getPitch(){ return pitch; }
	inline int getHeight(){ return height; }
	inline int getPitchedSize(){ return pitch * height; }
	inline bool isAquired(){ return aquired; }
	inline void setAutoGenable(bool val){ autoGenable = val; }
	inline D3DLOCKED_RECT* getRectAddr(){ return &rect; }
	static int GetBufferTextureSize(){ return BufferedTextureSize; }

	TextureHelper(int _pitch, int _height, bool _autoGen = true);
	virtual ~TextureHelper();

	bool allocateTextureBuffer();
	bool copyTextureData(unsigned char *dst);
};
