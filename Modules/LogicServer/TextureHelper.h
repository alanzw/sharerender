#pragma once

// this is for Texture, CubeTexture and VolumeTexture, get and store the top level surface data and auto generate the sub mipmap in render proxy
#include <d3d9.h>
#include <d3dx9.h>

#ifndef USE_TEXTURE_HELPER
#define USE_TEXTURE_HELPER
#endif

enum TEXTURE_TYPE{
	NO_TEXTURE_TYPE = 0,
	TEXTURE,
	CUBE_TEXTURE,
	VOLUME_TEXTURE
};


// represent a surface for texture
class SurfaceHelper{
private:
	unsigned char *		surfaceData;
	unsigned char *		ptr;
	int					pitch;
	int					height;
	short				face;
	short				level;				// in case the surface belongs to a cube texture
	bool				aquired;
	short				compression;
	TEXTURE_TYPE		type;
	void *				realSurfacePtr;		// the real ptr for surface content
	static int			TotalBufferedTextureSize;
	DWORD				lockFlags;
public:
	unsigned int		updateFlag;


	inline unsigned char *getSurfaceData(){ return surfaceData; }
	inline int			getPitch(){ return pitch; }
	inline int			getHeight(){ return height; }
	//inline int			getPitchedSize(){ return pitch * height; }
	inline bool			isAquired(){ return aquired; }
	inline void			setLockFlags(DWORD flags){lockFlags = flags;}
	inline DWORD		getLockFlags(){ return lockFlags; }

	SurfaceHelper(short level, short _compression);
	SurfaceHelper(short level, short face, short _compression);
	virtual ~SurfaceHelper();

	static int			GetBufferTextureSize(){ return TotalBufferedTextureSize; }
	inline void			setRealSurfacePointer(void * _ptr){ realSurfacePtr = _ptr; }  // must be called before changing the locked rect's pbits
	unsigned char *		allocateSurfaceBuffer(int pitch, int height);  // allocate the memory and set aquired
	bool				copyTextureData();   // copy buffered surface data to video memory and set the video pointer to NULL
	int					getPitchedSize();
};
// for cube texture
class CubeTextureHelper{
private:
	bool				autoGenable;
	bool				aquired;
	int					bufferSize;
	short				levels;

public:
	inline bool			isAutoGenable(){ return autoGenable;}
	inline bool			isAquired(int face, int level);
};
class TextureHelper{
private:
	bool				autoGenable;		// whether the texture can use auto gen
	bool				aquired;			// indicate the top level mipmap is got or not
	int					bufferSize;
	short				levels;				// the levels of the texture
	SurfaceHelper **	surfaceArray;
	short				validLevels;		// how many levels is in use
	short				compression;			// DXT1-DXT5

public:
	inline short		getCompression(){ return compression; }
	inline bool			isAutoGenable(){ return autoGenable; }
	inline bool			isAquired(int i){ return surfaceArray[i] && surfaceArray[i]->isAquired(); }

	TextureHelper(short _levels, D3DFORMAT format, bool _autogenable = false);
	virtual ~TextureHelper();
	
	SurfaceHelper *		getSurfaceHelper(short level);  // return the surface helper for given level, if NULL, create one
	int					getBufferSize();
};

class DeviceHelper{
	bool				supportAutoGenTex;
	bool				supportAuoGenCubeTex;
public:

	bool				isSupportAutoGenTex(){ return supportAutoGenTex; }
	bool				isSupportAutoGenCubeTex(){ return supportAuoGenCubeTex; }
	bool				checkSupportForAutoGenMipmap(IDirect3DDevice9 *device);
};
