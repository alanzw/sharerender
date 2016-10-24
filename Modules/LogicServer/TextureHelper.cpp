#include "TextureHelper.h"
#include "..\LibCore\InfoRecorder.h"

//#define ENABLE_LOG_TEXTURE_HELPER


int SurfaceHelper::TotalBufferedTextureSize = 0;

TextureHelper::TextureHelper(short _levels, D3DFORMAT format, bool _autogenable /* = false */)
	: autoGenable(_autogenable), aquired(false), levels(_levels), bufferSize(0), compression(1){
		surfaceArray = new SurfaceHelper *[levels];
		for(int i = 0; i < levels; i++){
			surfaceArray[i] = NULL;
		}
		validLevels = autoGenable ? 1 : levels;
#if 0

		if(D3DFMT_DXT1 == format || D3DFMT_DXT2 == format ||
			D3DFMT_DXT3 == format || D3DFMT_DXT4 == format ||
			D3DFMT_DXT5 == format){
				compression = 4;
		}
#endif
		char formatStr[100] = {0};
		switch(format){
		case D3DFMT_DXT1:
			strcpy(formatStr, "DXT1");
			compression = 4;
			break;
		case D3DFMT_DXT2:
			strcpy(formatStr, "DXT2");
			compression = 4;
			break;
		case D3DFMT_DXT3:
			strcpy(formatStr, "DXT3");
			compression = 4;
			break;
		case D3DFMT_DXT4:
			strcpy(formatStr, "DXT4");
			compression = 4;
			break;
		case D3DFMT_DXT5:
			compression = 4;
			strcpy(formatStr, "DXT5");
			break;
		}
#ifdef ENABLE_LOG_TEXTURE_HELPER
		cg::core::infoRecorder->logTrace("[TextureHelper]: auto gen:%s, compression:%d, format:%s.\n", autoGenable ? "true": "false", compression, formatStr);
#endif
}


TextureHelper::~TextureHelper(){
	if(surfaceArray){
		// release the surface helper
		for(int i = 0; i < validLevels; i++){
			if(surfaceArray[i]){
				delete surfaceArray[i];
				surfaceArray[i] = NULL;
			}
		}

		delete [] surfaceArray;
		surfaceArray = NULL;
	}
	autoGenable = false;
	validLevels = 0;
	levels = 0;
}
int TextureHelper::getBufferSize(){
	int ret = 0;
	for(int i =0; i< levels; i++){
		if(surfaceArray[i]){
			ret += (surfaceArray[i]->getPitchedSize() + sizeof(SurfaceHelper));
		}
	}
	bufferSize = ret;
	//cg::core::infoRecorder->logError("[TextureHelper]: buffer size:%d.\n", ret);
	return ret;
}

SurfaceHelper* TextureHelper::getSurfaceHelper(short level){
	if(level >= validLevels){
		// may error
		return NULL;
	}
	if(!surfaceArray[level]){
		// not created, create one
		surfaceArray[level] = new SurfaceHelper(level, compression);
	}
	return surfaceArray[level];
}


bool DeviceHelper::checkSupportForAutoGenMipmap(IDirect3DDevice9 *device){
	HRESULT hr = D3D_OK;
	D3DCAPS9 caps;
	IDirect3D9 *d3d9 = NULL;
	D3DDISPLAYMODE mode;
	D3DDEVICE_CREATION_PARAMETERS  params;

	if(!device){
		return false;
	}
	hr = device->GetDeviceCaps(&caps);
	hr = device->GetDirect3D(&d3d9);
	hr = device->GetCreationParameters(&params);
	hr = device->GetDisplayMode(0, &mode);

	if(!(caps.Caps2 & D3DCAPS2_CANAUTOGENMIPMAP)){
		supportAuoGenCubeTex = false;
		supportAutoGenTex = false;
		return true;
	}
	hr = d3d9->CheckDeviceFormat(params.AdapterOrdinal, params.DeviceType, mode.Format, D3DUSAGE_AUTOGENMIPMAP, D3DRTYPE_TEXTURE, D3DFMT_A8R8G8B8);

	supportAutoGenTex = (SUCCEEDED(hr));
	hr = d3d9->CheckDeviceFormat(params.AdapterOrdinal, params.DeviceType, mode.Format, D3DUSAGE_AUTOGENMIPMAP, D3DRTYPE_CUBETEXTURE, D3DFMT_A8R8G8B8); 
	supportAuoGenCubeTex = (SUCCEEDED(hr));

	d3d9->Release();
	#ifdef ENABLE_LOG_TEXTURE_HELPER
	cg::core::infoRecorder->logTrace("[DeviceHelper]: support texture auto gen:%s, support cube texture auto gen:%s.\n", supportAutoGenTex ? "true" : "false", supportAuoGenCubeTex? "true" : "false");
#endif
	return true;
}



//////// for Surface Helper ///////
SurfaceHelper::SurfaceHelper(short _level, short _compression):
	surfaceData(NULL), ptr(NULL), pitch(0), height(0), face(-1), level(_level),
	aquired(false), type(TEXTURE), realSurfacePtr(NULL), compression(_compression), updateFlag(0)
{

}
SurfaceHelper::SurfaceHelper(short _level, short _face, short _compression):
	surfaceData(NULL), ptr(NULL), pitch(0), height(0), face(_face), level(_level),
	aquired(false), type(CUBE_TEXTURE), realSurfacePtr(NULL), compression(_compression), updateFlag(0)
{

}

SurfaceHelper::~SurfaceHelper(){
	if(surfaceData){
		free(surfaceData);
		surfaceData = NULL;
	}
	ptr = NULL;
	pitch = 0;
	height = 0;
	face = -1;
	level = -1;
	aquired = false;
	compression = 1;
	realSurfacePtr = NULL;
}

unsigned char * SurfaceHelper::allocateSurfaceBuffer(int _pitch, int _height){
	
	pitch = _pitch;
	height = _height;
	//int divide = compressed ? 4 : 1;
	int compressedHeight = (height + compression -1) /compression;
	if(0 != pitch * height){
		
		surfaceData = (unsigned char *)malloc(sizeof(char) *(pitch * compressedHeight+ 100));
		if(surfaceData){
			aquired = true;
			TotalBufferedTextureSize += pitch * compressedHeight;
		}
	}
	#ifdef ENABLE_LOG_TEXTURE_HELPER
	cg::core::infoRecorder->logError("[SurfaceHelper]: allocate memory %d(%d x %d), ptr:%p.\n", pitch * compressedHeight, pitch, height, surfaceData);
#endif
	return surfaceData;
}
bool SurfaceHelper::copyTextureData(){
	bool ret = true;
	if(!realSurfacePtr || !surfaceData){
		return false;
	}
	D3DLOCK_DISCARD;
	//int devide = compressed ? 4 : 1;
	int compressedHeight = (height + compression -1)/compression;

	int copySize = pitch * compressedHeight;
#ifdef ENABLE_LOG_TEXTURE_HELPER
	cg::core::infoRecorder->logError("[SurfaceHelper]: memcpy, src:%p, dst: %p, size:%d ( pitch:%d x real height:%d).\n", surfaceData, realSurfacePtr, copySize, pitch, compressedHeight);
#endif
	memcpy(realSurfacePtr, surfaceData, copySize);
#if 0
	for(int j = 0; j < height; j++){
		memcpy((char *)realSurfacePtr + j * pitch, surfaceData + j * pitch, pitch);
	}
#endif
	realSurfacePtr = NULL;  // reset the surface buffer pointer, because when unlock, the pointer will be invalid.
	updateFlag = 0x8fffffff;
	return true;
}

int SurfaceHelper::getPitchedSize(){
	//int divide = compressed ? 4 : 1;
	int compressedHeight = (height + compression -1)/compression;
	return pitch * compressedHeight;
}