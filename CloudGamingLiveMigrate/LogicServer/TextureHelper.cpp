#include "TextureHelper.h"
#include "..\LibCore\InfoRecorder.h"
int SurfaceHelper::TotalBufferedTextureSize = 0;

TextureHelper::TextureHelper(short _levels, D3DFORMAT format, bool _autogenable /* = false */)
	: autoGenable(_autogenable), aquired(false), levels(_levels), bufferSize(0), compressed(false){
		surfaceArray = new SurfaceHelper *[levels];
		for(int i = 0; i < levels; i++){
			surfaceArray[i] = NULL;
		}
		validLevels = autoGenable ? 1 : levels;
		if(D3DFMT_DXT1 == format || D3DFMT_DXT2 == format ||
			D3DFMT_DXT3 == format || D3DFMT_DXT4 == format ||
			D3DFMT_DXT5 == format){
				compressed = true;
		}
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
		surfaceArray[level] = new SurfaceHelper(level, compressed);
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
	cg::core::infoRecorder->logError("[DeviceHelper]: support texture auto gen:%s, support cube texture auto gen:%s.\n", supportAutoGenTex ? "true" : "false", supportAuoGenCubeTex? "true" : "false");
	return true;
}



//////// for Surface Helper ///////
SurfaceHelper::SurfaceHelper(short _level, bool _compressed):
	surfaceData(NULL), ptr(NULL), pitch(0), height(0), face(-1), level(_level),
	aquired(false), type(TEXTURE), realSurfacePtr(NULL), compressed(_compressed)
{

}
SurfaceHelper::SurfaceHelper(short _level, short _face, bool _compressed):
	surfaceData(NULL), ptr(NULL), pitch(0), height(0), face(_face), level(_level),
	aquired(false), type(CUBE_TEXTURE), realSurfacePtr(NULL), compressed(_compressed)
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
	realSurfacePtr = NULL;
}

unsigned char * SurfaceHelper::allocateSurfaceBuffer(int _pitch, int _height){
	
	pitch = _pitch;
	height = _height;
	int divide = compressed ? 4 : 1;
	int compressedHeight = (height + divide -1) /divide;
	if(0 != pitch * height){
		
		surfaceData = (unsigned char *)malloc(sizeof(char) *(pitch * compressedHeight+ 100));
		if(surfaceData){
			aquired = true;
			TotalBufferedTextureSize += pitch * compressedHeight;
		}
	}
	//cg::core::infoRecorder->logError("[SurfaceHelper]: allocate memory %d, ptr:%p.\n", pitch * compressedHeight, surfaceData);
	return surfaceData;
}
bool SurfaceHelper::copyTextureData(){
	bool ret = true;
	if(!realSurfacePtr || !surfaceData){
		return false;
	}
	D3DLOCK_DISCARD;
	int devide = compressed ? 4 : 1;
	int compressedHeight = (height + devide -1)/devide;

	int copySize = pitch * compressedHeight;
	cg::core::infoRecorder->logError("[SurfaceHelper]: memcpy, src:%p, dst: %p, size:%d ( pitch:%d x real height:%d).\n", surfaceData, realSurfacePtr, copySize, pitch, compressedHeight);
	memcpy(realSurfacePtr, surfaceData, copySize);
#if 0
	for(int j = 0; j < height; j++){
		memcpy((char *)realSurfacePtr + j * pitch, surfaceData + j * pitch, pitch);
	}
#endif
	realSurfacePtr = NULL;  // reset the surface buffer pointer, because when unlock, the pointer will be invalid.

	return true;
}

int SurfaceHelper::getPitchedSize(){
	int divide = compressed ? 4 : 1;
	int compressedHeight = (height + divide -1)/divide;
	return pitch * compressedHeight;
}