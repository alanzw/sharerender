#include "TextureHelper.h"
#include "..\LibCore\InfoRecorder.h"
int SurfaceHelper::TotalBufferedTextureSize = 0;

TextureHelper::TextureHelper(short _levels, bool _autogenable /* = false */)
	: autoGenable(_autogenable), aquired(false), levels(_levels), bufferSize(0){
		surfaceArray = new SurfaceHelper *[levels];
		for(int i = 0; i < levels; i++){
			surfaceArray[i] = NULL;
		}
		validLevels = autoGenable ? levels : 1;
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

SurfaceHelper* TextureHelper::getSurfaceHelper(short level){
	if(level >= validLevels){
		// may error
		return NULL;
	}
	if(!surfaceArray[level]){
		// not created, create one
		surfaceArray[level] = new SurfaceHelper(level);
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
SurfaceHelper::SurfaceHelper(short _level):
	surfaceData(NULL), ptr(NULL), pitch(0), height(0), face(-1), level(_level),
	aquired(false), type(TEXTURE), realSurfacePtr(NULL)
{

}
SurfaceHelper::SurfaceHelper(short _level, short _face):
	surfaceData(NULL), ptr(NULL), pitch(0), height(0), face(_face), level(_level),
	aquired(false), type(CUBE_TEXTURE), realSurfacePtr(NULL)
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
	if(0 != pitch * height){
		
		surfaceData = (unsigned char *)malloc(sizeof(char) *(pitch * height + 100));
		if(surfaceData){
			aquired = true;
			TotalBufferedTextureSize += pitch * height;
		}
	}
	cg::core::infoRecorder->logError("[SurfaceHelper]: allocate memory %d, ptr:%p.\n", pitch * height, surfaceData);
	return surfaceData;
}
bool SurfaceHelper::copyTextureData(){
	bool ret = true;
	if(!realSurfacePtr || !surfaceData){
		return false;
	}
	cg::core::infoRecorder->logError("[SurfaceHelper]: memcpy, src:%p, dst: %p, size:%d.\n", surfaceData, realSurfacePtr, pitch * height);
	memcpy(realSurfacePtr, surfaceData, pitch * height);
	realSurfacePtr = NULL;  // reset the surface buffer pointer, because when unlock, the poitner will be invalid.

	return true;
}