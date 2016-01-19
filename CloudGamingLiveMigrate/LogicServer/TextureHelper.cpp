#include "TextureHelper.h"

int TextureHelper::BufferedTextureSize = 0;

TextureHelper::TextureHelper(int _pitch, int _height, bool _autoGen)
	:pitch(_pitch), height(_height), textureData(NULL), ptr(NULL), autoGenable(_autoGen), aquired(false){}


TextureHelper::~TextureHelper(){
	if(textureData){
		free(textureData);
		textureData = NULL;
	}
	ptr = NULL;
	pitch = 0;
	height = 0;
	autoGenable = false;
	aquired = false;
}

bool TextureHelper::allocateTextureBuffer(){
	textureData = (unsigned char *)malloc(sizeof(char) * height * pitch + 1024);
	ptr = textureData;
	return NULL == textureData;
}
bool TextureHelper::copyTextureData(unsigned char *dst){
	if(!dst || !textureData){
		return false;
	}
	int len = pitch * height;
	
	aquired = true;
	memcpy(dst, textureData, len);
	
	return true;
}

