#include "videocommon.h"
#include "MemOp.h"
#include "inforecoder.h"
//#include "inforecoder.h:"


int align_malloc(int size, void **ptr, int *alignment){
	if((*ptr = malloc(size + 16)) == NULL)
		return -1;
#ifdef __X86_64__
	*alignment = 16 - (((long long)*ptr)&0x0f);
#else
	*alignment = 16 - (((unsigned) * ptr) & 0x0f);
#endif
	return 0;
}

ImageFrame::ImageFrame(){ this->type = IMAGE; }

ImageFrame::ImageFrame(int w, int h, int s){
	this->type = IMAGE;
}

ImageFrame::~ImageFrame(){}



bool ImageFrame::init(){return true; }

bool ImageFrame::init(int maxW, int maxH, int maxS){
	int i ;
	for(i = 0; i< MAX_STRIDE; i++){
		lineSize[i] = maxS;
	}
	this->maxStride = maxS;
	imgBufSize = maxH * maxS;
	if(align_malloc(imgBufSize, (void **)&imgBufInternal, &alignment) < 0){
		// error
		return false;
	}
	imgBuf = imgBufInternal + alignment;
	bzero(imgBuf, imgBufSize);
	return true;
}
void ImageFrame::release(){
	if(imgBuf != NULL){
		free(imgBuf);
		imgBuf = NULL;
	}
	return;
}

int ImageFrame::setup(const char * pipeformat, struct VsourceConfig * conf, int id){
	return true;
}

void ImageFrame::DupFrame(ImageFrame * src, ImageFrame * dst){
	int j = 0;
	dst->imgPts = src->imgPts;
	
	for(j = 0; j< MAX_STRIDE; j++){
		dst->lineSize[j] = src->lineSize[j];
	}
	dst->realHeight = src->realHeight;
	dst->realWidth = src->realWidth;
	dst->realStride = src->realStride;
	dst->realSize = src->realSize;
	bcopy(src->imgBuf, dst->imgBuf, dst->imgBufSize);
	
}

//////// surfaceFrame //////////
SurfaceFrame::SurfaceFrame(){ this->type = SURFACE; }

SurfaceFrame::~SurfaceFrame(){}

bool SurfaceFrame::init(){
	dxSurface = NULL;
	
	width = 0;
	height = 0;
	pitch = 0;
	return true;
}

void SurfaceFrame::release(){

}

int SurfaceFrame::setup(const char * pipeformat, struct VsourceConfig * conf, int id){
	return 0;
}



