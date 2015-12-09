#ifndef __VIDEOCOMMON_H__
#define __VIDEOCOMMON_H__
// include all the enums and defines

#include <d3d9.h>
#include <d3d10_1.h>
#include <d3dx9.h>
#include <d3dx9tex.h>
#include <d3d11.h>
#include <cuda_d3d9_interop.h>

#include "../LibCore/DXDataTypes.h"
//#include "../LibCore/InfoRecorder.h"
#include "../LibVideo/Pipeline.h"

//#include <libavformat/avformat.h>
#include <libavutil/pixfmt.h>

#define RGBA_SIZE 4 /* in bytes*/

#define SOURCES 1
#define MAX_STRIDE 4
#define IMAGE_SOURCE_CHANNEL_MAX 4

#define POOLSIZE 8
#define BITSPERPIXEL 32
#define D3D_WINDOW_MODE true   // not in full screen mode


//extern InfoRecorder * infoRecorder;


// the source type for channel
enum SOURCE_TYPE{
    SOURCE_NONE,
    IMAGE,
    SURFACE,
};

// encoder type

enum ENCODER_TYPE{
	ENCODER_NONE = 0,
	X264_ENCODER,
	CUDA_ENCODER,
	NVENC_ENCODER,
	ADAPTIVE_NVENC
};
// the hardware supportion
enum SUPPORT_TYPE{
	SUPPORT_CUDA = 1,
	SUPPORT_NVENC,
	SUPPORT_X264,
};




struct VsourceConfig {
    int rtpId;	// RTP channel id
    int maxWidth;
    int maxHeight;
    int maxStride;
    // do not touch - filled by video_source_setup functions
    int id;		// image source id
};

class VFrame{
public:
	SOURCE_TYPE type; // the types of source, is image or surface
	VsourceConfig * conf;
    long long imgPts;		// presentation timestamp
    //PixelFormat pixelFormat;
    
    //enum vsource_format pixelformat; // rgba or yuv420p
};

class NoneFrame: public VFrame{

};

// the source is an image
class ImageFrame: public VFrame{
public:
    unsigned char *imgBuf;
    unsigned char *imgBufInternal;

    int lineSize[MAX_STRIDE];	// strides for YUV

	PixelFormat pixelFormat;

    int realWidth;
    int realHeight;
    int realStride;
    int realSize;
    // internal data - should not change after initialized
    int maxStride;
    int imgBufSize;
    int alignment;

    ImageFrame();
    ImageFrame(int w, int h, int s);
    ~ImageFrame();
    //ImageFrame(const ImageFrame &);
    //ImageFrame & operator=(const ImageFrame &);

    bool init(int w, int h, int s);
	unsigned char * getImgBuf(){ return imgBuf; }
	inline int getWidth(){ return realWidth; }
	inline int getHeight(){ return realHeight; }
	inline int getStride(){ return realStride; }
	inline int getImgBufSize(){ return imgBufSize; }

	static void DupFrame(ImageFrame * src, ImageFrame * dst);

    // virtual function from parent
    virtual bool init();
    virtual void release();
    virtual int setup(const char * pipeformat, struct VsourceConfig *conf, int id);
};


// the source is a D3D surface
class SurfaceFrame: public VFrame{
public:
	DX_VERSION dxVersion;
	DXSurface * dxSurface;

	int width, height;
	int pitch;

    SurfaceFrame();
    ~SurfaceFrame();
    //SurfaceFrame(const SurfaceFrame &);
    //SurfaceFrame & operator=(const SurfaceFrame &);

	IDirect3DSurface9 * getD3D9Surface(){ return dxSurface->d9Surface; }
	void setD3D9Surface(IDirect3DSurface9 * surface, int width, int height);
	ID3D10Texture2D * getD3D10Surface(){ return dxSurface->d10Surface; }
	ID3D11Texture2D * getD3D11Surface(){ return dxSurface->d11Surface; }

	//void setSurface(IDirect3DSurface9 * s, int width, int height);

    // virtual function
    virtual bool init();
    virtual void release();
    virtual int setup(const char * pipeformat, struct VsourceConfig *conf, int id);
};

struct ccgImage {
	int width;
	int height;
	int bytes_per_line;
};
struct ccgRect {
	int left, top;
	int right, bottom;
	int width, height;
	int linesize;
	int size;
};



#endif