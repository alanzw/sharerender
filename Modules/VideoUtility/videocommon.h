#ifndef __VIDEOCOMMON_H__
#define __VIDEOCOMMON_H__
// include all the enums and defines

#include <d3d9.h>
#include <d3d10_1.h>
#include <d3dx9.h>
#include <d3dx9tex.h>
#include <d3d11.h>
#include <cuda.h>
//#include <cuda_d3d9_interop.h>
#include "../LibCore/DXDataTypes.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

//#include "../VideoGen/rtspcontext.h"

#include <WinSock2.h>
#ifdef __cplusplus
extern "C"{
#endif
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#ifdef __cplusplus
}
#endif
#define RGBA_SIZE					4 /* in bytes*/

#define SOURCES						1
#define MAX_STRIDE					4
//#define IMAGE_SOURCE_CHANNEL_MAX	4

#define POOLSIZE					8
#define BITSPERPIXEL				32
#define D3D_WINDOW_MODE				true   // not in full screen mode

namespace cg{
	// the source type for channel
	enum SOURCE_TYPE{
		SOURCE_NONE,
		IMAGE,
		SURFACE,
	};

	// encoder type
	enum ENCODER_TYPE{
		ENCODER_NONE = -1,
		X264_ENCODER = 1,
		CUDA_ENCODER,
		NVENC_ENCODER,
		ADAPTIVE_CUDA,   // cuda and x264
		ADAPTIVE_NVENC   // nvenc and x264
	};
	// the hardware support
	enum SUPPORT_TYPE{
		SUPPORT_CUDA = 1,
		SUPPORT_NVENC,
		SUPPORT_X264,
	};

	struct VsourceConfig {
		int						rtpId;	// RTP channel id
		int						maxWidth;
		int						maxHeight;
		int						maxStride;
		// do not touch - filled by video_source_setup functions
		int						id;		// image source id
	};

	class Frame{
	public:
		SOURCE_TYPE				type; // the types of source, is image or surface
		long long				imgPts;		// presentation timestamp
	};

	// the source is an image
	class ImageFrame: virtual public Frame{
	public:
		unsigned char *			imgBufInternal;

		int						realWidth;
		int						realHeight;
		int						realStride;
		int						realSize;
		// internal data - should not change after initialized
		int						maxStride;
		int						imgBufSize;
		int						alignment;
	public:

		PixelFormat				pixelFormat;
		unsigned char *			imgBuf;
		int						lineSize[MAX_STRIDE];	// strides for YUV
		ImageFrame();

		virtual ~ImageFrame();
		//ImageFrame(const ImageFrame &);
		//ImageFrame & operator=(const ImageFrame &);

		bool					init(int w, int h, int s);
		unsigned char *			getImgBuf(){ return imgBuf; }
		inline int				getWidth(){ return realWidth; }
		inline int				getHeight(){ return realHeight; }
		inline int				getStride(){ return realStride; }
		inline int				getImgBufSize(){ return imgBufSize; }

		static void				DupFrame(ImageFrame * src, ImageFrame * dst);
		void					print();
	};


	// the source is a D3D surface
	class SurfaceFrame: public virtual Frame{
	public:
		cg::core::DX_VERSION	dxVersion;
		cg::core::DXSurface *	dxSurface;

		int						width, height;
		int						pitch;
		bool					registered;    // whether register to CUDA

	public:
		CUgraphicsResource		cuResource;   // the cuGraphic resource for the 

		SurfaceFrame();
		virtual ~SurfaceFrame();

		IDirect3DSurface9 *		getD3D9Surface(){ return dxSurface->d9Surface; }
		ID3D10Texture2D *		getD3D10Surface(){ return dxSurface->d10Surface; }
		ID3D11Texture2D *		getD3D11Surface(){ return dxSurface->d11Surface; }

		bool					registerToCUDA();
		bool					unregisterToCUDA();
		void					print();
	};

	class SourceFrame: public ImageFrame, public SurfaceFrame{
	public:
		void					print();
	};

	struct ccgImage {
		int						width;
		int						height;
		int						bytes_per_line;
	};

	struct ccgRect {
		int						left, top;
		int						right, bottom;
		int						width, height;
		int						linesize;
		int						size;
	};

	
	int							align_malloc(int size, void **ptr, int *alignment);
	int							ve_malloc(int size, void **ptr, int * alignment);
	long						ccg_gettid();
	struct ccgRect *			ccg_fillrect(struct ccgRect *rect, int left, int top, int right, int bottom);

}
#endif