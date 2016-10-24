// BlockDim = 16 x 16
// GridDim = w / 16 * h /16

#include "..\..\LibCore\Glob.hpp"
#include "..\..\LibCore\GpuMat.hpp"

using namespace cg;
using namespace cg::core;
#define USE_MSDN


extern "C" __global__ void InterleaveUV(unsigned char * yuv_cb, unsigned char * yuv_cr, unsigned char * nv12_chroma, int chroma_width, int chroma_height, int cb_pitch, int cr_pitch, int nv12_pitch){
	int x = 0, y = 0;
	unsigned char * pCb;
	unsigned char * pCr;
	unsigned char * pDst;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	y = blockIdx.y * blockDim.y + threadIdx.y;

	if((x < chroma_width) && (y < chroma_height)){
		pCb = yuv_cb + (y * cb_pitch);
		pCr = yuv_cr + (y * cr_pitch);
		pDst = nv12_chroma + y * nv12_pitch;
		pDst[x << 1]		= pCb[x];
		pDst[(x << 1) + 1]	= pCr[x];
	}
}

__device__ __forceinline__ void rgb_to_y(const unsigned char b, const unsigned char g, const unsigned char r, unsigned char & y){
#ifndef USE_MSDN
	y = static_cast<unsigned char>(((int)(30 * r) + (int)(59 * g)+(int)(11 *b))/100);
#else
	y = static_cast<unsigned char>((((int)(66 * r) + (int)(129 * g) + (int)( 25 * b) + 128) >> 8) + 16);

#endif
}
__device__ __forceinline__ void rgb_to_yuv(const unsigned char b, const unsigned char g, const unsigned char r, unsigned char & y, unsigned char & u, unsigned char & v){
	rgb_to_y(b, g, r, y);
#ifndef USE_MSDN
	u = static_cast<unsigned char>(((int)(-17 * r) - (int)(33 * g) + (int)(50 * b) + 12800) / 100);
	v = static_cast<unsigned char>(((int)(50 * r) - (int)(42 * g) - (int)(8 * b) + 12800) / 100);
#else
	u = static_cast<unsigned char>((((int)(-38 * r) - (int)(74 * g) + (int)(112 * b) + 128)>>8)+128);
	v = static_cast<unsigned char>((((int)(112 * r) - (int)(94 * g) - (int)(19 * b) + 128)>>8)+ 128);

#endif
}


#if 0
template <typename T>
__global__ void _BGRAMatToNV12(const GlobPtrSz<T> src, GlobPtr<uchar> dst){
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

	if( x + 1 >= src.cols || y + 1 >= src.rows)
		return;

	const size_t planeSize = src.rows * dst.step;
	unsigned char * y_plane = dst.data;
	unsigned char * u_plane = y_plane + planeSize;
	int uvOff = 0;

	T pix;
	unsigned char y_val, u_val, v_val;
	pix = src(y, x);
	rgb_to_y(pix.x, pix.y, pix.z, y_val);
	y_plane[y * dst.step + x] = y_val;

	pix = src(y, x + 1);
	rgb_to_y(pix.x, pix.y, pix.z, y_val);
	y_plane[y * dst.step + x + 1] = y_val;
	pix = src(y + 1, x);
	rgb_to_y(pix.x, pix.y, pix.z, y_val);
	y_plane[(y + 1) * dst.step + x] = y_val;

	pix = src(y + 1, x + 1);
	rgb_to_yuv(pix.x, pix.y, pix.z, y_val, u_val, v_val);
	y_plane[(y + 1) * dst.step + x + 1] = y_val;
	uvOff = y / 2 * dst.step + x / 2 * 2;

	u_plane[uvOff] = u_val;
	u_plane[uvOff + 1] = v_val;

}

extern "C" __global__ void RGBAToNV12(const GpuMat & src, GpuMat & dst){
	switch(src.channels()){
	case 3:
		_BGRAMatToNV12(globPtr<uchar3>(src), globPtr<uchar>(dst));
		break;
	case 4:
		_BGRAMatToNV12(globPtr<uchar4>(src), globPtr<uchar>(dst));
		break;
	}
}

extern "C" __global__ void RGBToNV12(const GpuMat & src, GpuMat & dst){
	switch(src.channels()){
	case 3:
		_BGRAMatToNV12(globPtr<uchar3>(src), globPtr<uchar>(dst));
		break;
	case 4:
		_BGRAMatToNV12(globPtr<uchar4>(src), globPtr<uchar>(dst));
		break;

	default:
		break;
	}
}
#endif

extern "C" __global__ void RGBAToNV12_2(unsigned char * pARGB, unsigned char * pNV, int srcPitch, int dstPitch, int width, int height){
	
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	int rgbaSize = 4;
    if (x + 1 >= width || y + 1 >= height)
        return;

	const int planeSize = height * dstPitch;
	unsigned char * y_plane = pNV;
	unsigned char * u_plane = y_plane + planeSize;

	unsigned char y_val, u_val, v_val;
	unsigned char r, g, b;
	
	int uv_off = 0;
#if 0
	// the (x, y)
	r = pARGB[ y * srcPitch + x * rgbaSize + 0];
	g = pARGB[ y * srcPitch + x * rgbaSize + 1];
	b = pARGB[ y * srcPitch + x * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[y * dstPitch + x] = y_val;

	// the (x + 1, y)
	r = pARGB[ y * srcPitch + (x + 1) * rgbaSize + 0];
	g = pARGB[ y * srcPitch + (x + 1) * rgbaSize + 1];
	b = pARGB[ y * srcPitch + (x + 1) * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[y * dstPitch + x + 1] = y_val;

	// the (x , y + 1)
	r = pARGB[ (y+1) * srcPitch + x * rgbaSize + 0];
	g = pARGB[ (y+1) * srcPitch + x * rgbaSize + 1];
	b = pARGB[ (y+1) * srcPitch + x * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[ (y+1) * dstPitch + x] = y_val;

	// the (x +1, y + 1)
	r = pARGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 0];
	g = pARGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 1];
	b = pARGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 2];

#else
	// the (x, y)
	b = pARGB[ y * srcPitch + x * rgbaSize + 0];
	g = pARGB[ y * srcPitch + x * rgbaSize + 1];
	r = pARGB[ y * srcPitch + x * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[y * dstPitch + x] = y_val;

	// the (x + 1, yb
	b = pARGB[ y * srcPitch + (x + 1) * rgbaSize + 0];
	g = pARGB[ y * srcPitch + (x + 1) * rgbaSize + 1];
	r = pARGB[ y * srcPitch + (x + 1) * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[y * dstPitch + x + 1] = y_val;

	// the (x , y + 1)
	b = pARGB[ (y+1) * srcPitch + x * rgbaSize + 0];
	g = pARGB[ (y+1) * srcPitch + x * rgbaSize + 1];
	r = pARGB[ (y+1) * srcPitch + x * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[ (y+1) * dstPitch + x] = y_val;

	// the (x +1, y + 1)
	b = pARGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 0];
	g = pARGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 1];
	r = pARGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 2];


#endif
	rgb_to_yuv(b, g, r, y_val, u_val, v_val);
	y_plane[ (y + 1) * dstPitch + x + 1] = y_val;
	uv_off = (y / 2) * dstPitch + x / 2 * 2;
	u_plane[ uv_off ] = u_val;
	u_plane[ uv_off + 1] = v_val;
}

extern "C" __global__ void RGBToNV12_2(unsigned char * pRGB, unsigned char * pNV, int srcPitch, int dstPitch, int width, int height){
	
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
	int rgbaSize = 3;

    if (x + 1 >= width * rgbaSize || y + 1 >= height)
        return;

	const int planeSize = height * dstPitch;
	unsigned char * y_plane = pNV;
	unsigned char * u_plane = y_plane + planeSize;

	unsigned char y_val, u_val, v_val;
	unsigned char r, g, b;
	
	int uv_off = 0;

	// the (x, y)
	r = pRGB[ y * srcPitch + x * rgbaSize + 0];
	g = pRGB[ y * srcPitch + x * rgbaSize + 1];
	b = pRGB[ y * srcPitch + x * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[y * dstPitch + x] = y_val;

	// the (x + 1, y)
	r = pRGB[ y * srcPitch + (x + 1) * rgbaSize + 0];
	g = pRGB[ y * srcPitch + (x + 1) * rgbaSize + 1];
	b = pRGB[ y * srcPitch + (x + 1) * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[y * dstPitch + x + 1] = y_val;

	// the (x , y + 1)
	r = pRGB[ (y+1) * srcPitch + x * rgbaSize + 0];
	g = pRGB[ (y+1) * srcPitch + x * rgbaSize + 1];
	b = pRGB[ (y+1) * srcPitch + x * rgbaSize + 2];

	rgb_to_y(b, g, r, y_val);
	y_plane[ (y+1) * dstPitch + x] = y_val;

	// the (x +1, y + 1)
	r = pRGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 0];
	g = pRGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 1];
	b = pRGB[ (y+1) * srcPitch + (x+1) * rgbaSize + 2];

	rgb_to_yuv(b, g, r, y_val, u_val, v_val);
	y_plane[ (y + 1) * dstPitch + x + 1] = y_val;
	uv_off = (y / 2) * dstPitch + x / 2 * 2;
	u_plane[ uv_off ] = u_val;
	u_plane[ uv_off + 1] = v_val;
}
