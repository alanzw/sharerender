#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
//#include "..\..\CloudGamingLiveMigrate\Core

#include "..\..\CloudGamingLiveMigrate\LibCore\glob.hpp"
#include "..\..\CloudGamingLiveMigrate\LibCore\GpuMat.hpp"
#include <vector_types.h>

using namespace cudev;

#define USE_MSDN

//void RGB_to_YV12(const GpuMat & src, GpuMat& dst);
namespace{
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

	__global__ void Gray_to_YV12(const GlobPtrSz<unsigned char> src, GlobPtr<unsigned char > dst){
		const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
		const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

		if(x + 1 >= src.cols || y + 1 >= src.rows)
			return;

		// get pointers to the data
		const size_t planeSize = src.rows * dst.step;

		GlobPtr<unsigned char> y_plane = globPtr(dst.data, dst.step);
		GlobPtr<unsigned char> u_plane = globPtr(y_plane.data + planeSize, dst.step / 2);
		GlobPtr<unsigned char> v_plane = globPtr(u_plane.data + (planeSize / 4), dst.step / 2);

		unsigned char pix;
		unsigned char y_val, u_val, v_val;

		pix = src(y, x);
        rgb_to_y(pix, pix, pix, y_val);
        y_plane(y, x) = y_val;

        pix = src(y, x + 1);
        rgb_to_y(pix, pix, pix, y_val);
        y_plane(y, x + 1) = y_val;

        pix = src(y + 1, x);
        rgb_to_y(pix, pix, pix, y_val);
        y_plane(y + 1, x) = y_val;

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix, pix, pix, y_val, u_val, v_val);
        y_plane(y + 1, x + 1) = y_val;
        u_plane(y / 2, x / 2) = u_val;
        v_plane(y / 2, x / 2) = v_val;
	}
	
#if 1	
	template <typename T>
    __global__ void RGB_to_YV12(const GlobPtrSz<T> src, GlobPtr<uchar> dst)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

        if (x + 1 >= src.cols || y + 1 >= src.rows)
            return;

        // get pointers to the data
        const size_t planeSize = src.rows * dst.step;
#ifdef USE_UV_PLANE
        GlobPtr<unsigned char> y_plane = globPtr(dst.data, dst.step);
		// the u,v plane is not right, because there is padding

        GlobPtr<unsigned char> u_plane = globPtr(y_plane.data + planeSize, dst.step / 2);
        GlobPtr<unsigned char> v_plane = globPtr(u_plane.data + (planeSize / 4), dst.step / 2);
#else
		unsigned char *y_plane = dst.data;
		unsigned char *u_plane = y_plane + planeSize;
		unsigned char *v_plane = u_plane + (planeSize>>2);
		int uvOff = 0;
#endif
		// not right here, cause the cuda pitch is not equal the surface pitch

        T pix;
        unsigned char y_val, u_val, v_val;
#ifdef BGR
        pix = src(y, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x) = y_val;

        pix = src(y, x + 1);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x + 1) = y_val;

        pix = src(y + 1, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y + 1, x) = y_val;

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix.z, pix.y, pix.x, y_val, u_val, v_val);
#else
		pix = src(y, x);
        rgb_to_y(pix.x, pix.y, pix.z, y_val);
#ifdef USE_UV_PLANE
        y_plane(y, x) = y_val;
#else
		y_plane[ y * dst.step + x ] = y_val;
#endif

        pix = src(y, x + 1);
        rgb_to_y(pix.x, pix.y, pix.z, y_val);
#ifdef USE_UV_PLANE] = y_val;
        y_plane(y, x + 1) = y_val;
#else
		y_plane[ y * dst.step + x + 1] = y_val;
#endif

        pix = src(y + 1, x);
        rgb_to_y(pix.x, pix.y, pix.z, y_val);
#ifdef USE_UV_PLANE
        y_plane(y + 1, x) = y_val;
#else
		y_plane[ (y + 1) * dst.step + x ]= y_val;
#endif

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix.x, pix.y, pix.z, y_val, u_val, v_val);
#endif

#ifdef USE_UV_PLANE
        y_plane(y + 1, x + 1) = y_val;
        u_plane(y / 2, x / 2) = u_val;
        v_plane(y / 2, x / 2) = v_val;
#else
		y_plane[( y + 1) * dst.step + x + 1]= y_val;
		// here, consider the padding
		uvOff =  y / 4 * dst.step + ((y/2)%2)* src.cols / 2 + x /2;
		u_plane[uvOff] = u_val;
		v_plane[uvOff] = v_val;
#endif

    }

	template <typename T>
    __global__ void RGB_to_NV12(const GlobPtrSz<T> src, GlobPtr<uchar> dst)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

        if (x + 1 >= src.cols || y + 1 >= src.rows)
            return;

        // get pointers to the data
        const size_t planeSize = src.rows * dst.step;
#ifdef USE_UV_PLANE
        GlobPtr<unsigned char> y_plane = globPtr(dst.data, dst.step);
		// the u,v plane is not right, because there is padding

        GlobPtr<unsigned char> u_plane = globPtr(y_plane.data + planeSize, dst.step / 2);
        GlobPtr<unsigned char> v_plane = globPtr(u_plane.data + (planeSize / 4), dst.step / 2);
#else
		unsigned char *y_plane = dst.data;
		unsigned char *u_plane = y_plane + planeSize;
		
		int uvOff = 0;
#endif
		// not right here, cause the cuda pitch is not equal the surface pitch

        T pix;
        unsigned char y_val, u_val, v_val;
#ifdef BGR
        pix = src(y, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x) = y_val;

        pix = src(y, x + 1);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x + 1) = y_val;

        pix = src(y + 1, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y + 1, x) = y_val;

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix.z, pix.y, pix.x, y_val, u_val, v_val);
#else
		pix = src(y, x);
        rgb_to_y(pix.x, pix.y, pix.z, y_val);
#ifdef USE_UV_PLANE
        y_plane(y, x) = y_val;
#else
		y_plane[ y * dst.step + x ] = y_val;
#endif

        pix = src(y, x + 1);
        rgb_to_y(pix.x, pix.y, pix.z, y_val);
#ifdef USE_UV_PLANE] = y_val;
        y_plane(y, x + 1) = y_val;
#else
		y_plane[ y * dst.step + x + 1] = y_val;
#endif

        pix = src(y + 1, x);
        rgb_to_y(pix.x, pix.y, pix.z, y_val);
#ifdef USE_UV_PLANE
        y_plane(y + 1, x) = y_val;
#else
		y_plane[ (y + 1) * dst.step + x ]= y_val;
#endif

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix.x, pix.y, pix.z, y_val, u_val, v_val);
#endif

#ifdef USE_UV_PLANE
        y_plane(y + 1, x + 1) = y_val;
        u_plane(y / 2, x / 2) = u_val;
        v_plane(y / 2, x / 2) = v_val;
#else
		y_plane[( y + 1) * dst.step + x + 1]= y_val;
		uvOff = y / 2 * dst.step + x/ 2 * 2;
		
		u_plane[uvOff] = u_val;
		u_plane[uvOff + 1] = v_val;
		// here, consider the padding
		//uvOff =  y / 4 * dst.step + ((y/2)%2)* src.cols / 2 + x /2;
		//u_plane[uvOff] = u_val;
		//v_plane[uvOff] = v_val;
#endif

    }
#else
	// use the elem size to solve the cuda pitch not equal surface pitch 
	template <typename T>
    __global__ void RGB_to_YV12(const GlobPtrSz<T> src, GlobPtr<uchar> dst)
    {
        const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
        const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

        if (x + 1 >= src.cols || y + 1 >= src.rows)
            return;

        // get pointers to the data
        const size_t planeSize = src.rows * dst.step;
        GlobPtr<unsigned char> y_plane = globPtr(dst.data, dst.step);
        GlobPtr<unsigned char> u_plane = globPtr(y_plane.data + planeSize, dst.step / 2);
        GlobPtr<unsigned char> v_plane = globPtr(u_plane.data + (planeSize / 4), dst.step / 2);
		// not right here, cause the cuda pitch is not equal the surface pitch

        T pix;
        unsigned char y_val, u_val, v_val;

        pix = src(y, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x) = y_val;

        pix = src(y, x + 1);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y, x + 1) = y_val;

        pix = src(y + 1, x);
        rgb_to_y(pix.z, pix.y, pix.x, y_val);
        y_plane(y + 1, x) = y_val;

        pix = src(y + 1, x + 1);
        rgb_to_yuv(pix.z, pix.y, pix.x, y_val, u_val, v_val);
        y_plane(y + 1, x + 1) = y_val;
        u_plane(y / 2, x / 2) = u_val;
        v_plane(y / 2, x / 2) = v_val;
    }

#endif

}

__global__ void RGB_to_YV12_2(const unsigned char * pARGB, unsigned char * pYV, int srcPitch, int dstPitch, int width, int height){

	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

   // if (x + 1 >= src.cols || y + 1 >= src.rows)
        //return;

	const int planeSize = height * dstPitch;
	unsigned char * y_plane = pYV;
	unsigned char * u_plane = y_plane + planeSize;
	unsigned char * v_plane = u_plane + (planeSize >> 2);

	unsigned char y_val, u_val, v_val;
	unsigned char r, g, b;
	int rgbaSize = 4;
	int uv_off = 0;

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

	rgb_to_yuv(b, g, r, y_val, u_val, v_val);
	y_plane[ (y + 1) * dstPitch + x + 1] = y_val;
	uv_off = (y / 4) * dstPitch + (( y / 2) % 2) * width /2 + x /2;
	u_plane[ uv_off ] = u_val;
	v_plane[ uv_off ] = v_val;
}

extern "C" 
	void RGB_to_YV12(const GpuMat& src, GpuMat& dst)
{
    const dim3 block(32, 8);
    const dim3 grid(divUp(src.cols, block.x * 2), divUp(src.rows, block.y * 2));

    switch (src.channels())
    {
    case 1:
        Gray_to_YV12<<<grid, block>>>(globPtr<uchar>(src), globPtr<uchar>(dst));
        break;
    case 3:
        RGB_to_YV12<<<grid, block>>>(globPtr<uchar3>(src), globPtr<uchar>(dst));
        break;
    case 4:
        RGB_to_YV12<<<grid, block>>>(globPtr<uchar4>(src), globPtr<uchar>(dst));
        break;
    }

    cudaGetLastError() ;
    cudaDeviceSynchronize() ;
}

// use the plane pointer to achieve the RGB to YV12 convertion
extern "C"
	void RGB_to_YV12_plane(const GpuMat& src, GpuMat& dst){
	const dim3 block(32, 8);
    const dim3 grid(divUp(src.cols, block.x * 2), divUp(src.rows, block.y * 2));
	switch(src.channels()){
	case 4:
		//RGB_to_YV12_plane<<<grid, block>>>(globPtr<uchar4>(src), globPtr<uchar>(dst));
		break;
	}
	cudaGetLastError();
	cudaDeviceSynchronize();
}

extern "C"
	void RGBA_to_NV12(const GpuMat & src, GpuMat & dst){
	const dim3 block(32, 8);
    const dim3 grid(divUp(src.cols, block.x * 2), divUp(src.rows, block.y * 2));
	switch(src.channels()){
	case 4:
		RGB_to_NV12<<<grid, block>>>(globPtr<uchar4>(src), globPtr<uchar>(dst));
		break;
	}
	cudaGetLastError();
	cudaDeviceSynchronize();
}