#include <cuda.h>

#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <iostream>
#include "GpuMat.hpp"
#include "glob.hpp"

using namespace std;
using namespace cudev;
// convertion from YV12 to RGB


#define COLOR_COMPONENT_BIT_SIZE 10

#define COLOR_COMPONENT_MASK     0x3FF
 
__constant__ float constHueColorSpaceMat[9]={1.1644f,0.0f,1.596f,1.1644f,-0.3918f,-0.813f,1.1644f,2.0172f,0.0f};

 
__device__ static void YUV2RGB(const int* yuvi,float* red,float* green,float* blue)
{
    float luma, chromaCb, chromaCr;

    // Prepare for hue adjustment

    luma     =(float)((int)yuvi[0]- 64.0f);
    chromaCb =(float)((int)yuvi[1]-512.0f);
    chromaCr =(float)((int)yuvi[2]-512.0f);

   // Convert YUV To RGB with hue adjustment

   *red   =(luma     * constHueColorSpaceMat[0])+
            (chromaCb * constHueColorSpaceMat[1])+
            (chromaCr * constHueColorSpaceMat[2]);

   *green =(luma     * constHueColorSpaceMat[3])+
            (chromaCb * constHueColorSpaceMat[4])+
            (chromaCr * constHueColorSpaceMat[5]);

   *blue  =(luma     * constHueColorSpaceMat[6])+
            (chromaCb * constHueColorSpaceMat[7])+
            (chromaCr * constHueColorSpaceMat[8]);

}

 

__device__ static int RGBA_pack_10bit(float red,float green,float blue,int alpha)
{
    int ARGBpixel =0;
    // Clamp final 10 bit results

    red   = min(max(red,  0.0f),1023.f);
    green = min(max(green,0.0f),1023.f);
    blue  = min(max(blue, 0.0f),1023.f);

    // Convert to 8 bit unsigned integers per color component

    ARGBpixel = (((int)blue  >>2 << 8)     |
                (((int)green >>2)<< 16)|
                (((int)red   >>2)<< 24)|
                  (int)alpha);

    return ARGBpixel;

}

__device__ static int RGB_pack_8bit(float red,float green,float blue, int alpha)
{
    int ARGBpixel =0;
    // Clamp final 10 bit results

    red   = min(max(red,  0.0f),1023.f);
    green = min(max(green,0.0f),1023.f);
    blue  = min(max(blue, 0.0f),1023.f);

    // Convert to 8 bit unsigned integers per color component

    ARGBpixel = (((int)blue  >>2)     |
                (((int)green >>2)<< 8)|
                (((int)red   >>2)<<16)|
                  (int)alpha);

    return ARGBpixel;

}

 

__global__ void YV12ToARGB_FourPixel(const unsigned char* pYV12,unsigned int* pARGB,int width,int height)
{
    // Pad borders with duplicate pixels, and we multiply by 2 because we process 4 pixels per thread
    const int x = blockIdx.x *(blockDim.x <<1)+(threadIdx.x <<1);
    const int y = blockIdx.y *(blockDim.y <<1)+(threadIdx.y <<1);

    if((x +1)>= width ||(y +1)>= height)
       return;

    // Read 4 Luma components at a time
    int yuv101010Pel[4];
    yuv101010Pel[0]=(pYV12[y * width + x    ])<<2;
    yuv101010Pel[1]=(pYV12[y * width + x +1])<<2;
    yuv101010Pel[2]=(pYV12[(y +1)* width + x    ])<<2;
    yuv101010Pel[3]=(pYV12[(y +1)* width + x +1])<<2;

    const unsigned int vOffset = width * height;
    const unsigned int uOffset = vOffset +(vOffset >>2);
    const unsigned int vPitch = width >>1;

    const unsigned int uPitch = vPitch;
    const int x_chroma = x >>1;
    const int y_chroma = y >>1;

    int chromaCb = pYV12[uOffset + y_chroma * uPitch + x_chroma];      //U
    int chromaCr = pYV12[vOffset + y_chroma * vPitch + x_chroma];      //V

    yuv101010Pel[0]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE       +2));
    yuv101010Pel[0]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1)+2));
    yuv101010Pel[1]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE       +2));
    yuv101010Pel[1]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1)+2));
    yuv101010Pel[2]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE       +2));
    yuv101010Pel[2]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1)+2));
    yuv101010Pel[3]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE       +2));
    yuv101010Pel[3]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1)+2));

    // this steps performs the color conversion

    int yuvi[12];

    float red[4], green[4], blue[4];

    yuvi[0]=(yuv101010Pel[0]&   COLOR_COMPONENT_MASK    );
    yuvi[1]=((yuv101010Pel[0]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[2]=((yuv101010Pel[0]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[3]=(yuv101010Pel[1]&   COLOR_COMPONENT_MASK    );
    yuvi[4]=((yuv101010Pel[1]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[5]=((yuv101010Pel[1]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[6]=(yuv101010Pel[2]&   COLOR_COMPONENT_MASK    );
    yuvi[7]=((yuv101010Pel[2]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[8]=((yuv101010Pel[2]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[9]=(yuv101010Pel[3]&   COLOR_COMPONENT_MASK    );
    yuvi[10]=((yuv101010Pel[3]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[11]=((yuv101010Pel[3]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    // YUV to RGB Transformation conversion
    YUV2RGB(&yuvi[0],&red[0],&green[0],&blue[0]);
    YUV2RGB(&yuvi[3],&red[1],&green[1],&blue[1]);
    YUV2RGB(&yuvi[6],&red[2],&green[2],&blue[2]);
    YUV2RGB(&yuvi[9],&red[3],&green[3],&blue[3]);

    pARGB[y * width + x     ]= RGBA_pack_10bit(red[0], green[0], blue[0],((int)0xff<<24));
    pARGB[y * width + x +1]= RGBA_pack_10bit(red[1], green[1], blue[1],((int)0xff<<24));
    pARGB[(y +1)* width + x     ]= RGBA_pack_10bit(red[2], green[2], blue[2],((int)0xff<<24));
    pARGB[(y +1)* width + x +1]= RGBA_pack_10bit(red[3], green[3], blue[3],((int)0xff<<24));

}
#if 0
template <typename T>
__global__ void YV12_to_RGBA(const GlobPtrSz<T> src, GlobPtrSz<T> dst){
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

	if (x + 1 >= src.cols || y + 1 >= src.rows)
		return;

	unsigned char * pYV12 = src.data;
	unsigned int * pARGB = dst.data;

	 // Read 4 Luma components at a time
    int yuv101010Pel[4];
	yuv101010Pel[0]=(pYV12[ y *    src.step + x    ])<<2;
    yuv101010Pel[1]=(pYV12[ y *    src.step + x +1 ])<<2;
    yuv101010Pel[2]=(pYV12[(y +1)* src.step + x    ])<<2;
    yuv101010Pel[3]=(pYV12[(y +1)* src.step + x +1 ])<<2;



    const unsigned int vOffset = src.step * dst.rows;
    const unsigned int uOffset = vOffset + (vOffset >> 2);
    const unsigned int vPitch = src.step >> 1;

    const unsigned int uPitch = vPitch;
    const int x_chroma = x >>1;
    const int y_chroma = y >>1;

    int chromaCb = pYV12[uOffset + y_chroma * uPitch + x_chroma];      //U
    int chromaCr = pYV12[vOffset + y_chroma * vPitch + x_chroma];      //V

    yuv101010Pel[0]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[0]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));
    yuv101010Pel[1]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[1]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));
    yuv101010Pel[2]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[2]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));
    yuv101010Pel[3]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[3]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));

    // this steps performs the color conversion

    int yuvi[12];

    float red[4], green[4], blue[4];

    yuvi[0]=( yuv101010Pel[0]&   COLOR_COMPONENT_MASK    );
    yuvi[1]=((yuv101010Pel[0]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[2]=((yuv101010Pel[0]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[3]=( yuv101010Pel[1]&   COLOR_COMPONENT_MASK    );
    yuvi[4]=((yuv101010Pel[1]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[5]=((yuv101010Pel[1]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[6]=( yuv101010Pel[2]&   COLOR_COMPONENT_MASK    );
    yuvi[7]=((yuv101010Pel[2]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[8]=((yuv101010Pel[2]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[9]=( yuv101010Pel[3]&   COLOR_COMPONENT_MASK    );
    yuvi[10]=((yuv101010Pel[3]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[11]=((yuv101010Pel[3]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    // YUV to RGB Transformation conversion
    YUV2RGB(&yuvi[0],&red[0],&green[0],&blue[0]);
    YUV2RGB(&yuvi[3],&red[1],&green[1],&blue[1]);
    YUV2RGB(&yuvi[6],&red[2],&green[2],&blue[2]);
    YUV2RGB(&yuvi[9],&red[3],&green[3],&blue[3]);

    pARGB[y * dst.cols + x        ]= RGBA_pack_10bit(red[0], green[0], blue[0],((int)0xff<<24));
    pARGB[y * dst.cols + x + 1    ]= RGBA_pack_10bit(red[1], green[1], blue[1],((int)0xff<<24));
    pARGB[(y +1)* dst.cols + x    ]= RGBA_pack_10bit(red[2], green[2], blue[2],((int)0xff<<24));
    pARGB[(y +1)* dst.cols + x + 1]= RGBA_pack_10bit(red[3], green[3], blue[3],((int)0xff<<24));

}
#else
__global__ void YV12_to_RGBA(const char * pYV12, int src_step, int src_cols, int src_rows, unsigned int * pARGB, int dst_pitch, int dst_rows){
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

	if (x + 1 >= src_cols || y + 1 >= dst_rows)
		return;

	//unsigned char * pYV12 = src.data;
	//unsigned int * pARGB = dst.data;

	 // Read 4 Luma components at a time
    int yuv101010Pel[4];
	yuv101010Pel[0]=(pYV12[ y *    src_step + x    ])<<2;
    yuv101010Pel[1]=(pYV12[ y *    src_step + x +1 ])<<2;
    yuv101010Pel[2]=(pYV12[(y +1)* src_step + x    ])<<2;
    yuv101010Pel[3]=(pYV12[(y +1)* src_step + x +1 ])<<2;



    const unsigned int vOffset = src_step * dst_rows;
    const unsigned int uOffset = vOffset + (vOffset >> 2);
   
	const unsigned int vPitch = src_step >> 1;
	//const unsigned int vPitch = src_cols >> 1;

    const unsigned int uPitch = vPitch;
    const int x_chroma = x >>1;
    const int y_chroma = y >>1;

    int chromaCb = pYV12[uOffset + y_chroma * uPitch + x_chroma];      //U
    int chromaCr = pYV12[vOffset + y_chroma * vPitch + x_chroma];      //V

    yuv101010Pel[0]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[0]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));
    yuv101010Pel[1]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[1]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));
    yuv101010Pel[2]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[2]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));
    yuv101010Pel[3]|=(chromaCb <<( COLOR_COMPONENT_BIT_SIZE      + 2));
    yuv101010Pel[3]|=(chromaCr <<((COLOR_COMPONENT_BIT_SIZE <<1) + 2));

    // this steps performs the color conversion

    int yuvi[12];

    float red[4], green[4], blue[4];

    yuvi[0]=( yuv101010Pel[0]&   COLOR_COMPONENT_MASK    );
    yuvi[1]=((yuv101010Pel[0]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[2]=((yuv101010Pel[0]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[3]=( yuv101010Pel[1]&   COLOR_COMPONENT_MASK    );
    yuvi[4]=((yuv101010Pel[1]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[5]=((yuv101010Pel[1]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[6]=( yuv101010Pel[2]&   COLOR_COMPONENT_MASK    );
    yuvi[7]=((yuv101010Pel[2]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[8]=((yuv101010Pel[2]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    yuvi[9]=( yuv101010Pel[3]&   COLOR_COMPONENT_MASK    );
    yuvi[10]=((yuv101010Pel[3]>>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[11]=((yuv101010Pel[3]>>(COLOR_COMPONENT_BIT_SIZE <<1))& COLOR_COMPONENT_MASK);

    // YUV to RGB Transformation conversion
    YUV2RGB(&yuvi[0],&red[0],&green[0],&blue[0]);
    YUV2RGB(&yuvi[3],&red[1],&green[1],&blue[1]);
    YUV2RGB(&yuvi[6],&red[2],&green[2],&blue[2]);
    YUV2RGB(&yuvi[9],&red[3],&green[3],&blue[3]);

    pARGB[y * dst_pitch / sizeof(int) + x        ]= RGBA_pack_10bit(red[0], green[0], blue[0],((int)0x000000ff));
    pARGB[y * dst_pitch / sizeof(int) + x + 1    ]= RGBA_pack_10bit(red[1], green[1], blue[1],((int)0x000000ff));
    pARGB[(y +1)* dst_pitch / sizeof(int) + x    ]= RGBA_pack_10bit(red[2], green[2], blue[2],((int)0x000000ff));
    pARGB[(y +1)* dst_pitch / sizeof(int) + x + 1]= RGBA_pack_10bit(red[3], green[3], blue[3],((int)0x000000ff));

}
#endif

extern "C" 
bool YV12ToARGB(unsigned char* pYV12,unsigned char* pARGB,int width,int height)
{
    unsigned char* d_src;
    unsigned char* d_dst;
    unsigned int srcMemSize =sizeof(unsigned char)* width * height *3/2;
    unsigned int dstMemSize =sizeof(unsigned char)* width * height *4;

    cudaMalloc((void**)&d_src,srcMemSize);
    cudaMalloc((void**)&d_dst,dstMemSize);
    cudaMemcpy(d_src,pYV12,srcMemSize,cudaMemcpyDeviceToDevice);

    dim3 block(32,8);
    int gridx =(width +2*block.x -1)/(2*block.x);
    int gridy =(height +2*block.y -1)/(2*block.y);

    dim3 grid(gridx,gridy);

    YV12ToARGB_FourPixel<<<grid, block>>>(d_src,(unsigned int*)d_dst,width,height);
    cudaMemcpy(pARGB,d_dst,dstMemSize,cudaMemcpyDeviceToHost);
    return true;

}

extern "C"
	bool YV12MatToARGB(GpuMat &src, GpuMat &dst){
		//unsigned char * d_src;
		unsigned char * d_dst;
		
		GpuMat d_src(src);

		//cudaMallocPitch((void **)&d_src, src.step, src.cols, src.rows);
			
		//cudaMallocPitch((void **)&d_src, src.step, src.cols, src.rows);
		const dim3 block(32,8);
		const dim3 grid(divUp(src.cols, block.x * 2), divUp(src.rows, block.y * 2));
		
		// launch the kernel
		YV12_to_RGBA<<<grid, block>>>((const char *)d_src.data, (size_t)d_src.step, d_src.cols, d_src.rows, (unsigned int *)dst.data, (size_t)dst.step, dst.rows);
		//cudaMemcpy(pARGB,d_dst,dstMemSize,cudaMemcpyDeviceToHost)

		cudaGetLastError();
		cudaDeviceSynchronize();
		return true;
}