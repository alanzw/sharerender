#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "glob.hpp"
#include <algorithm>
#include <iostream>
#include "GpuMat.hpp"

using namespace std;


using namespace cudev;

//extern struct GpuMat;
//extern "C"
//	void videoDecPostProcessFrame(const GpuMat& decodedFrame, GpuMat& _outFrame, int width, int height);

namespace
{
    __constant__ float constHueColorSpaceMat[9] = {1.1644f, 0.0f, 1.596f, 1.1644f, -0.3918f, -0.813f, 1.1644f, 2.0172f, 0.0f};

    __device__ static void YUV2RGB(const uint* yuvi, float* red, float* green, float* blue)
    {
        float luma, chromaCb, chromaCr;

        // Prepare for hue adjustment
        luma     = (float)yuvi[0];
        chromaCb = (float)((int)yuvi[1] - 512.0f);
        chromaCr = (float)((int)yuvi[2] - 512.0f);

       // Convert YUV To RGB with hue adjustment
       *red   = (luma     * constHueColorSpaceMat[0]) +
                (chromaCb * constHueColorSpaceMat[1]) +
                (chromaCr * constHueColorSpaceMat[2]);

       *green = (luma     * constHueColorSpaceMat[3]) +
                (chromaCb * constHueColorSpaceMat[4]) +
                (chromaCr * constHueColorSpaceMat[5]);

       *blue  = (luma     * constHueColorSpaceMat[6]) +
                (chromaCb * constHueColorSpaceMat[7]) +
                (chromaCr * constHueColorSpaceMat[8]);
    }

    __device__ static unsigned int RGBA_pack_10bit(float red, float green, float blue, unsigned int alpha)
    {
        unsigned int ARGBpixel = 0;

        // Clamp final 10 bit results
		
        red   = min(max(red,   0.0f), 1023.f);
        green = min(max(green, 0.0f), 1023.f);
        blue  = min(max(blue,  0.0f), 1023.f);

        // Convert to 8 bit unsigned integers per color component
        ARGBpixel = (((unsigned int)blue  >> 2) |
                    (((unsigned int)green >> 2) << 8)  |
                    (((unsigned int)red   >> 2) << 16) |
                    (unsigned int)alpha);

        return ARGBpixel;
    }

    // CUDA kernel for outputing the final ARGB output from NV12

    #define COLOR_COMPONENT_BIT_SIZE 10
    #define COLOR_COMPONENT_MASK     0x3FF

    __global__ void NV12_to_RGB(
		const unsigned char* srcImage, size_t nSourcePitch,
        unsigned int* dstImage, size_t nDestPitch,
        unsigned int width, uint height)
    {
        // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
        const int x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
        const int y = blockIdx.y *  blockDim.y       +  threadIdx.y;

        if (x >= width || y >= height)
            return;

        // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
        // if we move to texture we could read 4 luminance values

        uint yuv101010Pel[2];

        yuv101010Pel[0] = (srcImage[y * nSourcePitch + x    ]) << 2;
        yuv101010Pel[1] = (srcImage[y * nSourcePitch + x + 1]) << 2;

        const size_t chromaOffset = nSourcePitch * height;

        const int y_chroma = y >> 1;

        if (y & 1)  // odd scanline ?
        {
            uint chromaCb = srcImage[chromaOffset + y_chroma * nSourcePitch + x    ];
            uint chromaCr = srcImage[chromaOffset + y_chroma * nSourcePitch + x + 1];

            if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
            {
                chromaCb = (chromaCb + srcImage[chromaOffset + (y_chroma + 1) * nSourcePitch + x    ] + 1) >> 1;
                chromaCr = (chromaCr + srcImage[chromaOffset + (y_chroma + 1) * nSourcePitch + x + 1] + 1) >> 1;
            }

            yuv101010Pel[0] |= (chromaCb << ( COLOR_COMPONENT_BIT_SIZE       + 2));
            yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

            yuv101010Pel[1] |= (chromaCb << ( COLOR_COMPONENT_BIT_SIZE       + 2));
            yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
        }
        else
        {
            yuv101010Pel[0] |= ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x    ] << ( COLOR_COMPONENT_BIT_SIZE       + 2));
            yuv101010Pel[0] |= ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

            yuv101010Pel[1] |= ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x    ] << ( COLOR_COMPONENT_BIT_SIZE       + 2));
            yuv101010Pel[1] |= ((uint)srcImage[chromaOffset + y_chroma * nSourcePitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
        }

        // this steps performs the color conversion
        uint yuvi[6];
        float red[2], green[2], blue[2];

        yuvi[0] =  (yuv101010Pel[0] &   COLOR_COMPONENT_MASK    );
        yuvi[1] = ((yuv101010Pel[0] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
        yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

        yuvi[3] =  (yuv101010Pel[1] &   COLOR_COMPONENT_MASK    );
        yuvi[4] = ((yuv101010Pel[1] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
        yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

        // YUV to RGB Transformation conversion
        YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
        YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);

        // Clamp the results to RGBA

        const size_t dstImagePitch = nDestPitch >> 2;

        dstImage[y * dstImagePitch + x     ] = RGBA_pack_10bit(red[0], green[0], blue[0], ((uint)0xff << 24));
        dstImage[y * dstImagePitch + x + 1 ] = RGBA_pack_10bit(red[1], green[1], blue[1], ((uint)0xff << 24));
    }
}
extern "C"
	void videoDecPostProcessFrame(const GpuMat& decodedFrame, GpuMat _outFrame, int width, int height)
{
    // Final Stage: NV12toARGB color space conversion

    _outFrame.create(height, width, CV_8UC4);
    GpuMat outFrame = _outFrame;

    dim3 block(32, 8);
    dim3 grid(divUp(width, 2 * block.x), divUp(height, block.y));

	//decodedFrame.step;
	//decodedFrame.ptr();

    NV12_to_RGB<<<grid, block>>>(decodedFrame.ptr<uchar>(), decodedFrame.step,
                                 outFrame.ptr<uint>(), outFrame.step,
                                 width, height);

    cudaGetLastError();
    cudaDeviceSynchronize();
}