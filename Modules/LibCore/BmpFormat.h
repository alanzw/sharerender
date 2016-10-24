#ifndef __BMPFORMAT_H__
#define __BMPFORMAT_H__

// the BMP file format

extern "C"
{
    #include <stdio.h>
    #include <stdlib.h>
	#ifndef WIN32_LEAN_AND_MEAN
	#define WIN32_LEAN_AND_MEAN
	#endif
	#include <Windows.h>
}

typedef long LONG;
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned char uchar;

typedef struct {
        WORD    bfType;
        DWORD   bfSize;
        WORD    bfReserved1;
        WORD    bfReserved2;
        DWORD   bfOffBits;
} BMPFILEHEADER_T;

typedef struct{
        DWORD      biSize;
        LONG       biWidth;
        LONG       biHeight;
        WORD       biPlanes;
        WORD       biBitCount;
        DWORD      biCompression;
        DWORD      biSizeImage;
        LONG       biXPelsPerMeter;
        LONG       biYPelsPerMeter;
        DWORD      biClrUsed;
        DWORD      biClrImportant;
} BMPINFOHEADER_T;

// save BGR buffer to a BMP file
bool SaveBMP ( BYTE* Buffer, int width, int height, long paddedsize, LPCTSTR bmpfile );

// convert the YV12 plane data to RBG format
bool YV12ToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height);

// convert the NV12 plane data to RGB format
bool NV12ToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height);

// convert the YUV420P plane data to RGB format
bool YUV420PToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height);

// convert the RGB buffer to BMP buffer
BYTE* ConvertRGBToBMPBuffer ( BYTE* Buffer, int width, int height, long* newsize );
BYTE* ConvertRGBAToBMPBuffer ( BYTE* Buffer, int width, int height, long* newsize );
bool ConvertBMPToRGBBuffer ( BYTE* Buffer, BYTE * dstBuffer, int width, int height );

#endif