#ifndef __BMPFORMAT_H__
#define __BMPFORMAT_H__
// the bmp file format

extern "C"
{
    #include <stdio.h>
    #include <stdlib.h>
    //#include <jpeglib.h>
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

void savebmp(uchar * pdata, char * bmp_file, int width, int height , int rgb_step);

bool SaveBMP ( BYTE* Buffer, int width, int height, long paddedsize, LPCTSTR bmpfile );

bool YV12ToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height);

bool NV12ToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height);

BYTE* ConvertRGBToBMPBuffer ( BYTE* Buffer, int width, int height, long* newsize );

#endif