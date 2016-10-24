#include "bmpformat.h"

void savebmp(uchar * pdata, char * bmp_file, int width, int height, int rgb_step )

{      //分别为rgb数据，要保存的bmp文件名，图片长宽
       int size = width*height*rgb_step*sizeof(char); // 3 bytes per pixel for RGB and 4 bytes for RGBA
       // 位图第一部分，文件信息
       BMPFILEHEADER_T bfh;
       bfh.bfType = (WORD)0x4d42;  //bm
       bfh.bfSize = size  // data size
              + sizeof( BMPFILEHEADER_T ) // first section size
              + sizeof( BMPINFOHEADER_T ) // second section size
              ;
       bfh.bfReserved1 = 0; // reserved
       bfh.bfReserved2 = 0; // reserved
       bfh.bfOffBits = sizeof( BMPFILEHEADER_T )+ sizeof( BMPINFOHEADER_T );//真正的数据的位置

       // 位图第二部分，数据信息
       BMPINFOHEADER_T bih;
       bih.biSize = sizeof(BMPINFOHEADER_T);
       bih.biWidth = width;
       bih.biHeight = -height;//BMP图片从最后一个点开始扫描，显示时图片是倒着的，所以用-height，这样图片就正了
       bih.biPlanes = 1;//为1，不用改
       bih.biBitCount = 24;
       bih.biCompression = BI_RGB;//不压缩
	   //bih.biCompression = BI_RGB;

       bih.biSizeImage = size;
       bih.biXPelsPerMeter = 0x0ec4 ;//像素每米
       bih.biYPelsPerMeter = 0x0ec4 ;
       bih.biClrUsed = 0;//已用过的颜色，24位的为0
       bih.biClrImportant = 0;//每个像素都重要
       FILE * fp = fopen( bmp_file,"wb" );
       if( !fp ) return;

#if 0
       fwrite( &bfh, 8, 1,  fp );//由于linux上4字节对齐，而信息头大小为54字节，第一部分14字节，第二部分40字节，所以会将第一部分补齐为16自己，直接用sizeof，打开图片时就会遇到premature end-of-file encountered错误
       fwrite(&bfh.bfReserved2, sizeof(bfh.bfReserved2), 1, fp);
       fwrite(&bfh.bfOffBits, sizeof(bfh.bfOffBits), 1, fp);
#else
	   fwrite(&bfh, sizeof(BMPFILEHEADER_T), 1, fp);
#endif
       fwrite( &bih, sizeof(BMPINFOHEADER_T),1,fp );
       fwrite(pdata,size,1,fp);
       fclose( fp );
}


/***************************************************************
bool SaveBMP ( BYTE* Buffer, int width, int height, 
		long paddedsize, LPCTSTR bmpfile )

Function takes a buffer of size <paddedsize> 
and saves it as a <width> * <height> sized bitmap 
under the supplied filename.
On error the return value is false.

***************************************************************/

bool SaveBMP ( BYTE* Buffer, int width, int height, long paddedsize, LPCTSTR bmpfile )
{
	// declare bmp structures 
	BITMAPFILEHEADER bmfh;
	BITMAPINFOHEADER info;
	
	// andinitialize them to zero
	memset ( &bmfh, 0, sizeof (BITMAPFILEHEADER ) );
	memset ( &info, 0, sizeof (BITMAPINFOHEADER ) );
	
	// fill the fileheader with data
	bmfh.bfType = 0x4d42;       // 0x4d42 = 'BM'
	bmfh.bfReserved1 = 0;
	bmfh.bfReserved2 = 0;
	bmfh.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + paddedsize;
	bmfh.bfOffBits = 0x36;		// number of bytes to start of bitmap bits
	
	// fill the infoheader

	info.biSize = sizeof(BITMAPINFOHEADER);
	info.biWidth = width;
	info.biHeight = height;
	info.biPlanes = 1;			// we only have one bitplane
	info.biBitCount = 24;		// RGB mode is 24 bits
	info.biCompression = BI_RGB;	
	info.biSizeImage = 0;		// can be 0 for 24 bit images
	info.biXPelsPerMeter = 0x0ec4;     // paint and PSP use this values
	info.biYPelsPerMeter = 0x0ec4;     
	info.biClrUsed = 0;			// we are in RGB mode and have no palette
	info.biClrImportant = 0;    // all colors are important

	// now we open the file to write to
	HANDLE file = CreateFile ( bmpfile , GENERIC_WRITE, FILE_SHARE_READ,
		 NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL );
	if ( file == NULL )
	{
		CloseHandle ( file );
		return false;
	}
	
	// write file header
	unsigned long bwritten;
	if ( WriteFile ( file, &bmfh, sizeof ( BITMAPFILEHEADER ), &bwritten, NULL ) == false )
	{	
		CloseHandle ( file );
		return false;
	}
	// write infoheader
	if ( WriteFile ( file, &info, sizeof ( BITMAPINFOHEADER ), &bwritten, NULL ) == false )
	{	
		CloseHandle ( file );
		return false;
	}
	// write image data
	if ( WriteFile ( file, Buffer, paddedsize, &bwritten, NULL ) == false )
	{	
		CloseHandle ( file );
		return false;
	}
	
	// and close file
	CloseHandle ( file );

	return true;
}

bool YV12ToBGR24_Native(unsigned char* pYUV,unsigned char* pBGR24,int width,int height)
{
    if (width < 1 || height < 1 || pYUV == NULL || pBGR24 == NULL)
        return false;
    const long len = width * height;
    unsigned char* yData = pYUV;
    unsigned char* vData = &yData[len];
    unsigned char* uData = &vData[len >> 2];

    int bgr[3];
    int yIdx,uIdx,vIdx,idx;
    for (int i = 0;i < height;i++){
        for (int j = 0;j < width;j++){
            yIdx = i * width + j;
            vIdx = (i/2) * (width/2) + (j/2);
            uIdx = vIdx;
        
            bgr[0] = (int)(yData[yIdx] + 1.732446 * (uData[vIdx] - 128));                                    // b分量
            bgr[1] = (int)(yData[yIdx] - 0.698001 * (uData[uIdx] - 128) - 0.703125 * (vData[vIdx] - 128));    // g分量
            bgr[2] = (int)(yData[yIdx] + 1.370705 * (vData[uIdx] - 128));                                    // r分量

            for (int k = 0;k < 3;k++){
                idx = (i * width + j) * 3 + k;
                if(bgr[k] >= 0 && bgr[k] <= 255)
                    pBGR24[idx] = bgr[k];
                else
                    pBGR24[idx] = (bgr[k] < 0)?0:255;
            }
        }
    }
    return true;
}

bool NV12ToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height){
	if (width < 1 || height < 1 || pYUV == NULL || pBGR24 == NULL)
        return false;
    const long len = width * height;
    unsigned char* yData = pYUV;
    unsigned char* vData = &yData[len];
    unsigned char* uData = &yData[len];

    int bgr[3];
    int yIdx,uIdx,vIdx,idx;
    for (int i = 0;i < height;i++){
        for (int j = 0;j < width;j++){
            yIdx = i * width + j;
            vIdx = (i/2) * (width) + (j/2) * 2;
            uIdx = vIdx + 1;
        
            bgr[0] = (int)(yData[yIdx] + 1.732446 * (uData[vIdx] - 128));                                    // b分量
            bgr[1] = (int)(yData[yIdx] - 0.698001 * (uData[uIdx] - 128) - 0.703125 * (vData[vIdx] - 128));    // g分量
            bgr[2] = (int)(yData[yIdx] + 1.370705 * (vData[uIdx] - 128));                                    // r分量

            for (int k = 0;k < 3;k++){
                idx = (i * width + j) * 3 + k;
                if(bgr[k] >= 0 && bgr[k] <= 255)
                    pBGR24[idx] = bgr[k];
                else
                    pBGR24[idx] = (bgr[k] < 0)?0:255;
            }
        }
    }
    return true;
}

BYTE* ConvertRGBToBMPBuffer ( BYTE* Buffer, int width, int height, long* newsize )
{

	// first make sure the parameters are valid
	if ( ( NULL == Buffer ) || ( width == 0 ) || ( height == 0 ) )
		return NULL;

	// now we have to find with how many bytes
	// we have to pad for the next DWORD boundary	

	int padding = 0;
	int scanlinebytes = width * 3;
	while ( ( scanlinebytes + padding ) % 4 != 0 )     // DWORD = 4 bytes
		padding++;
	// get the padded scanline width
	int psw = scanlinebytes + padding;
	
	// we can already store the size of the new padded buffer
	*newsize = height * psw;

	// and create new buffer
	BYTE* newbuf = new BYTE[*newsize];
	
	// fill the buffer with zero bytes then we dont have to add
	// extra padding zero bytes later on
	memset ( newbuf, 0, *newsize );

	// now we loop trough all bytes of the original buffer, 
	// swap the R and B bytes and the scanlines
	long bufpos = 0;   
	long newpos = 0;
	for ( int y = 0; y < height; y++ )
		for ( int x = 0; x < 3 * width; x+=3 )
		{
			bufpos = y * 3 * width + x;     // position in original buffer
			newpos = ( height - y - 1 ) * psw + x;           // position in padded buffer

			newbuf[newpos] = Buffer[bufpos+2];       // swap r and b
			newbuf[newpos + 1] = Buffer[bufpos + 1]; // g stays
			newbuf[newpos + 2] = Buffer[bufpos];     // swap b and r
		}

	return newbuf;
}