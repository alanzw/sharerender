#include "BmpFormat.h"

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
	// declare BMP structures 
	BITMAPFILEHEADER bmfh;
	BITMAPINFOHEADER info;
	
	// and initialize them to zero
	memset ( &bmfh, 0, sizeof (BITMAPFILEHEADER ) );
	memset ( &info, 0, sizeof (BITMAPINFOHEADER ) );
	
	// fill the file header with data
	bmfh.bfType = 0x4d42;       // 0x4d42 = 'BM'
	bmfh.bfReserved1 = 0;
	bmfh.bfReserved2 = 0;
	bmfh.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + paddedsize;
	bmfh.bfOffBits = 0x36;		// number of bytes to start of bitmap bits
	
	// fill the info header

	info.biSize = sizeof(BITMAPINFOHEADER);
	info.biWidth = width;
	info.biHeight = height;
	info.biPlanes = 1;			// we only have one bit plane
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
	// write info header
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

BYTE * LoadBMP(LPCTSTR bmpfile){
	return NULL;
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

bool YUV420PToBGR24_Native(unsigned char * pYUV, unsigned char * pBGR24, int width, int height){
	if (width < 1 || height < 1 || pYUV == NULL || pBGR24 == NULL)
		return false;
	const long len = width * height;
	unsigned char* yData = pYUV;
	unsigned char* vData = &yData[len * 5 / 4];
	unsigned char* uData = &yData[len];

	int bgr[3];
	int yIdx,uIdx,vIdx,idx;
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width;j++){
			yIdx = i * width + j;
			vIdx = (i/2) * (width) + (j/2);
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
    for (int i = 0; i < height; i++){
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

BYTE* ConvertRGBAToBMPBuffer ( BYTE* Buffer, int width, int height, long* newsize )
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
		for ( int x = 0; x < 4 * width; x+=4 )
		{
			bufpos = y * 4 * width + x;     // position in original buffer
			newpos = ( height - y - 1 ) * psw + x;           // position in padded buffer
#if 0
			newbuf[newpos] = Buffer[bufpos+2];       // swap r and b
			newbuf[newpos + 1] = Buffer[bufpos + 1]; // g stays
			newbuf[newpos + 2] = Buffer[bufpos];     // swap b and r
#else
			// argb format
			newbuf[newpos + 0] = Buffer[bufpos + 1];
			newbuf[newpos + 1] = Buffer[bufpos + 2];
			newbuf[newpos + 2] = Buffer[bufpos + 3];
#endif
		}

		return newbuf;
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

// convert the bmp format data to a pre-allocated rgba buffer
bool ConvertBMPToRGBBuffer ( BYTE* Buffer, BYTE * dstBuffer, int width, int height)
{ 
	// first make sure the parameters are valid 
	if ( ( NULL == Buffer ) || (NULL == dstBuffer) || ( width == 0 ) || ( height == 0 ) ) 
		return false; 

	// find the number of padding bytes 

	int padding = 0; 
	int scanlinebytes = width * 3; 
	while ( ( scanlinebytes + padding ) % 4 != 0 )     // DWORD = 4 bytes 
		padding++; 
	// get the padded scanline width 
	int psw = scanlinebytes + padding; 

	// create new buffer 
	BYTE* newbuf = dstBuffer;     // RGBA format

	// now we loop trough all bytes of the original buffer, 
	// swap the R and B bytes and the scanlines 
	long bufpos = 0; 
	long newpos = 0; 
	for ( int y = 0; y < height; y++ ) 
		for(int x = 0; x < width; x++)
			//for ( int x = 0; x < 3 * width; x+=3 ) 
		{ 
			newpos = y * 4 * width + x * 4; 
			bufpos = ( height - y - 1 ) * psw + x * 3; 

			newbuf[newpos] = Buffer[bufpos + 2]; 
			newbuf[newpos + 1] = Buffer[bufpos+1]; 
			newbuf[newpos + 2] = Buffer[bufpos]; 
		} 

		return newbuf; 
} 