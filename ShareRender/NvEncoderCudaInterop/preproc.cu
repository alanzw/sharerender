
// BlockDim = 16x16
//GridDim = w/16*h/16
extern "C" __global__ void InterleaveUV( unsigned char *yuv_cb, unsigned char *yuv_cr, unsigned char *nv12_chroma,
                  int chroma_width, int chroma_height, int cb_pitch, int cr_pitch, int nv12_pitch )
{
    int x,y;
    unsigned char *pCb;
    unsigned char *pCr;
    unsigned char *pDst;
    x = blockIdx.x*blockDim.x+threadIdx.x;
    y = blockIdx.y*blockDim.y+threadIdx.y;

    if ((x < chroma_width) && (y < chroma_height))
    {
        pCb = yuv_cb + (y*cb_pitch);
        pCr = yuv_cr + (y*cr_pitch);
        pDst = nv12_chroma + y*nv12_pitch;
        pDst[x << 1]       = pCb[x];
        pDst[(x << 1) + 1] = pCr[x];
    }
}

