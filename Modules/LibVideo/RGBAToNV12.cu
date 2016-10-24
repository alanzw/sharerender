

#ifdef BT601
#define Ycoeff ((float4)(0.299f, 0.587f, 0.114f, 0.f))
#define Ucoeff ((float4)(-0.14713f, -0.28886f, 0.436f, 0.f))
#define Vcoeff ((float4)(0.615f, -0.51499f, -0.10001f, 0.f))

// BGR
#define YcoeffB ((float4)(0.114f, 0.587f, 0.299f, 0.f))
#define UcoeffB ((float4)(0.436f, -0.28886f, -0.14713f, 0.f))
#define VcoeffB ((float4)(-0.10001f, -0.51499, 0.615f, 0.f))

#else
#ifdef BT709

#define Ycoeff ((float4)(0.2126f, 0.7152f, 0.0722f, 0.f))
#define Ucoeff ((float4)(-0.09991f, -0.33609f, -.436f, 0.f))
#define Vcoeff ((float4)(0.615f, -0.55861f, -0.05639f, 0.f))

// BGR
#define YcoeffB ((float4)(0.0722f, 0.7152f, 0.2126f, 0.f))
#define UcoeffB ((float4)(0.436f, -0.33609f, -0.09991f, 0.f))
#define VcoeffB ((float4)(-0.05639f, -0.55861f, 0.615f, 0.f))

#else

#define Ycoeff ((float4)(0.257f, 0.504f, 0.098f, 0.f))
#define Ucoeff ((float4)(-0.148f, -0.291f, 0.439f, 0.f))
#define Vcoeff ((flaot4)(0.439f, -0.368f, -0.071f, 0.f))

#define YcoeffB ((float4)(0.098f, 0.504f, 0.257f, 0.f))
#define UcoeffB ((float4)(0.439f, -0.291f, -0.148f, 0))
#define VcoeffB ((float4)(-0.071f, -0.368f, 0.439f, 0))

#endif // BT709
#endif // BT601
// the cuda kernel to convert rgba data to nv12
__global__ void cuda_kernel_rgba_to_nv12(uchar4 * input, uchar4 * output, int alignedWidth){

}