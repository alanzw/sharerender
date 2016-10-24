// this file is testing the cuda encoder with CUDA 6.0
#define WINDOWS_LEAN_AND_MEAN
#pragma push_macro("_WINSOCKAPI_")
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#include "..\..\CloudGamingLiveMigrate\VideoUtility\CudaEncoder.h"
//#include "..\..\CloudGamingLiveMigrate\LibCore\Glob.hpp"
//#include "..\..\CloudGamingLiveMigrate\LibCore\GpuMat.hpp"

#pragma comment(lib, "ws2_32.lib")
#ifndef _DEBUG
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.lib")
#pragma comment(lib, "libgroupsock.lib")
#pragma comment(lib, "libBasicUsageEnvironment.lib")
#pragma comment(lib, "libUsageEnvironment.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#else

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.d.lib")
#pragma comment(lib, "libgroupsock.d.lib")
#pragma comment(lib, "libBasicUsageEnvironment.d.lib")
#pragma comment(lib, "libUsageEnvironment.d.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#pragma comment(lib, "nvcuvenc.lib")

#endif

int main(int argc, char ** argv){
#if 0
	cg::core::infoRecorder = new cg::core::InfoRecorder("testcuda");

	// here, open the yuv input file
	if(argc < 2){
		printf("[Main]: TestCudaEncoder [input file]");
		return 1;
	}
	// test GPU memory
	cg::core::GpuMat tmp(1, 1, CV_8U);
	tmp.release();


	CUcontext cuContext;
	CUresult err = CUDA_SUCCESS;

	err = cuCtxGetCurrent(&cuContext);
	assert(err == CUDA_SUCCESS);

	//cg::core::GpuMat src;
	//src.create(900, 800, CV_8UC1);

	FILE * input = NULL;
	typedef  unsigned char BYTE;
	BYTE * yuv_cpu = new BYTE[800 * 600 * 3];
	BYTE * yuv_array = new BYTE[src.step * src.rows];

	// create cuda encoder
	//cg::core::Size size(800, 600);
#if 0
	CV_CudaEncoder * encoder = CreateCVEncoder("test.cuda.t.264", size, 30, SF_NV12);


	memset(yuv_cpu, 0, 600 * 800 * 3);

	cudaError_t err_cuda = cudaSuccess;

	input = fopen(argv[1], "rb");
	while(! feof(input)){
		fread(yuv_cpu, 800 * 600 * 3 /2, 1, input);
		//copy to CUDA memory

		for(int i = 0; i< src.rows; i++){
			memcpy(yuv_array + i * src.step, yuv_cpu + i * 800, 800);
		}

		err_cuda = cudaMemcpy((void *)src.data, (void *)yuv_array, src.step * src.rows, cudaMemcpyHostToDevice);
		if(err != cudaSuccess){
			printf("[Main]: cudaMemcpy failed with:%d.\n", err_cuda);
			break;
		}
		encoder->write(src);

	}

#endif
#endif
	return 0;
}