// the file is mainly repsonsible for run the NVECN encoder

#include <d3d9.h>
#include <cuda.h>

#include <cuda_d3d9_interop.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


//#include "..\NVENCEncoder\nvencencoder.h"
#include "..\VideoUtility\generator.h"

#include "..\VideoUtility\inforecoder.h"

#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "nvcuvenc.lib")
#pragma comment(lib, "nvcuvid.lib")

VideoGen * generator = NULL;
InfoRecorder * infoRecorder = NULL;

// call when creating the device, along with the window size and height
void onDeviceCreation(IDirect3DDevice9 * device, int height, int width, HWND hwnd){
	if(infoRecorder == NULL){
		infoRecorder = new InfoRecorder("tesetD3D");
	}

	if(generator == NULL){
		generator = new VideoGen(hwnd, true);
	}

	if(generator == NULL){
		// error
		infoRecorder->logError("create generator failed.\n");
		return;
	}

	cudaError ret;
	if((ret = cudaD3D9SetDirect3DDevice(device))!= cudaSuccess){
		infoRecorder->logError("[cudaD3D9SetDirect3DDevice failed.\n");
	}

	generator->initVideoGen(DX9, device);
	if(!generator->isUseNVENC()){
		generator->initCudaEncoder(device);

	}else{
		generator->initNVENCEncoder(device);
	}

	generator->start();

}

// do the capture work when presenting the image
void onPresent(){


	// set the event
	infoRecorder->logError("trigger the present event.\n");
	SetEvent(generator->getPresentEvent());
}