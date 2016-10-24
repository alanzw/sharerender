#include "Generator.h"
#include "CudaFilter.h"

// the generator currently is only for cuda encoder

VideoGen::VideoGen(HWND hwnd, bool useNVENC){
	this->windowHandle = hwnd;
	encoder = NULL;
	dxWrapper = new DXWrapper();
	dxWrapper->d10Wrapper = NULL;
	dxVer = DXNONE;
	swapChain = NULL;
	d9Device = NULL;

	dxDevice = new DXDevice();

	windowHandle = NULL;
	cudaPipe = NULL;
	imagePipe = NULL;

	useNvenc = useNVENC;
#ifdef NVENC
	nvEncoder = NULL;
#endif
	
	// get the window size
	RECT rect;
	GetWindowRect(hwnd, &rect);
	width = rect.right - rect.left;
	height = rect.bottom - rect.top;

	infoRecorder->logTrace("[VideoGen]: constructor called. width:%d, height:%d\n", width, height);
}	


VideoGen::VideoGen(bool useNVENC){
	encoder = NULL;

#ifdef NVENC
	nvEncoder = NULL;
#endif

	dxWrapper = new DXWrapper();
	dxWrapper->d10Wrapper = NULL;
	dxVer = DXNONE;
	swapChain = NULL;
	d9Device = NULL;
	dxDevice = new DXDevice();

	windowHandle = NULL;
	cudaPipe = NULL;
	imagePipe = NULL;;
	height = 0;
	width = 0;

	useNvenc = useNVENC;
}
VideoGen::~VideoGen(){
	infoRecorder->logTrace("[VideoGen]: destructor called.\n");
	if(imagePipe){
		
		delete imagePipe;
		imagePipe = NULL;
	}
	if(cudaPipe){

		delete cudaPipe;
		cudaPipe = NULL;
	}

	if(dxWrapper){
		delete dxWrapper;
		dxWrapper = NULL;
	}
	if(encoder){
		delete encoder;
		encoder = NULL;
	}

#ifdef NVENC
	if(nvEncoder){
		delete nvEncoder;
		nvEncoder = NULL;
	}
#endif
}

int VideoGen::initVideoGen(DX_VERSION  ver, void * device){
	infoRecorder->logTrace("[VideoGen]: initVideoGen called.\n");
	dxVer = ver;

	presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	setupSurfaceSource();

	if(dxVer == DX9){
 		//DebugBreak();

		infoRecorder->logTrace("[VideoGen]: directx version is D3D9.\n");
		d9Device = (IDirect3DDevice9 *)device;
		
		dxWrapper->d9Wrapper = new D3DWrapper(d9Device);
		//dxWrapper->d9Wrapper->setPipe(pipe);

		// register the cuda pipe, cause we need test cuda encoder
		dxWrapper->d9Wrapper->registerCudaSource(cudaPipe);

	}else if(dxVer == DX10){
		infoRecorder->logTrace("[VideoGen]: directx version is D3D10.\n");
		swapChain = (IDXGISwapChain *)device;
		dxWrapper->d10Wrapper = new D3D10Wrapper(swapChain);
		dxWrapper->d10Wrapper->registerCudaSource(cudaPipe);
	}
	else if(dxVer == DX10_1){
		infoRecorder->logTrace("[VideoGen]: directx version is D3D10_1.\n");
		swapChain = (IDXGISwapChain *)device;
		dxWrapper->d10Wrapper1 = new D3D10Wrapper1(swapChain);
		dxWrapper->d10Wrapper1->registerCudaSource(cudaPipe);
	}
	else if(dxVer == DX11){
		infoRecorder->logTrace("[VideoGen]: directx version is D3D11.\n");
		swapChain = (IDXGISwapChain *)device;
		dxWrapper->d11Wrapper = new D3D11Wrapper(swapChain);

		dxWrapper->d11Wrapper->registerCudaSource(cudaPipe);
	}

	if(useNvenc){

	}else{
		if(encoder == NULL){
			//initCudaEncoder((void *)dxDevice->d10Device);
		}
	}

	return 0;
}
// setup the pipeline for surface
int VideoGen::setupSurfaceSource(){
	
	infoRecorder->logTrace("[VideoGen]: setupSurfaceSource called.\n");
	struct pooldata * data = NULL;
	if((cudaPipe = new pipeline(0))== NULL){
		// init the pipeline failed.
		return -1;
	}
	//if(pipe->allocate_data())
	//no private data ?
	//if((data = cudaPipe->datapool_init(POOLSIZE, sizeof(SurfaceFrame))) == NULL){
	if((data = cudaPipe->datapool_init(POOLSIZE)) == NULL){
		infoRecorder->logTrace("[VideoGen]: data pool init failed.\n");
		// alloc failed
		delete cudaPipe;
		cudaPipe = NULL;
		return -1;
	}
	// per frame init
	for(; data != NULL; data = data->next){
		//SurfaceFrame * frame = (SurfaceFrame *)data->ptr;
		SurfaceFrame * frame = new SurfaceFrame();
		frame->dxVersion = dxVer;
		data->ptr = frame;
		frame->dxSurface = new DXSurface();
		frame->dxSurface->d9Surface = NULL;
	}
	return 0;
}

BOOL VideoGen::stop(){
	//DebugBreak();
	infoRecorder->logTrace("[VideoGen]: stop called.\n");
	return CThread::stop();
}
// the loop logic
void VideoGen::run(){
	struct pooldata * data = NULL;
	struct SurfaceImage * frame = NULL;

	//infoRecorder->logTrace("[VideoGen]: run called. present event:%p\n", presentEvent);
	DWORD ret = WaitForSingleObject(presentEvent, 30);
	if(ret == WAIT_TIMEOUT){
		return;
	}
	
	// present event arrive
	if(dxVer == DX9){
		dxWrapper->d9Wrapper->capture(captureTv, initialTv, freq, frameInterval);
	}
	else if(dxVer == DX10){
		dxWrapper->d10Wrapper->capture(captureTv, initialTv, freq, frameInterval);
	}
	else if(dxVer == DX10_1){
		dxWrapper->d10Wrapper1->capture(captureTv, initialTv, freq, frameInterval);
	}
	else if(dxVer == DX11){
		dxWrapper->d11Wrapper->capture(captureTv, initialTv, freq, frameInterval);
	}
	else{
		// error
		return;
	}
	
}
// deal with msg
void VideoGen::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
	infoRecorder->logTrace("[VideoGen]: onThreadMsg called.\n");
}
// release
void VideoGen::onQuit(){
	infoRecorder->logTrace("[VideoGen]: onQuit called.\n");
}
// init before enter loop
void VideoGen::onThreadStart(){
	// start the modules
	infoRecorder->logTrace("[VideoGen]: onThreadStart called\n");
	if(dxWrapper->d10Wrapper == NULL){
		// error
		infoRecorder->logTrace("[VideoGen]: get NULL wrapper.\n");
	}
	if(cudaPipe == NULL){
		//error
		infoRecorder->logTrace("[VideoGen]: get NULL cuda pipe.\n");
	}
	if(useNvenc == false){
		if(this->encoder == NULL){
			initCudaEncoder((void *)dxDevice->d10Device);
		}
		encoder->startEncoding();
	}else{
#ifdef NVENC
		if(nvEncoder == NULL){
			initNVENCEncoder((void *)dxDevice->d10Device);
		}
		nvEncoder->start();  // start the NVENC encoder main thread.
#endif
	}
	//encoder->setOutputFileName(name);
	
	

	
}



/// for d11, the context is the DeviceContext
int VideoGen::initCudaEncoder(void * device, void * context){
	infoRecorder->logTrace("[VideoGen]: D11 init the cuda encoder.\n");
	dxDevice->d10Device = (ID3D10Device *)device;
	return 0;
}

#ifdef NVENC
int VideoGen::initNVENCEncoder(void * device){
	dxDevice->d10Device = (ID3D10Device *)device;
	infoRecorder->logTrace("[VideoGen]: init the NVENC encoder.\n");
	// error
	char name[100] = {0};
	//sprintf(name, "%s.264", __argv[0]);
	sprintf(name, "%s.264", "textOutput");

	nvEncoder = new CNvEncoderH264(this->width, this->height, name); //(this->height, this->width, this->height, this->width, name);
	//nvEncoder->setInputType(GPUIMAGE);
	//encoder->registerSource(NULL);
	if(cudaPipe){
		nvEncoder->setCudaPipe(this->cudaPipe);
		// register event
		if(this->dxVer == DX9)
			nvEncoder->setSourceNotifier(dxWrapper->d9Wrapper->getCudaNotifier());
			//encoder->registerSourceNotifier(dxWrapper->d9Wrapper->getCudaNotifier());
		else if(dxVer == DX10)
			nvEncoder->setSourceNotifier(dxWrapper->d10Wrapper->getCudaNotifier());
			//encoder->registerSourceNotifier(dxWrapper->d10Wrapper->getCudaNotifier());
		else if(dxVer == DX10_1)
			nvEncoder->setSourceNotifier(dxWrapper->d10Wrapper1->getCudaNotifier());
			//encoder->registerSourceNotifier(dxWrapper->d10Wrapper1->getCudaNotifier());
		else if(dxVer == DX11)
			nvEncoder->setSourceNotifier(dxWrapper->d11Wrapper->getCudaNotifier());
			//encoder->registerSourceNotifier(dxWrapper->d11Wrapper->getCudaNotifier());

	}else{
		//error
		return -1;
	}
	//DebugBreak();
	//encoder->InitEncoder();
	//encoder->SetEncodeParameters();

	//set the event
	//HANDLE evet = CreateEvent(NULL, FALSE, FALSE, NULL);
	infoRecorder->logTrace("[VideoGen]: before init cuda filter, width:%d, height:%d.\n", this->width, this->height);
	nvEncoder->InitCudaFilter(dxVer, device);
	

	return 0;
}
#endif

int VideoGen::initCudaEncoder(void * device){

	//DebugBreak();

	dxDevice->d10Device = (ID3D10Device *)device;
	infoRecorder->logTrace("[VideoGen]: init the cuda encoder.\n");
	// error
	char name[100] = {0};
	sprintf(name, "%s.264", GetCommandLine());
	encoder = new CudaEncoder(this->height, this->width, this->height, this->width, name);
	encoder->setInputType(GPUIMAGE);
	//encoder->registerSource(NULL);
	if(cudaPipe){
		encoder->registerCudaSource(this->cudaPipe);
		// register event
		if(this->dxVer == DX9)
			encoder->registerSourceNotifier(dxWrapper->d9Wrapper->getCudaNotifier());
		else if(dxVer == DX10)
			encoder->registerSourceNotifier(dxWrapper->d10Wrapper->getCudaNotifier());
		else if(dxVer == DX10_1)
			encoder->registerSourceNotifier(dxWrapper->d10Wrapper1->getCudaNotifier());
		else if(dxVer == DX11)
			encoder->registerSourceNotifier(dxWrapper->d11Wrapper->getCudaNotifier());

	}else{
		//error
		return -1;
	}
	//DebugBreak();
	encoder->InitEncoder();
	encoder->SetEncodeParameters();

	//set the event
	//HANDLE evet = CreateEvent(NULL, FALSE, FALSE, NULL);
	infoRecorder->logTrace("[VideoGen]: before init cuda filter, width:%d, height:%d.\n", this->width, this->height);
	encoder->InitCudaFilter(dxVer, device);
	

	return 0;
}