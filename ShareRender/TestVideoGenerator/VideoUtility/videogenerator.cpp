#include "videogenerator.h"
#include "evendefine.h"


// this is for the video generator
// usage: new VideoGenerator();
//        initVideoGenerator();
//        startThread();
//
//

// 2D game is simple

int VideoGenerator::loopFor2DGame(){
	int frame_interval;
	struct timeval tv;
	LARGE_INTEGER initialTv, captureTv, freq;
	struct pooldata * data = NULL;
	struct ImageFrame * frame = NULL;

	//pipeline * pipe[SOURCES];

	RTSPConf * rtspConf = NULL;
	rtspConf = RTSPConf::GetRTSPConf();

	if(rtspConf == NULL){
		Log::slog("[VideoGenerator]: get NULL rtspConfig.\n");
		return -1;
	}

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&initialTv);

	frame_interval = 1000000 / rtspConf->video_fps;

	if(getCurSourceType() == SURFACE){
		Log::slog("[VideoGenerator]: 2D game do not support SURFACE.\n");
		return -1;
	}

	// start the filter and encoder?
	if(this->filter){

	}else{
		// null filter, error
		Log::slog("[VideoGenerator]: NULL filter.\n");
		return -1;
	}

	if(this->x264Encoder){

	}else{
		// null encoder, error
		Log::slog("[VideoGenerator]: NULL x264Encoder.\n");
		return -1;
	}

	// enter the loop
	while(isRunning()){

		struct timeval * ptv= &tv;
		getTimeOfDay(ptv, NULL);

		if(!x264Encoder->encoderRunning()){
			Sleep(1);
			continue;
		}

		// copy image
		data = imagePipe[0]->allocate_data();
		frame = (struct ImageFrame *)data->ptr;
		frame->pixelFormat = PIX_FMT_BGRA;

		frame->realWidth = windowRect->width;
		frame->realHeight = windowRect->height;
		frame->realStride = windowRect->width << 2;
		frame->realSize = windowRect->height * frame->realStride;

		frame->lineSize[0] = frame->realStride;
		QueryPerformanceCounter(&captureTv);
		

		windowWrapper->capture((char *)frame->imgBuf, frame->imgBufSize, windowRect);
		ccg_win32_draw_sysytem_cursor(frame);

		frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
		// duplicate frame channel 0 to other channels
		for(int i = 0; i< SOURCES; i++){
			struct pooldata * dupdata;
			struct ImageFrame * dupframe;
			dupdata = imagePipe[i]->allocate_data();
			dupframe = (struct ImageFrame *)dupdata->ptr;

			dupframe->DupFrame(frame, dupframe);

			imageSource[i]->store_data(dupdata);
			imageSource[i]->notify_all();
		}

		imageSource[0]->store_data(data);
		imageSource[0]->notify_all();

		ccg_usleep(frame_interval, &tv);
	}
}

// D3D game is complicated, wrapper must be created and set first
int VideoGenerator::loopForD3DGame(){
	int frame_interval;
	struct timeval tv;
	LARGE_INTEGER initialTv, captureTv, freq;

	RTSPConf * rtspConf = NULL;
	rtspConf = RTSPConf::GetRTSPConf();

	if(rtspConf == NULL){
		Log::slog("[VideoGenerator]: get NULL rtspConfig.\n");
		return -1;
	}

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&initialTv);

	frame_interval = 1000000 / rtspConf->video_fps;

	// enter the loop
	while(isRunning()){

		struct timeval * ptv= &tv;
		getTimeOfDay(ptv, NULL);

		if(!x264Encoder->encoderRunning() || !cudaEncoder->encoderRunning()){
			Sleep(1);
			continue;
		}

#ifdef DEBUG
		if(presentEvent == NULL){
			Log::slog("[VideoGenerator]: NULL present event for 3D games.\n");
		}
#endif
		if(presentEvent)
			WaitForSingleObject(presentEvent, INFINITE);

		QueryPerformanceCounter(&captureTv);
		if(dxVersion == DX9){
			// call D3DWrapper to complete capturing
			doCapture(this->d3dDevice, rtspConf, captureTv, initialTv, freq, frame_interval);
			dxWrapper->d3dWrapper->capture();
		}
		else if(dxVersion == DX10){
			dxWrapper->d3d10Wrapper->capture();
		}else if(dxVersion == DX10_1){
			dxWrapper->d3d10Wrapper1->capture();
		}else if(dxVersion == DX11){
			dxWrapper->d3d11Wrapper->capture();
		}

		if(presentEvent == NULL){
			ccg_usleep(frame_interval, &tv);
		}
	}
}

// for D3D game only
bool VideoGenerator::doCapture(IDirect3DDevice9 * device, RTSPConf * rtspConf, LARGE_INTEGER captureTv, LARGE_INTEGER initialTv, LARGE_INTEGER freq, int frame_interval){
	HRESULT hr;
	IDirect3DSurface9 * capturedSurface = NULL;

	D3DSURFACE_DESC desc;
	D3DLOCKED_RECT lockedRect;
	int i = 0;
	struct pooldata * data = NULL;
	struct ImageFrame * frame = NULL;

	d3dWrapper->captureInit(rtspConf->video_fps);

	capturedSurface = d3dWrapper->capture(this->getCurEncoderType(), this->windowRect->width, this->windowRect->height);
	
	if(this->getCurSourceType() == IMAGE){
	//if(this->sourceType == SOURCE_TYPE::IMAGE){
		// start to lock screen from offline surface
		hr = capturedSurface->LockRect(&lockedRect, NULL, NULL);
		if (FAILED(hr)){
			Log::log("[capture]: LcokRect failed.\n");
			return false;
		}

		// copy image
		do{
			int image_size = 0;
			unsigned char * src, *dst;
			data = imagePipe[0]->allocate_data();
			frame = (ImageFrame *)data->ptr;
			frame->pixelFormat = PIX_FMT_BGRA;
			frame->realWidth = desc.Width;
			frame->realHeight = desc.Height;
			frame->realStride = desc.Width << 2;
			frame->realSize = frame->realWidth * frame->realStride;
			frame->lineSize[0] = frame->realStride;

			src = (unsigned char *)lockedRect.pBits;
			dst = (unsigned char *)frame->imgBuf;

			for (i = 0; i < windowRect->height; i++){
				CopyMemory(dst, src, frame->realStride);
				src += lockedRect.Pitch;
				dst += frame->realStride;
				image_size += lockedRect.Pitch;
			}
			Log::log("[capture]: get image data:%d, height:%d, pitch:%d.\n", image_size, windowRect->height, lockedRect.Pitch);
			frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
		} while (0);

		capturedSurface->UnlockRect();

		// duplicate from channel 0 to other channel
		for (i = 1; i< SOURCES; i++){
			struct pooldata * dupdata;
			struct ImageFrame * dupframe;
			dupdata = imagePipe[i]->allocate_data();
			dupframe = (struct ImageFrame *)dupdata->ptr;

			imagePipe[i]->store_data(dupdata);
			imagePipe[i]->notify_all();
		}
		imagePipe[0]->store_data(data);
		Log::log("[Channel]: source pipeline notify the client.\n");
		imagePipe[0]->notify_all();
		
	}
	else if(this->getCurSourceType() == SURFACE){
	//else if(this->sourceType == SOURCE_TYPE::SURFACE){
		// storee surface to pipeline
		data = cudaPipe[0]->allocate_data();
		SurfaceFrame * sframe = (SurfaceFrame *)data->ptr;
		IDirect3DSurface9 * sur = sframe->getSurface(); // get the surface
		if(sur == NULL){
			// create the surface
			
			hr = device->CreateOffscreenPlainSurface(windowRect->width, windowRect->height, desc.Format, D3DPOOL_DEFAULT, &sur, NULL);
			if (FAILED(hr)){
				Log::log("[capture]: Create offscreen surface failed.\n");
				return false;
			}
			sframe->setSurface(sur, windowRect->width, windowRect->height);
		}
		hr = device->StretchRect(capturedSurface, NULL, sur, NULL, D3DTEXF_LINEAR);
		//hr = device->UpdateSurface(offscreenSurface, NULL, sur, NULL);   // copy the surface
		if(FAILED(hr)){
			Log::log("[Channel]: copy the surface failed.\n");
		}

		for(i = 0; i< SOURCES; i++){
			struct pooldata * dupdata = cudaPipe[i]->allocate_data();
			struct SurfaceFrame * dupframe = (struct SurfaceFrame *)dupdata->ptr;

			IDirect3DSurface9 * sur = sframe->getSurface(); // get the surface
			if(sur == NULL){
				// create the surface

				hr = device->CreateOffscreenPlainSurface(windowRect->width, windowRect->height, desc.Format, D3DPOOL_DEFAULT, &sur, NULL);
				if (FAILED(hr)){
					Log::log("[capture]: Create offscreen surface failed.\n");
					return false;
				}
				sframe->setSurface(sur, windowRect->width, windowRect->height);
			}
			hr = device->StretchRect(capturedSurface, NULL, sur, NULL, D3DTEXF_LINEAR);
			//hr = device->UpdateSurface(offscreenSurface, NULL, sur, NULL);   // copy the surface
			if(FAILED(hr)){
				Log::log("[Channel]: copy the surface failed.\n");
			}

			cudaPipe[i]->store_data(dupdata);
			cudaPipe[i]->notify_all();
		}
		cudaPipe[0]->store_data(data);
		Log::log("[Channel]: source cudapipeline notify the client.\n");
		cudaPipe[0]->notify_all();

	}
	//xxx: disable until we have found a good place to safely release

	return true;
}

#ifndef USE_CTHREAD
// thread proc for video generator, 
DWORD VideoGenerator::VideoGenertorThreadProc(LPVOID param){

	Log::slog("[VideoGenerator]: enter the thread proc.\n");
	VideoGenerator * videoGenerator = (VideoGenerator *)param;


	// start the filter, encoder ?



	if(videoGenerator->isD3DGame()){
		// if it is D3D gane, complicated strategy for wrapper and encoder
		if(videoGenerator->loopForD3DGame()){
			// error
		}

	}
	else{
		// if it is 2D game, it simple, use window wrapper and CPU encoder
		if(videoGenerator->loopFor2DGame()){
			// error
		}

	}
	Log::slog("[VideoGenerator]: exit the video streaming.\n");
	return 0;
}
#endif

// constructor and destructor
VideoGenerator::VideoGenerator(){
	x264Encoder = NULL;
	cudaEncoder = NULL;
	encoder = NULL;
	filter = NULL;
	captureWindow = NULL;
	outsideSocket = NULL;
	winHeight = 0;
	winWidth = 0;
	pSection = NULL;
#if 0
	d3dWrapper = NULL;
#else
	dxWrapper = NULL;
#endif

	windowWrapper = NULL;

	pEncoderSection = NULL;
	pWrapperSection = NULL;
	pRunningSection = NULL;

	windowRect = new ccgRect();

	InitializeCriticalSection(&section);
	pSection = &section;
	InitializeCriticalSection(&encoderSection);
	pEncoderSection = &encoderSection;
	InitializeCriticalSection(&wrapperSection);
	pWrapperSection = &wrapperSection;
	InitializeCriticalSection(&runningSection);
	pRunningSection = &runningSection;

	running = true;
}

VideoGenerator::~VideoGenerator(){
	if(x264Encoder){
		delete x264Encoder;
		x264Encoder = NULL;
	}

	if(cudaEncoder){
		delete cudaEncoder;
		cudaEncoder = NULL;
	}

	if(filter){
		delete filter;
		filter = NULL;
	}

	if(captureWindow){
		
	}

	if(outsideSocket){
		closesocket(outsideSocket);
		outsideSocket = NULL;
	}
	if(pSection){
		DeleteCriticalSection(pSection);
		pSection =  NULL;
	}
#if 0
	if(d3dWrapper){
		delete d3dWrapper;
		d3dWrapper = NULL;
	}
#endif

	if(windowWrapper){
		delete windowWrapper;
		windowWrapper = NULL;
	}

	if(pEncoderSection){
		DeleteCriticalSection(pEncoderSection);
		pEncoderSection = NULL;
	}
	if(pWrapperSection){
		DeleteCriticalSection(pWrapperSection);
		pWrapperSection = NULL;
	}

	if(windowRect){
		delete windowRect;
		windowRect = NULL;
	}

}


int VideoGenerator::initVideoGenerator(){


	// the initializatioin for video generator
	WaitForSingleObject(windowEvent, INFINITE);
	// now the window is created
	RECT wRect;
	GetWindowRect(this->captureWindow, &wRect);

#if 0
	this->winWidth = wRect.right - wRect.left;
	this->winHeight = wRect.bottom - wRect.top;
	if(SetWindowPos(captureWindow, HWND_TOP, 0, 0, 0, SWP_NOSIZE | SWP_SHOWWINDOW) == 0){
		Log::slog("[VideoGenerator]: Set window pos failed.\n");
		return -1;
	}
#else
	setAndSetupWindow(this->captureWindow);
#endif

	// the encoder

	// the filter

	// 

}

// called only by 2D game
int VideoGenerator::setAndSetupWindow(HWND hwnd){
	RECT rect;
	POINT lt, rb;
#if 1
	if(SetWindowPos(hwnd, HWND_TOP, 0, 0, 0, 0, SWP_NOSIZE|SWP_SHOWWINDOW) == 0){
		Log::slog("[VideoGenerator]: SetWindowPos failed.\n");
		return -1;
	}
#endif

	if(GetClientRect(hwnd, &rect) == 0){
		Log::slog("[VideoGenerator]: GetClientRect failed.\n");
		return -1;
	}
	lt.x = rect.left;
	lt.y = rect.top;
	rb.x = rect.right - 1;
	rb.y = rect.bottom - 1;

	if(ClientToScreen(hwnd, &lt) == 0 || ClientToScreen(hwnd, &rb) == 0){
		Log::slog("[VideoGenerator]: map from client coordinate to screen coordingate failed.\n");
		return -1;
	}

	//windowRect = new ccgRect();
	if(windowRect == NULL){
		Log::slog("[VideoGenerator]: window rect is NULL.\n");
		//return -1;
		windowRect = new ccgRect();
	}
	windowRect->left = lt.x;
	windowRect->top = lt.y;
	windowRect->right = rb.x;
	windowRect->bottom = rb.y;

	// size check
	if((windowRect->right - windowRect->left + 1) % 2 != 0)
		windowRect->left -- ;
	if((windowRect->bottom - windowRect->top + 1) % 2 != 0)
		windowRect->top --;

	if(windowRect->left < 0 || windowRect->top < 0){
		Log::slog("[VideoGenerator]: Invalid window: (%d, %d) - (%d, %d).\n", windowRect->left, windowRect->top, windowRect->right, windowRect->bottom);
		return -1;
	}

	// setup the cuda pipeline
	//setupCudaPipeline();

	//setup the image pipeline
	//return setupImagePipeline(this->w);

	return  0;
}

///setup the cuda pipeline 
int VideoGenerator::setupCudaPipeline(){
	struct pooldata * data = NULL;

	for(int i = 0; i< SOURCES; i++){
		if((surfaceSource[i] = new pipeline()) == NULL){
			Log::slog("[VideoGenerator]: init pipeline for cuda failed.\n");
			return -1;
		}


		if((data = surfaceSource[i]->datapool_init(POOLSIZE, sizeof(struct SurfaceFrame))) == NULL){
				Log::slog("[VideoGenerator]: cannot allocate data pool.\n");
				delete surfaceSource[i];
				surfaceSource[i] = NULL;
				return -1;
			}
			// per frame init
			for(; data != NULL; data = data->next){
				struct SurfaceFrame * f = (struct SurfaceFrame *)data->ptr;
				if(f->init() == false){
					Log::slog("[VideoGenerator]: init frame failed.\n");
					return -1;
				}
			}

	}
}

// setup the image pipeline , allocate the pipeline memory
int VideoGenerator::setupImagePipeline(struct ccgImage *image){
	do{

		for(int i = 0; i < SOURCES; i++){
			struct VsourceConfig * conf = (struct VsourceConfig *)malloc(sizeof(struct VsourceConfig));
			struct pooldata * data = NULL;
			
			conf->id = i;
			conf->maxWidth = windowRect ? windowRect->width : image->width;
			conf->maxHeight = windowRect ? windowRect->height : image->height;
			conf->maxStride = windowRect ? windowRect->linesize : image->bytes_per_line;

			if((imageSource[i] = new pipeline()) == NULL){
				Log::slog("[VideoGenerator]: init pipeline failed.\n");
				return -1;
			}

			if(imageSource[i]->alloc_privdata(sizeof(struct VsourceConfig)) == NULL){
				delete imageSource[i];
				imageSource[i] = NULL;
			}

			imageSource[i]->set_privdata(conf, sizeof(struct VsourceConfig));
			// create data pool for the pipe
			if((data = imageSource[i]->datapool_init(POOLSIZE, sizeof(struct ImageFrame))) == NULL){
				Log::slog("[VideoGenerator]: cannot allocate data pool.\n");
				delete imageSource[i];
				imageSource[i] = NULL;
				return -1;
			}
			// per frame init
			for(; data != NULL; data = data->next){
				struct ImageFrame * f = (struct ImageFrame *)data->ptr;
				if(f->init(conf->maxWidth, conf->maxHeight, conf->maxStride) == NULL){
					Log::slog("[VideoGenerator]: init frame failed.\n");
					return -1;
				}
			}

			// register to pipeline

		}
	}while(0);
	return 0;
}
// called by 2d server, with window handle , height and width
// create all the capture, filter and encoder
// call this means that the game is not support d3d.
int VideoGenerator::initVideoGenerator(HWND hwnd, int height, int width){
	this->supportD3D = false;

	this->captureWindow = hwnd;
	this->winHeight = height;
	this->winWidth = width;
	this->outputHeight = height;
	this->outputWidth = width;

	this->windowWrapper->setHWND(hwnd);
	setAndSetupWindow(hwnd);

	ccgImage * image = new ccgImage();

	if(windowWrapper->init(image, hwnd) < 0){
		Log::slog("[VideoGenerator]: window wrapper init failed.\n");
		return -1;
	}

#ifdef SOURCES
	this->setupImagePipline(image);
#endif


	delete image;
	image = NULL;


	// setup the filter

	// setup the x264 encoder
}

int VideoGenerator::setupFilterPipeline(){
	// setup the filter pipeline
	Log::log("[VideoGenerator]: init the filter pipeline.\n");

	struct pooldata * data = NULL;
	VsourceConfig * vsourceConf = (struct VsourceConfig *)this->imageSource[0]->get_privdata();

	int iid = vsourceConf->id;
	int iwidth = vsourceConf->maxWidth;
	int iheight = vsourceConf->maxHeight;
	int istride = vsourceConf->maxStride;

	pipeline * pipe = imageDst[0];

	if(!pipe){
		if((pipe = new Pipeline()) == NULL){
			Log::slog("[VideoGenerator]: cannot init filter pipeline.\n");
			return -1;
		}
	}
	imageDst[0]= pipe;

	vsourceConf->maxHeight = outputHeight;
	vsourceConf->maxWidth = outputWidth;
	vsourceConf->maxStride = outputWidht << 2;
	if(pipe->alloc_privdata(struct VsourceConfig) == NULL){
		Log::slog("[VideoGenerator]: cannot allocate the private data for dst pipeline.\n");
		return -1;
	}
	// set private data (the VsourceConfig)
	pipe->set_privdata((void *)vsourceConf, sizeof(struct VsourceConfig));
	if((data = pipe->datapool_init(POOLSIZE, sizeof(struct ImageFrame))) == NULL){
		Log::slog("[VideoGenerator]: cannot allocate data pool for filter pipeline.\n");
		return -1;
	}
	// per frame init
	for(; data != NULL; data = data->next){
		ImageFrame * frame = (ImageFrame *) data->ptr;
		if(frame->init(outputHeight, outputWidth, vsourceConf->maxStride) == NULL){
			Log::slog("[VideoGenrator]: init frame for filter pipeline failed.\n");

		}
	}

	return 0;
}

// for scale the output only for 2D games, use windowWrapper
int VideoGenerator::initVideoGenerator(HWND hwnd, int h, int w, int outH, int outW){
	this->supportD3D = false;

	this->captureWindow = hwnd;
	this->winHeight = h;
	this->winWidth = w;
	this->outputHeight = outH;
	this->outputWidth = outW;

	this->windowWrapper->setHWND(hwnd);
	setAndSetupWindow(hwnd);

#if 0
	if(!this->windowRect){
		windowRect = new ccgRect();
	}
#else
	windowRect = NULL;
#endif

	ccgImage * image = new ccgImage();

	if(windowWrapper->init(image, hwnd) < 0){
		Log::slog("[VideoGenerator]: window wrapper init failed.\n");
		return -1;
	}

#ifdef SOURCES
	this->setupImagePipeline(image);
#endif

	image-width = w;
	image->height = h;
	image->bytes_per_line = (BITSPERPIXEL >> 3) * w;

	// init the source pipeline here
	this->setupImagePipeline(image);

	// init the destination pipeline here
	this->setupFilterPipeline();

	// setup the filter
	if(!this->filter){
		filter = new Filter();
	}

	// init the filter, source is not regisetered
	if(filter->init( h, w, outH, outW)){
		// init failed
	}
	// regiter pipelines to filter
	filter->registerSourcePipe(imageSource[0]);
	filter->registerFilterPipe(imageDst[0]);

	// setup the x264 encoder
	if(!this->x264Encoder){
		this->x264Encoder = new X264Encoder();
	}
	//setup the encoder, and register the source pipe for encoder
	x264Encoder->InitEncoder(RTSPConf::GetRTSPConf(), this->imageDst[0]);

	delete image;
	image = NULL;
}

// for D3D9, D3D10, D3D10.1 D3D11
// hwnd: is the game window handle
// device: is the device to capture surface
// w, h: is the game window size
// x, y: is the game window position
// outH, outW: is the output window size, that is to say the output image size
int VideoGenerator::initVideoGenerator(HWND hwnd, DX_VERSION dxVer, void * device, int h, int w, int x, int y, int outH, int outW){

	supportD3D = true;
	captureWindow = hwnd;
	winHeight = h;
	winWidth = w;
	this->outputHeight = outH;
	this->outputWidth = outW;

	dxVersion = dxVer;
	ccgImage * image = new ccgImage();
	image->width = w;
	image->height = h;
	image->bytes_per_line = (BITSPERPIXEL >> 3) * w;

	setAndSetupWindow(hwnd);

	// init the pipeline
	//this->setupCudaPipeline();  
	this->setupImagePipeline();   // the source pipeline
	this->setupFilterPipeline();  // the destination pipeline

	// init the wrapper
	switch(dxVer){
	case DX9:

		dxWrapper->d3dWrapper = new D3DWrapper((IDirect3DDevice9 *) device);
		dxWrapper->d3dWrapper->setHWND(hwnd);
		// init wrapper for image

		// init wrapper for surface

		break;
	case DX10:
		// init wrapper for image
		dxWrapper->d3d10Wrapper = new D3D10Wrapper((IDXGISwapChain *)device);
		dxWrapper->d3d10Wrapper->setHWND(hwnd);
		break;
	case DX10_1:
		// init the wrapper for image
		dxWrapper->d3d10Wrapper1 = new D3D10Wrapper1((IDXGISwapChain *)device);
		dxWrapper->d3d10Wrapper1->setHWND(hwnd);
		break;
	case DX11:

		// init the wrapper for image
		dxWrapper->d3d11Wrapper = new D3D11Wrapper((IDXGISwapChain * )device);
		dxWrapper->d3d11Wrapper->setHWND(hwnd);
		break;
	default:
		// use dx9
		this->dxVersion = DX9;
		dxWrapper->d3dWrapper = new D3DWrapper((IDirect3DDevice9 *) device);
		dxWrapper->d3dWrapper->setHWND(hwnd);
		break;
	}

	// setup the filter
	if(!this->filter){
		filter = new Filter();
	}

	// init the filter, source is not regisetered
	if(filter->init( h, w, outH, outW)){
		// init failed
	}
	// regiter pipelines to filter
	filter->registerSourcePipe(imageSource[0]);
	filter->registerFilterPipe(imageDst[0]);

	// setup the x264 encoder
	if(!this->x264Encoder){
		this->x264Encoder = new X264Encoder();
	}
	//setup the encoder, and register the source pipe for encoder
	x264Encoder->InitEncoder(RTSPConf::GetRTSPConf(), this->imageDst[0]);
	if(!this->cudaEncoder){
		this->cudaEncoder = new CudaEncoder();
	}
	// init the cuda encoder
}

// surpport d3d9, but we need to prepare the 2D video streaming, too.
int VideoGenerator::initVideoGenerator(HWND hwnd, IDirect3DDevice9 *device, int height, int width, int x, int y){

	this->supportD3D = true;

	this->captureWindow = hwnd;
	this->winHeight = height;
	this->winWidth = width;

	this->dxWrapper->d3dWrapper->setDevice(device);   // set the device to d3d wrapper
	this->dxWrapper->d3dWrapper->setHWND(hwnd);
	//this->windowWrapper->setHWND(hwnd);    // for 2d video streaming

	if(!windowRect){
		windowRect = new ccgRect();
	}
	windowRect->left = x;
	windowRect->right = x + width;
	windowRect->top = y; 
	windowRect->bottom = y + height;

	ccgImage * image = new ccgImage();
	image->:width = width;
	image->height = height;
	image->bytes_per_line = (BITSPERPIXEL >> 3) * width;

	if(dxWrapper->d3dWrapper->init(image, hwnd) < 0){
		Log::slog("[VideoGenerator]: init d3d wrapper failed.\n");
		return -1;
	}
#ifdef SOURCES
	this->setupImagePipline(image);
	this->setupCudaPipeline();

#endif
	delete image;
	image = NULL;


	// setup the filter

	// setup the x264 encoder

	// setup the cuda encoder

}

#ifndef USE_CTHREAD

void VideoGenerator::startThread(){
	vidoeThreadHandle = chBEGINTHREADEX(NULL, 0, VideoGenertorThreadProc, this, FALSE, &this->videoThreadId);
}
#endif

ENCODER_TYPE VideoGenerator::getCurEncoderType(){
	ENCODER_TYPE ret = ENCODER_TYPE::X264_ENCODER;
	EnterCriticalSection(pEncoderSection);
	ret = this->useEncoderType;
	LeaveCriticalSection(pEncoderSection);
	return ret;
}

SOURCE_TYPE VideoGenerator::getCurSourceType(){
	SOURCE_TYPE ret = SOURCE_TYPE::IMAGE;
	EnterCriticalSection(pWrapperSection);
	ret = this->useSourceType;
	LeaveCriticalSection(pWrapperSection);
	return ret;
}

bool VideoGenerator::isRunning(){
	bool ret = false;

	EnterCriticalSection(pRunningSection);
	ret = this->running;
	LeaveCriticalSection(pRunningSection);
	return ret;
}


// stop the thread manually.
BOOL VideoGenerator::stop(){


}

// the loop logic for video generator, not conclude the msg dealing, 
void VideoGenerator::run(){
	struct pooldata * data = NULL;
	struct ImageFrame * iframe = NULL;
	struct SurfaceFrame * sframe = NULL;

	// manager
	if(getCurEncoderType() == X264_ENCODER){
		if(!x264Encoder->isStart()){
			x264Encoder->start();
		}
	}else if(getCurEncoderType() == CUDA_ENCODER){
		if(!cudaEncoder->isStart()){
			cudaEncoder->start();
		}
	}

	QueryPerformanceCounter(&captureTv);

	if(!this->isD3DGame()){
#if 1
		windowWrapper->capture(captureTv, initialTv, freq, frameInterval);
#else
		// 2D game, time driven
		data = imageSource[0]->allocate_data();
		iframe = (struct ImageFrame *)data->ptr;
		iframe->pixelFormat = PIX_FMT_BGRA;

		iframe->realWidth = windowRect->width;
		iframe->realHeight = windowRect->height;
		iframe->realStride = windowRect->width << 2;
		iframe->realSize = windowRect->height * iframe->realStride;
		iframe->lineSize[0] = iframe->realStride;

		windowWrapper->capture((char *)iframe->imgBuf, iframe->imgBufSize, windowRect);
		ccg_win32_draw_system_cursor(iframe);
		iframe->imgPts = pcdiff_us(captureTv, initialTv, freq)/ frameInterval;
		// duplicate frame channel 0 to other channels
		for(int i = 0; i < SOURCES; i++){
			struct pooldata * dupdata = NULL;
			struct ImageFrame * dupframe = NULL;
			dupdata = imageSource[i]->allocate_data();
			dupframe = (struct ImageFrame *)dupdata->ptr;

			dupframe->DupFrame(iframe, dupframe);
			imageSource[i]->store_data(dupdata);
			imageSource[i]->notify_all();
		}
		imageSource[0]->store_data(data);
		imageSource[0]->notify_all();

		ccg_usleep(frameInterval, &tv);
#endif
	}
	else{
		// 3D games
		if(!x264Encoder->encoderRunning() || !cudaEncoder->encoderRunning()){
			Sleep(1);
			return;
		}

		// set the wrapper source type
		if(useSourceType == SOURCE_TYPE::IMAGE){
			Wrapper * base = dxWrapper->d3d10Wrapper;
			base->setSourceType(useSourceType);
		}else if(useSourceType == SOURCE_TYPE::SURFACE){
			Wrapper * base = dxWrapper->d3d10Wrapper;
			base->setSourceType(useSourceType);
		}
		//
#ifdef DEBUG
		if(presentEvent == NULL){
			Log::slog("[VideoGenerator]: NULL present event for 3D game.\n");
		}
#endif
		if(presentEvent){
			WaitForSingleObject(presentEvent, 30);
		}

		
		if(dxVersion == DX9){
			// call D3DWrapper to complete capturing
			doCapture(this->d3dDevice, rtspConf, captureTv, initialTv, freq, frameInterval);
			//dxWrapper->d3dWrapper->capture();
		}else if(dxVersion == DX10){
			dxWrapper->d3d10Wrapper->capture(captureTv, initialTv, freq, frameInterval);
		}else if(dxVersion == DX10_1){
			dxWrapper->d3d10Wrapper1->capture(captureTv, initialTv, freq, frameInterval);
		}else if(dxVersion == DX11){
			dxWrapper->d3d11Wrapper(captureTv, initialTv, freq, frameInterval);
		}
		
	}
}


// deal the thread message, it the major modification for video generator
void VideoGenerator::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){

	// start the modules for initialization
	switch(msg){
	case USER_VIDEO_RENDER:   // TODO: delete this, never used
		// the render event, add/decline
		break;

	case USER_VIDEO_FILTER:
		// filter event usually 
		lockFilter();

		unlockFilter();
		break;
	case USER_VIDEO_WRAPPER:
		lockWrapper();

		unlockWrapper();
		break;
	case USER_VIDEO_ENCODER:
		//change the encoder
		// maybe trigger the filter and wrapper event
		if(wParam = CHANGE_ENCODER){
			lockEncoder();

			if(this->getCurEncoderType() == ENCODER_TYPE::X264_ENCODER){
				// change to cuda encoder
				changeEncoderDevice(CUDA_ENCODER);

			}else if(this->getCurEncoderType() == CUDA_ENCODER){
				// change to x264Encoder
				changeEncoderDevice(X264_ENCODER);
			}

			unlockEncoder();
		}
		else if(wParam == START_ENCODER){
			// 
		}
		break;
	default:

		break;
	}
}

// qutit the thread
void VideoGenerator::onQuit(){

}

// start the modules in the generator.
int VideoGenerator::startModules(){
	if(useEncoderType == X264_ENCODER){
		if(x264Encoder){
			// check filter
			if(filter == NULL){
				// create the filter

			}
			if(filter == NULL){
				Log::slog("[VideoGenerator]: startModules get NULL filter for X264Encoder.\n");
				return -1;
			}
			//
			if(!filter->isRunning()){
				// start the filter
				if(!filter->isInited()){

				}
				if(!filter->isRunning()){
					filter->start();
				}
			}
			if(!x264Encoder->isStart()){
				// start the encoder
				x264Encoder->start();
			}
		}
		else{
			//error
			Log::slog("[VideoGenerator]: startModules get NULL x264Encoder.\n");
			return -1;
		}
	}
	else if(useEncoderType == CUDA_ENCODER){
		if(cudaEncoder == NULL){
			Log::slog("[VidoeGenerator]: startModules get NULL cudaEncoder.\n");
			return -1;
		}
		if(!cudaEncoder->isStart()){
			// start the encoder
			cudaEncoder->start();
		}
	}

	if(supportD3D){
		// use DXWrapper
		// check wrapper
		if(dxWrapper->d3dWrapper == NULL){
			Log::slog("[VideoGenerator]: startModules get NULL DXWrapper.\n");
			return -1;
		}

	}else{
		// use windowWrapper
		if(windowWrapper == NULL){
			Log::slog("[VideoGenerator]: startModules get NULL windowWrapper.\n");
		}
	}
}


// the initialize procedure for video generator thread
void VideoGenerator::onThreadStart(LPVOID param){

	// check the pipeline
	if(!imageSource[0] || !imageDst[0]){
		Log::slog("[VideoGenerator]: image source or destination pipeline is NULL.\n");
	}
	

	// start the modules
	startModules();

	if(this->isD3DGame()){
		if(!surfaceSource[0] || !surfaceDst[0]){
			Log::slog("[VideoGenerator]: surface source or destination pipeline is NULL.\n");
		}
	}
	else{
		// 2D games
		if(getCurSourceType() == SURFACE){
			Log::slog("[VideoGenerator]: 2D game does not support SURFACE.\n");
			return;
		}
	}

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&initialTv);
}


// change the encoder device, along with the filter, cause, the cuda filter may not gain performance improvement.
int VideoGenerator::changeEncoderDevice(ENCODER_TYPE type){
	useEncoderType = type;
	if(type == X264_ENCODER){
		if(this->filter){
			if(!filter->isInited()){

			}
			if(!filter->isRunning()){
				filter->StartFilterThread();
			}
		}else{
			// error
			Log::slog("[VideoGenerator]: use x264Encoder with NULL filter.\n");
			return -1;
		}
		if(!x264Encoder->isStart()){
			x264Encoder->start();
		}
	}else if(type == CUDA_ENCODER){
		if(!cudaEncoder->isStart()){
			cudaEncoder->start();
		}
	}else{
		// invalid
		Log::slog("[VideoGenerator]: invalid encoder type to change.\n");
		return -1;
	}
	return 0;
}