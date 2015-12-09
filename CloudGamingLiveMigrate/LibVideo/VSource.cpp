#include "Commonwin32.h"
#include "Pipeline.h"
#include "Config.h"
#include "RtspConf.h"
#include "RtspContext.h"
#include "Encoder.h"
#include "FilterRGB2YUV.h"
#include "VSource.h"

#include "CudaEncoder.h"
#include "X264Encoder.h"
#include "EncoderManager.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"

#include "Wrapper.h"

#include <d3d10.h>
#include <dxgi.h>


//const char * pipeformat = "pipe%d";
extern const char * imagepipefmt;
extern const char * filterpipefmt;
//static IDirect3D9 * g_pD3D = NULL;
//extern HANDLE presentMutex;

HANDLE Channel::presentMutex;
HANDLE Channel::initMutex;

void Channel::Release(){
	if (initMutex){
		CloseHandle(initMutex);
		initMutex = NULL;
	}
	///IDXGIOutput * duplicator;
	//dumplicator->
}

Channel::~Channel(){

	for (int i = 0; i < SOURCES; i++){
		if (pipe[i]){
			delete pipe[i];
			pipe[i] = NULL;
		}
	}
	if (encoder){
		delete encoder;
		encoder = NULL;
	}
	if (source){
		delete source;
		source = NULL;
	}
	if (wrapper){
		delete wrapper;
		wrapper = NULL;
	}
	if (filter){
		delete filter;
		filter = NULL;
	}
	if (imagepipename){
		free(imagepipename);
		imagepipename = NULL;
	}
	if (filterpipename){
		free(filterpipename);
		filterpipename = NULL;
	}
	if (screenRect){
		free(screenRect);
		screenRect = NULL;
	}
	if (cimage){
		free(cimage);
		cimage = NULL;
	}

	if (deviceEvent){
		CloseHandle(deviceEvent);
		deviceEvent = NULL;
	}
	if (windowHandleEvent){
		CloseHandle(windowHandleEvent);
		windowHandleEvent = NULL;
	}

}

Channel::Channel(){

	dxVersion = DXNONE;

	this->channelId = -1;
	for (int i = 0; i < SOURCES; i++)
		pipe[i] = NULL;
	sourceType = SOURCE_TYPE::SOURCE_NONE;

	encoder = NULL;
	source = NULL;
	wrapper = NULL;
	filter = NULL;

	maxWidth = 0;
	maxHeight = 0;
	maxStride = 0;

	cimage = NULL;
	//context = NULL;
	screenRect = NULL;
	windowHwnd = NULL;

	//sourcePipeName = NULL;
	//filterPipeName = NULL;
	imagepipename = NULL;
	filterpipename = NULL;

	deviceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	windowHandleEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	if (presentMutex == NULL){
		CreateMutex(NULL, FALSE, NULL);
	}

}

Channel::Channel(ENCODER_TYPE type){ 

	dxVersion = DXNONE;

	encoderType = type; 
	this->channelId = -1;
	for (int i = 0; i < SOURCES; i++)
		pipe[i] = NULL;
	//sourceType = SOURCE_TYPE::SOURCE_NONE;

	encoder = NULL;
	source = NULL;
	wrapper = NULL;
	filter = NULL;

	maxWidth = 0;
	maxHeight = 0;
	maxStride = 0;

	cimage = NULL;
	//context = NULL;
	screenRect = NULL;
	windowHwnd = NULL;

	//sourcePipeName = NULL;
	//filterPipeName = NULL;
	imagepipename = NULL;
	filterpipename = NULL;

	deviceEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	windowHandleEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	if (presentMutex == NULL){
		CreateMutex(NULL, FALSE, NULL);
	}
}
Channel::Channel(const Channel & ch){
	dxVersion = ch.dxVersion;
	channelId = ch.channelId;
	for (int i = 0; i < SOURCES; i++)
		pipe[i] = NULL;
	sourceType = ch.sourceType;

	encoder = ch.encoder;
	source = ch.source;
	wrapper = ch.wrapper;

	maxWidth = ch.maxWidth;
	maxHeight = ch.maxHeight;
	maxStride = ch.maxStride;

	deviceEvent = ch.deviceEvent;
	windowHandleEvent = ch.windowHandleEvent;
}

Channel & Channel::operator=(const Channel &ch){
	dxVersion = ch.dxVersion;
	channelId = ch.channelId;
	for (int i = 0; i < SOURCES; i++)
		pipe[i] = ch.pipe[i];
	sourceType = ch.sourceType;

	encoder = ch.encoder;
	source = ch.source;
	wrapper = ch.wrapper;

	maxWidth = ch.maxWidth;
	maxHeight = ch.maxHeight;
	maxStride = ch.maxStride;
	return *this;
}

const char * Channel::getPipeName(){
	//return pipename;
	return NULL;
}

// reconstruct the code
bool Channel::setDevice(DX_VERSION version, void *data){
	if(version == DXNONE || data == NULL){
		dxVersion = DXNONE;
		infoRecorder->logError("[Channel]: set device with invalid version or invalid device.\n");
		return false;
	}
	else{
		dxVersion = version;
		dxEntry = data;
		if(dxVersion == DX9){
			this->device = (IDirect3DDevice9 *)data;
		}
		return true;
	}
}


// must call setDevice, setDXVersion, setWindowHandle first
void Channel::waitForDeviceAndWindowHandle(){
	infoRecorder->logError("[Channel]: wait for the device event and the window handle event.\n");
	// wait for the window handle and the d3d device event.
	HANDLE waitFor[2] = { deviceEvent, windowHandleEvent };
	WaitForMultipleObjects(2, waitFor, TRUE, INFINITE);

	if(dxVersion == DX9){
		wrapper = new D3DWrapper(sourceType);
		if(device){
			wrapper->setDevice(device);
			getD3DResolution();
		}else{
			infoRecorder->logTrace("[Channel]: d3d device is NULL when registering to channel.\n");
			return;
		}
		registerD3DWrapper(wrapper);
		//set the dxWrapper
		dxWrapper->d9Wrapper = wrapper;
	}
	else if(dxVersion == DX10 ){
		dxWrapper->d10Wrapper = new D3D10Wrapper((IDXGISwapChain *)this->dxEntry);

	}else if(dxVersion == DX10_1){
		dxWrapper->d10Wrapper1 = new D3D10Wrapper1((IDXGISwapChain *)dxEntry);
	}else if(dxVersion == DX11){
		// set the IDXGISwapChain 
		dxWrapper->d11Wrapper = new D3D11Wrapper((IDXGISwapChain *)dxEntry);

	}
	else{
		infoRecorder->logTrace("[Channel]: directx version is not supported.\n");

	}

}

//// use the image source as default
bool Channel::init(int maxWidth, int maxHeight, int maxStride, SOURCE_TYPE sourceType){
	this->maxWidth = maxWidth;
	this->maxHeight = maxHeight;
	this->maxStride = maxStride;
	this->sourceType = sourceType;

	// check the hardware
	int cuda = Encoder::CheckDevice();
	switch (cuda){
	case 0:
	default:
		break;
	}
	// check encoder
	if (!encoder){
		infoRecorder->logTrace("create a new encoder for channel %d\n", this->channelId);
		//create and init
		switch (encoderType){
		case X264_ENCODER:
			encoder = new X264Encoder();
			break;
		case CUDA_ENCODER:
			encoder = new CudaEncoder();
			break;
		default:
			encoderType = X264_ENCODER;
			encoder = new X264Encoder();
			break;
		}
	}

	// check source
	if (!source){
		infoRecorder->logTrace("create a new source for channel %d\n", this->channelId);
		// create and init
		switch (sourceType){
		case IMAGE:
			source = new ImageFrame();
			break;
		case SURFACE:
			source = new SurfaceFrame(); 
			break;
		default:
			sourceType = IMAGE;
			source = new ImageFrame();
			break;
		}
		// init the source
		// source->init();
	}

	// check wrapper
	if (!wrapper){
		infoRecorder->logTrace("create a new wrapper for channel %d\n", this->channelId);
		// create and init
		wrapper = new D3DWrapper(sourceType);
	}

	return true;
}

bool Channel::init(ENCODER_TYPE encoderType, SOURCE_TYPE type){
	// create the encoder and the source according to the types
	Log::logscreen("[channel]: channel init called.\n");
	this->sourceType = type;
	this->encoderType = encoderType;

	Encoder * e = this->encoder;
	if (type == SOURCE_TYPE::SOURCE_NONE){

	}

	EncoderManager * encoderManager = EncoderManager::GetEncoderManager();
	// check the hardware
	int cuda = encoderManager->checkEncodingDevice();
	switch (cuda){
	case SUPPORT_CUDA:
		encoderType = CUDA_ENCODER;
		// notice that, cuda encoder can take images and surface, we need to create a encoder according to GPU usage
		break;
	case SUPPORT_NVENC:

		break;
	default:
		encoderType = X264_ENCODER;
		break;
	}

	// check encoder
	if (!e){
		infoRecorder->logTrace("create a new encoder for channel %d\n", this->channelId);
		//create and init
		switch (encoderType){
		case X264_ENCODER:
			sourceType = SOURCE_TYPE::IMAGE;
			break;
		case CUDA_ENCODER:
			sourceType = SOURCE_TYPE::SURFACE;
			break;
		default:
			encoderType = X264_ENCODER;
			sourceType = SOURCE_TYPE::IMAGE;
			break;
		}
		e = encoderManager->getEncoder(encoderType);
		if (!e){
			Log::logscreen("[channel]: NULLL encoder from encodermanager.\n");
		}
	}
	else{
		infoRecorder->logTrace("encoder exist!\n");

	}
	// register the encoder to channel
	encoder = e;	

	// check wrapper
	if (!wrapper){
		infoRecorder->logTrace("create a new wrapper for channel %d\n", this->channelId);
		// create and init
		wrapper = new D3DWrapper(sourceType);
	}

	// check source
	infoRecorder->logTrace("[Channel]: pipeline for channel:%p.\n", pipe[0]);

	// init the source pipeline for channel
	if (!pipe[0]){
		if(initMutex = NULL){
			// create the init mutex for channel
			initMutex = CreateMutex(NULL, FALSE, NULL);
		}
		infoRecorder->logError("create a new source for channel %d\n", this->channelId);
		// create and init
		switch (sourceType){
		case IMAGE:
			// init the image source

			// init wrapper screen rect
			cimage = new ccgImage();
#if 0
			if(wrapper->init(cimage, windowHwnd) < 0){
				infoRecorder->logError("CHANNEL: init wrapper failed\n");
				return -1;
			}
			else{
				infoRecorder->logTrace("[Channel]: cimage (%d x %d).\n", cimage->width, cimage->height);
			}
#else
			// init the source pipe
			if (wrapper->init(cimage, game_width, game_height) < 0){
				infoRecorder->logTrace("[Channel]: init wrapper failed.\n");
				return false;
			}
			else{
				infoRecorder->logTrace("[Channel]: cimage (%d x %d).\n", cimage->width, cimage->height);
			}
#endif
#ifdef SOURCES
			do{
				int i = 0;
				// construct a VsourceConfig
				VsourceConfig config[SOURCES];
				bzero(&config, sizeof(config));
				for (i = 0; i < SOURCES; i++){
					config[i].rtpId = i;
					config[i].maxWidth = cimage->width;
					config[i].maxHeight = cimage->height;
					config[i].maxStride = cimage->bytes_per_line;
				}
				// setup
				if (setup(imagepipefmt, config, SOURCES) < 0){
					return false;
				}
			}
			while (0);
#endif
			break;
		case SURFACE:
			// init the surface source





			break;
		default:
			sourceType = IMAGE;
			source = new ImageFrame();
			break;
		}
	}

	Log::logscreen("[channel]: channel pipe:%p.\n", pipe);
	// filter regiter the channel's source pipeline
#if 0
	if(filter){
		char ** name = (char **)malloc(sizeof(char *) *2);
		name[0] = getChannelPipeName();
		name[1] = _strdup("filter1");
		//filter->registerSourcePipe(pipe);
		Log::logscreen("[channel]: init filter with pipe name:%s filter name:%s.\n", name[0], name[1]);
		int ret = filter->init((void *)name); // init the filter for the x264 encoder

	}
#else
	if (filter){
		infoRecorder->logTrace("[Channel]:init the filter.\n");
		//char *name[2];
		//sprintf(name[0], )
		name[0] = imagepipename;
		name[1] = filterpipename;
		infoRecorder->logTrace("[channel]: param for filter: name[0]:%s, name[1]:%s.\n", name[0], name[1]);
		int ret = filter->init((void *)name); // init the filter for the x264 encoder
		if (ret != -1){
			// filter init succeeded
			//filter->StartFilterThread((void *)name);
		}
		//ch->setFilter(filter);  // set the filter for the channel
	}
#endif
	else{
		Log::logscreen("[channel]: null filter in channel.\n");
	}
	if (encoder){
		infoRecorder->logTrace("[channel]: init encoder\n");
		if (filter)
			encoder->init(this->rtspConf, filter->getFilterPipe());
		else
			encoder->init(this->rtspConf, NULL);

		encoder->setSrcPipeName(filterpipename);
	}
	// encoder regiter the filter's pipeline
	Log::logscreen("[chanel]: get the filter pipe:%p.\n", filter->getFilterPipe());
	infoRecorder->logTrace("[chanel]: get the filter pipe:%p.\n", filter->getFilterPipe());
#if 0
	if(encoder->registerSourcePipe(filter->getFilterPipe()) == false){
		// register the source failed.
		infoRecorder->logError("[channel]:ERROR: encoder register the source pipe failed.\n");
		return false;
	}
#endif

	onThreadStart(NULL);
	return true;
}
void Channel::startFilter(){
	if (filter){
		infoRecorder->logTrace("[Channel]:init the filter.\n");
		//char *name[2];
		//sprintf(name[0], )
		name[0] = imagepipename;
		name[1] = filterpipename;
		infoRecorder->logTrace("[channel]: param for filter: name[0]:%s, name[1]:%s.\n", name[0], name[1]);

		// filter init succeeded
		filter->StartFilterThread((void *)name);

	}
}
// this setup use the cuda pipe for the cuda encoder, 
// wrap the surface
int Channel::setup(const char* pipeformat){
	if (channelId > IMAGE_SOURCE_CHANNEL_MAX) {
		infoRecorder->logTrace("image source: too many sources (%d > %d)\n",
			channelId, IMAGE_SOURCE_CHANNEL_MAX);
		return -1;
	}
	return 0;
}
int Channel::setup(const char* pipeformat, struct VsourceConfig * config, int nConfig){
	int idx;
	//
	if (config == NULL || nConfig <= 0) {
		infoRecorder->logTrace("image source: invalid image source configuration ( %d %p)\n",
			nConfig, config);
		return -1;
	}
	if (nConfig > IMAGE_SOURCE_CHANNEL_MAX) {
		infoRecorder->logTrace("image source: too many sources (%d > %d)\n",
			nConfig, IMAGE_SOURCE_CHANNEL_MAX);
		return -1;
	}

	//channelId = id;
	for (idx = 0; idx < nConfig; idx++){
		struct pooldata *data = NULL;

		maxWidth = config[idx].maxWidth;
		maxHeight = config[idx].maxHeight;
		maxStride = config[idx].maxStride;

		width[idx] = maxWidth;
		height[idx] = maxHeight;
		stride[idx] = maxStride;

		char pipename[64];

		// create pipe
		if ((pipe[idx] = new pipeline()) == NULL) {
			infoRecorder->logTrace("image source: init pipeline failed.\n");
			return -1;
		}
		if (pipe[idx]->alloc_privdata(sizeof(struct VsourceConfig)) == NULL) {

			infoRecorder->logTrace("image source: cannot allocate private data.\n");
			delete pipe[idx];
			pipe[idx] = NULL;
			return -1;
		}
		config[idx].id = idx;
		pipe[idx]->set_privdata(&config[idx], sizeof(struct VsourceConfig));
		// create data pool for the pipe
		// fing he source type, if X264, use ImageFrame, and if CUDA, use SurfaceFrame
		if ((data = pipe[idx]->datapool_init(POOLSIZE, sizeof(ImageFrame))) == NULL) {
			infoRecorder->logTrace("image source: cannot allocate data pool.\n");
			delete pipe[idx];
			pipe[idx] = NULL;
			return -1;
		}
		// per frame init
		for (; data != NULL; data = data->next) {
			// init the source
#if 0
			if (source->init((struct vsource_rame*) data->ptr, maxWidth, maxHeight, maxStride) == NULL) {
				infoRecorder->logTrace("image source: init frame failed.\n");
				return -1;
			}
#else
			ImageFrame * f = (ImageFrame *)data->ptr;
			if (f->init(maxWidth, maxHeight, maxStride) == NULL){
				infoRecorder->logError("channel init : init frame falied.\n");
				return -1;
			}
#endif
		}
		//
		snprintf(pipename, sizeof(pipename), pipeformat, idx);
		if (pipeline::do_register(pipename, this->pipe[idx]) < 0) {
			infoRecorder->logTrace("image source: register pipeline failed (%s)\n",
				pipename);
			return -1;
		}
	}
	gSources = idx;
	return 0;
}

unsigned char * Channel::getImage(){
	return ((ImageFrame*)source)->imgBuf;
}

IDirect3DSurface9 * Channel::getSurface(){
	return ((SurfaceFrame *)source)->dxSurface->d9Surface;
}

bool Channel::registerD3DWrapper(D3DWrapper * wrapper){
	if (this->wrapper){
		infoRecorder->logError("CHANNEL ERROR: wrapper not null \n");
		delete this->wrapper;
		this->wrapper = wrapper;
		//return false;
	}

	this->wrapper = wrapper;
	// check the encoder type and set the source type
	if (this->encoder){
		if (this->encoder->getEncoderType() == X264_ENCODER){
			wrapper->setSourceType(SOURCE_TYPE::IMAGE);
		}
		else if (this->encoder->getEncoderType() == CUDA_ENCODER){
			wrapper->setSourceType(SOURCE_TYPE::SURFACE);
		}
		else{
			infoRecorder->logError("CHANNEL ERROR: set the wrapper type failed, unknown type of encoder type\n");
			return false;
		}
	}
	return true;
}

bool Channel::registerEncoder(Encoder * encoder){
	if (this->encoder){
		infoRecorder->logError("[channel]:CHANNEL ERROR: encoder not null!\n");
		return false;
	}
	else{
		this->encoder = encoder;
		if (this->rtspConf){
			encoder->setRTSPConf(rtspConf);
		}
		else{
			infoRecorder->logError("[channel]: regiseter encoder without rtps config.\n");
		}
		return true;
	}
}

// the thread proc for 2d games
DWORD Channel2D::Chanel2DThreadProc(LPVOID param){

	Channel2D * ch = (Channel2D *)param;

	int frame_interval;
	LARGE_INTEGER initialTv, captureTv, freq;
	struct timeval tv, * ptv = &tv;

	struct pooldata * data = NULL;
	struct ImageFrame * frame = NULL;
	pipeline * pipe[SOURCES];

	//LARGE_INTEGER initialTv, captureTv, freq;

	struct RTSPConf * rtspConf = ch->getRtspConf();
	frame_interval = 1000000 / rtspConf->video_fps;   // in the unif of us
	frame_interval ++;

	int i = 0;
	for(i = 0; i < SOURCES; i++){
		char pipename[64];
		snprintf(pipename, sizeof(pipename), imagepipefmt, i);
		if((pipe[i] = pipeline::lookup(pipename)) == NULL){
			infoRecorder->logError("[Channel2D]: image source, cannot find the pipeline '%s'.\n", pipename);
			exit(-1);
		}
	}

	infoRecorder->logError("[Channel2D]: iamge source thread started, tid = %ld.\n", ccg_gettid());

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&initialTv);

	while(true){
		getTimeOfDay(ptv, NULL);
		if(!ch->encoder->encoderRunning()){
			Sleep(1);
			continue;
		}
		// copy image 
		data = pipe[0]->allocate_data();
		frame = (struct ImageFrame *)data->ptr;
		frame->pixelFormat = PIX_FMT_BGRA;


		if(ch->windowRect == NULL){
			//// use the screen rect
			frame->realWidth = ch->screenRect->width;
			frame->realHeight = ch->screenRect->height;
			frame->realStride = ch->screenRect->width<<2;
			frame->realSize = ch->screenRect->height * frame->realStride;
		}else{
			///
			frame->realWidth = ch->windowRect->width;
			frame->realHeight = ch->windowRect->height;
			frame->realStride = ch->windowRect->width<<2;
			frame->realSize = ch->windowRect->height * frame->realStride;	
		}

		frame->lineSize[0] = frame->realStride;
		// get the source
		QueryPerformanceCounter(&captureTv);

		ch->wrapper->capture((char *)frame->imgBuf, frame->imgBufSize, ch->windowRect);

		ccg_win32_draw_sysytem_cursor(frame);
		frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
		// duplicate from channel 0 to other channels

		for(i = 1; i< SOURCES; i++){
			struct pooldata * dupdata;
			struct ImageFrame * dupframe;
			dupdata = pipe[i]->allocate_data();
			dupframe = (struct ImageFrame*)dupdata->ptr;

			ImageFrame::DupFrame(frame, dupframe);

			pipe[i]->store_data(dupdata);
			pipe[i]->notify_all();
		}

		pipe[0]->store_data(data);
		pipe[0]->notify_all();

		ccg_usleep(frame_interval, &tv);
	}
	infoRecorder->logError("[Channel2D]: image capture thread terminated.\n");
	return 0;

}


// init the thread
void Channel::onThreadStart(LPVOID param){
	//struct RTSPConf * conf = NULL;
	if (rtspConf == NULL){
		infoRecorder->logError("CHANNEL ERROR: get NULL rtspConfig\n");
		return;
	}
	const char * pipeformat = imagepipefmt;   // get the pipeline name format for channel, like 'image-0'

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&initialTv);

	frame_interval = 1000000 / rtspConf->video_fps;
	frame_interval++;

	for (int i = 0; i < SOURCES; i++){
		char pipename[64];
		snprintf(pipename, sizeof(pipename), pipeformat, i);
		if ((pipe[i] = pipeline::lookup(pipename)) == NULL){
			infoRecorder->logTrace("[Channel]: cannot find pipeline '%s'\n", pipename);
			exit(-1);
		}
	}
	infoRecorder->logTrace("[Channel]: video fps = %d.\n", rtspConf->video_fps);
	// start the encoder assigned to channel
	//encoder->start();
	infoRecorder->logTrace("[Channel]: encoder_width:%d, encoder_height:%d.\n", encoder_width, encoder_height);
	vsource_initialized = 1;
	encoder->encoder_width = encoder_width;
	encoder->encoder_height = encoder_height;
}
// the thread function for the channel
void Channel::run(){
	struct timeval * ptv = &tv;
	getTimeOfDay(ptv, NULL);
	if (encoder->encoderRunning()){
		Sleep(1);
		return;
	}
	// copy image
	// get the source

	if (presentEvent){
		infoRecorder->logTrace("[Channel]: wait for present event.\n");
		WaitForSingleObject(presentEvent, INFINITE);
	}
	else{
		infoRecorder->logTrace("[Channel]: channel get NULL present event.\n");
		return;
	}

	if (sourceType == SOURCE_TYPE::IMAGE){
		infoRecorder->logError("[channel]: run ... get the image frame...\n");
		//if (encoderType == ENCODER_TYPE::X264_ENCODER){
		// copy image
#if 0
		struct pooldata *data = pipe[0]->allocate_data();
		ImageFrame *frame = (ImageFrame *)data->ptr;
		frame->pixelFormat = PIX_FMT_BGRA;
		frame->realWidth = wrapper->screenWidth;
		frame->realHeight = wrapper->screenHeight;
		frame->realStride = wrapper->screenWidth << 2;
		frame->realSize = wrapper->screenHeight * frame->realStride;
		frame->lineSize[0] = frame->realStride;

		QueryPerformanceCounter(&captureTv);

		// capture
		int getsize = wrapper->capture((char*)frame->imgBuf, frame->getImgBufSize(), &wrapper->rect);
		infoRecorder->logTrace("[Channel]: d3d wrapper get imge, size:%d\n.", getsize);
		ccg_win32_draw_system_cursor(frame);

		frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
		for (int i = 1; i < SOURCES; i++){
			struct pooldata * dupdata;
			struct ImageFrame* dupframe;
			dupdata = pipe[i]->allocate_data();
			dupframe = (ImageFrame *)dupdata->ptr;

			pipe[i]->store_data(data);
			pipe[i]->notify_all();
		}
		pipe[0]->store_data(data);
		pipe[0]->notify_all();

		//notify
		//pipe[0]->releasePipeMutex();

		//ccg_usleep(frame_interval, &tv);
#else
		WaitForSingleObject(presentMutex, INFINITE);
		doCapture(device);
		ReleaseMutex(presentMutex);
#endif
	}
	else if (sourceType == SOURCE_TYPE::SURFACE){
		// can only use the cuda encoder 
		// get the surface pointer and store

		struct pooldata * data = pipe[0]->allocate_data();
		SurfaceFrame * frame = (SurfaceFrame *)data->ptr;
		QueryPerformanceCounter(&captureTv);

		// capture
		IDirect3DSurface9 * surface = wrapper->capture(this->encoderType, game_width, game_height);
		frame->dxSurface->d9Surface = surface;
		frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
		pipe[0]->store_data(data);
		ccg_usleep(frame_interval, &tv);
	}
}

BOOL Channel::stop(){
	BOOL ret = CThread::stop();

	return ret;
}

void Channel::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
	CThread::onThreadMsg(msg, wParam, lParam);
}

void Channel::onQuit(){
	CThread::onQuit();
}


bool Channel::doCapture(IDirect3DDevice9 * device){

	HRESULT hr;
	IDirect3DSurface9 * capturedSurface = NULL;

	D3DSURFACE_DESC desc;
	D3DLOCKED_RECT lockedRect;
	int i = 0;
	struct pooldata * data = NULL;
	struct ImageFrame * frame = NULL;

	wrapper->captureInit(rtspConf->video_fps);

	capturedSurface = wrapper->capture(this->encoderType, this->game_width, this->game_height);

#if 0

	static int frame_interval;
	static LARGE_INTEGER initialTv, captureTv, freq;
	static int capture_initialized = 0;

	HRESULT hr;
	D3DSURFACE_DESC desc;
	IDirect3DSurface9 * renderSurface = NULL, *oldRenderSurface = NULL;
	D3DLOCKED_RECT lockedRect;
	int i = 0;
	struct pooldata * data = NULL;
	struct ImageFrame * frame = NULL;

	if (vsource_initialized == 0){
		return false;
	}
	infoRecorder->logTrace("[Channel]: capture the screen.\n");
	renderSurface = oldRenderSurface = NULL;

	hr = device->GetRenderTarget(0, &renderSurface);
	if (FAILED(hr)){
		if (hr == D3DERR_INVALIDCALL){
			infoRecorder->logTrace("[capture]: GetRenderTarget failed (INVALIDCALL).\n");
		}
		else if (hr == D3DERR_NOTFOUND){
			infoRecorder->logTrace("[capture]: GetRenderTarget failed (D3DERR_NOTFOUND).\n");
		}
		else{
			infoRecorder->logTrace("[caputre]: GetRenderTarget failed. (Other).\n");
		}
	}
	if (renderSurface == NULL){
		infoRecorder->logTrace("[capture]: renderSurface is NULL.\n");
		return false;
	}

	renderSurface->GetDesc(&desc);

	if (desc.Width != game_width
		|| desc.Height != game_height){
			infoRecorder->logTrace("[capture]: game width and height are not match !\n");
			return false;
	}

	if (capture_initialized == 0){
		frame_interval = 1000000 / rtspConf->video_fps; // in the unif of us
		frame_interval++;
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&initialTv);
		capture_initialized = 1;
	}
	else{
		QueryPerformanceCounter(&captureTv);
	}

	// check if the surfaceo of local game enable multisampling 
	// multisampling enable will avoid locking in the surface
	// if yes, create a no-multisampling surface and copy frame into it
	if (desc.MultiSampleType != D3DMULTISAMPLE_NONE){
		if (resolvedSurface == NULL){
			hr = device->CreateRenderTarget(game_width, game_height, desc.Format, D3DMULTISAMPLE_NONE,
				0, FALSE, &resolvedSurface, NULL);
			if (FAILED(hr)){
				infoRecorder->logTrace("[capture]: CreateRenderTarget (resolvedSurface) failed.\n");
				return false;
			}
		}
		hr = device->StretchRect(renderSurface, NULL, resolvedSurface, NULL, D3DTEXF_NONE);
		if (FAILED(hr)){
			infoRecorder->logTrace("[capture]: StretchRect failed.\n");
			return false;
		}

		oldRenderSurface = renderSurface;
		renderSurface = resolvedSurface;
	}
	else{
		infoRecorder->logTrace("[capture]: render target is Multisampled.\n");
	}

	// create offline surface in system memeory

	if (offscreenSurface == NULL){
		if(this->encoderType == ENCODER_TYPE::X264_ENCODER){
			hr = device->CreateOffscreenPlainSurface(game_width, game_height, desc.Format, D3DPOOL_SYSTEMMEM, &offscreenSurface, NULL);
			if (FAILED(hr)){
				infoRecorder->logTrace("[capture]: Create offscreen surface failed.\n");
				return false;
			}
		}else if(this->encoderType == ENCODER_TYPE::CUDA_ENCODER){
			hr = device->CreateOffscreenPlainSurface(game_width, game_height, desc.Format, D3DPOOL_MANAGED, &offscreenSurface, NULL);
			if (FAILED(hr)){
				infoRecorder->logTrace("[capture]: Create offscreen surface failed.\n");
				return false;
			}
		}
	}

	// copy the render-target data from device memory to system memory
	hr = device->GetRenderTargetData(renderSurface, offscreenSurface);
	if (FAILED(hr)){
		infoRecorder->logTrace("[capture]: GetRenderTargetData failed.\n");
		if (hr == D3DERR_DRIVERINTERNALERROR){
			infoRecorder->logTrace("[capture]: GetRenderTargetData failed code: D3DERR_DIRVERINTERNALERROR.\n");
		}
		else if (hr == D3DERR_DEVICELOST){
			infoRecorder->logTrace("[capture]: GetRenderTargetData failed code: D3DERR_DEVICELOST.\n");
		}
		else if (hr == D3DERR_INVALIDCALL){
			infoRecorder->logTrace("[capture]: GetRenderTargetData failed code: D3DERR_INVALIDCALL.\n");
		}
		else{
			infoRecorder->logTrace("[capture]: Get render target data failed with code:%d\n", hr);
		}
		if (oldRenderSurface)
			oldRenderSurface->Release();
		else
			renderSurface->Release();
		return false;
	}

	if (oldRenderSurface)
		oldRenderSurface->Release();
	else
		renderSurface->Release();

#endif
	if(this->sourceType == SOURCE_TYPE::IMAGE){
		// start to lock screen from offline surface
		hr = capturedSurface->LockRect(&lockedRect, NULL, NULL);
		if (FAILED(hr)){
			infoRecorder->logTrace("[capture]: LcokRect failed.\n");
			return false;
		}

		// copy image
		do{
			int image_size = 0;
			unsigned char * src, *dst;
			data = pipe[0]->allocate_data();
			frame = (ImageFrame *)data->ptr;
			frame->pixelFormat = PIX_FMT_BGRA;
			frame->realWidth = desc.Width;
			frame->realHeight = desc.Height;
			frame->realStride = desc.Width << 2;
			frame->realSize = frame->realWidth * frame->realStride;
			frame->lineSize[0] = frame->realStride;

			src = (unsigned char *)lockedRect.pBits;
			dst = (unsigned char *)frame->imgBuf;

			for (i = 0; i < encoder_height; i++){
				CopyMemory(dst, src, frame->realStride);
				src += lockedRect.Pitch;
				dst += frame->realStride;
				image_size += lockedRect.Pitch;
			}
			infoRecorder->logTrace("[capture]: get image data:%d, height:%d, pitch:%d.\n", image_size, encoder_height, lockedRect.Pitch);
			frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
		} while (0);

		// duplicate from channel 0 to other channel
		for (i = 1; i< SOURCES; i++){
			struct pooldata * dupdata;
			struct ImageFrame * dupframe;
			dupdata = pipe[i]->allocate_data();
			dupframe = (struct ImageFrame *)dupdata->ptr;

			pipe[i]->store_data(dupdata);
			pipe[i]->notify_all();
		}
		pipe[0]->store_data(data);
		infoRecorder->logTrace("[Channel]: source pipeline notify the client.\n");
		pipe[0]->notify_all();
		//offscreenSurface->UnlockRect();
		capturedSurface->UnlockRect();
	}

	else if(this->sourceType == SOURCE_TYPE::SURFACE){
		// storee surface to pipeline
		data = cudaPipe[0]->allocate_data();
		SurfaceFrame * sframe = (SurfaceFrame *)data->ptr;
		IDirect3DSurface9 * sur = sframe->getD3D9Surface(); // get the surface
		if(sur == NULL){
			// create the surface

			hr = device->CreateOffscreenPlainSurface(game_width, game_height, desc.Format, D3DPOOL_DEFAULT, &sur, NULL);
			if (FAILED(hr)){
				infoRecorder->logTrace("[capture]: Create offscreen surface failed.\n");
				return false;
			}
			sframe->setD3D9Surface(sur, game_width, game_height);
		}
		hr = device->StretchRect(capturedSurface, NULL, sur, NULL, D3DTEXF_LINEAR);
		//hr = device->UpdateSurface(offscreenSurface, NULL, sur, NULL);   // copy the surface
		if(FAILED(hr)){
			infoRecorder->logTrace("[Channel]: copy the surface failed.\n");
		}

		for(i = 0; i< SOURCES; i++){
			struct pooldata * dupdata = cudaPipe[i]->allocate_data();
			struct SurfaceFrame * dupframe = (struct SurfaceFrame *)dupdata->ptr;

			IDirect3DSurface9 * sur = sframe->getD3D9Surface(); // get the surface
			if(sur == NULL){
				// create the surface

				hr = device->CreateOffscreenPlainSurface(game_width, game_height, desc.Format, D3DPOOL_DEFAULT, &sur, NULL);
				if (FAILED(hr)){
					infoRecorder->logTrace("[capture]: Create offscreen surface failed.\n");
					return false;
				}
				sframe->setD3D9Surface(sur, game_width, game_height);
			}
			hr = device->StretchRect(capturedSurface, NULL, sur, NULL, D3DTEXF_LINEAR);
			//hr = device->UpdateSurface(offscreenSurface, NULL, sur, NULL);   // copy the surface
			if(FAILED(hr)){
				infoRecorder->logTrace("[Channel]: copy the surface failed.\n");
			}

			cudaPipe[i]->store_data(dupdata);
			cudaPipe[i]->notify_all();
		}
		cudaPipe[0]->store_data(data);
		infoRecorder->logTrace("[Channel]: source cudapipeline notify the client.\n");
		cudaPipe[0]->notify_all();

	}
	//xxx: disable until we have found a good place to safely release

	return true;
}
int Channel::getD3DResolution(DX_VERSION version){
	if(version = DX9){
		return getD3D9Resolution();
	}else if(version == DX10){
		return getD3D10Resolution();
	}else if(version == DX10_1){
		return getD3D101Resolution();
	}else if(version == DX11){
		return getD3D11Resolution();
	}else{
		infoRecorder->logError("[Channel]: get resolution failed with invalid dx version.\n");
		return -1;
	}
	return 0;
}

int Channel::getD3D10Resolution(){

}

int Channel::getD3D101Resolution(){

}
int Channel::getD3D11Resolution(){

}

int Channel::getD3D9Resolution(){
	infoRecorder->logTrace("[Channel]: get resolution.\n");
	HRESULT hr;
	D3DSURFACE_DESC desc;
	int resolution[2];
	IDirect3DSurface9 * renderSurface = NULL;

	if (device == NULL){
		infoRecorder->logTrace("[Channel]: device NULL.\n");
		return -1;
	}
	hr = device->GetRenderTarget(0, &renderSurface);
	if (!renderSurface || FAILED(hr)){
		infoRecorder->logTrace("[Channel]: get render target failed.\n");
		return -1;
	}
	renderSurface->GetDesc(&desc);
	renderSurface->Release();

	if (game_width <= 0 || game_height <= 0){
		game_width = desc.Width;
		game_height = desc.Height;
	}
	if (game_width == desc.Width && game_height == desc.Height){
		if (rtspConf->confReadInts("max-resolution", resolution, 2) == 2){
			encoder_width = resolution[0];
			encoder_height = resolution[1];
		}
		else{
			encoder_width = (game_width / ENCODING_MOD_BASE) * ENCODING_MOD_BASE;
			encoder_height = (game_height / ENCODING_MOD_BASE) * ENCODING_MOD_BASE;
		}
		infoRecorder->logTrace("[Channel]: resolution fitted: game %dx%d; encoder %dx%d.\n",
			game_width, game_height, encoder_width, encoder_height);
		return 0;
	}
	else{
		infoRecorder->logTrace("[Channel]: resolution not fitted (%dx%d).\n", desc.Width, desc.Height);
	}
	return -1;
}
// Surface Frame class

SurfaceFrame::SurfaceFrame(){this->type = SURFACE;}

SurfaceFrame::~SurfaceFrame(){}

#if 1
void SurfaceFrame::setD3D9Surface(IDirect3DSurface9 * s, int width, int height){
	dxSurface->d9Surface = s;
	D3DLOCKED_RECT rect;
	HRESULT hr = s->LockRect(&rect, NULL, D3DLOCK_DISCARD);
	if(FAILED(hr)){
		infoRecorder->logError("[SurfaceFrame]: lock rect failed.\n");
		return ;
	}
	pitch = rect.Pitch;
	this->width = width;
	this->height = height;
}
#endif

bool SurfaceFrame::init(){
	dxSurface = new DXSurface();
	dxSurface->d9Surface = NULL;
	width = 0;
	height = 0;
	pitch = 0;
	return true;
}

void SurfaceFrame::release(){}

int SurfaceFrame::setup(const char * pipeformat, struct VsourceConfig * conf, int id){
	return 0;
}

// Image Frame Class

ImageFrame::ImageFrame(){this->type = IMAGE;}

ImageFrame::ImageFrame(int w, int h, int s){}

ImageFrame::~ImageFrame(){}

bool ImageFrame::init(){
	return true;
}

bool ImageFrame::init(int maxW, int maxH, int maxS){
	int i;
	//
	//bzero(frame, sizeof(struct VsourceFrame));
	//
	for (i = 0; i < MAX_STRIDE; i++) {
		lineSize[i] = maxS;
	}
	this->maxStride = maxS;
	imgBufSize = maxH * maxS;
	if (ve_malloc(imgBufSize, (void **)&imgBufInternal, &alignment) < 0) {
		infoRecorder->logError("error: malloc buf failed in VsourceFrame\n");
		return false;
	}
	imgBuf = imgBufInternal + alignment;
	bzero(imgBuf, imgBufSize);
	return true;
}

void ImageFrame::release(){
	if (imgBuf != NULL){
		free(imgBuf);
		imgBuf = NULL;
	}
	return;
}

int ImageFrame::setup(const char * pipeformt, struct VsourceConfig * conf, int id){
	return true;
}


void ImageFrame::DupFrame(struct ImageFrame * src, struct ImageFrame * dst){
	int j;
	dst->imgPts = src->imgPts;
	dst->pixelFormat = src->pixelFormat;
	for(j = 0; j< MAX_STRIDE; j++){
		dst->lineSize[j] = src->lineSize[j];
	}
	dst->realWidth = src->realWidth;
	dst->realHeight = src->realHeight;
	dst->realStride = src->realStride;
	dst->realSize = src->realSize;
	bcopy(src->imgBuf, dst->imgBuf, dst->imgBufSize);
	return;
}

int ccg_win32_draw_system_cursor(ImageFrame * frame){
	static int capture_cursor = -1;
	static unsigned char bitmask[8] = {
		0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
	};
	int i, j, ptx, pty;
	int ret = -1;
	if (capture_cursor < 0){
		capture_cursor = 1;
	}

	ICONINFO iinfo;
	CURSORINFO cinfo;
	HCURSOR hc;
	BITMAP mask, cursor;
	int msize, csize;

	bzero(&cinfo, sizeof(cinfo));
	cinfo.cbSize = sizeof(cinfo);
	if (GetCursorInfo(&cinfo) == FALSE){
		infoRecorder->logError("CHANNEL ERROR: GetCursorInfoFailed, capture-cursor disabled.\n");
		capture_cursor = 0;
		return -1;
	}
	if (cinfo.flags != CURSOR_SHOWING)
		return 0;
	if ((hc = CopyCursor(cinfo.hCursor)) == NULL){
		infoRecorder->logError("CHANNEL ERROR: CopyCursor failed, err=0x%08x.\n", GetLastError());
		return -1;
	}
	if (GetIconInfo((HICON)hc, &iinfo) == FALSE){
		infoRecorder->logError("CHANNEL ERROR: GetIconInfo failed.\n");
		goto quitFreeCursor;
	}

	GetObject(iinfo.hbmMask, sizeof(mask), &mask);
	msize = mask.bmHeight * mask.bmWidthBytes;
	if (iinfo.hbmColor != NULL){
		GetObject(iinfo.hbmColor, sizeof(cursor), &cursor);
		csize = cursor.bmHeight * cursor.bmWidthBytes;
	}
	if (iinfo.hbmColor == NULL){
		unsigned char mbits[8192];
		unsigned char *mcurr, *ccurr, *fcurr;
		if (mask.bmBitsPixel != 1){
			infoRecorder->logError("CHANNEL ERROR: unsupported B/W cursor bitsPixel - m:%d%s1",
				mask.bmBitsPixel, mask.bmBitsPixel == 1 ? "==" : "!=");
			goto quitFreeIconinfo;
		}
		if (msize > sizeof(mbits)){
			infoRecorder->logError("CHANNEL ERROR: B/W cursor too larget, ignored.\n");
			goto quitFreeIconinfo;
		}
		if (mask.bmHeight != mask.bmWidth << 1){
			infoRecorder->logError("CHANNEL ERROR: Bad B/W cursor size (%dx%d)\n", mask.bmWidth, mask.bmHeight);
			goto quitFreeIconinfo;
		}
		GetBitmapBits(iinfo.hbmMask, msize, mbits);

		mask.bmHeight = mask.bmHeight >> 1;
		for (i = 0; i < mask.bmHeight; i++){
			pty = cinfo.ptScreenPos.y - iinfo.yHotspot + i;
			if (pty >= frame->realHeight)
				break;
			mcurr = mbits + i * mask.bmWidthBytes;
			ccurr = mbits + (mask.bmHeight + i) *  mask.bmWidthBytes;
			fcurr = frame->imgBuf + (pty  * frame->realStride);
			for (j = 0; j < mask.bmWidth; j++){
				ptx = cinfo.ptScreenPos.x - iinfo.xHotspot + j;
				if (ptx >= frame->realWidth)
					break;
				if ((mcurr[j >> 3] & bitmask[j & 0x07]) == 0){
					if ((ccurr[j >> 3] & bitmask[j & 0x07]) != 0){
						fcurr[ptx * 4 + 0] = 0xff;
						fcurr[ptx * 4 + 1] = 0xff;
						fcurr[ptx * 4 + 2] = 0xff;

					}
					else{
						if ((ccurr[j >> 3] & bitmask[j & 0x07]) != 0){
							fcurr[ptx * 4 + 0] ^= 0xff;
							fcurr[ptx * 4 + 1] ^= 0xff;
							fcurr[ptx * 4 + 2] ^= 0xff;


						}
					}
				}
			}
		}
	}
	else{
		unsigned char mbits[8192];
		unsigned char cbits[262144];
		unsigned char * mcurr, *ccurr, *fcurr;

		if (mask.bmBitsPixel != 1 || cursor.bmBitsPixel != 32){
			infoRecorder->logError("CHANNEL ERROR: unsupported currsor bitsPixel - m:%d %d1, c:%d%s32\n",
				mask.bmBitsPixel, mask.bmBitsPixel == 1 ? "==" : "!=",
				cursor.bmBitsPixel, cursor.bmBitsPixel == 32 ? " ==" : "!=");
			goto quitFreeIconinfo;
		}
		if (msize > sizeof(mbits) || csize > sizeof(cbits)){
			infoRecorder->logError("CHANNEL ERROR: cursor too large (> 256x256), ignored.\n");
			goto quitFreeIconinfo;
		}
		GetBitmapBits(iinfo.hbmMask, msize, mbits);
		GetBitmapBits(iinfo.hbmColor, csize, cbits);

		for (i = 0; i < mask.bmHeight; i++){
			pty = cinfo.ptScreenPos.y - iinfo.yHotspot + i;
			if (pty >= frame->realHeight)
				break;
			mcurr = mbits + i * mask.bmWidthBytes;
			ccurr = cbits + i * cursor.bmWidthBytes;
			fcurr = frame->imgBuf + (pty * frame->realStride);
			for (j = 0; j < mask.bmWidth; j++){
				ptx = cinfo.ptScreenPos.x - iinfo.xHotspot + j;
				if (ptx >= frame->realWidth)
					break;
				if ((mcurr[j >> 3] & bitmask[j & 0x07]) == 0){
					fcurr[ptx * 4 + 0] = ccurr[j * 4 + 0];
					fcurr[ptx * 4 + 1] = ccurr[j * 4 + 1];
					fcurr[ptx * 4 + 2] = ccurr[j * 4 + 2];
				}
				else{
					fcurr[ptx * 4 + 0] ^= ccurr[j * 4 + 0];
					fcurr[ptx * 4 + 1] ^= ccurr[j * 4 + 1];
					fcurr[ptx * 4 + 2] ^= ccurr[j * 4 + 2];
				}
			}
		}
	}
	ret = 0;
quitFreeIconinfo:
	if (iinfo.hbmMask != NULL) DeleteObject(iinfo.hbmMask);
	if (iinfo.hbmColor != NULL) DeleteObject(iinfo.hbmColor);
quitFreeCursor:
	DestroyCursor(hc);

	return ret;
}

// thread proc for channel, get get the source image

DWORD ChannelThreadProc(LPVOID param){
	DWORD ret = 0;
	Channel * ch = (Channel *)param;
	// init
	int frame_interval;
	struct timeval tv;
	LARGE_INTEGER initialTv, captureTv, freq;
	RTSPConf * rtspConf = NULL;

	rtspConf = ch->getRtspConf();

	//struct RTSPConf * conf = NULL;
	if (rtspConf == NULL){
		infoRecorder->logError("CHANNEL ERROR: get NULL rtspConfig\n");
		return 0;
	}
	const char * pipeformat = (const char *)param;

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&initialTv);

	frame_interval = 1000000 / rtspConf->video_fps;
	frame_interval++;

	for (int i = 0; i < SOURCES; i++){
		char pipename[64];
		snprintf(pipename, sizeof(pipename), pipeformat, i);
		if ((ch->pipe[i] = pipeline::lookup(pipename)) == NULL){
			infoRecorder->logTrace("[Channel]: cannot find pipeline '%s'\n", pipename);
			exit(-1);
		}
	}
	// start the encoder assigned to channel
	ch->encoder->start();
	// enter the loop
	while (!ch->encoder->encoderRunning()){
		struct timeval * ptv = &tv;
		getTimeOfDay(ptv, NULL);
		if (ch->encoder->encoderRunning()){
			Sleep(1);
			return 0;
		}
		// copy image
		// get the source
		if(ch->presentEvent != NULL){
			if (ch->presentEvent)
				WaitForSingleObject(ch->presentEvent, INFINITE);
			else{
				infoRecorder->logTrace("[Channel]: channel get NULL present event.\n");
				break;
			}

			if (ch->sourceType == SOURCE_TYPE::IMAGE){
				infoRecorder->logError("[channel]: run ... get the image frame...\n");
				//if (encoderType == ENCODER_TYPE::X264_ENCODER){
#if 0
				// copy image
				struct pooldata *data = ch->pipe[0]->allocate_data();
				ImageFrame *frame = (ImageFrame *)data->ptr;
				frame->pixelFormat = PIX_FMT_BGRA;
				frame->realWidth = ch->wrapper->screenWidth;
				frame->realHeight = ch->wrapper->screenHeight;
				frame->realStride = ch->wrapper->screenWidth << 2;
				frame->realSize = ch->wrapper->screenHeight * frame->realStride;
				frame->lineSize[0] = frame->realStride;

				QueryPerformanceCounter(&captureTv);

				// capture
				int getsize = ch->wrapper->capture((char*)frame->imgBuf, frame->getImgBufSize(), &ch->wrapper->rect);
				infoRecorder->logTrace("[Channel]: d3d wrapper get imge, size:%d\n.", getsize);
				ccg_win32_draw_system_cursor(frame);

				frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
				for (int i = 1; i < SOURCES; i++){
					struct pooldata * dupdata;
					struct ImageFrame* dupframe;
					dupdata = ch->pipe[i]->allocate_data();
					dupframe = (ImageFrame *)dupdata->ptr;

					ch->pipe[i]->store_data(data);
					ch->pipe[i]->notify_all();
				}
				ch->pipe[0]->store_data(data);
				ch->pipe[0]->notify_all();   // notify all waited thread

				//notify
				ch->pipe[0]->releasePipeMutex();

				ccg_usleep(frame_interval, &tv);
#else
				ch->doCapture(ch->device);
#endif
				//}
				//else if (encoderType == ENCODER_TYPE::CUDA_ENCODER){
				// use the cuda encoder encoding the image

				//}

			}
			else if (ch->sourceType == SOURCE_TYPE::SURFACE){
				// can only use the cuda encoder 
				// get the surface pointer and store

				struct pooldata * data = ch->pipe[0]->allocate_data();
				SurfaceFrame * frame = (SurfaceFrame *)data->ptr;
				QueryPerformanceCounter(&captureTv);

				// capture
				IDirect3DSurface9 * surface = ch->wrapper->capture(ch->encoderType, ch->game_width, ch->game_height);
				frame->dxSurface->d9Surface = surface;
				frame->imgPts = pcdiff_us(captureTv, initialTv, freq) / frame_interval;
				ch->pipe[0]->store_data(data);
				//ccg_usleep(frame_interval, &tv);
			}
		}else{
			// use the window wrapper, get the image source every 30ms 
			infoRecorder->logError("[Channel]: invalid present event.\n");
			ccg_usleep(frame_interval, &tv);  // sleep
		}
	}

	return ret;
}

bool Channel::startChannelThread(LPVOID param){
	threadHandle = chBEGINTHREADEX(NULL, 0, ChannelThreadProc, param, NULL, &this->channelThreadId);
	return true;
}




///////////////////for 