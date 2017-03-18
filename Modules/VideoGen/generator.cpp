#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "generator.h"
#include "../LibCore/TimeTool.h"

// this definition will enable trace log for VideoGen
#define ENABLE_GEN_LOG 

namespace cg{

	// the generator currently is only for cuda encoder
	int VideoGen::GenCounter = 0;
	map<IDENTIFIER, VideoGen*> VideoGen::genMap;
	HANDLE VideoGen::genNotifier = NULL;
	bool VideoGen::globalInited = false;

	VideoGen * VideoGen::findVideoGen(IDENTIFIER _id){
		VideoGen * ret = NULL;
#ifdef ENABLE_GEN_LOG
		cg::core::infoRecorder->logError("[VideoGen]: find VideoGen for %p, wait for object:%p.\n", _id, genNotifier);
#endif  // ENABLE_GEN_LOG
		std::map<IDENTIFIER, VideoGen *>::iterator it;
		if(genMap.size() == 0){
			
			cg::core::infoRecorder->logError("[VideoGen]: no VideoGen, wait for one.\n");
			DWORD waitRet = WaitForSingleObject(genNotifier, INFINITE);
			switch(waitRet){
			case WAIT_ABANDONED:
				cg::core::infoRecorder->logError("[VideoGen]: wait ret 'WAIT_ABANDONED'\n");
				break;
			case WAIT_OBJECT_0:
				cg::core::infoRecorder->logError("[VideoGen]: wait ret 'WAIT_OBJECT_0'\n");
				break;
			case WAIT_TIMEOUT:
				cg::core::infoRecorder->logError("[VideoGen]: wait ret 'WAIT_TIMEOUT'\n");
				break;
			case WAIT_FAILED:
				cg::core::infoRecorder->logError("[VideoGen]: wait ret 'WAIT_FAILED'\n");
				break;
			}
		}else{
			cg::core::infoRecorder->logError("[VideoVen]: gen map is not EMPTY, to find one.\n");
		}
		
		if((it = genMap.find(_id)) != genMap.end()){
			// find
			return it->second;
		}
		else{
#ifdef ENABLE_GEN_LOG
			cg::core::infoRecorder->logTrace("[VideoGen]: VideoGen for id: %p find.\n", _id);
#endif
			WaitForSingleObject(genNotifier, INFINITE);
			if((it = genMap.find(_id)) != genMap.end()){
				// find
				return it->second;
			}
			cg::core::infoRecorder->logError("[VideoGen]: find VideoGen for id: %p, get NULL gen.\n", _id);
			return NULL;
		}
	}

	void VideoGen::addMap(IDENTIFIER _id, VideoGen * gen){
#ifdef ENABLE_GEN_LOG
		cg::core::infoRecorder->logTrace("[VideoGen]: add video gen for '%p'.\n", _id);
#endif
		SetEvent(genNotifier);
		genMap.insert(map<IDENTIFIER, VideoGen *>::value_type(_id, gen));
	}

	// create a VideoGen with d3d wrapper, create VideoWriter when inited
	VideoGen::VideoGen(HWND hwnd, void * _device, DX_VERSION _version, bool _useNvenc /* = false */){
		enableWriteToFile = false;
		enableWriteToNet = false;
		videoOutputName = NULL;
		ctx = NULL;
		inited = false;
		useSourceType = SOURCE_NONE;
		id = GenCounter++;
		windowHandle = hwnd;

		isChangedDevice = false;

		encoder = NULL;
		x264Encoder = NULL;
		wrapper = NULL;
		writer = NULL;

		dxVer = _version;

		swapChain = NULL;
		d9Device = (IDirect3DDevice9 *)_device;
		dxDevice = new DXDevice();
		dxDevice->d9Device = (IDirect3DDevice9 *)_device;

		sourcePipe = NULL;
		useNvenc = _useNvenc;
		encoderType = useNvenc ? ADAPTIVE_NVENC : ADAPTIVE_CUDA;
		useEncoderType = NVENC_ENCODER;
#ifdef NVENC
		nvEncoder = NULL;
#endif
		x264Inited = false, cudaInited = false, nvecnInited = false;
		width = 0;
		height = 0;
		RECT rect;
		GetWindowRect(hwnd, &rect);
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;
		intraMigrationTimer = new PTimer();

#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: window %p has rect (%d x %d).\n",hwnd, width, height);
#endif
	}

	// create a VideoGen using Window Wrapper, create VideoWriter when inited
	VideoGen::VideoGen(HWND hwnd, bool useNVENC){

		enableWriteToFile = false;
		enableWriteToNet = false;
		useSourceType = SOURCE_NONE;
		ctx = NULL;
		id = GenCounter++;

		windowHandle = hwnd;
		encoder = NULL;
		x264Encoder = NULL;
		wrapper = NULL;
		writer = NULL;

		dxVer = DXNONE;
		swapChain = NULL;
		d9Device = NULL;

		dxDevice = new DXDevice();
		sourcePipe = NULL;	
		isChangedDevice = false;

		encoderType = useNVENC ? ADAPTIVE_NVENC : X264_ENCODER;
		useEncoderType = X264_ENCODER;

		useNvenc = useNVENC;
#ifdef NVENC
		nvEncoder = NULL;
#endif // NVENC
		x264Inited = false, cudaInited = false, nvecnInited = false;
		// get the window size
		RECT rect;
		GetWindowRect(hwnd, &rect);
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;
		videoOutputName = NULL;
		intraMigrationTimer = new PTimer();

#ifdef ENABLE_GEN_LOG
		infoRecorder->logError("[VideoGen]: constructor called. width:%d, height:%d\n", width, height);
#endif
	}	
	//////// for test //////////////////

	VideoGen::VideoGen(HWND hwnd, void * _device, DX_VERSION _version, bool _useNVENC, bool writeToFile, bool writeNet){
		enableWriteToFile = writeToFile;
		enableWriteToNet = writeNet;
		ctx = NULL;
		id = GenCounter++;
		char name[100] = {0};
		sprintf(name, "video-%d.264", id);
		if(enableWriteToFile)
			videoOutputName = _strdup(name);   // give the name
		else
			videoOutputName = NULL; 

		inited = false;
		useSourceType = SOURCE_NONE;
		windowHandle = hwnd;
		isChangedDevice = false;

		encoder = NULL;
		x264Encoder = NULL;
		wrapper = NULL;
		writer = NULL;

		dxVer = _version;
		swapChain = NULL;
		d9Device = (IDirect3DDevice9 *)_device;
		dxDevice = new DXDevice();
		dxDevice->d9Device = (IDirect3DDevice9 *)_device;

		sourcePipe = NULL;	
		useNvenc = _useNVENC;
		encoderType = (useNvenc ? ADAPTIVE_NVENC : ADAPTIVE_CUDA);
		useEncoderType = NVENC_ENCODER;
#ifdef NVENC
		nvEncoder = NULL;
#endif // NVENC
		x264Inited = false, cudaInited = false, nvecnInited = false;
		width = 0;
		height = 0;

		RECT rect;
		GetWindowRect(hwnd, &rect);
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;

		intraMigrationTimer = new PTimer();

#ifdef ENABLE_GEN_LOG
		infoRecorder->logError("[VideoGen]: window %p has rect (%d x %d), rect(%d, %d, %d, %d).\n", hwnd, width, height, rect.right, rect.left, rect.bottom, rect.top);
#endif

	}
	VideoGen::VideoGen(HWND hwnd, bool useNVENC, bool writeToFile, bool writeNet){
		enableWriteToFile = writeToFile;
		enableWriteToNet = writeNet;
		ctx = NULL;
		dxVer = DXNONE;

		id = GenCounter++;
		char name[100] = {0};
		sprintf(name, "video-%d.264", id);
		if(enableWriteToFile)
			videoOutputName = _strdup(name);   // give the name
		else
			videoOutputName = NULL; 

		inited = false;
		useSourceType = SOURCE_NONE;
		isChangedDevice = false;
		windowHandle = hwnd;
		encoder = NULL;
		x264Encoder = NULL;

		wrapper = NULL;
		writer = NULL;
		dxVer = DXNONE;
		swapChain = NULL;
		d9Device = NULL;
		dxDevice = NULL;

		encoderType = useNVENC ? ADAPTIVE_NVENC : X264_ENCODER;
		useEncoderType = X264_ENCODER;

		sourcePipe = NULL;	
		useNvenc = useNvenc;
#ifdef NVENC
		nvEncoder = NULL;
#endif // NVENC
		x264Inited = false, cudaInited = false, nvecnInited = false;
		width = 0;
		height = 0;
		RECT rect;
		GetWindowRect(hwnd, &rect);
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;

		intraMigrationTimer = new PTimer();

#ifdef ENABLE_GEN_LOG
		infoRecorder->logError("[VideoGen]: window %p has rect (%d x %d).\n", width, height);
#endif

	}

	VideoGen::VideoGen(HWND hwnd, bool useNVENC, ENCODER_TYPE _encodeType, bool writeToFile, bool writeNet){
		enableWriteToFile = writeToFile;
		enableWriteToNet = writeNet;
		ctx = NULL;
		dxVer = DXNONE;

		id = GenCounter++;
		char name[100] = {0};
		sprintf(name, "video-%d.264", id);
		if(enableWriteToFile)
			videoOutputName = _strdup(name);   // give the name
		else
			videoOutputName = NULL; 

		inited = false;
		useSourceType = SOURCE_NONE;
		windowHandle = hwnd;
		encoder = NULL;
		x264Encoder = NULL;

		wrapper = NULL;
		writer = NULL;
		dxVer = DXNONE;
		swapChain = NULL;
		d9Device = NULL;
		dxDevice = NULL;

		encoderType = _encodeType;
		useEncoderType = useNVENC ? NVENC_ENCODER : X264_ENCODER;
		isChangedDevice = false;

		sourcePipe = NULL;	
		useNvenc = useNvenc;
#ifdef NVENC
		nvEncoder = NULL;
#endif // NVENC
		x264Inited = false, cudaInited = false, nvecnInited = false;
		width = 0;
		height = 0;
		RECT rect;
		GetWindowRect(hwnd, &rect);
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;

		intraMigrationTimer = new PTimer();

#ifdef ENABLE_GEN_LOG
		infoRecorder->logError("[VideoGen]: window %p has rect (%d x %d).\n", width, height);
#endif

	}

	VideoGen::VideoGen(HWND hwnd, void * device, core::DX_VERSION version, bool _useNVENC, ENCODER_TYPE _encodeType, bool writeFile, bool writeNet){
		enableWriteToFile = writeFile,
		enableWriteToNet = writeNet;
		ctx = NULL;
		id = GenCounter++;
		char name[100] = {0};
		sprintf(name, "video-%d.264", id);
		if(enableWriteToFile)
			videoOutputName = _strdup(name);   // give the name
		else
			videoOutputName = NULL; 

		inited = false;
		useSourceType = SOURCE_NONE;
		windowHandle = hwnd;

		encoder = NULL;
		x264Encoder = NULL;
		wrapper = NULL;
		writer = NULL;

		dxVer = version;
		swapChain = NULL;
		d9Device = (IDirect3DDevice9 *)device;
		dxDevice = new DXDevice();
		dxDevice->d9Device = (IDirect3DDevice9 *)device;

		sourcePipe = NULL;	
		useNvenc = _useNVENC;
		encoderType = _encodeType;
		isChangedDevice = false;
		useEncoderType = useNvenc ? NVENC_ENCODER : CUDA_ENCODER;
#ifdef NVENC
		nvEncoder = NULL;
#endif // NVENC
		x264Inited = false, cudaInited = false, nvecnInited = false;
		width = 0;
		height = 0;

		RECT rect;
		GetWindowRect(hwnd, &rect);
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;

		intraMigrationTimer = new PTimer();

#ifdef ENABLE_GEN_LOG
		infoRecorder->logError("[VideoGen]: window %p has rect (%d x %d).\n", width, height);
#endif

	}
	///////////////////////////////////

	VideoGen::~VideoGen(){
		infoRecorder->logError("[VideoGen]: destructor called.\n");
		if(sourcePipe){
			delete sourcePipe;
			sourcePipe = NULL;
		}	
		if(wrapper){
			delete wrapper;
			wrapper = NULL;
		}

		if(encoder){
			delete encoder;
			encoder = NULL;
		}
		if(x264Encoder){
			delete x264Encoder;
			x264Encoder = NULL;
		}

#ifdef NVENC
		if(nvEncoder){
			delete nvEncoder;
			nvEncoder = NULL;
		}
#endif // NVENC
		if(intraMigrationTimer){
			delete intraMigrationTimer;
			intraMigrationTimer = NULL;
		}
	}

	// init a video generator for non-d3d games, the fps is from the RTSP config.
	// for non-d3d games, only the X264 encoder would be efficient. 
	int VideoGen::initVideoGen(){
		struct pooldata * data = NULL;
		if(width < 100 || height < 100){
			infoRecorder->logError("[VideoGen]: resolution not set, VideoGen init failed.\n");
			return -1;
		}

		if(videoOutputName == NULL){
			videoOutputName = _strdup("null.video.264");
		}
		if(enableWriteToFile){
			videoOutput = fopen(videoOutputName, "wb");
		}
		else{
			
		}

		// use image as default
		if(useSourceType == SOURCE_NONE)
			useSourceType = IMAGE;

		// create the pipeline outside
		if((sourcePipe = new pipeline(sizeof(VsourceConfig), "image")) == NULL){
			infoRecorder->logError("[VideoGen]: create pipeline for image failed.\n");
			return ERR_NULL_PIPE;
		}

		// init the pool
		if((data = sourcePipe->datapool_init(POOLSIZE)) == NULL){
			infoRecorder->logError("[VideoGen]: data pool init failed.\n");
			delete sourcePipe;
			sourcePipe = NULL;
			return ERR_NULL_PIPE;
		}
		for(; data != NULL; data = data->next){
			data->ptr = new SourceFrame();
		}

		if(setupImageSource()){
			// error
			return -1;
		}

		if(dxVer == DXNONE){
			// use window wrapper
			setupWrapper();
			isD3D = false;
			return WRAPPER_OK;
		}
		else{
			if(d9Device == NULL && dxDevice->d9Device == NULL){
				// the device not specified
				infoRecorder->logError("[VideoGen]: NULL device.\n");
				return ERR_NULL_DEVICE;
			}
		}
		// use d3d
		isD3D = true;
		useSourceType = SURFACE;   // if support d3d
		// check device
		setupWrapper();
		
		if(setupSurfaceSource()){
			// error
			return ERR_FAILED;
		}

		ctx = new RTSPContext();
		RTSPConf * conf = RTSPConf::GetRTSPConf();
		ctx->setConf(conf);
		ctx->clientInit(1, width, height);
		strcpy(ctx->object, object.c_str());
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: create rtsp context with name:%s.", object.c_str());
		infoRecorder->logTrace("[VideoGen]: create video writer, file out:%p, ctx:%p.\n", videoOutput, ctx);
#endif

		if(enableWriteToFile)
			writer = new VideoWriter(videoOutput, ctx);
		else
			writer = new VideoWriter(ctx);

		inited = true;
		return WRAPPER_OK;
	}

	// init the wrapper
	int VideoGen::setupWrapper(){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: create the wrapper with dx version: %d", dxVer);
#endif
		// if support directx
		switch(dxVer){
		case DX9:
			//infoRecorder->logError("DX9\n");
			wrapper = new D3DWrapper(windowHandle, d9Device, width, height, sourcePipe);
			break;
		case DX10:
			//infoRecorder->logError("DX10\n");
			swapChain = (IDXGISwapChain *)d9Device;
			wrapper = new D3D10Wrapper(windowHandle, swapChain, width, height, sourcePipe);
			break;
		case DX10_1:
			//infoRecorder->logError("DX10_1\n");
			swapChain = (IDXGISwapChain *)d9Device;
			wrapper = new D3D10Wrapper1(windowHandle, swapChain, width, height, sourcePipe);	
			break;
		case DX11:
			//infoRecorder->logError("DX11\n");
			swapChain = (IDXGISwapChain *)d9Device;
			wrapper = new D3D11Wrapper(windowHandle, swapChain, width, height, sourcePipe);
			break;
		case DXNONE:
		default:
			infoRecorder->logError("DX_NONE\n");
			// use window wrapper
			wrapper = new WindowWrapper(windowHandle, width, height, sourcePipe);
			break;
		}
		return 0;
	}

	// init the video generator,
	int VideoGen::initVideoGen(DX_VERSION  ver, void * device){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: initVideoGen called.\n");
#endif
		struct pooldata * data = NULL;
		if(enableWriteToFile){
#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("[VideoGen]: output file name:%s.\n", videoOutputName);
#endif
			if(videoOutputName == NULL){
				videoOutputName = _strdup("null.video.264");
			}
			videoOutput = fopen(videoOutputName, "wb");
		}

		/////// setup source pipeline  ////////////
		if((sourcePipe = new pipeline(sizeof(VsourceConfig), "image")) == NULL){
			infoRecorder->logError("[VideoGen]: init the image pipeline failed.\n");
			return ERR_NULL_PIPE;
		}

		if((data = sourcePipe->datapool_init(POOLSIZE)) == NULL){
			infoRecorder->logError("[VideoGen]: data pool init failed.\n");
			delete sourcePipe;
			sourcePipe = NULL;
			return ERR_NULL_PIPE;
		}
		for(; data != NULL; data = data->next){
			data->ptr = new SourceFrame();
		}

		if(setupImageSource()){
			// error 
			return ERR_FAILED;
		}
		//default use image as source
		
		d9Device = (IDirect3DDevice9 *)device;
		dxVer = ver;

		setupWrapper();

		ctx = new RTSPContext();
		RTSPConf * conf = RTSPConf::GetRTSPConf();
		ctx->setConf(conf);
		ctx->clientInit(1, width, height);
		strcpy(ctx->object, object.c_str());

#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: create rtsp context with name:%s.", object.c_str());

		infoRecorder->logTrace("[VideoGen]: create video writer, file out:%p, ctx:%p.\n", videoOutput, ctx);
#endif

		if(enableWriteToFile)
			writer = new VideoWriter(videoOutput, ctx);
		else
			writer = new VideoWriter(ctx);

		inited = true;
		return WRAPPER_OK;
	}

	// set the source type or change the source type
	void VideoGen::setSourceType(SOURCE_TYPE srcType){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: try to set source type: %s", srcType == IMAGE ? "IMAGE" : "SURFACE");
#endif
		if(isInited()){
			// change the source type, change the wrapper and encoder
			useSourceType = dxVer != DXNONE ? srcType : (useSourceType == SURFACE ? IMAGE: srcType);
		}
		else{
			// set the source type
			useSourceType = dxVer != DXNONE ? srcType : IMAGE;
		}
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: wrapper setup source type: %s.\n", useSourceType == IMAGE ? "IMAGE" : "SURFACE");
#endif
		wrapper->changeSourceType(useSourceType);
	}

	// setup the pipeline for image
	int VideoGen::setupImageSource(){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: setupImageSource called.\n");
#endif
		struct pooldata * data = NULL;
		SourceFrame * frame = NULL;

		// form the private data
		VsourceConfig config;
		bzero(&config, sizeof(config));
		config.id = 0;

		config.rtpId = 0;
		config.maxWidth =  width;
		config.maxHeight = height;
		config.maxStride = (BITSPERPIXEL >> 3) * width;   // ARGB format

		sourcePipe->set_privdata(&config, sizeof(struct VsourceConfig));

		// per frame init
		for(data = sourcePipe->datapool_getpool(); data != NULL; data = data->next){
			frame = (SourceFrame *)data->ptr;
			//frame->pixelFormat = PIX_FMT_RGBA;//PIX_FMT_YUV420P;
			frame->pixelFormat = PIX_FMT_BGRA;//PIX_FMT_YUV420P;

#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("[VideoGen]: frame init, w:%d, h:%d, stride:%d.\n", width, height, config.maxStride);
#endif
			if(frame->init(width, height, config.maxStride) == NULL){
				// init frame failed
				return ERR_FAILED;
			}
		}

		return WRAPPER_OK;
	}

	// setup the pipeline for surface
	int VideoGen::setupSurfaceSource(){
		HRESULT hr = D3D_OK;
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: setupSurfaceSource called.\n");
#endif
		struct pooldata * data = NULL;
		SourceFrame * frame = NULL;
		//DebugBreak();
		// per frame init
		for(data = sourcePipe->datapool_getpool(); data != NULL; data = data->next){
			frame = static_cast<SourceFrame *>(data->ptr);
			frame->dxVersion = dxVer;
			
			//frame->dxSurface = new DXSurface();
			frame->width = width;
			frame->height = height;
			frame->dxSurface->d9Surface = NULL;  // create the surface first
			hr = dxDevice->d9Device->CreateRenderTarget(width, height, D3DFMT_A8R8G8B8, D3DMULTISAMPLE_NONE, 0, TRUE, &frame->dxSurface->d9Surface, NULL);
			if(FAILED(hr)){
				// create surface failed
				infoRecorder->logError("[VideoGen]: create render target for pipeline failed with:%d.\n",hr);
				return -1;
			}
		}

		return 0;
	}
	void VideoGen::setValueTag(unsigned char valueTag){
		if(writer)
			writer->setValueTag(valueTag);

	}

	void VideoGen::setVideoSpecialTag(unsigned char tag){
		if(writer){
			writer->setSpecialTag(tag);
		}
		else{
			infoRecorder->logError("[VideoGen]: NULL writer.\n");
		}
	}
	int VideoGen::changeEncodeDevice(ENCODER_TYPE dstEncoderType){
		// 
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: change encoder device.\n");
#endif
		changeToEncoder = dstEncoderType;
		isChangedDevice = true;
		return 0;
	}
	// the encoder device change callback
	// TODO: check the wrapper type, for x264encoder, no surface supported, for device encoder, window wrapper may not suitable
	int VideoGen::onEncodeDeviceChange(){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: on encoder device change.\n");
#endif
		// the image pipeline is always created cause image source or surface source all need image pipe
		ENCODER_TYPE oldType = useEncoderType;
		if(isChangedDevice){
			isChangedDevice = false;
			if(useEncoderType == changeToEncoder){
				// the same
				return 0;
			}
			// check the encoder is changeable????
			if(encoderType == ADAPTIVE_CUDA || encoderType == ADAPTIVE_NVENC){
				useEncoderType = changeToEncoder;
			}
			else{
				// fixed encoder type
				return 0;
			}
		}else{
			return 0;
		}

		ENCODER_TYPE nowType = useEncoderType;
		// determine the encoder type
		if(encoderType == ADAPTIVE_CUDA){
			// only x264 and cuda allowed
			useEncoderType = (nowType == X264_ENCODER ? nowType : (CUDA_ENCODER)); 
		}else if(encoderType == ADAPTIVE_NVENC){
			// only x264 and nvenc allowed
			useEncoderType = (nowType == X264_ENCODER ? nowType : (NVENC_ENCODER));
		}else{
			useEncoderType = encoderType;
		}

		// tell wrapper to capture right data
		setSourceType(useEncoderType == X264_ENCODER ? IMAGE : SURFACE);
		intraMigrationTimer->Start();

#ifdef ENABLE_GEN_LOG
		infoRecorder->logError("[VideoGen]: old encoder type:%s, now use encoder type: %s.\n", oldType == X264_ENCODER ? "x264": (oldType == CUDA_ENCODER ? "CUDA": (oldType == NVENC_ENCODER ? "NVENC" : "NONE")), useEncoderType == X264_ENCODER ? "x264" : (useEncoderType == CUDA_ENCODER ? "cuda" : (useEncoderType == NVENC_ENCODER ? "nvenc" : "none")));

		infoRecorder->logTrace("[VideoGen]: use encoder type: %s.\n", useEncoderType == X264_ENCODER ? "x264" : (useEncoderType == CUDA_ENCODER ? "cuda" : (useEncoderType == NVENC_ENCODER ? "nvenc" : "none")));
#endif

		return activeEncoder(useEncoderType) ? 0 : -1;
	}

	BOOL VideoGen::stop(){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: stop called.\n");
#endif
		if(encoderType == X264_ENCODER && x264Encoder){
			if(x264Filter)
				x264Filter->stop();
			x264Encoder->stop();
		}
		else if(encoderType == NVENC_ENCODER)
			nvEncoder->stop();
		else if(encoderType == CUDA_ENCODER)
			encoder->stop();

		return CThread::stop();
	}

	char * EncoderTypeString(ENCODER_TYPE type){
		switch(type){
		case X264_ENCODER:
			return _strdup("X264 Encoder");
			break;
		case NVENC_ENCODER:
			return _strdup("NVENC Encoder");
			break;
		case CUDA_ENCODER:
			return _strdup("CUDA Encoder");
		case ADAPTIVE_NVENC:
			return _strdup("ADAPTIVE NVENC");
			break;
		case ADAPTIVE_CUDA:
			return _strdup("ADAPTIVE CUDA");
		default:
			return _strdup("UNKNOWN");
		}
	}

	// the loop logic
	BOOL VideoGen::run(){
		int err = WRAPPER_OK;
#ifdef _DEBUG
		// test change encoder
#if 1
		static int runCount = 0;
		runCount++;
		if(runCount == 200){
			if(encoderType == ADAPTIVE_CUDA){
				if(useEncoderType != X264_ENCODER)
					changeEncodeDevice(X264_ENCODER);
				else
					changeEncodeDevice(CUDA_ENCODER);
			}else if(encoderType == ADAPTIVE_NVENC){
				if(useEncoderType != X264_ENCODER)
					changeEncodeDevice(X264_ENCODER);
				else
					changeEncodeDevice(NVENC_ENCODER);
			}
			runCount=0;
		}
#endif
#if 0
		char * sE = EncoderTypeString(encoderType);
		char * uE = EncoderTypeString(useEncoderType);
		infoRecorder->logError("[VideoGen]: encode type: %s, use encoder type: %s\n", sE, uE);
		free(sE);
		free(uE);
#endif
#endif  _DEBUG
		if(!(enableGen || (ctx && ctx->enableGen))){
			if(isStart()){
				Sleep(10);
			}
			return TRUE;
		}
		QueryPerformanceCounter(&captureTv);
		
		if(isChangedDevice){

#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("[VideoGen]: encoder device changed, change the source.\n");
#endif
			onEncodeDeviceChange();
		}

		if(!wrapper->capture(captureTv, initialTv, freq, frameInterval)){
			infoRecorder->logError("[VideoGen]: capture failed.\n");
			return FALSE;
		}

		if(!isD3D){
			getTimeOfDay(&tv, NULL);
			// not a d3d game, can only use window wrapper
			usleep(frameInterval, &tv);
		}
		return TRUE;
	}

	// deal with msg, now do nothing
	void VideoGen::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){

#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: onThreadMsg called.\n");
#endif
		switch(msg){
		case WM_USR_ENCODER:
			// change the encoder type
			if(encoderType == ADAPTIVE_CUDA){
				if(useEncoderType != X264_ENCODER)
					changeEncodeDevice(X264_ENCODER);
				else
					changeEncodeDevice(CUDA_ENCODER);
			}else if(encoderType == ADAPTIVE_NVENC){
				if(useEncoderType != X264_ENCODER)
					changeEncodeDevice(X264_ENCODER);
				else
					changeEncodeDevice(NVENC_ENCODER);
			}
			break;

		default:
			infoRecorder->logError("[VideoGen]: %d msg not delt by generator.\n", msg);
			break;	
		}
	}
	// release
	void VideoGen::onQuit(){
		infoRecorder->logError("[VideoGen]: onQuit called, TODO.\n");

		// release all resources

	}
	// init before enter loop
	BOOL VideoGen::onThreadStart(){
		initVideoGen();
#ifdef ENABLE_GEN_LOG
		// start the modules
		infoRecorder->logTrace("[VideoGen]: onThreadStart called\n");
#endif
		if(!inited){
			// init the video gen
			return FALSE;
		}

		if(!wrapper){
			infoRecorder->logError("[VideoGen]: the wrapper is not created yet. ERROR.\n");
			return FALSE;   // error
		}
		// active the encoder
#if 0
		onEncodeDeviceChange();
#else
		//DebugBreak();
		setSourceType(useEncoderType == X264_ENCODER ? IMAGE : SURFACE);
		activeEncoder(useEncoderType);
#endif
		// update time
		frameInterval = 100000 / RTSPConf::GetRTSPConf()->video_fps;
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&initialTv);
		return TRUE;
	}

	/// TODO, for d11, the context is the DeviceContext
	int VideoGen::initCudaEncoder(void * device, void * context){
		infoRecorder->logError("[VideoGen]: D11 init the CUDA encoder, TODO.\n");
		return 0;
	}

#ifdef NVENC

#if defined(USE_NVENC_CUDA)
	int VideoGen::initNVENCEncoder(void * device){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: init NVENC encoder: NvEncoder:%p, inited:%s\n", nvEncoder, nvecnInited ? "true" : "false");
#endif
		if(!nvecnInited){
			dxDevice->d9Device = (IDirect3DDevice9 *)device;
			useNvenc = true;
			if(useSourceType == IMAGE && sourcePipe){
				x264Filter = new Filter();
				x264Filter->setOutputFormat(PIX_FMT_NV12);
				//generator->setOutputFileName("test.video.264");
				x264Filter->init(height, width, height, width);
				x264Filter->registerSourcePipe(sourcePipe);

				nvEncoder = new cg::nvenc::CNvEncoderCudaInteropImpl(dxDevice->d9Device, width, height, /*IMAGE,*/ x264Filter->getFilterPipe(), writer);

				x264Filter->start();
			}else if(useSourceType == SURFACE && sourcePipe){
				nvEncoder = new cg::nvenc::CNvEncoderCudaInteropImpl(dxDevice->d9Device, width, height, /*SURFACE, */sourcePipe, writer);
			}else{
				infoRecorder->logError("[VideoGen]: init NVENC encoder, source type or pipe is INVALID, type:%d, pipe:%p.\n", useSourceType == IMAGE ?" IMAGE" : "SURFACE", sourcePipe);
			}
			nvecnInited = true;
		}
		return 0;
	}

#elif defined(USE_NVENC_D3D)
	int VideoGen::initNVENCEncoder(void * device){
		dxDevice->d9Device = (IDirect3DDevice9 *)device;
		useNvenc = true;

		if(useSourceType == IMAGE && imagePipe){
			// error
			infoRecorder->logError("[CNvEncoderD3DInteropImpl]: only support surface input, while current use IMAGE.\n");
			return -1;
		}else if(useSourceType == SURFACE && cudaPipe){
			nvEncoder = new NV::CNvEncoderD3DInteropImpl(dxDevice->d9Device, width, height, cudaPipe);
		}
		if(enableWriteToFile){
			nvEncoder->setEnableWriteToFile(true);
			nvEncoder->setVideoFile(videoOutput);
		}
		return 0;

	}
#elif defined(USE_NVENC_NOR)

#endif

#endif // NVENC
	/// for D3D9, init the CUDA encoder with only device
	int VideoGen::initCudaEncoder(void * device){
		pipeline * pipe = NULL;
		if(!cudaInited){
			dxDevice->d9Device = (IDirect3DDevice9 *)device;
#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("[VideoGen]: init the cuda encoder.\n");
#endif

			encoder = new cg::nvcuvenc::CudaEncoder(width, height, /*useSourceType, */sourcePipe, writer, (IDirect3DDevice9 *)device); 

			encoder->initEncoder();
			//set the event

#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("[VideoGen]: before init cuda filter, width:%d, height:%d.\n", width, height);
#endif
			cudaInited = true;
		}
		return 0;
	}

	int VideoGen::initX264Encoder(){
#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: init the x264 encoder.\n");
#endif

		if(!x264Inited){
			if(!sourcePipe){
				// error
				return -1;
			}
			// init the filter
			x264Filter = new Filter();
			x264Filter->init(height, width, height, width);
			x264Filter->setOutputFormat(PIX_FMT_YUV420P);
			x264Filter->registerSourcePipe(sourcePipe);
			x264Encoder = new X264Encoder(width, height, 0, x264Filter->getFilterPipe(), writer);

			x264Encoder->InitEncoder();
			// start the filter
#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("[VideoGen]:start the filter.\n");
#endif
			x264Filter->start();  /// start the filter
			x264Inited = true;
		}
		return 0;
	}

	bool VideoGen::activeEncoder(ENCODER_TYPE type){ 

#ifdef ENABLE_GEN_LOG
		infoRecorder->logTrace("[VideoGen]: active the encoder, device:%p \n", dxDevice->d9Device);
#endif
		switch(type){
		case CUDA_ENCODER:
#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("CUDA Encoder.\n");
#endif
			initCudaEncoder(dxDevice->d9Device);
			if(encoder){
				if(!encoder->isStart())
					encoder->start();
				else{
#ifdef ENABLE_GEN_LOG
					infoRecorder->logTrace("[VideoGen]: Cuda Encoder, register to source.\n");
#endif
					// inited and started, when change, register the cuda event
					encoder->registerEvent();
					encoder->encoderChanged();
				}
			}
			encoder->setRefIntraMigrationTimer(intraMigrationTimer);
			break;
		case X264_ENCODER:
#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("X264 Encoder.\n");
			infoRecorder->logError("video gen, use x264 encoder.\n");
#endif
			initX264Encoder();
			if(x264Encoder){
				if(!x264Encoder->isStart()){
					x264Encoder->start();
				}
				else{

#ifdef ENABLE_GEN_LOG
					infoRecorder->logTrace("[VideoGen]: X264 Encoder, register to source.\n");
#endif
					// register the filter event to wrapper
					x264Filter->registerEvent();
					x264Encoder->encoderChanged();
				}
			}
			x264Encoder->setRefIntraMigrationTimer(intraMigrationTimer);
			break;
		case NVENC_ENCODER:
#ifdef ENABLE_GEN_LOG
			infoRecorder->logTrace("NVENC encoder.\n");
#endif
			initNVENCEncoder(dxDevice->d9Device);
			if(nvEncoder){
				if(!nvEncoder->isStart()){
#ifdef ENABLE_GEN_LOG
					infoRecorder->logTrace("[VideoGen]: NVENC Encoder start, encoder:%p.\n", nvEncoder);
#endif
					//DebugBreak();
					nvEncoder->start();
				}
				else{
#ifdef ENABLE_GEN_LOG
					infoRecorder->logTrace("[VideoGen]: NVENC Encoder, register to source.\n");
#endif
					nvEncoder->registerEvent();
					nvEncoder->encoderChanged();
				}
			}else{
				infoRecorder->logError("[VideoGen]: NVENC Encoder get NULL nvEncoder.\n");
			}
			nvEncoder->setRefIntraMigrationTimer(intraMigrationTimer);
			break;

		case ADAPTIVE_CUDA:
			infoRecorder->logError("[VideoGen]: use x264 and CUDA encoder, not used????.\n");

			initX264Encoder();
			initCudaEncoder(dxDevice->d9Device);
			break;
		case ADAPTIVE_NVENC:
			infoRecorder->logError("[VideoGen]: use x264 and NVENC encoder. not used ????\n");
			initX264Encoder();
			initNVENCEncoder(dxDevice->d9Device);
			break;

		default:
			infoRecorder->logError("[VideoGen]: ERROR, invalid encoder type.\n");
			return false;
		}

		return true;
	}

	// init the rtsp context with given socket
	bool VideoGen::initRTSPContext(SOCKET s){
		infoRecorder->logError("[VideoGen]: init RTSP context, TODO.\n");
		return true;
	}

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////


	// tool function to write message on screen

	BOOL _stdcall DrawMyText(LPDIRECT3DDEVICE9 pDxdevice,TCHAR* strText ,int nbuf)
	{
		if(pDxdevice){

			RECT myrect;
			myrect.top=150; //y coordinate
			myrect.left=0; //x coordinate
			myrect.right=500+myrect.left;
			myrect.bottom=100+myrect.top;
			pDxdevice->BeginScene();//begin to draw text

			D3DXFONT_DESCA lf;
			ZeroMemory(&lf, sizeof(D3DXFONT_DESCA));
			lf.Height = 24; //font height
			lf.Width = 12; // font width
			lf.Weight = 100; 
			lf.Italic = false;
			lf.CharSet = DEFAULT_CHARSET;
			strcpy(lf.FaceName, "Times New Roman"); //font type 
			ID3DXFont* font=NULL;
			if(D3D_OK!=D3DXCreateFontIndirect(pDxdevice, &lf, &font)) // create font object
				return false;

			font->DrawText(
				NULL,
				strText, // text to draw 
				nbuf, 
				&myrect, 
				DT_TOP | DT_LEFT, // align to left
				D3DCOLOR_ARGB(255,255,255,0)); 

			pDxdevice->EndScene();// end drawing
			font->Release();//release objects
		}
		return true;
	}
}