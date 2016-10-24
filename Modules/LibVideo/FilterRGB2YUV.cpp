#include <map>
#include "Commonwin32.h"
#include "Pipeline.h"
#include "Config.h"
#include "RtspConf.h"
#include "RtspContext.h"

#include "Encoder.h"
#include "FilterRGB2YUV.h"

#include "VSource.h"
#include "AVCodecCommon.h"
#include "VConventer.h"

#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"

map<void *, bool > Filter::initialized;
HANDLE Filter::initMutex;// = NULL;
HANDLE Filter::filterMutex;
map<string, Filter *> Filter::filterMap;

Filter::Filter(){
	outputW = 0;
	outputH = 0;
	

	outputFormat = PIX_FMT_NONE;

	InitializeCriticalSection(&fmtSection);
}

void Filter::Release(){
	if (filterMutex){
		CloseHandle(filterMutex);
		filterMutex = NULL;
	}
	if (initMutex){
		CloseHandle(initMutex);
		initMutex = NULL;
	}
}
Filter::~Filter(){

	
}

int Filter::init(void * arg)
{
	int iid;
	int iwidth;
	int iheight;
	int istride;
	int resolution[2];

	Log::logscreen("[filter]: filter init called.\n");
	if (initMutex == NULL){
		// init the mutex
		initMutex = CreateMutex(NULL, FALSE, NULL);
	}
	if (filterMutex == NULL){
		filterMutex = CreateMutex(NULL, FALSE, NULL);
	}

	const char ** filterpipe= (const char **)arg;
	pipeline *srcpipe = pipeline::lookup(filterpipe[0]);
	pipeline * pipe = NULL;
	struct pooldata *data = NULL;
	struct SwsContext * swsctx = NULL;

	map<void *, bool>::iterator mi;
	WaitForSingleObject(initMutex, INFINITE);

	if ((mi = initialized.find(arg)) != initialized.end()){
		if (mi->second != false){
			// has been initialized
			ReleaseMutex(initMutex);
			return 0;
		}
	}

	ReleaseMutex(initMutex);

	if (srcpipe == NULL){
		infoRecorder->logError("RGB2YUV filter: init - NULL pipeline specified (%s). \n", filterpipe[0]);
		goto init_failed;
	}
	Log::logscreen("[filter]: src pipie:%p\n", srcpipe);
	//DebugBreak();
	VsourceConfig * vsourceConf = (struct VsourceConfig *)srcpipe->get_privdata();
	iid = vsourceConf->id;
	iwidth = vsourceConf->maxWidth;
	iheight = vsourceConf->maxHeight;
	istride = vsourceConf->maxStride;

	outputW = iwidth;
	outputH = iheight;
	// read the config file for the resolution
	//DebugBreak();
	RTSPConf * conf = RTSPConf::GetRTSPConf();
	if (!conf){
		Log::logscreen("[filter]: get rtspconfig failed.\n");
	}
	if (conf->confReadInts("output-resolution", resolution, 2) == 2){
		outputW = resolution[0];
		outputH = resolution[1];
		infoRecorder->logTrace("[Filter]: output-resolution specified: %dx%d.\n", outputW, outputH);
	}
	// create default conventers
	do{
		char pixelFmt[64];
		// read the pixel format from config file
		if (conf->confReadV("filter-source-pixelformat", pixelFmt, sizeof(pixelFmt)) != NULL){
			if (strcasecmp("rgba", (const char *)pixelFmt) == 0){
				swsctx = Conventer::createFrameConventer(
					iwidth, iheight, PIX_FMT_RGBA,
					outputW, outputH, PIX_FMT_YUV420P);
				infoRecorder->logTrace("RGB2YUV filter: RGBA source specified\n");
			}
			else if (strcasecmp("bgra", pixelFmt) == 0){
				swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_BGRA,
					outputW, outputH, PIX_FMT_YUV420P);
				infoRecorder->logTrace("RGB2YUV filter: BGRA source specified.\n");
			}
		}
		if (swsctx == NULL){
#ifdef __APPLE__
			swsctx = Conventer::createFrameConventer(
				iwidth, iheight, PIX_FMT_RGBA,
				outputW, outputH, PIX_FMT_YUV420P);
#else
			swsctx = Conventer::createFrameConventer(
				iwidth, iheight, PIX_FMT_BGRA,
				outputW, outputH, PIX_FMT_YUV420P);
#endif
		}
	} while (0);
	if (swsctx == NULL){
		infoRecorder->logError("RGB2YUV filter: connot initialize swsscale.\n");
		goto init_failed;
	}

	if ((pipe = new pipeline(0)) == NULL){
		infoRecorder->logError("RGB2YUV filter: init pipeline failed.\n");
		goto init_failed;
	}

	this->dstPipe = pipe;
	Log::logscreen("[filter]: filter pipeline: 0x%p\n", filterpipe);

	// has privadata from source?
	if (srcpipe->get_privdata_size() > 0){
		if (pipe->alloc_privdata(srcpipe->get_privdata_size()) == NULL){
			infoRecorder->logError("RGB2YUV filter: cannot allocate privdata.\n");
			goto init_failed;
		}
		pipe->set_privdata(srcpipe->get_privdata(), srcpipe->get_privdata_size());
	}
	// allocate the data for the image frame
	if ((data = pipe->datapool_init(POOLSIZE, sizeof(struct ImageFrame))) == NULL){
		infoRecorder->logError("RGB2YUV filter: cannot allocate data pool.\n");
		goto init_failed;
	}
	// per frame init
	for (; data != NULL; data = data->next){
		ImageFrame * frame = (ImageFrame *)data->ptr;
		if (frame->init(iwidth, iheight, istride) == NULL){
			infoRecorder->logError("RGB2YUV filter: init frame failed!\n");
			goto init_failed;
		}
	}

	//
	pipeline::do_register(filterpipe[1], pipe);

	WaitForSingleObject(initMutex, INFINITE);
	initialized[arg] = true;
	ReleaseMutex(initMutex);

	return 0;
init_failed:
	if (pipe){
		delete pipe;
		pipe = NULL;
	}
	return -1;


}

// the new init function to create the sws, the pipeline 
int Filter::init(int iheight, int iwidth, int outH, int outW){
	infoRecorder->logTrace("[Filter]: init the filter.\n");
	struct SwsContext * swsctx = NULL;
	RTSPConf * conf = RTSPConf::GetRTSPConf();

	outputH = outH;
	outputW = outW;

	// create the default conventers
	do{
		char pixelFmt[64];
		// read the pixel format from config file
		if(conf->confReadV("filter-source-pixelformat", pixelFmt, sizeof(pixelFmt)) != NULL){
			if(strcasecmp("rgba", (const char *)pixelFmt) == 0){
				swsctx = Conventer::createFrameConventer(
					iwidth, iheight, PIX_FMT_RGBA,
					outputW, outputH, PIX_FMT_YUV420P);
				infoRecorder->logTrace("[Filter]: RGBA source specificed.\n");
			}
			else if(strcasecmp("bgra", pixelFmt) == 0){
				swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_BGRA, outputW, outputH, PIX_FMT_YUV420P);
				infoRecorder->logTrace("[Filter]: BGRA source specificed.\n");
			}
		}
		if(swsctx == NULL){
			swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_RGBA, outputW, outputH, PIX_FMT_YUV420P);
		}
	}while(0);
	// check
	if(swsctx == NULL){
		infoRecorder->logError("[Filter]: cannot initialize swsscale.\n");
		return -1;
	}
	return 0;
}

#if 0
int Filter::init(pipeline * source, char * filterPipeName){
	int iid;
	int iwidth;
	int iheight;
	int istride;
	int resolution[2];

	if (initMutex == NULL){
		// init the mutex
		initMutex = CreateMutex(NULL, FALSE, NULL);
	}

	//const char ** filterpipe = (const char **)arg;

	registerSourcePipe(source);
	pipeline *srcpipe = source; 
	pipeline * pipe = NULL;
	struct pooldata *data = NULL;
	struct SwsContext * swsctx = NULL;

	map<void *, bool>::iterator mi;
	WaitForSingleObject(initMutex, INFINITE);

	if ((mi = initialized.find(filterPipeName)) != initialized.end()){
		if (mi->second != false){
			// has been initialized
			ReleaseMutex(initMutex);
			return 0;
		}
	}

	ReleaseMutex(initMutex);

	if (srcpipe = NULL){
		infoRecorder->logError("RGB2YUV filter: init - NULL pipeline specified (%s). \n", filterPipeName);
		goto init_failed;
	}
	VsourceConfig * vsourceConf = (struct VsourceConfig *)srcpipe->get_privdata();
	iid = vsourceConf->id;
	iwidth = vsourceConf->maxWidth;
	iheight = vsourceConf->maxHeight;
	istride = vsourceConf->maxStride;

	outputW = iwidth;
	outputH = iheight;
	// read the config file for the resolution

	// create default concerters
	do{
		char pixelFmt[64];
		// read the pixel format from config file
		if (strcasecmp("rgba", pixelFmt) == 0){
			swsctx = Conventer::createFrameConventer(
				iwidth, iheight, PIX_FMT_RGBA,
				outputW, outputH, PIX_FMT_YUV420P);
			infoRecorder->logTrace("RGB2YUV filter: RGBA source speciffied\n");
		}
		else if (strcasecmp("bgra", pixelFmt) == 0){
			swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_BGRA,
				outputW, outputH, PIX_FMT_YUV420P);
			infoRecorder->logTrace("RGB2YUV filter: BGRA source specified.\n");
		}
	} while (0);
	if (swsctx == NULL){
		infoRecorder->logError("RGB2YUV filter: connot initialize swsscale.\n");
		goto init_failed;
	}

	if ((pipe = new pipeline()) == NULL){
		infoRecorder->logError("RGB2YUV filter: init pipeline failed.\n");
		goto init_failed;
	}

	// has privadata from source?
	if (srcpipe->get_privdata_size() > 0){
		if (pipe->alloc_privdata(srcpipe->get_privdata_size()) == NULL){
			infoRecorder->logError("RGB2YUV filter: cannot allocate privdata.\n");
			goto init_failed;
		}
		pipe->set_privdata(srcpipe->get_privdata(), srcpipe->get_privdata_size());
	}
	// allocate the data for the image frame
	if ((data = pipe->datapool_init(POOLSIZE, sizeof(struct ImageFrame))) == NULL){
		infoRecorder->logError("RGB2YUV filter: cannot allocate data pool.\n");
		goto init_failed;
	}
	// per frame init
	for (; data != NULL; data = data->next){
		ImageFrame * frame = (ImageFrame *)data->ptr;
		if (frame->init(iwidth, iheight, istride) == NULL){
			infoRecorder->logError("RGB2YUV filter: init frame failed!\n");
			goto init_failed;
		}
	}
	//
	pipeline::do_register(filterPipeName, pipe);
	WaitForSingleObject(initMutex, INFINITE);
	initialized[filterPipeName] = true;
	ReleaseMutex(initMutex);

	return 0;
init_failed:
	if (pipe){
		delete pipe;
		pipe = NULL;
	}
	return -1;
}

int Filter::init(char * sourcePipeName, char * filterPipeName){
	int iid;
	int iwidth;
	int iheight;
	int istride;
	int resolution[2];

	if (initMutex == NULL){
		// init the mutex
		initMutex = CreateMutex(NULL, FALSE, NULL);
	}

	//const char ** filterpipe = (const char **)arg;

	pipeline *srcpipe = pipeline::lookup(sourcePipeName);
	pipeline * pipe = NULL;
	struct pooldata *data = NULL;
	struct SwsContext * swsctx = NULL;

	map<void *, bool>::iterator mi;
	WaitForSingleObject(initMutex, INFINITE);

	if ((mi = initialized.find(filterPipeName)) != initialized.end()){
		if (mi->second != false){
			// has been initialized
			ReleaseMutex(initMutex);
			return 0;
		}
	}

	ReleaseMutex(initMutex);

	if (srcpipe = NULL){
		infoRecorder->logError("RGB2YUV filter: init - NULL pipeline specified (%s). \n", filterPipeName);
		goto init_failed;
	}
	VsourceConfig * vsourceConf = (struct VsourceConfig *)srcpipe->get_privdata();
	iid = vsourceConf->id;
	iwidth = vsourceConf->maxWidth;
	iheight = vsourceConf->maxHeight;
	istride = vsourceConf->maxStride;

	outputW = iwidth;
	outputH = iheight;
	// read the config file for the resolution

	// create default concerters
	do{
		char pixelFmt[64];
		// read the pixel format from config file
		if (strcasecmp("rgba", pixelFmt) == 0){
			swsctx = Conventer::createFrameConventer(
				iwidth, iheight, PIX_FMT_RGBA,
				outputW, outputH, PIX_FMT_YUV420P);
			infoRecorder->logTrace("RGB2YUV filter: RGBA source speciffied\n");
		}
		else if (strcasecmp("bgra", pixelFmt) == 0){
			swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_BGRA,
				outputW, outputH, PIX_FMT_YUV420P);
			infoRecorder->logTrace("RGB2YUV filter: BGRA source specified.\n");
		}
	} while (0);
	if (swsctx == NULL){
		infoRecorder->logError("RGB2YUV filter: connot initialize swsscale.\n");
		goto init_failed;
	}

	if ((pipe = new pipeline()) == NULL){
		infoRecorder->logError("RGB2YUV filter: init pipeline failed.\n");
		goto init_failed;
	}

	// has privadata from source?
	if (srcpipe->get_privdata_size() > 0){
		if (pipe->alloc_privdata(srcpipe->get_privdata_size()) == NULL){
			infoRecorder->logError("RGB2YUV filter: cannot allocate privdata.\n");
			goto init_failed;
		}
		pipe->set_privdata(srcpipe->get_privdata(), srcpipe->get_privdata_size());
	}
	// allocate the data for the image frame
	if ((data = pipe->datapool_init(POOLSIZE, sizeof(struct ImageFrame))) == NULL){
		infoRecorder->logError("RGB2YUV filter: cannot allocate data pool.\n");
		goto init_failed;
	}
	// per frame init
	for (; data != NULL; data = data->next){
		ImageFrame * frame = (ImageFrame *)data->ptr;
		if (frame->init(iwidth, iheight, istride) == NULL){
			infoRecorder->logError("RGB2YUV filter: init frame failed!\n");
			goto init_failed;
		}
	}
	//
	pipeline::do_register(filterPipeName, pipe);
	WaitForSingleObject(initMutex, INFINITE);
	initialized[filterPipeName] = true;
	ReleaseMutex(initMutex);

	return 0;
init_failed:
	if (pipe){
		delete pipe;
		pipe = NULL;
	}
	return -1;
}
#endif
// this the thread procedure for filter
// param[0]: source pipe name
// param[1]: dst pipeline name
// param[2]: used fitler name
DWORD WINAPI FilterThreadProc(LPVOID param)
{
	const char ** filterpipe = (const char **)param;
	// arg is pointer to source pip
	//const char ** filterpipe = (const char **)param;
	// param is the filter
	Filter * filter = NULL;
	infoRecorder->logTrace("[Filter]: thread proc param[0]:%s, param[1]:%s\n", filterpipe[0], filterpipe[1]);

	pipeline *srcpipe = pipeline::lookup(filterpipe[0]);
	//pipeline * srcpipe = filter->getSrcPipe();
	pipeline * dstpipe = NULL;
	struct pooldata * srcdata = NULL;
	struct pooldata * dstdata = NULL;
	struct ImageFrame * srcframe = NULL;
	struct ImageFrame * dstframe = NULL;

	// image info

	unsigned char * src[] = { NULL, NULL, NULL, NULL };
	unsigned char * dst[] = { NULL, NULL, NULL, NULL };
	int srcstride[] = { 0, 0, 0, 0 };
	int dststride[] = { 0, 0, 0, 0 };

	struct SwsContext * swsCtx = NULL;

	struct SwsContext * n12SwsCtx = NULL;// for cuda encoder

	HANDLE condMutex = NULL;
	HANDLE cond = NULL;

	if (srcpipe == NULL){
		infoRecorder->logTrace("RGB2YUV fitler: NULL pipeline specified.\n");
		goto filter_quit;
	}
	// init variables
#if 0
	if ((dstpipe = filter->getFilterPipe()) == NULL){
#else
	if ((dstpipe = pipeline::lookup(filterpipe[1])) == NULL){
#endif
		infoRecorder->logError("RGB2YUV filter: cannot find filter pipeline \n");
		goto filter_quit;
	}
	//if ((dstpipe = pipeline::lookup(filterpipe[1])) == NULL){
	//	infoRecorder->logError("RGB2YUV filter: cannot find pipeline '%s'\n", filterpipe[1]);
	//	goto filter_quit;
	//}
	if ((filter = Filter::lookup(filterpipe[1])) == NULL){
		infoRecorder->logError("[Filter]: cannot find the filter.\n");
		goto filter_quit;
	}

	infoRecorder->logTrace("RGB2YUV filter: pipe from '%s' to '%s' (output-resolution=%dx%d)\n",
		srcpipe->name(), dstpipe->name(), filter->getW(), filter->getH());

	condMutex = CreateMutex(NULL, FALSE, NULL);
	cond = CreateEvent(NULL, FALSE, FALSE, NULL);

	// register the event 
	srcpipe->client_register(ccg_gettid(), cond);

	while (true){
		// wait for notification
		srcdata = srcpipe->load_data();

		if (srcdata == NULL){
			srcpipe->wait(cond, condMutex);
			//ResetEvent(cond);
			srcdata = srcpipe->load_data();
			if (srcdata == NULL){
				infoRecorder->logError("RGB2YUB fitler: unexpected NULL frame received (from '%s', data = %d, buf = %d).\n",
					srcpipe->name(), srcpipe->data_count(), srcpipe->buf_count());
				//exit(-1);
				continue;
				// should never be here
				goto filter_quit;
			}
		}
		srcframe = (struct ImageFrame *)srcdata->ptr;
		dstdata = dstpipe->allocate_data();
		dstframe = (struct ImageFrame *)dstdata->ptr;

		// basic info 
		dstframe->imgPts = srcframe->imgPts;


		if (filter->getOutputFormat() == PIX_FMT_YUV420P){
			dstframe->pixelFormat = PIX_FMT_YUV420P;
		}
		else if (filter->getOutputFormat() == PIX_FMT_NV12){
			dstframe->pixelFormat = PIX_FMT_NV12;
		}

		// scale image: xxx: RGBA or BGRA
		if (srcframe->pixelFormat == PIX_FMT_RGBA ||
			srcframe->pixelFormat == PIX_FMT_BGRA){
				swsCtx = Conventer::lookupFrameConventer(srcframe->realWidth,
					srcframe->realHeight, srcframe->pixelFormat, dstframe->pixelFormat);

				if (swsCtx == NULL){
					swsCtx = Conventer::createFrameConventer(srcframe->realWidth, srcframe->realHeight, srcframe->pixelFormat,
						filter->getW(), filter->getH(), dstframe->pixelFormat);
				}


				if (swsCtx == NULL){
					infoRecorder->logTrace("RGB2YUV filter: fatal - cannot create frame conventer (%d, %d)->(%x, %d).\n",
						srcframe->realWidth, srcframe->realHeight, filter->getW(), filter->getH());
				}

				if (dstframe->pixelFormat == PIX_FMT_YUV420P){
					src[0] = srcframe->imgBuf;
					src[1] = NULL;
					srcstride[0] = srcframe->realStride;
					srcstride[1] = 0;

					dst[0] = dstframe->imgBuf;
					dst[1] = dstframe->imgBuf + filter->getH() * filter->getW();
					dst[2] = dstframe->imgBuf + filter->getH() * filter->getW() + (filter->getH() * filter->getW() >> 2);
					dst[3] = NULL;
					dstframe->lineSize[0] = dststride[0] = filter->getW();
					dstframe->lineSize[1] = dststride[1] = filter->getW() >> 1;
					dstframe->lineSize[2] = dststride[2] = filter->getW() >> 1;
					dstframe->lineSize[3] = dststride[3] = 0;

					sws_scale(swsCtx, src, srcstride, 0, srcframe->realHeight,
						dst, dstframe->lineSize);
				}
				else if (dstframe->pixelFormat == PIX_FMT_NV12){
					src[0] = srcframe->imgBuf;
					src[1] = srcframe->imgBuf + srcframe->realStride * srcframe->realHeight;
					int linesize[2];
					linesize[0] = srcframe->realStride;
					linesize[1] = srcframe->realStride;

					dst[0] = dstframe->imgBuf;
					int outlinesize[1];
					outlinesize[0] == filter->getW() * 4;

					// convert the format
					sws_scale(swsCtx, src, srcstride, 0, srcframe->realHeight, dst, dstframe->lineSize);
				}
		}
		else
		{
			infoRecorder->logTrace("[filter]: src frame format not supported.\n");
		}
		srcpipe->release_data(srcdata);
		dstpipe->store_data(dstdata);
		infoRecorder->logTrace("[Filter]: notify encoder.\n");
		dstpipe->notify_all();
	}

filter_quit:
	if (srcpipe){
		srcpipe->client_unregister(ccg_gettid());
		if (srcpipe)
			delete srcpipe;
		srcpipe = NULL;
	}
	if (dstpipe){
		// free dstpipe
		delete dstpipe;
		dstpipe = NULL;
	}
	if (swsCtx) sws_freeContext(swsCtx);
	infoRecorder->logTrace("RGB2YUV filter: thread terminated!\n");
	return NULL;
}

DWORD Filter::StartFilterThread(LPVOID arg){
	// start the filter proc
	thread = chBEGINTHREADEX(NULL, 0, FilterThreadProc, arg, NULL, &threadId);
	return threadId;
}

int Filter::do_register(const char * provider, Filter * filter){
	DWORD ret = WaitForSingleObject(filterMutex, INFINITE);
	if (filterMap.find(provider) != filterMap.end()){
		// already registered
		ReleaseMutex(filterMutex);
		infoRecorder->logError("[filter]: deplicated filter '%s'\n", provider);
		return -1;
	}
	filterMap[provider] = filter;
	ReleaseMutex(filterMutex);
	filter->myname = provider;
	infoRecorder->logTrace("[filter]: new filter '%s' registered.\n", provider);
	return 0;
}
void Filter::do_unregister(const char * provider){
	DWORD ret = WaitForSingleObject(filterMutex, INFINITE);
	filterMap.erase(provider);
	ReleaseMutex(filterMutex);
	infoRecorder->logTrace("[filter]: filter '%s' unregistered.\n", provider);
	return;
}
Filter * Filter::lookup(const char * provider){
	map<string, Filter *>::iterator mi;
	Filter * filter = NULL;
	DWORD ret = WaitForSingleObject(filterMutex, INFINITE);
	if ((mi = filterMap.find(provider)) == filterMap.end()){
		ReleaseMutex(filterMutex);
		return NULL;
	}
	filter = mi->second;
	ReleaseMutex(filterMutex);
	return filter;
}
const char * Filter::name(){
	return myname.c_str();
}

PixelFormat Filter::getOutputFormat(){
	PixelFormat ret = PIX_FMT_NONE;
	EnterCriticalSection(&fmtSection);

	ret = outputFormat;
	LeaveCriticalSection(&fmtSection);
	return ret;
}

void Filter::setOutputFormat(PixelFormat fmt){
	EnterCriticalSection(&fmtSection);
	outputFormat = fmt;
	LeaveCriticalSection(&fmtSection);
}

// stop the fitler 
BOOL Filter::stop(){

}
// loop logic to execute in fitler

void Filter::run(){
	struct pooldata * srcdata = NULL;
	struct pooldata * dstdata = NULL;
	struct ImageFrame * iframe = NULL;
	struct ImageFrame * dstframe = NULL;

	unsigned char * src[] = { NULL, NULL, NULL, NULL };
	unsigned char * dst[] = { NULL, NULL, NULL, NULL };
	int srcstride[] = { 0, 0, 0, 0 };
	int dststride[] = { 0, 0, 0, 0 };

	srcdata = srcPipe->load_data();
	if(srcdata == NULL){
		// wait the data
		srcPipe->wait(cond, condMutex);
		srcdata = srcPipe->load_data();

		if(srcdata == NULL){
			infoRecorder->logError("Filter]: get NULL src data from source pipeline .\n");
			return;
		}
	}

	iframe = (struct ImageFrame *)srcdata->ptr;
	dstdata = dstPipe->allocate_data();
	dstframe = (struct ImageFrame *)dstdata->ptr;

	dstframe->imgPts = iframe->imgPts;

	if(getOutputFormat() == PIX_FMT_YUV420P){
		dstframe->pixelFormat = PIX_FMT_YUV420P;
	}
	else if(getOutputFormat() == PIX_FMT_NV12){
		dstframe->pixelFormat = PIX_FMT_NV12;
	}
	// scale the image: xxx: RGBA or BGRA
	if(iframe->pixelFormat == PIX_FMT_RGBA || iframe->pixelFormat == PIX_FMT_BGRA){
		if(swsCtx == NULL){
			swsCtx = Conventer::lookupFrameConventer(iframe->realWidth, iframe->realHeight, iframe->pixelFormat, dstframe->pixelFormat);

			if(swsCtx == NULL){
				// not found, create new
				swsCtx == Conventer::createFrameConventer(iframe->realWidth, iframe->realHeight, iframe->pixelFormat, this->getW(), this->getH(), dstframe->pixelFormat);

			}
			if(swsCtx == NULL){
				infoRecorder->logError("[Filter]: fatal - cannot create frame conventer.\n");

			}
		}
		// get the swsctx
		if(dstframe->pixelFormat == PIX_FMT_YUV420P){
			src[0] = iframe->imgBuf;
			src[1] = NULL;
			srcstride[0] = iframe->realStride;
			srcstride[1] = 0;

			dst[0] = dstframe->imgBuf;
			dst[1] = dstframe->imgBuf + getH() * getW();
			dst[2] = dstframe->imgBuf + getH() * getW() + (getH() * getW() >> 2);
			dst[3] = NULL;
			dstframe->lineSize[0] = dststride[0] = getW();
			dstframe->lineSize[1] = dststride[1] = getW() >> 1;
			dstframe->lineSize[2] = dststride[2] = getW() >> 1;
			dstframe->lineSize[3] = dststride[3] = 0;

			sws_scale(swsCtx, src, srcstride, 0, iframe->realHeight,
				dst, dstframe->lineSize);
		}
		else if(dstframe->pixelFormat == PIX_FMT_NV12){
			src[0] = iframe->imgBuf;
			src[1] = iframe->imgBuf + iframe->realStride * iframe->realHeight;
			int linesize[2];
			linesize[0] = iframe->realStride;
			linesize[1] = iframe->realStride;

			dst[0] = dstframe->imgBuf;
			int outlinesize[1];
			outlinesize[0] == getW() * 4;

			// convert the format
			sws_scale(swsCtx, src, srcstride, 0, iframe->realHeight, dst, dstframe->lineSize);
		}
	}
	else{
		infoRecorder->logError("[Filter]: src frame format is not supported.\n");
	}
	srcPipe->release_data(srcdata);
	dstPipe->store_data(dstdata);
	infoRecorder->logTrace("[Filter]: notify encoder.\n");
	dstPipe->notify_all();

}

// deal the msg
void Filter::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
	CThread::onThreadMsg(msg, wParam, lParam);
}

// quit the filter, release the resources
void Filter::onQuit(){
	if(srcPipe){
		srcPipe->client_unregister(ccg_gettid());
		if(srcPipe)
			delete srcPipe;
		srcPipe = NULL;
	}
	if(dstPipe){
		delete dstPipe;
		dstPipe = NULL;
	}
	if(swsCtx)
		sws_freeContext(swsCtx);
	infoRecorder->logTrace("[Filter]: thread terminated.\n");
}


// filter start thread, do the initial work
void Filter::onThreadStart(){
	if(!this->isInited()){
		infoRecorder->logError("[Filter]: enter thread proc without initialization.\n");
		return;
	}

	if(srcPipe == NULL){
		infoRecorder->logError("[Filter]: without source pipeline .\n");
		return;
	}
	if(dstPipe == NULL){
		infoRecorder->logError("[Filter]: without destination pipeline.\n");
		return;
	}

	// do the setup work here
	condMutex = CreateMutex(NULL, FALSE, NULL);
	cond = CreateEvent(NULL, FALSE, FALSE, NULL);

	srcPipe->client_register(ccg_gettid(), cond);

}