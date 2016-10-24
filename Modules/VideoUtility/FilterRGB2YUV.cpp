#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <map>
#include "avcodeccommon.h"
#include "pipeline.h"
#include "rtspconf.h"
#include "../VideoGen/rtspcontext.h"
#include "encoder.h"
#include "FilterRGB2YUV.h"
#include "avcodeccommon.h"
#include "vconventer.h"

#include "..\LibCore\BmpFormat.h"
#include "..\LibCore\InfoRecorder.h"

#define USE_NV12_RAW
#define USE_SWS_SCALE


//#define ENABLE_LOG_FILTER

//////////////////////////////////////////////////////////////////////////
//   for filter, the dst pipeline is created by itself, the src pipeline is form outside
//
//////////////////////////////////////////////////////////////////////////

using namespace cg::core;

namespace cg{

	std::map<void *, bool > Filter::initialized;
	HANDLE Filter::initMutex;// = NULL;
	HANDLE Filter::filterMutex;
	std::map<std::string, Filter *> Filter::filterMap;

	void RGB_TO_Y_CPU(const unsigned char b, const unsigned char g, const unsigned char r, unsigned char & y){
		y = static_cast<unsigned char >((((int ) 66 * r) + (int)(129 * g) + (int)(25 * b) + 128) + 16);
	}
	void RGB_TO_YUV_CPU(const unsigned char b, const unsigned char g, const unsigned char r, unsigned char & y, unsigned char & u, unsigned char & v){
		RGB_TO_Y_CPU(b, g, r, y);
		u = static_cast<unsigned char>((((int)(-38 * r) - (int)(74 * g) + (int)(112 * b) + 128)>>8)+128);
		v = static_cast<unsigned char>((((int)(112 * r) - (int)(94 * g) - (int)(19 * b) + 128)>>8)+ 128);
	}

	// convert RGB to VN12 format, src is four channeled
	void RGB_TO_NV12_CPU(unsigned char * src, int src_width, int src_height, int src_stride, int channels, unsigned char * dst, int dst_width, int dst_height, int dst_stride){
		unsigned char y_val = 0, u_val = 0, v_val = 0;
		unsigned char r = 0, g = 0, b = 0;
		unsigned char *y_plane = NULL, * uv_plane = NULL;
		int uv_off = 0;

		const size_t planSize=  src_height * dst_stride;   // plan size;
		y_plane = dst;
		uv_plane = y_plane + planSize;

		if(channels == 4){
			// format is ARGB ????
		}
		else if(channels == 3){
			// format is BGRA
		}
		// for each rows
		for(int y = 0; y < src_height; y += 2){
			for(int x = 0; x < src_width; x += 2){

				// ( x, y )
				b = *(src + y * src_stride + x * channels);
				g = *(src + y * src_stride + x * channels + 1);
				r = *(src + y * src_stride + x * channels + 2);

				RGB_TO_Y_CPU(b, g, r, y_val);
				y_plane[ y * dst_stride + x] = y_val;

				// ( x + 1, y)
				b = *(src + y * src_stride + (x + 1) * channels);
				g = *(src + y * src_stride + (x + 1) * channels + 1);
				r = *(src + y * src_stride + (x + 1) * channels + 2);
				RGB_TO_Y_CPU(b, g, r, y_val);
				y_plane[ y * dst_stride + x + 1] = y_val;

				// ( x, y + 1)
				b = *(src + (y + 1) * src_stride + (x) * channels);
				g = *(src + (y + 1) * src_stride + (x) * channels + 1);
				r = *(src + (y + 1) * src_stride + (x) * channels + 2);
				RGB_TO_Y_CPU(b, g, r, y_val);
				y_plane[ (y + 1) * dst_stride + x ] = y_val;
				// ( x + 1, y + 1)
				b = *(src + (y + 1) * src_stride + (x + 1) * channels);
				g = *(src + (y + 1) * src_stride + (x + 1) * channels + 1);
				r = *(src + (y + 1) * src_stride + (x + 1) * channels + 2);
				// get YUV

				RGB_TO_YUV_CPU(b, g, r, y_val, u_val, v_val);
				y_plane[ (y + 1) * dst_stride + x + 1] = y_val;
				uv_off = y / 2 * dst_stride + x / 2 * 2;

				uv_plane[uv_off] = u_val;
				uv_plane[uv_off + 1] = v_val;

			}
		}
	}

	Filter::Filter(){
		outputW = 0;
		outputH = 0;
		srcPipe = NULL;
		dstPipe = NULL;
		outputFormat =PIX_FMT_NONE; 
		swsCtx = NULL;

		InitializeCriticalSection(&fmtSection);
		myname = std::string("Filter");
		convertTime = 0;
		pTimer = new cg::core::PTimer();
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

		if (srcPipe){
			delete srcPipe;
			srcPipe = NULL;
		}
		if (dstPipe){
			delete dstPipe;
			dstPipe = NULL;
		}
		if(pTimer){
			delete pTimer;
			pTimer = NULL;
		}
	}

	// 
	bool Filter::initDestinationPipeline(){
#ifdef ENABLE_LOG_FILTER
		infoRecorder->logError("[Filter]: init the filter pipeline, w:%d, h:%d, stride:%d.\n", outputW, outputH, outputW);
#endif
		struct pooldata * data = NULL;
		if((dstPipe = new pipeline(sizeof(VsourceConfig), "filter")) == NULL){
			infoRecorder->logError("[Filter]: init the destination pipeline failed.\n");
			return false;
		}

		VsourceConfig config;
		bzero(&config, sizeof(config));
		config.id = 0;
		config.rtpId = 0;
		config.maxHeight = outputH;
		config.maxWidth = outputW;
		config.maxStride = outputW << 1;    //// ??????

		dstPipe->set_privdata(&config, sizeof(struct VsourceConfig));
		if((data = dstPipe->datapool_init(POOLSIZE)) == NULL){
			infoRecorder->logError("[Filter]: init destination pipeline failed.\n");
			delete dstPipe;
			dstPipe = NULL;
			return false;
		}
#ifdef ENABLE_LOG_FILTER
		infoRecorder->logError("[Filter]: to init each frame.\n");
#endif
		for(; data != NULL; data = data->next){
			ImageFrame * frame = new ImageFrame();
			if(frame->init(outputW, outputH * 2, outputW) == NULL){
				infoRecorder->logError("[Filter]: allocate memory for Filter Image failed.\n");
				return false;
			}
			data->ptr = frame;
		}
		return true;
	}

	// the new init function to create the sws, the pipeline 
	int Filter::init(int iheight, int iwidth, int outH, int outW){
		struct SwsContext * swsctx = NULL;
		RTSPConf * conf = RTSPConf::GetRTSPConf();

		outputH = outH;
		outputW = outW;

		// create the default conventers
#ifndef USE_NV12_RAW
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
#else
		// create YUV420 filter
#ifdef ENABLE_LOG_FILTER
		infoRecorder->logError("[Filter]: to create YUV420P filter.\n");
#endif
		do{
			char pixelFmt[64];
			// read the pixel format from config file
			if(conf->confReadV("filter-source-pixelformat", pixelFmt, sizeof(pixelFmt)) != NULL){
				if(strcasecmp("rgba", (const char *)pixelFmt) == 0){
					swsctx = Conventer::createFrameConventer(
						iwidth, iheight, PIX_FMT_RGBA,
						outputW, outputH, PIX_FMT_NV12);

					swsctx = Conventer::createFrameConventer(
						iwidth, iheight, PIX_FMT_RGBA,
						outputW, outputH, PIX_FMT_YUV420P);
					infoRecorder->logError("[Filter]: RGBA source specificed.\n");
				}
				else if(strcasecmp("bgra", pixelFmt) == 0){
					swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_BGRA, outputW, outputH, PIX_FMT_NV12);
					swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_BGRA, outputW, outputH, PIX_FMT_YUV420P);
					infoRecorder->logError("[Filter]: BGRA source specificed.\n");
				}
			}
			if(swsctx == NULL){
				swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_RGBA, outputW, outputH, PIX_FMT_NV12);
				swsctx = Conventer::createFrameConventer(iwidth, iheight, PIX_FMT_RGBA, outputW, outputH, PIX_FMT_YUV420P);
				infoRecorder->logTrace("[Filter]: no special source format, use RGBA.\n");
			}
		}while(0);
		// create NV12 filter
#ifdef ENABLE_LOG_FILTER
		infoRecorder->logError("[Filter]: to create NV12 filter.\n");
#endif
		do{
			char pixelFmt[64];
			// read the pixel format from config file
			if(conf->confReadV("filter-source-pixelformat", pixelFmt, sizeof(pixelFmt)) != NULL){
				if(strcasecmp("rgba", (const char *)pixelFmt) == 0){
					infoRecorder->logTrace("[Filter]: RGBA source specificed.\n");
				}
				else if(strcasecmp("bgra", pixelFmt) == 0){
					infoRecorder->logTrace("[Filter]: BGRA source specificed.\n");
				}
			}
			if(swsctx == NULL){

			}
		}while(0);
#endif
		// check
		if(swsctx == NULL){
			infoRecorder->logError("[Filter]: cannot initialize swsscale.\n");
			return -1;
		}
		inited = true;
		return 0;
	}

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
		//ImageFrame * srcframe = NULL;
		SourceFrame * srcframe = NULL;
		ImageFrame * dstframe = NULL;

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
		
		if ((filter = Filter::lookup(filterpipe[1])) == NULL){
			infoRecorder->logError("[Filter]: cannot find the filter.\n");
			goto filter_quit;
		}

#ifdef ENABLE_LOG_FILTER
		infoRecorder->logTrace("RGB2YUV filter: pipe from '%s' to '%s' (output-resolution=%dx%d)\n",
			srcpipe->name(), dstpipe->name(), filter->getW(), filter->getH());
#endif

		condMutex = CreateMutex(NULL, FALSE, NULL);
		cond = CreateEvent(NULL, FALSE, FALSE, NULL);
		// register the event 
		srcpipe->client_register(ccg_gettid(), cond);

		while (true){
			// wait for notification
			srcdata = srcpipe->load_data();
			if (srcdata == NULL){
				infoRecorder->logError("[Filter]: to wait %p, src pipe:%s\n", cond, srcpipe->name());
				srcpipe->wait(cond, condMutex);
				srcdata = srcpipe->load_data();
				if (srcdata == NULL){
					infoRecorder->logError("RGB2YUB fitler: unexpected NULL frame received (from '%s', data = %d, buf = %d).\n",
						srcpipe->name(), srcpipe->data_count(), srcpipe->buf_count());
					continue;
					// should never be here
					goto filter_quit;
				}
			}
			srcframe = (SourceFrame *)srcdata->ptr;
			dstdata = dstpipe->allocate_data();
			dstframe = (ImageFrame *)dstdata->ptr;

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
#ifdef SHARE_CONVENTER
					swsCtx = Conventer::lookupFrameConventer(srcframe->realWidth,
						srcframe->realHeight, srcframe->pixelFormat, dstframe->pixelFormat);

					if (swsCtx == NULL){
						swsCtx = Conventer::createFrameConventer(srcframe->realWidth, srcframe->realHeight, srcframe->pixelFormat,
							filter->getW(), filter->getH(), dstframe->pixelFormat);
					}

#endif
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
						outlinesize[0] = filter->getW() * 4;

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
		std::map<std::string, Filter *>::iterator mi;
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

	// stop the filter 
	BOOL Filter::stop(){
		return TRUE;
	}
	// loop logic to execute in filter

	BOOL Filter::run(){
		struct pooldata * srcdata = NULL;
		struct pooldata * dstdata = NULL;
		//ImageFrame * iframe = NULL;
		SourceFrame * iframe = NULL;
		ImageFrame * dstframe = NULL;

		unsigned char * src[] = { NULL, NULL, NULL, NULL };
		unsigned char * dst[] = { NULL, NULL, NULL, NULL };
		int srcstride[] = { 0, 0, 0, 0 };
		int dststride[] = { 0, 0, 0, 0 };

		

		srcdata = srcPipe->load_data();
		if(srcdata == NULL){
			// wait the data
			infoRecorder->logTrace("[Filter]: wait for pipe %s notify event %p.\n", srcPipe->name(), cond);
			srcPipe->wait(cond, condMutex);
			srcdata = srcPipe->load_data();

			if(srcdata == NULL){
				//infoRecorder->logError("Filter]: get NULL src data from source pipeline .\n");
				return FALSE;
			}
		}

		pTimer->Start();

		iframe = (SourceFrame *)srcdata->ptr;
		dstdata = dstPipe->allocate_data();

		dstframe = (ImageFrame *)dstdata->ptr;
		dstframe->imgPts = iframe->imgPts;

		if(getOutputFormat() == PIX_FMT_YUV420P){
			dstframe->pixelFormat = PIX_FMT_YUV420P;
		}
		else if(getOutputFormat() == PIX_FMT_NV12){
			dstframe->pixelFormat = PIX_FMT_NV12;
		}
		else if(getOutputFormat() == PIX_FMT_NONE){
			outputFormat = PIX_FMT_YUV420P;
			dstframe->pixelFormat = PIX_FMT_YUV420P;
		}
#ifdef ENABLE_LOG_FILTER
		infoRecorder->logError("[Filter]: output format: %s.\n", outputFormat == PIX_FMT_YUV420P ? "YUV420P" : "NV12");
#endif
		// scale the image: xxx: RGBA or BGRA
		if(iframe->pixelFormat == PIX_FMT_RGBA || iframe->pixelFormat == PIX_FMT_BGRA){
			if(swsCtx == NULL){
				swsCtx = Conventer::lookupFrameConventer(iframe->realWidth, iframe->realHeight, iframe->pixelFormat, dstframe->pixelFormat);
				if(swsCtx == NULL){
					// not found, create new
					swsCtx = Conventer::createFrameConventer(iframe->realWidth, iframe->realHeight, iframe->pixelFormat, this->getW(), this->getH(), dstframe->pixelFormat);
				}
				if(swsCtx == NULL){
					infoRecorder->logError("[Filter]: fatal - cannot create frame conventor.\n");
				}
			}
			// get the swsctx
			if(dstframe->pixelFormat == PIX_FMT_YUV420P){
#ifdef ENABLE_LOG_FILTER
				infoRecorder->logError("[Filter]: to convent to YUV420P: h:%d, w:%d, v stride:%d, source stride:%d, source H:%d, source W:%d.\n", getH(), getW(), getW() >> 1, iframe->realStride, iframe->realHeight, iframe->realWidth);
#endif
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

				dstframe->realHeight = getH();
				dstframe->realWidth = getW();

				sws_scale(swsCtx, src, srcstride, 0, iframe->realHeight,
					dst, dstframe->lineSize);

			}
			else if(dstframe->pixelFormat == PIX_FMT_NV12){
#ifdef USE_SWS_SCALE
				src[0] = iframe->imgBuf;
				src[1] = 0; //iframe->imgBuf + iframe->realStride * iframe->realHeight;
				srcstride[0] = iframe->realStride;
				srcstride[1] = 0;

				dst[0] = dstframe->imgBuf;
				dst[1] = dstframe->imgBuf + getH() * getW();
				dst[2] = 0;
				dst[3] = 0;

				dstframe->lineSize[0] = dststride[0] = getW();
#if 0
				dstframe->lineSize[1] = dststride[1] = getW() >> 1;
				dstframe->lineSize[2] = dststride[2] = getW() >> 1;
#else
				dstframe->lineSize[1] = dststride[1] = getW();
				dstframe->lineSize[2] = dststride[2] = 0;
#endif
				dstframe->realWidth = getW();
				dstframe->realHeight = getH();
				dstframe->lineSize[3] = dststride[3] = 0;
				//printf("convert to NV12, width:%d, height:%d", getW(), getH());
				// convert the format
				sws_scale(swsCtx, src, srcstride, 0, iframe->realHeight, dst, dstframe->lineSize);
#else
				RGB_TO_NV12_CPU(iframe->imgBuf, getW(), getH(), iframe->realStride, 4, dstframe->imgBuf, getW(), getH(), getW());
				dstframe->realWidth = getW();
				dstframe->realHeight = getH();
#endif
#if 0
				BYTE * rgb24 = new BYTE[getH() * getW() * 3];
				NV12ToBGR24_Native(dstframe->imgBuf, rgb24, getW(), getH());
				long bmp_size =0;
				BYTE * bmp = ConvertRGBToBMPBuffer(rgb24, getW(), getH(), & bmp_size);
				static int in = 0;
				char tmp[512] = {0};
				sprintf(tmp, "bmp_%d.bmp", in++);
				SaveBMP(bmp, getW(), getH(), bmp_size, tmp);

				delete[] rgb24;
				delete[] bmp;
#endif
			}
		}
		else{
			infoRecorder->logError("[Filter]: src frame format is not supported.\n");
		}
		srcPipe->release_data(srcdata);
		dstPipe->store_data(dstdata);

		convertTime = pTimer->Stop();

		infoRecorder->addConvertTime(getConvertTime());

#ifdef ENABLE_LOG_FILTER
		infoRecorder->logError("[Filter]: notify encoder.\n");
#endif
		dstPipe->notify_all();
		return TRUE;
	}

	// deal the msg
	void Filter::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
		//do nothing
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
	BOOL Filter::onThreadStart(){
		if(!this->isInited()){
			infoRecorder->logError("[Filter]: enter thread proc without initialization.\n");
			return FALSE;
		}

		if(srcPipe == NULL){
			infoRecorder->logError("[Filter]: without source pipeline .\n");
			return FALSE;
		}
		if(dstPipe == NULL){
			infoRecorder->logError("[Filter]: without destination pipeline.\n");
			return FALSE;
		}

		// do the setup work here
		condMutex = CreateMutex(NULL, FALSE, NULL);
		//if(cond == NULL)
		cond = CreateEvent(NULL, FALSE, FALSE, NULL);

		srcPipe->client_register(ccg_gettid(), cond);
		return TRUE;
	}

}