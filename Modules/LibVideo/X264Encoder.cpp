#include <sstream>
#include <fstream>
using namespace std;

#include "../LibCore/Log.h"
#if 0
#include "libswscale/swscale.h"
#include "libswresample/swresample.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/base64.h"

#include "Commonwin32.h"
#endif
#include "AVCodecCommon.h"
/// add the encoder include
#include "Pipeline.h"

#include "Config.h"
#include "RtspConf.h"
#include "RtspContext.h"
#include "Encoder.h"
#include "EncoderManager.h"
#include "FilterRGB2YUV.h"
#include "VSource.h"

#include "X264Encoder.h"
#include <libavutil\avutil.h>

#include "../LibCore/TimeTool.h"

#ifndef UINT64_C
#define UINT64_C(val) val##ui64
#endif

#define MULTI_SOURCE

X264Encoder::~X264Encoder(){
	if (sourcePipe){
		delete sourcePipe;
		sourcePipe = NULL;
	}
	if (avcodecOpenMutex){
		CloseHandle(avcodecOpenMutex);
		avcodecOpenMutex = NULL;
	}
	if (condMutex){
		CloseHandle(condMutex);
		condMutex = NULL;
	}
	if (cond){
		CloseHandle(cond);
		cond = NULL;
	}
	if (data){
		free(data);
		data = NULL;
	}
	if (pic_in_buf){
		free(pic_in_buf);
		pic_in_buf = NULL;
	}
}

int X264Encoder::init(void * arg, pipeline * pipe){
	RTSPConf * conf = (RTSPConf *)arg;
	if (conf == NULL){
		return -1;
	}
	this->sourcePipe = pipe;
	Log::logscreen("[X264Encoder]: init called. rtspconfig: %p.\n", conf);
	//InitEncoder(conf, pipe);
	return 0;
}
void X264Encoder::InitEncoder(RTSPConf * rtspConf, pipeline *p){
	// image info
	int iid;
	int iwidth;
	int iheight;
	int rtp_id;
	struct pooldata *data = NULL;
	struct vsource_frame *frame = NULL;
	pipeline *pipe = (pipeline*)p;
	AVCodecContext *encoder = NULL;
	//
	AVFrame *pic_in = NULL;
	unsigned char *pic_in_buf = NULL;
	int pic_in_size;
	unsigned char *nalbuf = NULL, *nalbuf_a = NULL;
	int nalbuf_size = 0, nalign = 0;
	long long basePts = -1LL, newpts = 0LL, pts = -1LL, ptsSync = 0LL;
#if 0
	pthread_mutex_t condMutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
#else
	HANDLE condMutex = NULL;
	HANDLE cond = NULL;
	// create the mutex and the event
	
	avcodecOpenMutex = CreateMutex(NULL, FALSE, NULL);
#endif
	//
	int resolution[2];
	int video_written = 0;
	//
	if (pipe == NULL) {
		infoRecorder->logTrace("video encoder: NULL pipeline specified.\n");
		goto video_quit;
	}
	
	this->setRtspConf(rtspConf);
	
	// init variables
	iid = ((struct VsourceConfig*) pipe->get_privdata())->id;
	struct VsourceConfig * sourceConf = (struct VsourceConfig *)pipe->get_privdata();
	// set the output widht and height
	iwidth = sourceConf->maxWidth;
	iheight = sourceConf->maxHeight;

	rtp_id = ((struct VsourceConfig*) pipe->get_privdata())->rtpId;
	//
	outputW = iwidth;	// by default, the same as max resolution
	outputH = iheight;
	// get the resolution from profile
	if (rtspConf->confReadInts("output-resolution", resolution, 2) == 2) {
		outputW = resolution[0];
		outputH = resolution[1];
	}
	//
	infoRecorder->logTrace("video encoder: image source from '%s' (%dx%d) via channel %d, resolution=%dx%d.\n",
		pipe->name(), iwidth, iheight, rtp_id, outputW, outputH);
	//
	this->codecContext = this->InitEncoder(
		//encoder = ga_avcodec_vencoder_init(
		NULL,
		rtspConf->video_encoder_codec,
		outputW, outputH,
		rtspConf->video_fps,
		rtspConf->vso);
	if (codecContext == NULL) {
		infoRecorder->logError("video encoder: cannot initialized the encoder.\n");
		goto video_quit;
	}
	//
	nalbuf_size = 100000 + 12 * outputW * outputH;

	if (ve_malloc(nalbuf_size, (void**)&nalbuf, &nalign) < 0) {
		infoRecorder->logTrace("video encoder: buffer allocation failed, terminated.\n");
		goto video_quit;
	}
	nalbuf_a = nalbuf + nalign;
	//
	if ((pic_in = avcodec_alloc_frame()) == NULL) {
		infoRecorder->logTrace("video encoder: picture allocation failed, terminated.\n");
		goto video_quit;
	}
	pic_in_size = avpicture_get_size(PIX_FMT_YUV420P, outputW, outputH);
	if ((pic_in_buf = (unsigned char*)av_malloc(pic_in_size)) == NULL) {
		infoRecorder->logTrace("video encoder: picture buffer allocation failed, terminated.\n");
		goto video_quit;
	}

	/// combine the frame and the buffer
	avpicture_fill((AVPicture*)pic_in, pic_in_buf,
		PIX_FMT_YUV420P, outputW, outputH);
	//ga_error("video encoder: linesize = %d|%d|%d\n", pic_in->linesize[0], pic_in->linesize[1], pic_in->linesize[2]);
	// start encoding
	infoRecorder->logTrace("video encoding started: tid=%ld %dx%d@%dfps, nalbuf_size=%d, pic_in_size=%d.\n",
		GetCurrentThreadId(),
		iwidth, iheight, rtspConf->video_fps,
		nalbuf_size, pic_in_size);

	//regist the client to server
	//pipe->client_register(ccg_gettid(), cond);
	return;

video_quit:
	if (pipe){
		pipe->client_unregister(ccg_gettid());
		pipe = NULL;
	}
	if (pic_in_buf) av_free(pic_in_buf);
	if (pic_in) av_free(pic_in);
	if (nalbuf) free(nalbuf);
	if (encoder) AVCodecCommon::AVCodecClose(encoder);
	return;
}

AVCodecContext * X264Encoder::InitEncoder(AVCodecContext * ctx, AVCodec *codec, int width, int height, int fps, vector<string> *vso){
	AVDictionary *opts = NULL;
	if (codec == NULL){
		return NULL;
	}
	if (ctx == NULL){
		if ((ctx = avcodec_alloc_context3(codec)) == NULL){
			return NULL;
		}
	}

	// parameters

	ctx->time_base.num = 1;
	ctx->time_base.den = fps;
	ctx->pix_fmt = PIX_FMT_YUV420P;
	ctx->width = width;
	ctx->height = height;
	//ctx->bit_rate =

	if (vso != NULL) {
		unsigned i, n = vso->size();
		for (i = 0; i < n; i += 2) {
			av_dict_set(&opts, (*vso)[i].c_str(), (*vso)[i + 1].c_str(), 0);
			infoRecorder->logTrace("vencoder-init: option %s = %s\n",
				(*vso)[i].c_str(),
				(*vso)[i + 1].c_str());
		}
	}
	else {
		infoRecorder->logError("vencoder-init: using default video encoder parameter.\n");
	}

	// lock 
	WaitForSingleObject(avcodecOpenMutex, INFINITE);
	if (avcodec_open2(ctx, codec, &opts) != 0){
		avcodec_close(ctx);
		av_free(ctx);
		ReleaseMutex(avcodecOpenMutex);
		return NULL;
	}

	ReleaseMutex(avcodecOpenMutex);
	return ctx;
}

AVCodecContext * X264Encoder::InitEncoder(AVCodecContext *ctx, AVCodec *codec, int bitrate, int samplerate, int channels, AVSampleFormat format, uint64_t chlayout){
	AVDictionary *opts = NULL;

	if (codec == NULL) {
		return NULL;
	}
	if (ctx == NULL) {
		if ((ctx = avcodec_alloc_context3(codec)) == NULL) {
			fprintf(stderr, "# audio-encoder: cannot allocate context\n");
			return NULL;
		}
	}
	// parameters
	ctx->thread_count = 1;
	ctx->bit_rate = bitrate;
	ctx->sample_fmt = format;	//AV_SAMPLE_FMT_S16;
	ctx->sample_rate = samplerate;
	ctx->channels = channels;
	ctx->channel_layout = chlayout;
#ifdef WIN32
	ctx->time_base.num = 1;
	ctx->time_base.den = ctx->sample_rate;
#else
	ctx->time_base = (AVRational) { 1, ctx->sample_rate };
#endif

	WaitForSingleObject(avcodecOpenMutex, INFINITE);
	//pthread_mutex_lock(&avcodec_open_mutex);
	if (avcodec_open2(ctx, codec, &opts) != 0) {
		avcodec_close(ctx);
		av_free(ctx);
		ReleaseMutex(avcodecOpenMutex);
		//pthread_mutex_unlock(&avcodec_open_mutex);
		fprintf(stderr, "# audio-encoder: open codec failed.\n");
		return NULL;
	}
	ReleaseMutex(avcodecOpenMutex);
	//pthread_mutex_unlock(&avcodec_open_mutex);

	return ctx;
}

BOOL X264Encoder::stop(){
	return TRUE;
}

void X264Encoder::run(){
	AVPacket pkt;
	int got_packet = 0;
	struct timeval tv;
	//struct timeval * ptv = new timeval();
	struct timespec to;
	// wait for notification
	infoRecorder->logTrace("[X264Encoder]: encoder pipe loader data.\n");
	data = sourcePipe->load_data();
	if (data == NULL){
		int err;
		
		getTimeOfDay(&tv, NULL);
		to.tv_sec = tv.tv_sec + 1;
		to.tv_nsec = tv.tv_usec * 1000;
		infoRecorder->logTrace("[X264Encoder]: encoder pipe wait data...\n");
		if ((err = sourcePipe->timedwait(cond, condMutex, &to)) != 0){
			infoRecorder->logError("video encoder: image source timed out. \n");
			return;
		}
		data = sourcePipe->load_data();
		if (data == NULL){
			infoRecorder->logError("video encoder: unexpected NULL frame recvived (from '%s', data = %d, buf=%d).\n",
				sourcePipe->name(), sourcePipe->data_count(), sourcePipe->buf_count());
			return;
		}
	}
	ImageFrame *frame = (struct ImageFrame *)data->ptr;

	// handle pts
	if (basePts == -1LL){
		basePts = frame->imgPts;
		ptsSync = encoderPtsSync(rtspConf->video_fps);
		newPts = ptsSync;
	}
	else{
		newPts = ptsSync + frame->imgPts - basePts;
	}
	// XXX: assume always YUV420P
	if (pic_in->linesize[0] == frame->lineSize[0] &&
		pic_in->linesize[1] == frame->lineSize[1] &&
		pic_in->linesize[2] == frame->lineSize[2]){
		bcopy(frame->imgBuf, pic_in_buf, pic_in_size);
	}
	else{
		infoRecorder->logError("video encoder: YUV mode failed - mismatched linesize(s) (src: %d,%d,%d; dst: %d,%d,%d\n",
			frame->lineSize[0], frame->lineSize[1], frame->lineSize[2],
			pic_in->linesize[0], pic_in->linesize[1], pic_in->linesize[2]);
		sourcePipe->release_data(data);
		return;
	}
	sourcePipe->release_data(data);
	// pts must be monotonically increasing
	if (newPts > pts){
		pts = newPts;
	}
	else{
		pts++;
	}

	infoRecorder->logError("[X264Encoder]: the packet pts is :%d.\n", pts);

	pic_in->pts = pts;
	av_init_packet(&pkt);
	pkt.data = nalbuf_a;
	pkt.size = nalbuf_size;
	infoRecorder->logTrace("[X264Encoder]: to encode frame data.\n");
	if (avcodec_encode_video2(codecContext, &pkt, pic_in, &got_packet) < 0){
		infoRecorder->logError("video encoder: encoder failed, terminated.\n");
		return;
	}
	if (got_packet){
		if (pkt.pts == (int64_t)AV_NOPTS_VALUE){
			pkt.pts = pts;
		}
		pkt.stream_index = 0;
		// send the packet
		infoRecorder->logTrace("[X264Encoder]: to send frame packet. pkt size:%d\n", pkt.size);

#ifndef MULTI_SOURCE
		if (EncoderManager::GetEncoderManager()->encoderSendPacketAll("[X264Encoder]", rtp_id, &pkt, pkt.pts)){
			infoRecorder->logTrace("[X264Encoder]: encoder manager send failed.\n");
		}
#else
		// the encoder send the packet
		if(!this->sendPacket(rtp_id, this->rtspContext, &pkt, pkt.pts)){
			infoRecorder->logTrace("[X264Encoder]: encoder send failed.\n");
		}

#endif
		if (pkt.side_data_elems > 0){
			int i;
			for (i = 0; i < pkt.side_data_elems; i++)
				av_free(pkt.side_data[i].data);
			av_freep(&pkt.side_data);
			pkt.side_data_elems = 0;
		}
		if (video_written == 0){
			video_written = 1;
			infoRecorder->logTrace("first video frame written (pts=%lld)\n", pts);
		}
	}
	else{
		infoRecorder->logTrace("[X264Encoder]: get no data packet.\n");
	}
}

void X264Encoder::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
	// the msg process function
}

void X264Encoder::onThreadStart(LPVOID param){
	// init the mutex
	//const char * srcpipename = (char *)param;
	condMutex = CreateMutex(NULL, FALSE, NULL);
	cond = CreateEvent(NULL, FALSE, FALSE, NULL);
	basePts = -1LL;
	newPts = 0LL;
	pts = -1LL;
	ptsSync = 0LL;
	pic_in = NULL;

	if (sourcePipe == NULL){
		sourcePipe = pipeline::lookup(srcpipename);
	}
	else{
		sourcePipe = pipeline::lookup(srcpipename);
	}

	// init variables
	if (sourcePipe == NULL){
		infoRecorder->logError("source pipe for the encoder is NULL\n");
		return;
	}
	VsourceConfig* vconf = (VsourceConfig *)sourcePipe->get_privdata();
	int iid = ((struct VsourceConfig *)sourcePipe->get_privdata())->id;
	int iwidth = vconf->maxWidth;
	int iheight = vconf->maxHeight;
	rtp_id = vconf->rtpId;

	infoRecorder->logTrace("[X264Encoder]: rtp_id is %d.\n", rtp_id);
	Log::logscreen("[X264Encoder]: rtsp config:%p.\n", rtspConf);

	codecContext = AVCodecCommon::AVCodecVEncoderInit(
		NULL,
		rtspConf->video_encoder_codec,
		iwidth, iheight,
		rtspConf->video_fps,
		rtspConf->vso);

	nalbuf_size = 100000 + 12 * iwidth * iheight;
	if (ve_malloc(nalbuf_size, (void **)&nalbuf, &nalign) < 0){
		infoRecorder->logError("video encoder: buffer allocation failed, terminated.\n");
		// post quit message and return 
		//goto quit1;
		return;
	}

	nalbuf_a = nalbuf + nalign;
	if ((pic_in = avcodec_alloc_frame()) == NULL){
		infoRecorder->logError("video encoder: buffer allocation failed, terminated.\n");
		return;
	}
	pic_in_size = avpicture_get_size(PIX_FMT_YUV420P, iwidth, iheight);
	if ((pic_in_buf = (unsigned char *)av_malloc(pic_in_size)) == NULL){
		infoRecorder->logError("video encoder: picture buffer allocation failed, terminated!\n");
		return;
	}
	avpicture_fill((AVPicture *)pic_in, pic_in_buf, PIX_FMT_YUV420P, iwidth, iheight);
	// start encoding
	infoRecorder->logError("video encoding started: tid=%ld %dx%d@%dfps, nabibuf_size = %d, pic_in_size = %d.\n",
		ccg_gettid(),
		iwidth, iheight, rtspConf->video_fps, nalbuf_size, pic_in_size);

	sourcePipe->client_register(ccg_gettid(), cond);
}
void X264Encoder::onQuit(){
	// release the thread resources
	// unregister the client
	sourcePipe->client_unregister(ccg_gettid());
	sourcePipe = NULL;

	if (pic_in_buf) av_free(pic_in_buf);
	if (pic_in) av_free(pic_in);
	if(nalbuf) free(nalbuf);
	if (codecContext) AVCodecCommon::AVCodecClose(codecContext);
	infoRecorder->logError("X264 encoder: thread terminatedf (tid=%ld).\n", ccg_gettid());
	return;
}

int X264Encoder::sendPacketAll(int channelId, AVPacket *pkt, int64_t encoderPts){
	map<RTSPContext *, RTSPContext *>::iterator mi;
	//for (mi = EncoderManager::
	return 0;
}

int X264Encoder::sendPacket(int channelId, RTSPContext * rtsp, AVPacket *pkt, int64_t encoderPts){
	int ioLen;
	uint8_t * iobuf;

	//RTSPContext * rtsp = this->rtspContext;

	if (rtsp->state != SERVER_STATE_PLAYING){
		infoRecorder->logTrace("[X264Encoder]: server state is not SERVER_STATE_PLAY, is %d!\n", rtsp->state);
	}

	if (rtsp->fmtCtx[channelId] == NULL){
		// not initialized -  disable?
		infoRecorder->logTrace("[X264Encoder]: channel %d fmtCtx is NULL.\n", channelId);
		return 0;
	}
	if (encoderPts != (int64_t)AV_NOPTS_VALUE){
		pkt->pts = av_rescale_q(encoderPts, rtsp->encoder[channelId]->time_base,
			rtsp->stream[channelId]->time_base);
	}
	if (ffio_open_dyn_packet_buf(&rtsp->fmtCtx[channelId]->pb, rtsp->mtu) < 0){
		infoRecorder->logError("x264Encoder %d: buffer allocation failed\n", channelId);
		return -1;
	}
	if (av_write_frame(rtsp->fmtCtx[channelId], pkt) != 0){
		infoRecorder->logError("x264Encoder %d: RTSP write failed!\n", channelId);
		return -1;
	}

	ioLen = avio_close_dyn_buf(rtsp->fmtCtx[channelId]->pb, &iobuf);
	if (rtsp->lowerTransport[channelId] == RTSP_LOWER_TRANSPORT_TCP){
		if (rtsp->rtspWriteBinData(channelId, iobuf, ioLen) < 0){
			av_free(iobuf);
			infoRecorder->logError("x264Encoder %d: RTSP write failed.\n", channelId);
			return -1;
		}
	}
	else{
		if (rtsp->rtpWriteBinData(channelId, iobuf, ioLen, this->frameIndex) < 0){
			av_free(iobuf);
			infoRecorder->logError("x264Encoder %d: RTP write failed.\n", channelId);
			return -1;
		}
	}
	av_free(iobuf);
	return 0;
}


#if 0

DWORD WINAPI X264ThreadFunc(PVOID param){
	// arg is pointer to source pipe
    
    X264Encoder *x264Encoder = (X264Encoder *)param;  // get the x264 encoder for encoding
	// image info
	int iid;
	int iwidth;
	int iheight;
	int rtp_id;

	struct pooldata * data = NULL;
	struct ImageFrame * frame = NULL;
	pipeline * pipe = (pipeline *)x264Encoder->getSourcePipe();
	AVCodecContext * encoder = NULL;

	RTSPConf * rtspConf = x264Encoder->rtspConf;  // get the rtspConf 
	// 
	AVFrame * pic_in = NULL;
	unsigned char *pic_in_buf = NULL;
	int pic_in_size;
	unsigned char * nalbuf = NULL, *nalbuf_a = NULL;
	int nalbuf_size = 0, nalign = 0;
	long long basePts = -1LL, newpts = 0LL, pts = -1LL, ptsSync = 0LL;
    
	HANDLE condMutex = NULL;
	HANDLE cond = NULL; // the notification event handle
	int resolution[2];
	int video_written = 0;
	if (pipe == NULL){
		infoRecorder->logError("video encoder: NULL pipeline specified.\n");
		goto quit1;
	}
    // create the mutex and the event handle
    condMutex = CreateMutex(NULL, false, "condMutex" );
    cond = CreateEvent(NULL, false, true, "event");
	// init variables
	VsourceConfig* vconf = (VsourceConfig *)pipe->get_privdata();
	iid = ((struct VsourceConfig *)pipe->get_privdata())->id;
	iwidth = vconf->maxWidth;
	iheight = vconf->maxHeight;
	rtp_id = vconf->rtpId;
	// 
    //
	x264Encoder->setOutW(iwidth); // by default, the same as max resolution
	x264Encoder->setOutH(iheight);
	//if ()
	// read thre resolution from the config
	infoRecorder->logTrace("video encoder: image source from '%s' (%dx%d) via channel %d, resolution =%dx%d.\n",
		pipe->name(), iwidth, iheight, rtp_id, iwidth, iheight);

	encoder = AVCodecCommon::AVCodecVEncoderInit(
		NULL,
		rtspConf->video_encoder_codec,
		iwidth, iheight,
		rtspConf->video_fps,
		rtspConf->vso);

	if (encoder == NULL){
		infoRecorder->logError("video encoder: cannot initialization the encoder.\n");
		goto quit1;
	}
	// 
	nalbuf_size = 100000 + 12 * iwidth * iheight;
	if (ve_malloc(nalbuf_size, (void **)&nalbuf, &nalign) < 0){
		infoRecorder->logError("video encoder: buffer allocation failed, terminated.\n");
		goto quit1;
	}

	nalbuf_a = nalbuf + nalign;
	if ((pic_in = avcodec_alloc_frame()) == NULL){
		infoRecorder->logError("video encoder: buffer allocation failed, terminated.\n");
		goto quit1;
	}
	pic_in_size = avpicture_get_size(PIX_FMT_YUV420P, iwidth, iheight);
	if ((pic_in_buf = (unsigned char *)av_malloc(pic_in_size)) == NULL){
		infoRecorder->logError("video encoder: picture buffer allocation failed, terminated!\n");
		goto quit1;
	}
	avpicture_fill((AVPicture *)pic_in, pic_in_buf, PIX_FMT_YUV420P, iwidth, iheight);
	// start encoding
	infoRecorder->logError("video encoding started: tid=%ld %dx%d@%dfps, nabibuf_size = %d, pic_in_size = %d.\n",
		ccg_gettid(),
		iwidth, iheight, rtspconf_global()->video_fps, nalbuf_size, pic_in_size);

	pipe->client_register(ccg_gettid(), &cond);

	EncoderManager * encoderManager = EncoderManager::GetEncoderManager();
	if (encoderManager == NULL){
		infoRecorder->logError("video encoder: get the encoder manager failed\n");
		goto quit1;
	}

	while (1){
		AVPacket pkt;
		int got_packet = 0;
		// wait for notification
		data = pipe->load_data();
		if (data == NULL){
			int err;
			struct timeval val;
			struct timespec to;
			getTimeOfDay(&tv, NULL);
			to.tv_sec = tv.tv_sec + 1;
			to.tv_nsec = tv.tv_usec * 1000;

			if ((err = pipe->timedwait(&cond, &condMutex, &to)) != 0){
				infoRecorder->logError("video encoder: image source timed out. \n");
				continue;
			}
			data = pipe->load_data();
			if (data == NULL){
				infoRecorder->logError("video encoder: unexpected NULL frame recvived (from '%s', data = %d, buf=%d).\n",
					pipe->name(), pipe->data_count(), pipe->buf_count());
				continue;
			}
		}
		frame = (struct ImageFrame *)data->ptr;

		// handle pts
		if (basePts == -1LL){
			basePts = frame->imgPts;
			ptsSync = encoderManager->encoderPtsSync(rtspConf->video_fps);
			newpts = ptsSync;
		}
		else{
			newpts = ptsSync + frame->imgPts - basePts;
		}
		// XXX: assume always YUV420P
		if (pic_in->linesize[0] == frame->lineSize[0] &&
			pic_in->linesize[1] == frame->lineSize[1] &&
			pic_in->linesize[2] == frame->lineSize[2]){
			bcopy(frame->imgBuf, pic_in_buf, pic_in_size);
		}
		else{
			infoRecorder->logError("video encoder: YUV mode failed - mismatched linesize(s) (src: %d,%d,%d; dst: %d,%d,%d\n",
				frame->lineSize[0], frame->lineSize[1], frame->lineSize[2],
				pic_in->linesize[0], pic_in->linesize[1], pic_in->linesize[2]);
			pipe->release_data(data);
			goto quit1;
		}
		pipe->release_data(data);
		// pts must be monotonically increasing
		if (newpts > pts){
			pts = newpts;
		}
		else{
			pts++;
		}
		
		pic_in->pts = pts;
		av_init_packet(&pkt);
		pkt.data = nalbuf_a;
		pkt.size = nalbuf_size;
		if (avcodec_encode_video2(encoder, &pkt, pic_in, &got_packet) < 0){
			infoRecorder->logError("video encoder: encoder failed, terminated.\n");
			goto quit1;
		}
		if (got_packet){
			if (pkt.pts == (int64_t)AV_NOPTS_VALUE){
				pkt.pts = pts;
			}
			pkt.stream_index = 0;
			// send the packet
		}
	}
}

#endif

/*
bool X264Encoder::registerSourcePipe(pipeline * pipe){
	this->sourcePipe = pipe;
}
pipeline * X264Encoder::getSourcePipe(){
	return this->sourcePipe;
}

*/