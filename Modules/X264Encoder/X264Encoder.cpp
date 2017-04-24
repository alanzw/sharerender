#include "X264Encoder.h"
#include "../Videoutility/FilterRGB2YUV.h"
#include "../LibCore/TimeTool.h"
#include <libavutil\avutil.h>

#include "../LibCore/InfoRecorder.h"

#ifndef UINT64_C
#define UINT64_C(val) val##ui64
#endif

using namespace  cg::core;
namespace cg{

	X264Encoder::~X264Encoder(){
		if(avcodecOpenMutex){
			CloseHandle(avcodecOpenMutex);
			avcodecOpenMutex = NULL;
		}
		if(pic_in_buf){
			free(pic_in_buf);
			pic_in_buf = NULL;
		}
	}

	X264Encoder::X264Encoder(int _width, int _height, int _rtp_id, pipeline * _pipe, VideoWriter *_writer)
		:Encoder(0, _height, _width, /*IMAGE,*/ X264_ENCODER, _pipe, _writer), 
		rtp_id(_rtp_id), avcodecOpenMutex(NULL), pic_data(NULL), codecContext(NULL), pic_in(NULL), pic_in_buf(NULL), nalbuf(NULL), nalbuf_a(NULL), nalbuf_size(0)
	{
		//rtspConf = RTSPConf::GetRTSPConf();
	}

	void X264Encoder::InitEncoder(){
		if(inited)
			return;

		struct RTSPConf * rtspConf = RTSPConf::GetRTSPConf();

		avcodecOpenMutex = CreateMutex(NULL, FALSE, NULL);
		codecContext = InitEncoder(NULL, rtspConf->video_encoder_codec, encoderWidth, encoderHeight, rtspConf->video_fps, rtspConf->vso);
		if(NULL == codecContext){
			infoRecorder->logError("[X264Encoder]: cannot initialize the encoder.\n");
			goto video_quit;
		}
		nalbuf_size = 100000 + 12 * encoderWidth * encoderHeight;     // 5.5MB  ??
		if(ve_malloc(nalbuf_size, (void **)&nalbuf, &nalign) < 0){
			infoRecorder->logError("[X264Encoder]: buffer allocation failed. terminate.\n");
			goto video_quit;
		}
		nalbuf_a = nalbuf + nalign;

		if((pic_in == NULL) && (pic_in  = avcodec_alloc_frame()) == NULL){
			infoRecorder->logError("[X264Encoder]: picture allocation failed, terminate\n");
			goto video_quit;
		}
		pic_in_size = avpicture_get_size(PIX_FMT_YUV420P, encoderWidth, encoderHeight);
		// the av_malloc is from avutil.h
		if((pic_in_buf == NULL) && (pic_in_buf = (unsigned char *)av_malloc(pic_in_size)) == NULL){
			infoRecorder->logError("[X264Encoder]: picture buffer allocation failed. terminated\n");
			goto video_quit;
		}

		avpicture_fill((AVPicture *)pic_in, pic_in_buf, PIX_FMT_YUV420P, encoderWidth, encoderHeight);

		infoRecorder->logTrace("[X264Encoder]: AVPicture line size, [0]:%d, [1]:%d, [3]:%d.\n", ((AVPicture *)pic_in)->linesize[0],((AVPicture *)pic_in)->linesize[1],((AVPicture *)pic_in)->linesize[2]);

		inited = true;

		return;
video_quit:

		if(pic_in_buf)
			av_free(pic_in_buf);
		if(pic_in)
			av_free(pic_in);
		if(nalbuf)
			free(nalbuf);
		if(codecContext) 
			AVCodecCommon::AVCodecClose(codecContext);
	}
#if 0    // not used
	void X264Encoder::InitEncoder(RTSPConf * conf, pipeline * pipe){

		infoRecorder->logError("[X264Encoder]: init Encoder with config and pipeline '%s'.\n", pipe->name());

		avcodecOpenMutex = CreateMutex(NULL, FALSE, NULL);

		struct VsourceConfig * sourceConf = (VsourceConfig *)pipe->get_privdata();
		int iid = sourceConf->id;
		iwidth = sourceConf->maxWidth;
		iheight = sourceConf->maxHeight;

		rtp_id = sourceConf->rtpId;
		outH = iheight;
		outW = iwidth;

		codecContext = InitEncoder(NULL, conf->video_encoder_codec, outW, outH, conf->video_fps, conf->vso);
		if(NULL == codecContext){
			infoRecorder->logError("[X264Encoder]: cannot initialize the encoder.\n");
			goto video_quit;
		}

		if(NULL == nalbuf){
			nalbuf_size = 100000 + 12 * outH * outW;     // 5.5MB  ??
			if(ve_malloc(nalbuf_size, (void **)&nalbuf, &nalign) < 0){
				infoRecorder->logError("[X264Encoder]: buffer allocation failed. terminate.\n");
				goto video_quit;
			}

			nalbuf_a = nalbuf + nalign;
		}
		if((pic_in == NULL) && (pic_in  = avcodec_alloc_frame()) == NULL){
			infoRecorder->logError("[X264Encoder]: picture allocation failed, terminate\n");
			goto video_quit;
		}
		pic_in_size = avpicture_get_size(PIX_FMT_YUV420P, outW, outH);
		// the av_malloc is from avutil.h
		if((pic_in_buf == NULL) && (pic_in_buf = (unsigned char *)av_malloc(pic_in_size)) == NULL){
			infoRecorder->logError("[X264Encoder]: picture buffer allocation failed. terminated\n");
			goto video_quit;
		}

		avpicture_fill((AVPicture *)pic_in, pic_in_buf, PIX_FMT_YUV420P, outW, outH);

		infoRecorder->logError("[X264Encoder]: init the AVPicture, w:%d,  h:%d.\n", outW, outH);
		infoRecorder->logError("[X264Encoder]: AVPicture line size, [0]:%d, [1]:%d, [3]:%d.\n", ((AVPicture *)pic_in)->linesize[0],((AVPicture *)pic_in)->linesize[1],((AVPicture *)pic_in)->linesize[2]);
#ifdef ENABLE_FILE_OUTPUT
		// create the outputfile
		outputFile = fopen(outputFileName, "wb");
		if(outputFile == NULL){
			// create file failed.
			infoRecorder->logError("[X264Encoder]: create the output file failed.\n");
			goto video_quit;
		}
#endif

		return;
video_quit:
		if(pipe){
			pipe->client_unregister(ccg_gettid());
			pipe = NULL;
		}
		if(pic_in_buf)
			av_free(pic_in_buf);
		if(pic_in)
			av_free(pic_in);
		if(nalbuf)
			free(nalbuf);
		if(codecContext) 
			AVCodecCommon::AVCodecClose(codecContext);
	}

#endif  // not used

	AVCodecContext * X264Encoder::InitEncoder(AVCodecContext * ctx, AVCodec * codec, int width, int height, int fps, std::vector<std::string> * vso){
		AVDictionary * opts = NULL;
		if(NULL == codec){
			return NULL;
		}
		if(NULL == ctx){
			if((ctx = avcodec_alloc_context3(codec)) == NULL){
				return NULL;
			}
		}

		// parameters
		ctx->time_base.num = 1;
		ctx->time_base.den = fps;
#ifndef USE_NV12_RAW
		ctx->pix_fmt = PIX_FMT_YUV420P;
#else
		ctx->pix_fmt = PIX_FMT_NV12;
#endif
		ctx->width = width;
		ctx->height = height;

		if(NULL != vso){
			unsigned i, n = vso->size();
			for(i = 0; i< n; i += 2){
				av_dict_set(&opts, (*vso)[i].c_str(), (*vso)[i + 1].c_str(), 0);
				infoRecorder->logTrace("[X264Encoder]: initEncoder: option %s = %s.\n", (*vso)[i].c_str(), (*vso)[i + 1].c_str());
			}
		}else{
			infoRecorder->logTrace("[X264Encoder]: initEncoder: using default video encoder parameter.\n");
		}

		// lock
		WaitForSingleObject(avcodecOpenMutex, INFINITE);
		if(avcodec_open2(ctx, codec, &opts) != 0){
			avcodec_close(ctx);
			av_free(ctx);
			ReleaseMutex(avcodecOpenMutex);
			return NULL;
		}
		ReleaseMutex(avcodecOpenMutex);
		return ctx;
	}

	BOOL X264Encoder::run(){
		AVPacket pkt;
		int got_packet = 0;
		struct pooldata * data  = NULL;
		long long pts = -1LL;

		if(!(data = loadFrame())){
			infoRecorder->logTrace("[X264Encoder]: load frame failed.\n");
			return TRUE;
		}

		pTimer->Start();

		ImageFrame* frame= (ImageFrame* )data->ptr;
		if(frame->type ==  SURFACE){
			return TRUE;   // never deal with surface
		}

		pts = writer->updataPts(frame->imgPts);

		// XXX: assume always YUV420P
		if (pic_in->linesize[0] == frame->lineSize[0] &&
			pic_in->linesize[1] == frame->lineSize[1] &&
			pic_in->linesize[2] == frame->lineSize[2]){
				bcopy(frame->imgBuf, pic_in_buf, pic_in_size);
		}
		else{
			infoRecorder->logError("[X264Encoder]: YUV mode failed - mismatched linesize(s) (src: %d, %d, %d; dst:%d,%d, %d\n",
				frame->lineSize[0], frame->lineSize[1], frame->lineSize[2],
				pic_in->linesize[0], pic_in->linesize[1], pic_in->linesize[2]);
			releaseData(data);	
			return FALSE;
		}
		releaseData(data);
		
		infoRecorder->logTrace("[X264Encoder]: the packet pts is:%d.\n", pts);

		pic_in->pts = pts;
		av_init_packet(&pkt);
		pkt.data = nalbuf_a + 2;
		pkt.size = nalbuf_size - 2;

		if(avcodec_encode_video2(codecContext, &pkt, pic_in, &got_packet) < 0){
			infoRecorder->logError("[X264Encoder]: encode failed. terminate.\n");
			return FALSE;
		}

		if(got_packet){
			if(pkt.pts == (int64_t)AV_NOPTS_VALUE){
				pkt.pts = pts;
			}
			pkt.stream_index = 0;
			infoRecorder->logTrace("[X264Encoder]: to send frame packet. pkt size:%d.\n", pkt.size);
			// send the nal data
			encodeTime = (UINT)pTimer->Stop();

			infoRecorder->addEncodeTime(getEncodeTime());

			//pTimer->Start();
			writer->sendPacket(0, &pkt, pts);
			//packTime = pTimer->Stop();

			if(refIntraMigrationTimer){
				UINT intramigration = (UINT)refIntraMigrationTimer->Stop();
				infoRecorder->logError("[Global]: intra-migration: %f (ms), in x264 encoder.\n", 1000.0 * intramigration / refIntraMigrationTimer->getFreq());
				refIntraMigrationTimer = NULL;
			}

			if(pkt.side_data_elems > 0){
				int i;
				for(i = 0; i < pkt.side_data_elems; i++){
					av_free(pkt.side_data[i].data);
				}
				av_free(&pkt.side_data);
				pkt.side_data_elems = 0;
			}
			if(video_written == 0){
				video_written = 1;
			}
		}else{
			infoRecorder->logError("[X264Encoder]: get no data packet.\n");
		}
		return TRUE;
	}

	void X264Encoder::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
		// the msg process function
	}

	BOOL X264Encoder::onThreadStart(){
		// init the mutex

		if(nalbuf_a == NULL){
			// error
			infoRecorder->logError("[X264Encoder]: not inited.\n");
			return FALSE;
		}

		if((pic_in == NULL) && (pic_in = avcodec_alloc_frame()) == NULL){
			infoRecorder->logError("[X264Encoder]: buffer allocation failed, terminated.\n");
			return FALSE;
		}

		if(pic_in_buf == NULL){
			pic_in_size = avpicture_get_size(PIX_FMT_YUV420P, encoderWidth, encoderHeight);
			if((pic_in_buf = (unsigned char *)av_malloc(pic_in_size)) == NULL){
				infoRecorder->logError("[X264Encoder]: picture buffer allocation failed, terminate.\n");
				return FALSE;
			}
			avpicture_fill((AVPicture *)pic_in, pic_in_buf, PIX_FMT_YUV420P, encoderWidth, encoderHeight);
		}

		registerEvent();
		return TRUE;
	}

	void X264Encoder::onQuit(){
		// release the thread resources
		unregisterEvent();
		if(pic_in_buf)	av_free(pic_in_buf);
		if(pic_in)		av_free(pic_in);
		if(nalbuf)		free(nalbuf);
		if(codecContext)AVCodecCommon::AVCodecClose(codecContext);

		Encoder::onQuit();
	}

}