#include "avcodeccommon.h"
#include "../LibCore/InfoRecorder.h"

#ifndef snprintf
#define snprintf(b, n, f, ...) _snprintf_s(b,n,_TRUNCATE, f, __VA_ARGS__)
#endif
namespace cg{
	HANDLE AVCodecCommon::openMutex;// = NULL;
	bool AVCodecCommon::avInited = false;
	void AVCodecCommon::Init(){
		if(avInited == false){
			av_register_all();
			avcodec_register_all();
			avformat_network_init();
			avInited = true;
		}
	}

	SwsContext * AVCodecCommon::SwScaleInit(PixelFormat format, int inW, int inH, int outW, int outH){
		struct SwsContext * swsctx = NULL;
		if ((swsctx = sws_getContext(inW, inH, format, outW, outH, PIX_FMT_YUV420P,
			SWS_FAST_BILINEAR, NULL, NULL, NULL)) == NULL){
				fprintf(stderr, "# AVCODECCOMMON: swscaleinit cannot create swscale context\n");

		}
		return swsctx;
	}

	SwsContext * AVCodecCommon::SwScaleInit(PixelFormat srcformat, int inW, int inH, PixelFormat dstFormat, int outW, int outH){
		struct SwsContext * swsctx = NULL;
		if ((swsctx = sws_getContext(inW, inH, srcformat, outW, outH, dstFormat,
			SWS_FAST_BILINEAR, NULL, NULL, NULL)) == NULL){
				fprintf(stderr, "# AVCODECCOMMON: swscaleinit cannot create swscale context\n");
		}
		return swsctx;
	}

	AVFormatContext *  AVCodecCommon::FormatInit(const char * filename){
		AVOutputFormat * fmt;
		AVFormatContext * ctx;
		if ((fmt = av_guess_format(NULL, filename, NULL)) == NULL){
			if ((fmt = av_guess_format("mkv", NULL, NULL)) == NULL){
				fprintf(stderr, "# cannot find suitable format.\n");
				return NULL;
			}
		}
		if ((ctx = avformat_alloc_context()) == NULL){
			fprintf(stderr, "# create vformat context failed.\n");
			return NULL;
		}
		ctx->oformat = fmt;
		snprintf(ctx->filename, sizeof(ctx->filename), "%s", filename);
		if ((fmt->flags & AVFMT_NOFILE) == 0){
			if (avio_open(&ctx->pb, ctx->filename, AVIO_FLAG_WRITE) < 0){
				fprintf(stderr, "# cannot create file '%s'\n", ctx->filename);
				return NULL;
			}
		}
		return ctx;
	}

	AVFormatContext * AVCodecCommon::RtpInit(const char * url){
		AVOutputFormat * fmt;
		AVFormatContext *ctx;
		if ((fmt = av_guess_format("rtp", NULL, NULL)) == NULL){
			fprintf(stderr, "# rtp is not supported\n");
			return NULL;

		}
		if ((ctx = avformat_alloc_context()) == NULL){
			fprintf(stderr, "# create avformat context failed.\n");
			return NULL;
		}
		ctx->oformat = fmt;
		snprintf(ctx->filename, sizeof(ctx->filename), "%s", url);
		if (avio_open(&ctx->pb, ctx->filename, AVIO_FLAG_WRITE) < 0){
			fprintf(stderr, "# cannot create file '%s'\n", ctx->filename);
			return NULL;
		}
		return ctx;
	}

	AVStream * AVCodecCommon::AVFormatNewStream(AVFormatContext *ctx, int id, AVCodec * codec){
		AVStream * st = NULL;
		if (codec == NULL)
			return NULL;
		if ((st = avformat_new_stream(ctx, codec)) == NULL){
			return NULL;
		}
		st->id = id;
		if (ctx->flags & AVFMT_GLOBALHEADER){
			st->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
		}
		if (codec->id == CODEC_ID_H264 || codec->id == CODEC_ID_AAC){
			st->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
		}
		
		return st;
	}

	AVCodec *AVCodecCommon::AVCodecFindEncoder(const char **names, enum AVCodecID cid){
		AVCodec *codec = NULL;
		if (names != NULL){
			while (*names != NULL){
				if ((codec = avcodec_find_encoder_by_name(*names)) != NULL){
					return codec;

				}
				//codec = avcodec_find_encoder_by_name("libx264");
				names++;
			}
		}
		//cid = AVCodecID::AV_CODEC_ID_H264;
		//codec = avcodec_find_encoder(cid);
		if (cid != AV_CODEC_ID_NONE)
			return avcodec_find_encoder(cid);
		return NULL;
	}

	AVCodec *AVCodecCommon::AVCodecFindDecoder(const char **names, enum AVCodecID cid){
		AVCodec * codec = NULL;
		if (names != NULL){
			while (*names != NULL){
				if ((codec = avcodec_find_decoder_by_name(*names)) != NULL){
					return codec;
				}
				names++;

			}
		}
		if (cid != AV_CODEC_ID_NONE)
			return avcodec_find_decoder(cid);
		return NULL;
	}

	AVCodecContext * AVCodecCommon::AVCodecVEncoderInit(AVCodecContext *ctx, AVCodec * codec, int width, int height, int fps, std::vector<std::string> *vso){
		cg::core::infoRecorder->logTrace("[AVCodecCommon]: AVCodecVEncoderInit called.\n");
		if (!openMutex)
			openMutex = CreateMutex(NULL, FALSE, NULL);
		AVDictionary *opts = NULL;
		if (codec == NULL){
			cg::core::infoRecorder->logTrace("[AVCodecCommon]: NULL codec specificed.\n");
			return NULL;
		}
		if (ctx == NULL){
			if ((ctx = avcodec_alloc_context3(codec)) == NULL){
				cg::core::infoRecorder->logTrace("[AVCodecCommon]: avcodec_alloc_context3 failed.\n");
				return NULL;
			}
		}
#if 1
		//ctx->flags = CODEC_FLAG_GLOBAL_HEADER;
#if 0
		ctx->time_base = (AVRational){1,fps};
		//ctx->
#else
		ctx->time_base.num = 1;
		ctx->time_base.den = fps;
#endif
		ctx->pix_fmt = PIX_FMT_YUV420P;
		ctx->width = width;
		ctx->height = height;
#else
		struct AVStream * stream = avformat_new_stream(ctx, codec);
#endif

		if (vso != NULL){
			unsigned i, n = vso->size();
			for (i = 0; i < n; i += 2){
				av_dict_set(&opts, (*vso)[i].c_str(), (*vso)[i + 1].c_str(), 0);
				// error
				cg::core::infoRecorder->logTrace("Init: option %s = %s\n", (*vso)[i].c_str(), (*vso)[i + 1].c_str());
			}

		}
		else{
			cg::core::infoRecorder->logError("init: using default video encoder parameter.\n");
		}
		WaitForSingleObject(openMutex, INFINITE);
		cg::core::infoRecorder->logTrace("[AVCodecCommon]: AVCodecVEncoderInit call avcodec_open2.\n");
		if (avcodec_open2(ctx, codec, &opts) != 0){
			avcodec_close(ctx);
			av_free(ctx);
			ReleaseMutex(openMutex);
			return NULL;
		}
		ReleaseMutex(openMutex);
		return ctx;
	}
	AVCodecContext * AVCodecCommon::AVCodecAEncoderInit(AVCodecContext * ctx, AVCodec * codec, int bitrate, int samplereate, int channels, AVSampleFormat  format, uint64_t chlayout){
		AVDictionary *opts = NULL;

		if (codec == NULL){
			return NULL;
		}
		if (ctx == NULL){
			if ((ctx = avcodec_alloc_context3(codec)) == NULL){
				fprintf(stderr, "[AVCodecCommon]: # audio-encoder: cannot allocate context.\n");
				return NULL;
			}
		}

		// parameters
		ctx->thread_count = 1;
		ctx->bit_rate = bitrate;
		ctx->sample_fmt = format; // AV_SAMPLE_FMT_S16
		ctx->sample_rate = samplereate;
		ctx->channels = channels;
		ctx->channel_layout = chlayout;

		ctx->time_base.num = 1;
		ctx->time_base.den = ctx->sample_rate;

		WaitForSingleObject(openMutex, INFINITE);
		if (avcodec_open2(ctx, codec, &opts) != 0){
			avcodec_close(ctx);
			av_free(ctx);
			ReleaseMutex(openMutex);
			fprintf(stderr, "[AVCodecCommon]: #audio-encoder: open codec failed.\n");
			return NULL;
		}
		ReleaseMutex(openMutex);
		return ctx;
	}
	void AVCodecCommon::AVCodecClose(AVCodecContext * ctx){
		if (ctx == NULL)
			return;
		WaitForSingleObject(openMutex, INFINITE);
		avcodec_close(ctx);
		ReleaseMutex(openMutex);
		return;
	}

}
