#include "AVCodecCommon.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"

#ifndef snprintf
#define snprintf(b, n, f, ...) _snprintf_s(b,n,_TRUNCATE, f, __VA_ARGS__)
#endif

HANDLE AVCodecCommon::openMutex;// = NULL;

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
			names++;
		}
	}
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

AVCodecContext * AVCodecCommon::AVCodecVEncoderInit(AVCodecContext *ctx, AVCodec * codec, int width, int height, int fps, vector<string> *vso){
	infoRecorder->logTrace("[AVCodecCommon]: AVCodecVEncoderInit called.\n");
	if (!openMutex)
		openMutex = CreateMutex(NULL, FALSE, NULL);
	AVDictionary *opts = NULL;
	if (codec == NULL){
		infoRecorder->logTrace("[AVCodecCommon]: NULL codec specificed.\n");
		return NULL;
	}
	if (ctx == NULL){
		if ((ctx = avcodec_alloc_context3(codec)) == NULL){
			infoRecorder->logTrace("[AVCodecCommon]: avcodec_alloc_context3 failed.\n");
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
			infoRecorder->logError("Init: option %s = %s\n", (*vso)[i].c_str(), (*vso)[i + 1].c_str());
		}

	}
	else{
		infoRecorder->logError("init: using default video encoder parameter.\n");
	}
	WaitForSingleObject(openMutex, INFINITE);
	infoRecorder->logTrace("[AVCodecCommon]: AVCodecVEncoderInit call avcodec_open2.\n");
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
/*

// functions to create new avstream
AVStream * AVCodecCommon::AVFormatNewStream(AVFormatContext * ctx, int id, AVCodec * codec){
AVStream * st = NULL;
if(codec == NULL)
return NULL;
if((st = avformat_new_stream(ctx, codec)) == NULL){
return NULL;
}
// format specific index
st->id = id;
if(ctx->flags & AVFMT_GLOBALHEADER)
st->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;

if(codec->id == CODEC_ID_H264 || codec->id == CODEC_ID_AAC){
// should we always set the global header?
//
st->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
}
return st;
}

AVCodec * AVCodecCommon::AVCodecFindEncoder(const char ** names, enum AVCodecID cid){
AVCodec * codec = NULL;
if(names != NULL){
while(*names != NULL){
if((codec = avcodec_find_encoder_by_name(*names)) != NULL)
return codec;
names ++;
}
}
if(cid == AV_CODEC_ID_NONE)
return avcodec_find_encoder(cid);
return NULL;
}

AVCodec * AVCodecCommon::AVCodecFindDecoder(const char ** names, enum AVCodecID cid){
AVCodec * codec = NULL;
if(names != NULL){
while(*names != NULL){
if((codec = avcodec_find_decoder_by_name(*names)) != NULL)
return codec;
names++;
}
}
if(cid = AV_CODEC_ID_NONE)
return avcodec_find_decoder(cid);
return NULL;
}

AVCodecContext * AVCodecCommon::AVCodecVEncoderInit(AVCodecContext * ctx, AVCodec * codec, int width, int height, int fps, vector<string> *vso){
AVDictionary * opts = NULL;

if(codec == NULL)
return NULL;
if(ctx == NULL){
if((ctx = avcodec_alloc_context3(codec)) == NULL)
return NULL;
}
// parameters
ctx->time_base.num = 1;
ctx->time_base.den = fps;

ctx->pix_fmt = PIX_FMT_YUV420P;
ctx->width = width;
ctx->height = height;

if(vso != NULL){
unsigned i, n = vso->size();
for(i = 0; i< n; i+= 2){
av_dict_set(&opts, (*vso)[i].c_str(), (*vso)[i+1].c_str(), 0);
infoRecorder->logError("vencoder-init: option %s = %s\n",
(*vso)[i].c_str(), (*vso)[i+1].c_str());
}
}
else{
infoRecorder->logError("vencoder-init: using default video encoder parameter.\n");
}

// open the codec using the function avcodec_open2
WaitForSingleObject(openMutex, INFINITE);

if(avcodec_open2(ctx, codec, &opts) != 0){
avcodec_close(ctx);
av_free(ctx);
ReleaseMutex(openMutex);
return NULL;
}
ReleaseMutex(openMutex);
return ctx;
}

AVCodecContext * AVCodecCommon::AVCodecAEncoder(AVCodecContext * ctx, AVCodec * codec, int bitrate, int samplerate, int channels, AVSampleFormat format, uint64_t chlayout){
AVDictionary *opts = NULL;
if(codec == NULL)
return NULL;

if(ctx == NULL){
if((ctx = avcodec_alloc_context3(codec)) == NULL){
infoRecorder->logError("audio encoder: cannot allocate context\n");
return NULL;
}
}
// parameter
ctx->thread_count = 1;
ctx->bit_rate = bitrate;
ctx->sample_fmt = format;
ctx->sample_rate = samplerate;
ctx->channels = channels;
ctx->channel_layout = chlayout;

ctx->time_base.num = 1;
ctx->time_base.den = ctx->sample_rate;

WaitForSingleObject(openMutex, INFINITE);

if(avcodec_open2(ctx, codec, &opts) != 0){
avcodec_close(ctx);
av_free(ctx);
ReleaseMutex(openMutex);
infoRecorder->logError("audio encoder: open codec failed\n");
return NULL;
}
ReleaseMutex(openMutex);
return ctx;
}

void AVCodecCommon::AVCodecClose(AVCodecContext * ctx){
if(ctx == NULL)
return;

WaitForSingleObject(openMutex, INFINITE);
avcodec_close(ctx);
ReleaseMutex(openMutex);
return;
}

struct SwsContext * AVCodecCommon::SwScaleInit(PixelFormat format, int inW, int inH, int outW, int outH){
struct SwsContext * swsctx = NULL;

if((swsctx = sws_getContext(inW, inH, format, outW, outH, PIX_FMT_YUV420P,
SWS_FAST_BILINEAR, NULL, NULL, NULL)) == NULL){
infoRecorder->logError("swsscale init: cannot create swscale context\n");

}
return swsctx;
}

AVFormatContext * AVCodecCommon::FormatInit(const char * filename){
AVOutputFormat *fmt;
AVFormatContext * ctx;

if((fmt = av_guess_format(NULL, filename, NULL)) == NULL){
if((fmt = av_guess_format("mkv", NULL, NULL)) == NULL){
infoRecorder->logError("# cannot fild suitable format.\n");
return NULL;
}
}
if((ctx = avformat_alloc_context()) == NULL){
infoRecorder->logError("# create avformat context failed\n");
return NULL;
}
ctx->oformat = fmt;
snprintf(ctx->filename, sizeof(ctx->filename), "%s", filename);

if((fmt->flags & AVFMT_NOFILE) == 0){
if(avio_open(&ctx->pb, ctx->filename, AVIO_FLAG_WRITE) < 0){
infoRecorder->logError("# cannot create file '%s'\n", ctx->filename);
return NULL;
}
}
return ctx;
}

AVFormatContext * AVCodecCommon::RtpInit(const char * url){
AVOutputFormat * fmt;
AVFormatContext *ctx;
if((fmt = av_guess_format("rtp", NULL, NULL)) == NULL){
infoRecorder->logError("# rtp is not supported.\n");
return NULL;
}
if((ctx = avformat_alloc_context()) == NULL){
infoRecorder->logError("# create avformat context failed\n");
return NULL;
}

ctx->oformat = fmt;
snprintf(ctx->filename, sizeof(ctx->filename), "%s", filename);

if(avio_open(&ctx->pb, ctx->filename, AVIO_FLAG_WRITE) < 0){
infoRecorder->logError("# cannot create file '%s'\n", ctx->filename);
return NULL;
}
return ctx;
}


*/