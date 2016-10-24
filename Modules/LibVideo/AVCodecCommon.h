#ifndef __AVCODECCOMMON_H__
#define __AVCODECCOMMON_H__
#ifndef __cplusplus
#define __cplusplus
#endif
#ifdef __cplusplus
extern "C" {
#endif
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/base64.h>
#ifdef __cplusplus
}
#endif

#include <map>
#include <string>
#include <vector>

#include "Commonwin32.h"

using namespace std;

class AVCodecCommon{
	static HANDLE openMutex;

public:

	static void Init();
	static SwsContext * SwScaleInit(PixelFormat srcFormat, int inW, int inH, PixelFormat dstFormat, int outW, int outH);

	static SwsContext * SwScaleInit(PixelFormat format, int inW, int inH, int outW, int outH);
	static AVFormatContext * FormatInit(const char * filename);
	static AVFormatContext * RtpInit(const char *url);
	static AVStream * AVFormatNewStream(AVFormatContext *ctx, int id, AVCodec * codec);
	static AVCodec* AVCodecFindEncoder(const char **names, enum AVCodecID cid = AV_CODEC_ID_NONE);
	static AVCodec* AVCodecFindDecoder(const char **names, enum AVCodecID cid = AV_CODEC_ID_NONE);
	static AVCodecContext*	AVCodecVEncoderInit(AVCodecContext *ctx, AVCodec *codec, int width, int height, int fps, std::vector<std::string> *vso = NULL);
	static AVCodecContext*	AVCodecAEncoderInit(AVCodecContext *ctx, AVCodec *codec, int bitrate, int samplerate, int channels, AVSampleFormat format, uint64_t chlayout);
	static void AVCodecClose(AVCodecContext *ctx);

	static AVCodecContext * AVCodecAEncoder(AVCodecContext * ctx, AVCodec * codec, int bitrate, int samplerate, int channels, AVSampleFormat format, uint64_t chlayout);
};

#endif