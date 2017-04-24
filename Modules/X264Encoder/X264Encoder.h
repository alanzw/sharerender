#ifndef __X264ENCODER__
#define __X264ENCODER__
#ifndef  WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "../VideoUtility/avcodeccommon.h"
#include "../VideoUtility/encoder.h"
#include "../VideoUtility/rtspconf.h"
#include "../VideoGen/rtspcontext.h"

#define ENABLE_FILE_OUTPUT

namespace cg{

	class X264Encoder: public Encoder{

		//RTSPConf *			rtspConf;
		HANDLE				avcodecOpenMutex;  // avcodec_open / close is not thread-safe
		char *				pic_data;
		AVCodec *			videoEncoderCodec;
		AVCodecContext *	codecContext;
		AVFrame *			pic_in;

		unsigned char *		pic_in_buf;
		int					pic_in_size;
		unsigned char *		nalbuf, * nalbuf_a;
		int					nalbuf_size, nalign;
		int					video_written;
		int					rtp_id;

		// video context
		AVCodecContext *	InitEncoder(AVCodecContext * ctx, AVCodec * codec, int width, int height, int fps, std::vector<std::string> * vso);

	public:

		~X264Encoder();

		X264Encoder(int _width, int _height, int _rtp_id, pipeline * _pipe,  VideoWriter * _writer);

		inline int			getOutW(){ return encoderWidth; }
		inline int			getoutH(){ return encoderHeight; }

#if 0  // no used
		void				InitEncoder(RTSPConf * conf, pipeline * pipe);
#endif // not used
		void				InitEncoder();

		// from encoder
		virtual BOOL		run();
		virtual void		onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
		virtual BOOL		onThreadStart();
		virtual void		onQuit();

	};

}

#endif