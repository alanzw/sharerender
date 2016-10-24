#ifndef __AUDIO_ENCODER_H__
#define __AUDIO_ENCODER_H__

class AudioEncoder{
protected:

	int					audio_bitrate;
	int					audio_samplerate;
	int					audio_channels;   // XXX: AVFrame->channesl is int64_t, use with care
	AVCodec *			audioEncoderCodec;
};

#endif   // __AUDIO_ENCODER_H__