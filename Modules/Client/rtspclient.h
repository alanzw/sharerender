#ifndef __RTSP_CLIENT_H__
#define __RTSP_CLIENT_H__
// this is for the rtsp client

#include "Pipeline.h"
#include "SDL2\SDL.h"
//#include 

#define MULTI_RTSP_STREAM


#define IMAGE_SOURCE_CHANNEL_MAX		8
#define	SDL_USEREVENT_CREATE_OVERLAY	0x0001
#define	SDL_USEREVENT_OPEN_AUDIO		0x0002
#define	SDL_USEREVENT_RENDER_IMAGE		0x0004
#define	SDL_USEREVENT_RENDER_TEXT		0x0008

#define SDL_AUDIO_BUFFER_SIZE			2048

extern int image_rendered;

#define	RTSP_VIDEOSTATE_NULL	0

#ifndef MULTI_RTSP_STREAM

struct RTSPThreadParam {
	const char *url;
	bool running;
#ifdef ANDROID
	bool rtpOverTCP;
#endif
	char quitLive555;
	// video
	int width[IMAGE_SOURCE_CHANNEL_MAX];
	int height[IMAGE_SOURCE_CHANNEL_MAX];
	PixelFormat format[IMAGE_SOURCE_CHANNEL_MAX];
#ifdef ANDROID
	JNIEnv *jnienv;
	pthread_mutex_t surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
	struct SwsContext *swsctx[IMAGE_SOURCE_CHANNEL_MAX];
	pipeline *pipe[IMAGE_SOURCE_CHANNEL_MAX];
#else
	//pthread_mutex_t surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
	//HANDLE surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
	CRITICAL_SECTION surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
#if 1	// only support SDL2
	unsigned int windowId[IMAGE_SOURCE_CHANNEL_MAX];
	SDL_Window *surface[IMAGE_SOURCE_CHANNEL_MAX];
	SDL_Renderer *renderer[IMAGE_SOURCE_CHANNEL_MAX];
	SDL_Texture *overlay[IMAGE_SOURCE_CHANNEL_MAX];
#endif
	struct SwsContext *swsctx[IMAGE_SOURCE_CHANNEL_MAX];
	pipeline *pipe[IMAGE_SOURCE_CHANNEL_MAX];
	// audio
	//pthread_mutex_t audioMutex;
	CRITICAL_SECTION audioMutex;
	bool audioOpened;
#endif
	int videostate;
};
#else
struct RTSPThreadParam {
	const char *url;
	bool running;
#ifdef ANDROID
	bool rtpOverTCP;
#endif
	char quitLive555;
	// video
	int width[IMAGE_SOURCE_CHANNEL_MAX];
	int height[IMAGE_SOURCE_CHANNEL_MAX];
	PixelFormat format[IMAGE_SOURCE_CHANNEL_MAX];
#ifdef ANDROID
	JNIEnv *jnienv;
	pthread_mutex_t surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
	struct SwsContext *swsctx[IMAGE_SOURCE_CHANNEL_MAX];
	pipeline *pipe[IMAGE_SOURCE_CHANNEL_MAX];
#else
	//pthread_mutex_t surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
	//HANDLE surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
	CRITICAL_SECTION surfaceMutex[IMAGE_SOURCE_CHANNEL_MAX];
#if 1	// only support SDL2
	unsigned int windowId[IMAGE_SOURCE_CHANNEL_MAX];
	SDL_Window *surface[IMAGE_SOURCE_CHANNEL_MAX];
	SDL_Renderer *renderer[IMAGE_SOURCE_CHANNEL_MAX];
	SDL_Texture *overlay[IMAGE_SOURCE_CHANNEL_MAX];
#endif
	struct SwsContext *swsctx[IMAGE_SOURCE_CHANNEL_MAX];
	pipeline *pipe[IMAGE_SOURCE_CHANNEL_MAX];
	// audio
	//pthread_mutex_t audioMutex;
	CRITICAL_SECTION audioMutex;
	bool audioOpened;
#endif
	int videostate;
};



#endif


extern struct RTSPConf *rtspconf;

void rtsperror(const char *fmt, ...);
DWORD WINAPI rtsp_thread(LPVOID param);

/* internal use only */
int audio_buffer_fill(void *userdata, unsigned char *stream, int ssize);
void audio_buffer_fill_sdl(void *userdata, unsigned char *stream, int ssize);
#ifdef ANDROID
void setRTSPThreadParam(struct RTSPThreadParam *param);
struct RTSPThreadParam * getRTSPThreadParam();
#endif


struct CodecEntry{
	const char * key;
	enum AVCodecID id;
	const char * mime;
	const char * ffmpeg_decoders[4];
};

class ccgClient{

public:
	static const char ** LookupDecoders(const char * key);
	static enum AVCodecID LookupCodecID(const char * key);
	static const char * LookupMime(const char *  key);
	static CodecEntry * LookupCore(const char * key);
};

#endif