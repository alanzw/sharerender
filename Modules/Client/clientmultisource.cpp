/*
* Copyright (c) 2013 Chun-Ying Huang
*
* This file is part of GamingAnywhere (GA).
*
* GA is free software; you can redistribute it and/or modify it
* under the terms of the 3-clause BSD License as published by the
* Free Software Foundation: http://directory.fsf.org/wiki/License:BSD_3Clause
*
* GA is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*
* You should have received a copy of the 3-clause BSD License along with GA;
* if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/


#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <stdarg.h>
#include <string.h>
#include <stdio.h>

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#ifndef ANDROID
#include <SDL2/SDL_ttf.h>
#endif /* ! ANDROID */
#ifndef __cplusplus
#define __cplusplus
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
}
#endif

#include "../LibInput/controller.h"
#include "../VideoUtility/rtspConf.h"
#ifndef MULTI_SOURCES
#define MULTI_SOURCES
#endif
#ifdef MULTI_SOURCES
#include "rtspclientmultisource.h"
#endif

#include "client.h"
#include "../LibCore/MemOp.h"
#include "../LibCore/TimeTool.h"
#include "../LibCore/InfoRecorder.h"

#define INCLUDE_DISTRIBUTOR
#ifdef	INCLUDE_DISTRIBUTOR
#include "disforclient.h"
#endif


#ifndef _DEBUG
#pragma comment(lib, "event.lib")
#pragma comment(lib, "event_core.lib")
#pragma comment(lib, "event_extra.lib")
#else
#pragma comment(lib, "event.d.lib")
#pragma comment(lib, "event_core.d.lib")
#pragma comment(lib, "event_extra.d.lib")
#endif

#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "SDL2main.lib")
#pragma comment(lib, "SDL2_ttf.lib")
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "d3d10.lib")

#ifndef _DEBUG
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.lib")
#pragma comment(lib, "libgroupsock.lib")
#pragma comment(lib, "libBasicUsageEnvironment.lib")
#pragma comment(lib, "libUsageEnvironment.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#else

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.d.lib")
#pragma comment(lib, "libgroupsock.d.lib")
#pragma comment(lib, "libBasicUsageEnvironment.d.lib")
#pragma comment(lib, "libUsageEnvironment.d.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")

#endif

#define	POOLSIZE					16
#define	IDLE_MAXIMUM_THRESHOLD		(8 * 3600000)	/* us */
#define	IDLE_DETECTION_THRESHOLD	(16 * 600000) /* us */

using namespace cg;
using namespace cg::core;
using namespace cg::input;


//extern cg::input::CtrlMessagerClient * ctrlClient;// = NULL;
extern cg::input::CtrlMessagerClient * gCtrlClient;

CtrlConfig * ctrlConf = NULL;
CRITICAL_SECTION watchdogMutex;
struct timeval watchdogTimer = { 0LL, 0LL };

// The time for response delay
BTimer * globalTimer = NULL;
bool responseTickStarted = false;

#ifndef MULTI_SOURCES
static RTSPThreadParam rtspThreadParam;
#else
GameStreams * gameStreams = NULL;
#endif

//static char logic_servername[100];
int relativeMouseMode = 0;
int showCursor = 1;
int windowSizeX;
int windowSizeY;

static cg::RTSPConf * rtspConf = NULL;

#ifndef ANDROID
#define	DEFAULT_FONT		"FreeSans.ttf"
#define	DEFAULT_FONTSIZE	24
static TTF_Font *defFont = NULL;
#endif

struct CodecEntry codecTable[] = {
	{ "H264", AV_CODEC_ID_H264, "video/avc", { "h264", NULL } },
	{ "VP8", AV_CODEC_ID_VP8, "video/x-vnd.on2.vp8", { "libvpx", NULL } },
	{ "MPA", AV_CODEC_ID_MP3, "audio/mpeg", { "mp3", NULL } },
	{ NULL, AV_CODEC_ID_NONE, NULL, { NULL } }
};

const char ** ccgClient::LookupDecoders(const char * key){
	struct CodecEntry * e = LookupCore(key);
	if (e == NULL || e->ffmpeg_decoders == NULL){
		infoRecorder->logError("lookup[%s]: ffmpeg decoders not found\n", key);
		return NULL;
	}
	return e->ffmpeg_decoders;
}

enum AVCodecID ccgClient::LookupCodecID(const char * key){
	struct CodecEntry * e = LookupCore(key);
	if (e == NULL){
		infoRecorder->logError("lookup[%s]: codec id not found\n", key);
		return AV_CODEC_ID_NONE;
	}
	return e->id;
}

const char *ccgClient::LookupMime(const char * key){
	struct CodecEntry * e = LookupCore(key);
	if (e == NULL || e->mime == NULL){
		infoRecorder->logError("LookupMime[%s]: mime not found\n", key);
		return NULL;
	}
	return e->mime;
}

CodecEntry * ccgClient::LookupCore(const char *key){
	int i = 0;
	while (i >= 0 && codecTable[i].key != NULL){
		if (strcasecmp(codecTable[i].key, key) == 0)
			return &codecTable[i];
		i++;
	}
	return NULL;
}

#ifdef MULTI_SOURCES
static void CreateOverlay(GameStreams * gameStreams){

	gameStreams->createOverlay();
	return;

	int w, h;
	PixelFormat format;

	unsigned int rendererFlags = SDL_RENDERER_SOFTWARE;
	SDL_Window * surface = NULL;
	SDL_Renderer * renderer = NULL;
	SDL_Texture * overlay = NULL;

	struct SwsContext * swsctx = NULL;
	pipeline * pipe = NULL;
	struct pooldata * data = NULL;
	char windowTitle[64];

	infoRecorder->logError("[CreateOverlay]: to init the pipeline in GameStreams.\n");
	cg::RTSPConf * rtspConf = cg::RTSPConf::GetRTSPConf();

	EnterCriticalSection(gameStreams->getSurfaceMutex());
	if(gameStreams->getSurface()!= NULL){
		LeaveCriticalSection(gameStreams->getSurfaceMutex());
		rtsperror("[ga-client]:duplicated create window request - image come to fast?\n");
		infoRecorder->logError("[ga-client]: duplicated create window request - image come too fast?");
		return;
	}

	w = gameStreams->getWidth();
	h = gameStreams->getHeight();
	format = gameStreams->getFormat();
	LeaveCriticalSection(gameStreams->getSurfaceMutex());

	infoRecorder->logError("[ga-client]: to create sws_scale context, width:%d, height;%d.\n", w, h);
	if((swsctx  = sws_getContext(w,h, format, w,h, PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL)) == NULL){
		rtsperror("[ga-client]: cannot create swsscale context.\n");
		infoRecorder->logError("[ga-client]: cannot create swsscale context.\n");
		exit(-1);
	}
	//pipeline
	if((pipe = new pipeline(0)) == NULL){
		rtsperror("[ga-client]: cannot create pipeline.\n");
		infoRecorder->logError("[ga-client]: cannot create pipeline, exit ?.\n");

		exit(-1);
	}
	if((data = pipe->datapool_init(POOLSIZE, sizeof(AVPicture))) == NULL){
		rtsperror("[ga-client]: cannot allocate the data pool.\n");
		infoRecorder->logError("[ga-client]: cannot allocate the data pool, to exit?.\n");
		exit(-1);
	}
	for(; data != NULL; data = data->next){
		bzero(data->ptr, sizeof(AVPicture));
		if (avpicture_alloc((AVPicture*)data->ptr, PIX_FMT_YUV420P, w, h) != 0) {
			rtsperror("ga-client: per frame initialization failed.\n");
			infoRecorder->logError("ga-client: per frame initialization failed, to exit?.\n");
			exit(-1);
		}
	}
	//pipeline::do_register("client pipe", pipe);

	int wflag = SDL_WINDOW_SHOWN | SDL_WINDOW_INPUT_GRABBED | SDL_WINDOW_MOUSE_FOCUS;
#ifdef	ANDROID
	wflag = SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS;
#else
	if (rtspConf->confReadBool("fullscreen", 0) != 0){
		//if (ga_conf_readbool("fullscreen", 0) != 0) {
		wflag |= SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS;
	}
#endif
	snprintf(windowTitle, sizeof(windowTitle), "Player Channel #%d", 0);
	surface = SDL_CreateWindow(windowTitle, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, w, h, wflag);
	windowSizeX = w;
	windowSizeY = h;
	if (surface == NULL) {
		rtsperror("ga-client: set video mode (create window) failed.\n");
		infoRecorder->logError("[ga-client]: set video mode (create window) failed, to exit?.\n");
		exit(-1);
	}
	// move mouse to center
#if 1	// only support SDL2
	//SDL_WarpMouseInWindow(surface, w / 2, h / 2);
#endif
	if (relativeMouseMode != 0) {
		SDL_SetRelativeMouseMode(SDL_TRUE);
		showCursor = 0;
		//SDL_ShowCursor(0);
		infoRecorder->logError("[ga-client]: relative mouse mode enbaled.\n");
	}
	SDL_ShowCursor(showCursor);

#if 1	// only support SDL2
	do {	// choose SW or HW renderer?
		// XXX: Windows crashed if there is not a HW renderer!
		int i, n = SDL_GetNumRenderDrivers();
		SDL_RendererInfo info;
		for (i = 0; i < n; i++) {
			if (SDL_GetRenderDriverInfo(i, &info) < 0)
				continue;
			rtsperror("ga-client: renderer#%d - %s (%s%s%s%s)\n",
				i, info.name,
				info.flags & SDL_RENDERER_SOFTWARE ? "SW" : "",
				info.flags & SDL_RENDERER_ACCELERATED ? "HW" : "",
				info.flags & SDL_RENDERER_PRESENTVSYNC ? ",vsync" : "",
				info.flags & SDL_RENDERER_TARGETTEXTURE ? ",texture" : "");
			infoRecorder->logError("[ga-client]: renderer#%d - %s (%s%s%s%s)\n",
				i, info.name,
				info.flags & SDL_RENDERER_SOFTWARE ? "SW" : "",
				info.flags & SDL_RENDERER_ACCELERATED ? "HW" : "",
				info.flags & SDL_RENDERER_PRESENTVSYNC ? ",vsync" : "",
				info.flags & SDL_RENDERER_TARGETTEXTURE ? ",texture" : "");

			if (info.flags & SDL_RENDERER_ACCELERATED)
				rendererFlags = SDL_RENDERER_ACCELERATED;
		}
	} while (0);
	renderer = SDL_CreateRenderer(surface, -1,
		rtspConf->video_renderer_software ?
SDL_RENDERER_SOFTWARE : rendererFlags);
	if (renderer == NULL) {
		rtsperror("ga-client: create renderer failed.\n");
		infoRecorder->logError("ga-client: create renderer failed, to exit?\n");
		exit(-1);
	}
	overlay = SDL_CreateTexture(renderer,
		SDL_PIXELFORMAT_YV12,
		SDL_TEXTUREACCESS_STREAMING,
		w, h);
#endif
	if (overlay == NULL) {
		rtsperror("ga-client: create overlay (textuer) failed.\n");
		infoRecorder->logError("ga-client: create overlay (textuer) failed. to exit?\n");
		exit(-1);
	}

	EnterCriticalSection(gameStreams->getSurfaceMutex());
	gameStreams->setPipe(pipe);
	gameStreams->setSwsctx(swsctx);
	gameStreams->setOverlay(overlay);
#if 1	// only support SDL2
	gameStreams->setRenderer(renderer);
	gameStreams->setWindowId(SDL_GetWindowID(surface));
#endif
	gameStreams->setSurface(surface);
	LeaveCriticalSection(gameStreams->getSurfaceMutex());

	rtsperror("[ga-client]: window created successfully (%dx%d).\n", w, h);
	infoRecorder->logError("[ga-client]: window created successfully (%dx%d).\n", w, h);

	// initialize watchdog
	EnterCriticalSection(&watchdogMutex);
	getTimeOfDay(&watchdogTimer, NULL);
	LeaveCriticalSection(&watchdogMutex);
	//
	return;
}


static void RenderImage(GameStreams * gameStreams, long long tag){
#if 0
	infoRecorder->logTrace("[EventDealing]: RenderImage.\n");
	DWORD start =  GetTickCount();
#endif
	gameStreams->renderImage(tag);
#if 0
	DWORD end = GetTickCount();
	infoRecorder->logTrace("[RenderImage]: use %d tick count to complete.\n", end -start);
#endif

	return;

	struct pooldata *data;
	AVPicture *vframe;
	SDL_Rect rect;
#if 1	// only support SDL2
	unsigned char *pixels;
	int pitch;
#endif
	// get the frame data to display
	if ((data = gameStreams->getPipe()->load_data()) == NULL) {
		infoRecorder->logError("[ga-client]: get NULL pipeline for channel:%d.\n", 0);
		return;
	}
	vframe = (AVPicture*)data->ptr;
	//
#if 1	// only support SDL2
	if (SDL_LockTexture(gameStreams->getOverlay(), NULL, (void**)&pixels, &pitch) == 0) {
		bcopy(vframe->data[0], pixels, gameStreams->getWidth() * gameStreams->getHeight());
		bcopy(vframe->data[1], pixels + ((pitch*gameStreams->getHeight() * 5) >> 2), gameStreams->getWidth() * gameStreams->getHeight() / 4);
		bcopy(vframe->data[2], pixels + pitch*gameStreams->getHeight(), gameStreams->getWidth() * gameStreams->getHeight() / 4);
		SDL_UnlockTexture(gameStreams->getOverlay());
	}
	else {
		rtsperror("ga-client: lock textture failed - %s\n", SDL_GetError());
	}
#endif
	gameStreams->getPipe()->release_data(data);
	rect.x = 0;
	rect.y = 0;
	rect.w = gameStreams->getWidth();
	rect.h = gameStreams->getHeight();
#if 1	// only support SDL2
	SDL_RenderCopy(gameStreams->getRenderer(), gameStreams->getOverlay(), NULL, NULL);
	SDL_RenderPresent(gameStreams->getRenderer());
#endif
	//
	image_rendered = 1;
	//
	return;
}

static void RenderText(SDL_Renderer * renderer, SDL_Window * window, int x, int y, int line, const char * text){
#ifdef ANDROID
	// not supported
#else
	SDL_Color textColor = { 255, 255, 255 };
	SDL_Surface *textSurface = TTF_RenderText_Solid(defFont, text, textColor);
	SDL_Rect dest = { 0, 0, 0, 0 }, boxRect;
	SDL_Texture *texture;
	int ww = 0, wh = 0;
	if (window == NULL || renderer == NULL) {
		rtsperror("render_text: Invalid window(%p) or renderer(%p) received.\n", window, renderer);
		return;
	}
	SDL_GetWindowSize(window, &ww, &wh);
	// centering X/Y?
	if (x >= 0) { dest.x = x; }
	else { dest.x = (ww - textSurface->w) / 2; }
	if (y >= 0) { dest.y = y; }
	else { dest.y = (wh - textSurface->h) / 2; }
	//
	dest.y += line * textSurface->h;
	dest.w = textSurface->w;
	dest.h = textSurface->h;
	//
	boxRect.x = dest.x - 6;
	boxRect.y = dest.y - 6;
	boxRect.w = dest.w + 12;
	boxRect.h = dest.h + 12;
	//
	if ((texture = SDL_CreateTextureFromSurface(renderer, textSurface)) != NULL) {
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
		SDL_RenderFillRect(renderer, &boxRect);
		SDL_RenderCopy(renderer, texture, NULL, &dest);
		SDL_DestroyTexture(texture);
	}
	else {
		rtsperror("render_text: failed on creating text texture: %s\n", SDL_GetError());
	}
	//
	SDL_FreeSurface(textSurface);
#endif
	return;
}
static void OpenAudio(GameStreams * gameStreams, AVCodecContext * adecoder){
	SDL_AudioSpec wanted, spec;
	cg::RTSPConf * rtspConf = cg::RTSPConf::GetRTSPConf();
	wanted.freq = cg::RTSPConf::GetRTSPConf()->audio_samplerate;
	wanted.format = -1;
	if(rtspConf->audio_device_format == AV_SAMPLE_FMT_S16){
		wanted.format = AUDIO_S16SYS;
	}
	else{
		rtsperror("[ga-client]: open audio - unsupported audio device format.\n");
		return;
	}
#if 0
	wanted.channels = rtspConf->audio_Channels;
	wanted.silence = 0;
	wanted.samples = SDL_AUDIO_BUFFER_SIZE;
	wanted.callback = audioBufferFillSdl;
	wanted.userdata = adecoder;

	EnterCriticalSection(&gameStreams->audioMutex);
	//pthread_mutex_lock(&rtspParam->audioMutex);
	if (gameStreams->audioOpened == true) {
		LeaveCriticalSection(&gameStreams->audioMutex);
		//pthread_mutex_unlock(&rtspParam->audioMutex);
		return;
	}
	if (SDL_OpenAudio(&wanted, &spec) < 0) {
		LeaveCriticalSection(&gameStreams->audioMutex);
		//pthread_mutex_unlock(&rtspParam->audioMutex);
		rtsperror("ga-client: open audio failed - %s\n", SDL_GetError());
		return;
	}
	//
	gameStreams->audioOpened = true;
	//
	SDL_PauseAudio(0);
	LeaveCriticalSection(&gameStreams->audioMutex);
	//pthread_mutex_unlock(&rtspParam->audioMutex);
	rtsperror("ga-client: audio device opened.\n");
#endif
	return;
}
#endif


void ProcessEvent(SDL_Event *event, CtrlMessagerClient * ctrlClient) {
		sdlmsg_t m;
		DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();
		switch (event->type) {
		case SDL_KEYUP:
			infoRecorder->logTrace("[EventDeal]: key up.\n");
			if (event->key.keysym.sym == SDLK_BACKQUOTE && relativeMouseMode != 0) {
					showCursor = 1 - showCursor;
					if (showCursor)
						SDL_SetRelativeMouseMode(SDL_FALSE);
					else
						SDL_SetRelativeMouseMode(SDL_TRUE);
			}

			if (rtspConf->ctrlEnable) {
				// for test response delay
				if(event->key.keysym.sym == SDLK_HOME){
					// to send special key
				}
				else if(event->key.keysym.sym == SDLK_F10){

				}else if(event->key.keysym.sym == SDLK_F11){
					if(!globalTimer){
						globalTimer = new PTimer();
					}
					
					if(!responseTickStarted){
						globalTimer->Start();
						responseTickStarted = true;
					}
					delayRecorder->startDelayCount();
				}

				sdlmsg_keyboard(&m, 0, 
					event->key.keysym.scancode, 
					event->key.keysym.sym, 
					event->key.keysym.mod, 0/*event->key.keysym.unicode*/);
				if(ctrlClient)
					ctrlClient->sendMsg(&m, sizeof(sdlmsg_keyboard_t));
			}
			break;

		case SDL_KEYDOWN:
			infoRecorder->logTrace("[EventDeal]: key down.\n");
			if (rtspConf->ctrlEnable) {
				sdlmsg_keyboard(&m, 1, event->key.keysym.scancode, event->key.keysym.sym, event->key.keysym.mod, 0/*event->key.keysym.unicode*/);
				if(ctrlClient)
					ctrlClient->sendMsg(&m, sizeof(sdlmsg_keyboard_t));
			}
			break;

		case SDL_MOUSEBUTTONUP:
			if (rtspConf->ctrlEnable) {
				sdlmsg_mousekey(&m, 0, event->button.button, event->button.x, event->button.y);
				if(ctrlClient)
					ctrlClient->sendMsg(&m, sizeof(sdlmsg_mouse_t));
			}
			break;

		case SDL_MOUSEBUTTONDOWN:
			infoRecorder->logTrace("[EventDeal]: mouse button down.\n");
			if (rtspConf->ctrlEnable) {
				sdlmsg_mousekey(&m, 1, event->button.button, event->button.x, event->button.y);
				if(ctrlClient)
					ctrlClient->sendMsg(&m, sizeof(sdlmsg_mouse_t));
			}
			break;
		case SDL_MOUSEMOTION:
			//infoRecorder->logTrace("[EventDeal]: mouse motion, (x, y): (%d, %d), (relx, rely):(%d, %d).\n", event->motion.x, event->motion.y, event->motion.xrel, event->motion.yrel);
			if (rtspConf->ctrlEnable && rtspConf->sendMouseMotion) {
				sdlmsg_mousemotion(&m, event->motion.x, event->motion.y, event->motion.xrel, event->motion.yrel, event->motion.state, relativeMouseMode == 0 ? 0 : 1);
				if(ctrlClient)
					ctrlClient->sendMsg(&m, sizeof(sdlmsg_mouse_t));
			}
			else{
				infoRecorder->logError("[EventDeal]: ctrl disable or send mouse motion disabled.\n");
			}
			break;
#if 1	// only support SDL2
		case SDL_MOUSEWHEEL:
			infoRecorder->logTrace("[EventDeal]: mouse wheel.\n");
			if (rtspConf->ctrlEnable && rtspConf->sendMouseMotion) {
				sdlmsg_mousewheel(&m, event->motion.x, event->motion.y);
				if(ctrlClient)
					ctrlClient->sendMsg(&m, sizeof(sdlmsg_mouse_t));
			}
			break;
#ifdef ANDROID
#define	DEBUG_FINGER(etf)	\
	rtsperror("XXX DEBUG: finger-event(%d) - x=%d y=%d dx=%d dy=%d p=%d\n", \
	(etf).type, (etf).x, (etf).y, (etf).dx, (etf).dy, (etf).pressure);
		case SDL_FINGERDOWN:
			// window size has not been registered
			if (windowSizeX[0] == 0)
				break;
			//DEBUG_FINGER(event->tfinger);
			if (rtspConf->ctrlEnable) {
				unsigned short mapx, mapy;
				mapx = (unsigned short)(1.0 * (windowSizeX[0] - 1) * event->tfinger.x / 32767.0);
				mapy = (unsigned short)(1.0 * (windowSizeY[0] - 1) * event->tfinger.y / 32767.0);
				sdlmsg_mousemotion(&m, mapx, mapy, 0, 0, 0, 0);
				ctrl_client_sendmsg(&m, sizeof(sdlmsg_mouse_t));
				//
				sdlmsg_mousekey(&m, 1, SDL_BUTTON_LEFT, mapx, mapy);
				ctrl_client_sendmsg(&m, sizeof(sdlmsg_mouse_t));
			}
			break;
		case SDL_FINGERUP:
			// window size has not been registered
			if (windowSizeX[0] == 0)
				break;
			//DEBUG_FINGER(event->tfinger);
			if (rtspConf->ctrlEnable) {
				unsigned short mapx, mapy;
				mapx = (unsigned short)(1.0 * (windowSizeX[0] - 1) * event->tfinger.x / 32767.0);
				mapy = (unsigned short)(1.0 * (windowSizeY[0] - 1) * event->tfinger.y / 32767.0);
				sdlmsg_mousemotion(&m, mapx, mapy, 0, 0, 0, 0);
				ctrl_client_sendmsg(&m, sizeof(sdlmsg_mouse_t));
				//
				sdlmsg_mousekey(&m, 0, SDL_BUTTON_LEFT, mapx, mapy);
				ctrl_client_sendmsg(&m, sizeof(sdlmsg_mouse_t));
			}
			break;
		case SDL_FINGERMOTION:
			// window size has not been registered
			if (windowSizeX[0] == 0)
				break;
			//DEBUG_FINGER(event->tfinger);
			if (rtspConf->ctrlEnable) {
				unsigned short mapx, mapy;
				mapx = (unsigned short)(1.0 * (windowSizeX[0] - 1) * event->tfinger.x / 32767.0);
				mapy = (unsigned short)(1.0 * (windowSizeY[0] - 1) * event->tfinger.y / 32767.0);
				sdlmsg_mousemotion(&m, mapx, mapy, 0, 0, 0, 0);
				ctrl_client_sendmsg(&m, sizeof(sdlmsg_mouse_t));
			}
			break;
#undef	DEBUG_FINGER
#endif	/* ANDROID */
		case SDL_WINDOWEVENT:
			if (event->window.event == SDL_WINDOWEVENT_CLOSE) {
#ifndef MULTI_SOURCES
				rtspThreadParam.running = false;
#else
				gameStreams->setRunning(false);  ///START_RTSP_SERVICE
#endif
				return;
			}
			else if (event->window.event == SDL_WINDOWEVENT_RESIZED) {
				rtsperror("event video resize w=%d h=%d\n",
					event->window.data1, event->window.data2);
			}
			break;
		case SDL_USEREVENT:
			if (event->user.code == SDL_USEREVENT_RENDER_IMAGE) {
				//long long ch = (long long)event->user.data2;
				// get the tag
				unsigned char tag = 0, valueTag = 0, *ptr = NULL;

				ptr = (unsigned char *)(&(event->user.data2));
				tag = *ptr;
				valueTag = *(ptr +1);
#if 1
				GameStreams * streams = (GameStreams *)event->user.data1;
				streams->renderImage(tag, valueTag);
				
#else
				RenderImage((GameStreams *)event->user.data1, tag);
#endif
				break;
			}
			if (event->user.code == SDL_USEREVENT_CREATE_OVERLAY) {
				long long ch = (long long)event->user.data2;
				CreateOverlay((GameStreams*)event->user.data1);
				break;
			}
			if (event->user.code == SDL_USEREVENT_OPEN_AUDIO) {
				//OpenAudio((GameStreams *)event->user.data1, (AVCodecContext *)event->user.data2);
				break;
			}
			if (event->user.code == SDL_USEREVENT_RENDER_TEXT) {
				//SDL_SetAlpha()
				GameStreams * streams = (GameStreams *)event->user.data1;
				SDL_SetRenderDrawColor(streams->getRenderer(), 0, 0, 0, 192/*SDL_ALPHA_OPAQUE/2*/);
				//SDL_RenderFillRect(rtspThreadParam.renderer[0], NULL);
				RenderText(streams->getRenderer(), streams->getSurface(), -1, -1, 0, (const char *)event->user.data1);
				SDL_RenderPresent(streams->getRenderer());
				break;
			}
			//add the sdl event to handle the render's exit and add
			if(event->user.code == SDL_USEREVENT_ADD_RENDER){
				infoRecorder->logTrace("[ProcessEvent]: SDL_USEREVENT_ADD_RENDER event triggered.\n");
				// the render is added in the logic server
				GameStreams * streams = (GameStreams *)event->user.data1;
				char * cmd = (char *)event->user.data2;
				// get the count and the command from the user event datz.
				streams->addRenders(cmd);
				break;
			}
			if(event->user.code == SDL_USEREVENT_DECLINE_RENDER){
				GameStreams * streams = (GameStreams *)event->user.data1;
				char * cmd = (char *)event->user.data2;
				// get the count and the urls from the user event data
				streams->declineRenders(cmd);
				break;
			}
			break;
#endif /* SDL_VERSION_ATLEAST(2,0,0) */
		case SDL_QUIT:
			gameStreams->setRunning(false);
			return;
		default:
			// do nothing
			break;
		}
		return;
}

static void *
	watchdog_thread(void *args) {
		static char idlemsg[128];
		struct timeval tv;
		SDL_Event evt;
		//
		rtsperror("watchdog: launched, waiting for audio/video frames ...\n");
		//
		while (true) {
#ifdef WIN32
			Sleep(1000);
#else
			sleep(1);
#endif
			EnterCriticalSection(&watchdogMutex);
			getTimeOfDay(&tv, NULL);
			if (watchdogTimer.tv_sec != 0) {
				long long d;
				d = tvdiff_us(&tv, &watchdogTimer);
				if (d > IDLE_MAXIMUM_THRESHOLD) {
					gameStreams->setRunning(false);
					break;
				}
				else if (d > IDLE_DETECTION_THRESHOLD) {
					// update message and show
					snprintf(idlemsg, sizeof(idlemsg),
						"Audio/video stall detected, waiting for %d second(s) to terminate ...",
						(int)(IDLE_MAXIMUM_THRESHOLD - d) / 1000000);
					//
					bzero(&evt, sizeof(evt));
					evt.user.type = SDL_USEREVENT;
					evt.user.timestamp = time(0);
					evt.user.code = SDL_USEREVENT_RENDER_TEXT;
					evt.user.data1 = idlemsg;
					evt.user.data2 = NULL;
					SDL_PushEvent(&evt);
					//
					rtsperror("watchdog: %s\n", idlemsg);
				}
				else {
					// do nothing
				}
			}
			else {
				//rtsperror("watchdog: initialized, but no frames received ...\n");
			}
			LeaveCriticalSection(&watchdogMutex);
		}
		//
		rtsperror("watchdog: terminated.\n");
		infoRecorder->logError("[Watchdog]: exit.\n");
		exit(-1);
		//
		return NULL;
}

int winsockInit(){
	WSADATA wd;
	if (WSAStartup(MAKEWORD(2, 2), &wd) != 0)
		return -1;
	return 0;
}

//RTSPConf * rtspConf = NULL;
WSADATA wd;
char * GetConfig(){
	char * conf = (char *)malloc(sizeof(100));
	sprintf(conf, "%s", "config\\client.rel.conf");
	return conf;
}
static int ClientInit(char * config){
	infoRecorder->logTrace("[client-init]: %s\n", config);
	srand(time(0));
#if 1
	if (WSAStartup(MAKEWORD(2, 2), &wd) != 0){
		infoRecorder->logTrace("[client]: WSAStartup failed.\n");
		return -1;
	}
#endif
	infoRecorder->logTrace("[Client]: star to init the av part.\n");
	av_register_all();   // here is the problem
	avcodec_register_all();
	avformat_network_init();

	infoRecorder->logTrace("[Client]: init av part finished, start to load config.\n");
	cg::RTSPConf * conf = cg::RTSPConf::GetRTSPConf((char *)config);
	if (conf == NULL){
		infoRecorder->logError("[client]: get rtsp config failed.\n");
		return -1;
	}
	return 0;
}
static int ClientInit(char * config, const char * url){
	//config = conffile;
	infoRecorder->logTrace("[client-init]: %s %s\n", config, url);

	srand(time(0));
	winsockInit();

	av_register_all();
	avcodec_register_all();
	avformat_network_init();

	cg::RTSPConf * conf = cg::RTSPConf::GetRTSPConf((char *)config);
	if (config != NULL){
		if (conf->confLoad(config) < 0){
			infoRecorder->logError("[client]: cannot load configure file '%s'\n", config);
			return -1;
		}
	}
	if (url != NULL){
		// get the server url and port
		// overwrite the server url and port
		if (conf->UrlParse(url) < 0){
			infoRecorder->logError("[client]: invalid URL '%s'\n", url);
			return -1;
		}
	}
	return 0;
}
//int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow){

#if 0
void TestRTSPFiles(){
	string fileNames[] ={
		string("client.abs.conf"),
		string("client.rel.conf"),
		string("server.controller.conf"),
		string("server.distributor.conf"),
		string("server.logic.conf"),
		string("server.render.conf")
	};
	string path = "config/";
	for each(std::string file in fileNames){
		string fullPath = path + file;
		std::cout << "load config file " << fullPath<< std::endl;
		ccgConfig * conf = new ccgConfig((char *)fullPath.c_str());
		conf->confLoad(fullPath.c_str());
		conf->print();
		delete conf;
	}
}
#endif


// the main function for multi-soruces game client, event based
int main(int argc, char * argv[]){

	DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();

	if(infoRecorder == NULL){
		infoRecorder = new InfoRecorder("client");
	}
	// read configure file , connect to distributor
	int i = 0;
	SDL_Event event;
	HANDLE rtspthread = NULL, ctrlthread = NULL, watchdog = NULL;

	for (int m = 0; m < argc; m++){
		infoRecorder->logTrace("[client]: argv[%d]: %s\n", m, argv[m]);
	}
	// init the log
#ifdef ANDROID
	if (ga_init("/sdcard/ga/android.conf", NULL) < 0) {
		rtsperror("cannot load configuration file '%s'\n", argv[1]);
		return -1;
	}
#else  // ANDROID
	// init the SDL to intercept everything
	//SDL_SetMainReady();
	if (SDL_Init(SDL_INIT_EVERYTHING) != 0){
		infoRecorder->logError("[Client]: unable to initialize SDL: %s\n", SDL_GetError());
		return -1;
	}

	infoRecorder->logTrace("SDL_Quit:%p.\n", SDL_Quit);
	atexit(SDL_Quit);
#ifdef INCLUDE_DISTRIBUTOR
	if (argc < 3) {
		rtsperror("usage: %s [config] [url] [game name]\n", argv[0]);
		return -1;
	}
#endif  // INCLUDE_DISTRIBUTOR

	char * configfilename = _strdup(argv[1]);

#ifdef INCLUDE_DISTRIBUTOR
	char * gameName = _strdup(argv[2]);
	infoRecorder->logTrace("%s %s\n", configfilename, gameName);
#else  // INCLUDE_DISTRIBUTOR
	infoRecorder->logError("[Client]: %s.\n", configfilename);
#endif // INCLUDE_DISTRIBUTOR
	//load the config file and set the server url
	if (ClientInit(configfilename) < 0){
		rtsperror("[client]: init failed, maybe cannot load configuration file '%s'\n", argv[1]);
		return -1;
	}

	rtspConf = cg::RTSPConf::GetRTSPConf();  // get the rtsp config
#endif  // ANDROID

	// init fonts
	if (TTF_Init() != 0) {
		rtsperror("cannot initialize SDL_ttf: %s\n", SDL_GetError());
		return -1;
	}
	if ((defFont = TTF_OpenFont(DEFAULT_FONT, DEFAULT_FONTSIZE)) == NULL) {
		rtsperror("open font '%s' failed: %s\n",
			DEFAULT_FONT, SDL_GetError());
		return -1;
	}
	
	///// launch watchdog
	// init the watchdog critical section
	InitializeCriticalSection(&watchdogMutex);
	if (rtspConf->confReadBool("enable-watchdog", 1) == 1) {
		// launch the watch dog thread
		DWORD watchdog_thread_id;
		// watchdog = chBEGINTHREADEX(NULL, 0, watchdog_thread, 0, NULL, &watchdog_thread_id);
		if (watchdog == 0){
			rtsperror("Cannot create watchdog thread.\n");
			infoRecorder->logError("[Client]: cannot create watchdog thread.\n");	
		}
	}
	else {
		infoRecorder->logError("[Client]: watchdog disabled.\n");
	}

	/////////// build the GameStreams and init //////////////
	if(gameStreams == NULL){
		gameStreams = GameStreams::GetStreams();
	}

	gameStreams->init();
	gameStreams->setRunning(true);
	gameStreams->name = _strdup(gameName);
	free(gameName);

#if 0   // request the distributor
	gameStreams->setDisUrl(rtspConf->getDisUrl());
	// create a thread to deal with network event
	DWORD netThreadId = 0;
	HANDLE netThread = chBEGINTHREADEX(NULL, 0, NetworkThreadProc, gameStreams, FALSE, &netThreadId);
#else
	// request the render proxy directly
	// get the render proxy's url and port
	UserClient * userClient = new UserClient();
	userClient->setName(gameStreams->name);
	//userClient->startRTSP(rtspConf->getDisUrl(), rtspConf->serverPort);
	userClient->startRTSP(argv[3], rtspConf->serverPort);

#endif

	// main thread to deal the SDL event
	while(gameStreams->isRunning()){
		if(SDL_WaitEvent(&event)){
		//if(SDL_WaitEvent(&event)){
			//infoRecorder->logError("process event, ctrl client:%p.\n", gCtrlClient);
			ProcessEvent(&event, gCtrlClient);
		}else{
			infoRecorder->logTrace("[main]: no event.\n");
		}
	}
	//gameStreams->quitLive555 = 1;

	rtsperror("terminating ...\n");
	infoRecorder->logError("[main]: terminating.\n");
	//
#ifndef ANDROID
	//TerminateThread(rtspthread, 0);
	if (rtspConf->ctrlEnable && ctrlthread){
		TerminateThread(ctrlthread, 0);
		ctrlthread = NULL;
	}
	if(watchdog){
		TerminateThread(watchdog, 0);
		watchdog = NULL;
	}
#endif // ANDROID
	//SDL_WaitThread(thread, &status);
	SDL_Quit();
	return 0;
}
