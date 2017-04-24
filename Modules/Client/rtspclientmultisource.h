#ifndef __RTSPCLIENTMULTISOURCE_H__
#define __RTSPCLIENTMULTISOURCE_H__

#define MULTI_RTSP_STREAM

#ifdef MULTI_RTSP_STREAM

// this is for the multi source rtsp client, basically, this file is similiar with rtspclient.h
#include "../VideoUtility/pipeline.h"
#include <SDL2\SDL.h>
#include "..\VideoUtility\rtspconf.h"
#include "..\VideoUtility\pipeline.h"
//#include "ccg_win32.h"


#include "../LibCore/TagDriver.h"

#ifndef __cplusplus
#define __cplusplus

#endif

#include <list>

using namespace std;


// use templated frame pool

#define USE_TEMPLATE_FRAME_POOL


#ifdef __cplusplus
extern "C"{
#include "libavformat/avformat.h"

};
#endif

#include <live555\liveMedia.hh>
#include <live555\BasicUsageEnvironment.hh>
#include <live555\UsageEnvironment.hh>


#define IMAGE_SOURCE_CHANNEL_MAX		8
#define	SDL_USEREVENT_CREATE_OVERLAY	0x0001
#define	SDL_USEREVENT_OPEN_AUDIO		0x0002
#define	SDL_USEREVENT_RENDER_IMAGE		0x0004
#define	SDL_USEREVENT_RENDER_TEXT		0x0008

#define SDL_USEREVENT_ADD_RENDER		0x0010
#define SDL_USEREVENT_DECLINE_RENDER	0x0011

#define SDL_AUDIO_BUFFER_SIZE			2048

#define PRIVATE_BUFFER_SIZE				1048576
#define RTSP_CLIENT_VERBOSITY_LEVEL		1 // by default, print verbose output from each "RTSPClient"
#define	COUNT_FRAME_RATE				600	// every N frames

// Even though we're not going to be doing anything with the incoming data, we still need to receive it.
// Define the size of the buffer that we'll use:
#define DUMMY_SINK_RECEIVE_BUFFER_SIZE	262144	//100000
#define DEBUG_PRINT_EACH_RECEIVED_FRAME 1

// define the max video source, each one is a seperated rtsp stream
#define MAX_RTSP_STREAM					8

#define RTSP_VIDEOSTATE_NULL			0

#define SINGLE_CHANNEL

#define YUV_TEXTURE

extern int								image_rendered;

void rtsperror(const char * fmt, ...);
DWORD WINAPI rtsp_thread_multi_source(LPVOID param);

/* internal use only */
struct CodecEntry{
	const char *	key;
	enum AVCodecID	id;
	const char *	mime;
	const char *	ffmpeg_decoders[4];
};

struct PacketQueue{
	bool packetQueueInitialized;
	list<AVPacket> _queue;
	int size;
	HANDLE mutex;
	HANDLE cond;

	PacketQueue();
	~PacketQueue();
	void packetQueueInit();
	int packetQueuePut(AVPacket * pkt);
	int packetQueueGet(AVPacket * pkt, int block);
};


class ccgClient{
public:
	static const char ** LookupDecoders(const char * key);
	static enum AVCodecID LookupCodecID(const char * key);
	static const char * LookupMime(const char *key);
	static CodecEntry * LookupCore(const char * key);
};


UsageEnvironment & operator<<(UsageEnvironment & env, const RTSPClient & rtspClient);
UsageEnvironment& operator<<(UsageEnvironment & env, const MediaSubsession & subsession);
void rtsperror(const char * fmt, ...);


struct DecoderBuffer{
	unsigned int	privBufLen;
	unsigned char * privBuf;
	struct timeval	lastpts;
};

#ifdef USE_TEMPLATE_FRAME_POOL
struct TaggedFrame{
	AVFrame * frame;  // the decoded frame data
	unsigned char tag;   // the tag to represent the frame is special.
	unsigned char valueTag; 
};
#endif


class VideoDecoder{
	bool inited;
	AVCodecContext * vDecoder;   // single channel 
#ifdef USE_TEMPLATE_FRAME_POOL
	TaggedFrame * frame;
#else
	AVFrame * vFrame;
#endif

	char *			videoCodecName;
	AVCodecID		videoCodecId;
	int				videoSessFmt;
public:

	VideoDecoder(){ 
		inited = false;
		videoCodecName = NULL; 
		videoCodecId = AV_CODEC_ID_NONE;
		vDecoder = NULL;
#ifdef USE_TEMPLATE_FRAME_POOL
		frame = NULL;
#else
		vFrame = NULL;
#endif
	}

	inline AVCodecContext * getVDecoder(){ return vDecoder; }
#ifndef USE_TEMPLATE_FRAME_POOL
	inline AVFrame * getVFrame(){ return vFrame; }
#endif

	int init(const char * sprop);
	AVFrame *decodeVideo(int * got_picture, int * step_len, AVPacket * pkt);



	inline void		setVideoSessFmt(int val){ videoSessFmt = val; }
	inline void		setCodecName(char * name){ videoCodecName = name; }
	inline int		getVideoSessFmt(){ return this->videoSessFmt; }
	inline bool		isInited(){ return inited; }
	inline void		setInited(bool v = true){ inited = v;}
};
class AudioDecoder{
	AVCodecContext * aDecoder;
	AVFrame *		aFrame;
	AVCodecID		audioCodecId;
	int				audioSessFmg;

	int				packetQueueInitialized;
	int				abmaxsize;
	unsigned char * audioBuf;

	char *			audioCodecName;

	PacketQueue *	audioq;
	unsigned int	absize;
	unsigned int	abpos;
	struct SwrContext * swrctx;
	unsigned char * convbuf;
	int				maxDecoderSize;
	int				audioStart;

public:
	int				initADecoder();

	unsigned char * audioBufferInit();
	int				audioBufferDecode(AVPacket * pkt, unsigned char * dstBuf, int dstLen);
	int				audioBufferFill(void * userData, unsigned char * stream, int ssize);
	int				audioBufferFillSDL(void * userData, unsigned char * stream, int ssize);

	void			playAudio(unsigned char * buffer, int bufSize, struct timeval pts);
	int				getAudioSessFmg(){ return audioSessFmg; }
};


// the sub game stream is one of the many video streams for a cloud gaming
// responsible for a rtsp connection, more like a RTSPThreadParam
class SubGameStream{
	int streamId;  // a stream id to identify the sub stream in gameStream

	char *			url; // the url of the source
	bool			running; // indicate whether the source is available
	HANDLE			notifier;
	HANDLE			thread;

	char			quitLive555;

	int				width, height;
	struct SwsContext * swsctx;

	DecoderBuffer * decoderBuffer;
	int				videoState;

	static int		RTSPClientCount;
	// store the map here

	RTSPClient *	rtspClient;
	UsageEnvironment * env;
	VideoDecoder *	videoDecoder;

public:

	void			playVideoPriv(unsigned char * buffer, int bufSize, struct timeval pts);
	void			playVideo(unsigned char * buffer, int bufSize, struct timeval pts, bool maker);

	inline VideoDecoder * getVideoDecoder(){ return videoDecoder; }
	inline void		setThread(HANDLE t){ thread = t; }
	inline HANDLE	getThread(){ return thread; }
	inline void		setVideoState(int state){ videoState = state; }
	inline char *	getURL(){return url;};
	inline bool		isQuitLive555(){ return quitLive555; }
	inline HANDLE	getNotifier(){ return notifier; }
	inline RTSPClient * getRTSPClient(){ return rtspClient; }
	static RTSPClient * OpenURL(UsageEnvironment &env, char const * rtspURL);

	bool			openUrl(UsageEnvironment *env, char * url);
	bool			openUrl(UsageEnvironment * env);

	static void		shutdownStream(RTSPClient * rtspClient, int exitCode = 1);

	SubGameStream(char * rtspUrl);
	SubGameStream(char * url, int port);
	~SubGameStream();
	bool			init();
	inline bool		setQuitLive555(char val){ this->quitLive555 = val; return true; }

	inline int		getId(){ return streamId ;}
	bool			isDecoderInited(){ return videoDecoder->isInited(); }
	int				initVDecoder(const char * sprop){ return videoDecoder->init(sprop); }
};

// the buffer to store the tagged frame
struct FrameKey{
	char frameIndex;
	unsigned char tag;

	bool operator < (const FrameKey & key1)const{
		if(tag < key1.tag){
			return true;
		}else if(tag == key1.tag && frameIndex < key1.frameIndex)
			return true;
		else
			return false;
	}
};

#ifdef USE_TEMPLATE_FRAME_POOL
template<class T>
class FramePool{
	map<FrameKey, T *> pool;

public:
	int addFrame(FrameKey key, T * frame){
		pool[key] = frame;
		return pool.size();
	}
	T * getFrame(FrameKey &lowBound){
		T * ret= NULL;
		map<FrameKey, T *>::iterator it = pool.upper_bound(lowBound);
		if(it != pool.end()){
			lowBound.frameIndex = it->first.frameIndex;
			lowBound.tag = it->first.tag;
			ret = it->second;
			pool.erase(it);
			return ret;
		}
		else
			return NULL;
	}
};

#else
class FramePool{
	map<FrameKey, AVFrame *> pool;
	
public:
	int addFrame(FrameKey key, AVFrame * frame){
		pool[key] = frame;
		return pool.size();
	}
	AVFrame * getFrame(FrameKey &lowBound){
		AVFrame * ret= NULL;
		map<FrameKey, AVFrame *>::iterator it = pool.upper_bound(lowBound);
		if(it != pool.end()){
			lowBound.frameIndex = it->first.frameIndex;
			lowBound.tag = it->first.tag;
			ret = it->second;
			pool.erase(it);
			return ret;
		}
		else
			return NULL;
	}
};

#endif


// game stream is formed nby SubGameStream, need to merge the video streams. Actually, it is the back end to display the result
class GameStreams{
	char * disUrl;

	struct cg::RTSPConf * rtspConf;
	CRITICAL_SECTION section;
	LPCRITICAL_SECTION pSection;

	AudioDecoder *	audioDecoder;
	SubGameStream * subStreams[MAX_RTSP_STREAM];
	//GameDecoder * gameDecoder[MAX_RTSP_STREAM];   // the decoder for this each stream

	static map<RTSPClient *, SubGameStream *> streamMap;

	int				videoSessFmt;
	int				audioSessFmg;
	char *			videoCodecName;
	char *			audioCodecName;
	enum AVCodecID	videoCodecId;
	enum AVCodecID	audoCodecId;
	int				videoFraming;
	int				audioFraming;

#ifdef COUNT_FRAME_RATE
	int				cf_frame;
	struct timeval	cf_tv0;
	struct timeval	cf_tv1;
	long long		cf_interval;
#endif

	// the attributes are for rendering
	CRITICAL_SECTION surfaceMutex;
#if 1 // only support SDL2
	unsigned int	windowId;
	SDL_Window *	surface;
	SDL_Renderer *	renderer;
	SDL_Texture *	overlay;

#endif

	struct SwsContext * swsctx;

	//////////////////
	int				width;
	int				height;
	PixelFormat		format;
	AVFrame *		vFrame;

	///////////////////
	cg::pipeline  *		pipe;

	CRITICAL_SECTION audioMutex;
	bool			audioOpened;

	int				totalStreams;

	char			currentFrameIndex;
	char			nextFrameIndex;
	char			maxFrameIndex, minFrameIndex;  // current the max and min index of the valid frames

	unsigned char	currentFrameTag;
	unsigned char	nextFrameTag;
	unsigned char	maxFrameTag, minFrameTag;    // current the max and min frame tag of the valid frames

	short			displayInterval;    // the time interval to display when there are multiple frames arriving.
	
#ifdef USE_TEMPLATE_FRAME_POOL
	FramePool<TaggedFrame> framePool;
#else
	FramePool		framePool;
#endif
	FrameKey		lowBound;

	bool running;
	HANDLE newAddEvent, declineEvent;
	HANDLE frameEvent; // set the event when store the frame
	//storeIndexedFrameToMatrix(short row, short col);
	GameStreams();
	static GameStreams * streams;

public:
	static GameStreams * GetStreams(){
		if(!streams){
			streams = new GameStreams();
		}
		return streams;

	}
	// used when distributor is enabled
	inline void		setDisUrl(char * url){ disUrl = _strdup(url); }
	inline char *	getDisUrl(){ return disUrl; }


	inline HANDLE	getNewAddEvent(){ return newAddEvent; }
	inline HANDLE	getDeclineEvent(){ return declineEvent; }
	inline HANDLE	getFrameEvent(){ return frameEvent; }
	inline bool		isRunning(){ return running; }
	inline void		setRunning(bool val){ running = val; }
	inline int		getTotalStreams(){ return totalStreams; }
	inline void		setTotalStreams(int val){ totalStreams = val; }

	inline CRITICAL_SECTION *	getSurfaceMutex(){ return &surfaceMutex; }
	inline SDL_Window *			getSurface(){ return surface; }
	inline void		setWidth(int val){ width = val; }
	inline void		setHeight(int val){ height = val; }
	inline int		getWidth(){ return width; }
	inline int		getHeight(){ return height; }
	inline PixelFormat			getFormat(){ return format; }
	inline void		setFormat(PixelFormat f){ format = f; }

	inline AVFrame *			getVFrame(){ return vFrame; }
	inline cg::pipeline *		getPipe(){ return pipe; }
	inline SDL_Texture *		getOverlay(){ return overlay; }
	inline SwsContext *			getSwsctx(){ return  swsctx; }
	inline SDL_Renderer *		getRenderer(){ return renderer; }
	inline unsigned int			getWindowId(){ return windowId; }
	//inline SDL_Window * getSurface(int ch){ return surface[ch]; }
	inline void		setPipe(cg::pipeline *p){ pipe = p; }
	inline void		setSwsctx(SwsContext * ctx){ swsctx = ctx; }
	inline void		setOverlay(SDL_Texture * t){ overlay = t; }
	inline void		setRenderer(SDL_Renderer * r){ renderer = r; }
	inline void		setWindowId(unsigned int id){ windowId = id; }
	inline void		setSurface(SDL_Window * w){ surface = w; }

	inline AudioDecoder *		getAudioDecoder(){ return this->audioDecoder; }

	inline char		getCurrentFrameIndex(){ return currentFrameIndex; }
	inline char		getNextFrameIndex(){ return (currentFrameIndex + 1)%totalStreams; }
	inline char		getNextFrameIndex(char frameIndex){ return (frameIndex + 1) % totalStreams; }
	inline void		setFrameIndex(char val){ currentFrameIndex = val; }

	inline unsigned char		getNextFrameTag(){ return (currentFrameTag + 1) % MAX_RTSP_STREAM; }
	inline unsigned char		getNextFrameTag(unsigned char frameTag){ return (frameTag +1 )% totalStreams; }
	inline unsigned char		getCurrentFrameTag(){ return currentFrameTag; }
	inline void		setFrameTag(unsigned char tag){ currentFrameTag = tag; }

	inline bool		lock(){ EnterCriticalSection(pSection); return true; }
	inline bool		unlock(){ LeaveCriticalSection(pSection); return true;}

	inline void		setSubStream(SubGameStream * stream, int index){ subStreams[index] = stream; }
	SubGameStream * getStream(int frameIndex);   // return the game stream using the frame index received from the render server
	SubGameStream * removeStream(char * url);
	SubGameStream * getStream(char * url);
	static SubGameStream * getStream(RTSPClient * client);  // returnt he game stream using the rtspclient
	//static void		addMap(RTSPClient * client, SubGameStream * stream);
	
	int				playVideo();
	bool			checkRunning(){ return true; }
	void			countFrameRate();


#ifdef USE_TEMPLATE_FRAME_POOL
	int				storeFrame(FrameKey key, TaggedFrame * frame);
	TaggedFrame *	getIndexedFrame(FrameKey &lowBound);
	int				formDisplayEvent(TaggedFrame* frame);
#else
	int				storeFrame(FrameKey key, AVFrame * frame);
	AVFrame *		getIndexedFrame(FrameKey &lowBound);
	int				formDisplayEvent(AVFrame * frame);
#endif
	
	~GameStreams();

	// tool function to add sub streams
	bool			addSubStream(SubGameStream * subStream);
	bool			init();
	bool			addRenders(char * cmd);
	bool			declineRenders(char * cmd);
	bool			createOverlay();
	bool			renderImage(long long special);
	bool			renderImage(unsigned char specialTag, unsigned char valueTag);

	HANDLE mutex;
	char * name;
};


// the procedure for handling the display, a gameStreams will wait for the subGameStream's event to load the recved frame, and then display it using a time driven model and a event driven
// if recved frame is the next frame to show, then ,display it immidiately, if not, wait for the right frame, but, 30ms later, diaplay the recved frame and corrent the frame index
DWORD WINAPI RTSPDisplayThread(LPVOID param);

//////////////////////////////////
class StreamClientState {
public:
	StreamClientState();
	virtual ~StreamClientState();

public:
	MediaSubsessionIterator* iter;
	MediaSession*	session;
	MediaSubsession* subsession;
	TaskToken		streamTimerTask;
	double			duration;
};
class ourRTSPClient : public RTSPClient {
public:
	static ourRTSPClient* createNew(UsageEnvironment& env, char const* rtspURL,
		int			verbosityLevel = 0,
		char const* applicationName = NULL,
		portNumBits tunnelOverHTTPPortNum = 0);

protected:
	ourRTSPClient(UsageEnvironment& env, char const* rtspURL,
		int			verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum);
	// called only by createNew();
	virtual ~ourRTSPClient();

public:
	StreamClientState scs;

	// the stream for the client
	SubGameStream * subStream;
	// the decoder for the client
	VideoDecoder *	streamDecoder;

	inline SubGameStream * getSubStream(){ return subStream; }
	inline VideoDecoder * getDecoder(){ return streamDecoder; }
};

class DummySink : public MediaSink {
public:
	static DummySink* createNew(UsageEnvironment& env,
		MediaSubsession& subsession, // identifies the kind of data that's being received
		char const* streamId = NULL); // identifies the stream itself (optional)

	static DummySink * createNew(UsageEnvironment & env, MediaSubsession & subsession,SubGameStream * subStream,char const * streamId = NULL);

private:
	DummySink(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId);
	// called only by "createNew()"
	virtual ~DummySink();

	static void		afterGettingFrame(void* clientData, unsigned frameSize,
		unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds);
	void			afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds);

private:
	// redefined virtual functions:
	virtual Boolean continuePlaying();

private: 
	u_int8_t*		fReceiveBuffer;
	MediaSubsession& fSubsession;
	char*			fStreamId;

	SubGameStream * subStream;    // the sub stream for this connection
	VideoDecoder *	videoDecoder; // the decoder for this connection

	int videoFraming;
	int audioFraming;
};


RTSPClient *		openUrl(UsageEnvironment& env, char const * rtspUrl);
void				continueAfterDESCRIBE(RTSPClient * rtspClient, int resultCode, char * resultString);
void				setupNextSubsession(RTSPClient * rtspClient);
void				NATHolePunch(RTPSource * rtpsrc, MediaSubsession * subsession);
void				continueAfterSETUP(RTSPClient * rtspClient, int resultCode, char * resultString);
void				continueAfterPLAY(RTSPClient * rtspClient, int resultCode, char * resultString);
void				subsessionAfterPlaying(void * clientData);
void				subsessionByeHandler(void * clientData);
void				streamTimerHandler(void * clientData);
void				shutdownStream(RTSPClient * rtspClient, int exitCode);


DWORD WINAPI rtspThreadForSubsession(LPVOID param);

#endif   // MULTI_SOURCE
#endif