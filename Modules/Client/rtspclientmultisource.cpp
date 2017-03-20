#ifndef ANDROID
#include <vector>
//#include "ccg_win32.h"
#include "../VideoUtility/ccg_config.h"
#include "../VideoUtility/rtspConf.h"
#include "rtspclientmultisource.h"
#include "../VideoGen/rtspcontext.h"
#include "../VideoUtility/pipeline.h"
#include "../VideoUtility/encoder.h"
#include "../VideoUtility/FilterRGB2YUV.h"
#include "../LibCore/TimeTool.h"
#endif

#include "../VideoUtility/avcodeccommon.h"
#include "../LibInput/ctrlconfig.h"
#include "../LibInput/controller.h"
#include "minih264.h"
#include "../LibCore/InfoRecorder.h"

#ifndef INCLUDE_DISTRIBUTOR
#define INCLUDE_DISTRIBUTOR
#endif
// the define will enable saving result to bmp file
//#define SAVE_SURFACE

#ifdef SAVE_SURFACE
#include "../CudaFilter/bmpformat.h"
#endif

#ifdef ANDROID
#include "android-decoders.h"
#include "libgaclient.h"
#endif

//#include <Windows.h>
#include <string.h>
#include <list>
#include <map>
//#include "ccg_win32.h"

using namespace std;
using namespace cg;
using namespace cg::core;
using namespace cg::input;

#ifndef	AVCODEC_MAX_AUDIO_FRAME_SIZE
#define	AVCODEC_MAX_AUDIO_FRAME_SIZE	192000 // 1 second of 48khz 32bit audio
#endif

#ifndef IMAGE_SOURCE_CHANNEL_MAX
#define IMAGE_SOURCE_CHANNEL_MAX 4
#endif

#define	COUNT_FRAME_RATE	600	// every N frames
#define MAX_FRAMING_SIZE 8

struct cg::RTSPConf * rtspConf;

static int audioSessFmt;
static char * audioCodecName;

// global variables, global streams which contains all the subStreams
extern GameStreams * gameStreams;
extern CRITICAL_SECTION watchdogMutex;
extern struct timeval watchdogTimer;
extern int relativeMouseMode;
extern int showCursor;
extern int windowSizeX;
extern int windowSizeY;

int image_rendered = 0;


static cg::core::BTimer * displayTimer = NULL;
////////////////////////// PacketQueue ///////////
PacketQueue::PacketQueue(){
	size = 0;
	mutex = NULL;
	cond = NULL;
	packetQueueInitialized = false;
}

PacketQueue::~PacketQueue(){
	if(mutex){
		CloseHandle(mutex);
		mutex = NULL;
	}
	if(cond){
		CloseHandle(cond);
		cond = NULL;
	}
	packetQueueInitialized = false;
}

void PacketQueue::packetQueueInit(){
	packetQueueInitialized = true;
	this->_queue.clear();
	// create the mutex and the event
	this->mutex = CreateMutex(NULL, FALSE, NULL);
	this->cond = CreateEvent(NULL, FALSE, FALSE, NULL);

}
int PacketQueue::packetQueueGet(AVPacket * pkt, int block){
	int ret = -1;
	if(packetQueueInitialized == false){
		rtsperror("[PacketQueue]: packet queue not initialized.\n");
		return -1;
	}
	WaitForSingleObject(mutex, INFINITE);
	for(;;){
		if(_queue.size() > 0){
			*pkt = _queue.front();
			_queue.pop_front();
			size -= pkt->size;
			ret = 1;
			break;
		}
		else if(!block){
			ret = 0;
			break;
		}
		else{
			struct cg::core::timespec ts;
#if defined(__APPLE__) || defined(WIN32)
			struct timeval tv;
			getTimeOfDay(&tv, NULL);
			ts.tv_sec = tv.tv_sec;
			ts.tv_nsec = tv.tv_usec * 1000;
#else
			clock_gettime(CLOCK_REALTIME, &ts);
#endif
			ts.tv_nsec += 50000000LL; // 50ms
			if(ts.tv_nsec >= 1000000000LL){
				ts.tv_sec++;
				ts.tv_nsec -= 1000000000LL;
			}
			DWORD r = WaitForSingleObject(cond, ts.tv_sec * 1000);
			if(r == WAIT_TIMEOUT){
				ret = -1;
				break;
			}
			//
		}
	}
	ReleaseMutex(mutex);
	return ret;
}
int PacketQueue::packetQueuePut(AVPacket * pkt){
	if(av_dup_packet(pkt) < 0){
		rtsperror("[PacketQueue]: packet queue put failed.\n");
		return -1;
	}
	WaitForSingleObject(mutex, INFINITE);
	_queue.push_back(*pkt);
	size += pkt->size;
	ReleaseMutex(mutex);
	SetEvent(cond);
	return 0;
}

UsageEnvironment&
	operator<<(UsageEnvironment& env, const RTSPClient& rtspClient) {
		return env << "[URL:\"" << rtspClient.url() << "\"]: ";
}

UsageEnvironment&
	operator<<(UsageEnvironment& env, const MediaSubsession& subsession) {
		return env << subsession.mediumName() << "/" << subsession.codecName();
}

void
	rtsperror(const char *fmt, ...) {
		//static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
		va_list ap;
		//pthread_mutex_lock(&mutex);
		va_start(ap, fmt);
#ifdef ANDROID
		__android_log_vprint(ANDROID_LOG_INFO, "ga_log", fmt, ap);
#else
		vfprintf(stderr, fmt, ap);
#endif
		va_end(ap);
		//pthread_mutex_unlock(&mutex);
		return;
}
////////////////////////// SubGameStream ////////////////////

int SubGameStream::RTSPClientCount = 0;
bool SubGameStream::init(){
	running = false;
	notifier = CreateEvent(NULL, FALSE, FALSE, NULL);
	thread = NULL;
	quitLive555 = 1;

	rtspClient = NULL;
	env = NULL;
	videoDecoder = new VideoDecoder();
	decoderBuffer = (DecoderBuffer *)malloc(sizeof(DecoderBuffer));
	memset(decoderBuffer, 0, sizeof(DecoderBuffer));
	streamId = 0;
	return true;
}

SubGameStream::~SubGameStream(){
	if(notifier){
		CloseHandle(notifier);
		notifier = NULL;
	}
	if(decoderBuffer){
		free(decoderBuffer);
		decoderBuffer = NULL;
	}
}

// constructor takes the url
SubGameStream::SubGameStream(char * url){
	// the url is the ip, we need to form the real rtsp url
	char rtspUrl[100] = {0};
	if(url[0] == 'r' || url[0] == 'R'){
		// the url is a rtps url
		this->url = _strdup(url);
	}
	else{
		// form the rtsp url
		sprintf(rtspUrl, "rtsp://%s:%d/desktop", url, cg::RTSPConf::GetRTSPConf()->serverPort);
		cg::core::infoRecorder->logError("[SubGameStream]: construct game stream with url '%s'.\n", rtspUrl);
		this->url = _strdup(rtspUrl);
	}

	init();
}

SubGameStream::SubGameStream(char * url, int port){
	char rtspUrl[100] = {0};
	sprintf(rtspUrl, "rtsp://%d:%d/desktop", url, cg::RTSPConf::GetRTSPConf()->serverPort);
	this->url = strdup(rtspUrl);

	init();
}

bool SubGameStream::openUrl(UsageEnvironment * env, char * url){
	this->env = env;
	char rtspUrl[100] = {0};

	cg::core::infoRecorder->logError("[SubGameStream]: openUrl, url = '%s'.\n", url);

	if(url[0] == 'r' || url[0] == 'R'){
		this->url = _strdup(url);
		rtsperror("open url, %s\n", this->url);
		//cg::core::infoRecorder->logError("[SubGameStream]: openUrl '%s'.\n", this->url);
	}
	else{
		sprintf(rtspUrl, "rtsp://%s:%d/desktop", url, cg::RTSPConf::GetRTSPConf()->serverPort);
		this->url = _strdup(rtspUrl);
		rtsperror("open url, %s\n", this->url);
		cg::core::infoRecorder->logError("[SubGameStream]: openURL '%s'.\n", this->url);
	}

	ourRTSPClient * c = ourRTSPClient::createNew(*env, this->url, RTSP_CLIENT_VERBOSITY_LEVEL, "RTSP Client");
	++RTSPClientCount;
	c->subStream = this;

	c->sendDescribeCommand(continueAfterDESCRIBE);
	rtspClient = c;
	return true;
}
bool SubGameStream::openUrl(UsageEnvironment * env){
	this->env = env;
	// check the url
	if(this->url == NULL){
		cg::core::infoRecorder->logError("[SubGameStream]: NULL url for the sub game stream.\n");
		return false;
	}
	RTSPClient * c = ourRTSPClient::createNew(*env, url, RTSP_CLIENT_VERBOSITY_LEVEL, "RTSP Client");
	++RTSPClientCount;
	c->sendDescribeCommand(continueAfterDESCRIBE);
	rtspClient = c;
	return true;
}
RTSPClient * SubGameStream::OpenURL(UsageEnvironment &env, char const * rtspURL){
	ourRTSPClient * rtspClient  = ourRTSPClient::createNew(env, rtspURL, RTSP_CLIENT_VERBOSITY_LEVEL, "RTSP Client");
	if(rtspClient == NULL){
		rtsperror("connect failed: %s \n", env.getResultMsg());
		return NULL;
	}
	else{
		//rtspClient->subStream = this;
	}

	++RTSPClientCount;
	rtspClient->sendDescribeCommand(continueAfterDESCRIBE);
	return rtspClient;
}

void continueAfterDESCRIBE(RTSPClient * rtspClient, int resultCode, char * resultString){
	do{
		UsageEnvironment & env = rtspClient->envir();  // alias
		StreamClientState & scs = ((ourRTSPClient*)rtspClient)->scs;  // alias
		if(resultCode != 0){
			env << rtspClient << "Failed to get a SDP description: " << resultString << " " << resultCode << "\n";
			break;
		}
		char * const sdpDescription = resultString;
		env << rtspClient << "Got a SDP description:\n" << sdpDescription << "\n";
		infoRecorder->logTrace("[RTSP]: %p Got a SDP description: %s.\n", rtspClient, sdpDescription);
		// create a media session object from this SDP description;
		scs.session = MediaSession::createNew(env, sdpDescription);
		delete[] sdpDescription; // because we don't need it anymore
		if (scs.session == NULL) {
			env << *rtspClient << "Failed to create a MediaSession object from the SDP description: " << env.getResultMsg() << "\n";
			infoRecorder->logTrace("[RTSP]: %p Failed to create a MediaSession object from the SDP description: %s.\n", rtspClient, env.getResultMsg());
			break;
		}
		else if (!scs.session->hasSubsessions()) {
			env << *rtspClient << "This session has no media subsessions (i.e., no \"m=\" lines)\n";
			infoRecorder->logTrace("[RTSP]: %d This session has no media subsessions (i.e., no \"m=\" lines)\n", rtspClient);
			break;
		}

		scs.iter = new MediaSubsessionIterator(*scs.session);
		setupNextSubsession(rtspClient);
		return;
	}while(0);
	// An unrecoverable error occurred with this stream.
	//shutdownStream(rtspClient);
	rtsperror("Connect to %s failed.\n", rtspClient->url());
	cg::core::infoRecorder->logError("[continueAfterDESCRIBE]: connect to %s failed.\n", rtspClient->url());
#ifdef ANDROID
	goBack(rtspParam->jnienv, -1);
#else
	((ourRTSPClient*)rtspClient)->subStream->setQuitLive555(1);
#endif
}
void setupNextSubsession(RTSPClient* rtspClient){
	infoRecorder->logTrace("[RTSP]: setup next subsession.\n");
	UsageEnvironment& env = rtspClient->envir(); // alias
	StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

	SubGameStream * subStream = ((ourRTSPClient *)rtspClient)->subStream;

	VideoDecoder * vDecoder = subStream->getVideoDecoder();
	bool rtpOverTCP = false;

	rtspConf = cg::RTSPConf::GetRTSPConf();

	if (rtspConf->proto == IPPROTO_TCP) {
		rtpOverTCP = true;
	}

	scs.subsession = scs.iter->next();
	do if (scs.subsession != NULL) {
			if (!scs.subsession->initiate()) {
				env << *rtspClient << "Failed to initiate the \"" << *scs.subsession << "\" subsession: " << env.getResultMsg() << "\n";
				infoRecorder->logTrace("[RTSP]: Failed to initialize the %p sub session: %s\n", *scs.subsession, env.getResultMsg());
				setupNextSubsession(rtspClient); // give up on this subsession; go to the next one
			}
			else {
				if (strcmp("video", scs.subsession->mediumName()) == 0) {
					// use the Decoder or just build the function in the sub stream????????
					vDecoder->setVideoSessFmt(scs.subsession->rtpPayloadFormat());
					//gameStreams->videoSessFmt = scs.subsession->rtpPayloadFormat();
					vDecoder->setCodecName(_strdup(scs.subsession->codecName()));
					//gameStreams->videoCodecName = _strdup(scs.subsession->codecName());
					cg::core::infoRecorder->logError("[ga-client]: video, client port num:%d.\n", scs.subsession->clientPortNum());

					// check the decoder initialization
					// just init the decoder
					if(!subStream->isDecoderInited()){
						if(subStream->initVDecoder(scs.subsession->fmtp_spropparametersets()) < 0){
							//video decoder init failed
							rtsperror("[ga-client]: subStream %d cannot initialize video decoder", subStream->getId());
							infoRecorder->logError("[RTSP]: substream %d cannot iniitalize video decoder", subStream->getId());
							subStream->setQuitLive555(1);
							return;
						}
						rtsperror("[ga-client]: video decoder for stream %d initialized (client prot %d)\n", subStream->getId(), scs.subsession->clientPortNum());
						infoRecorder->logTrace("[RTSP]: video decoder for stream %d initialized (client prot %d)\n", subStream->getId(), scs.subsession->clientPortNum());
					}
				}else if (strcmp("audio", scs.subsession->mediumName()) == 0) {
					const char *mime = NULL;
					audioSessFmt = scs.subsession->rtpPayloadFormat();
					audioCodecName = _strdup(scs.subsession->codecName());
					AudioDecoder *aDecoder = gameStreams->getAudioDecoder();

					if (aDecoder == NULL) {
						if (aDecoder->initADecoder() < 0) {
							rtsperror("cannot initialize audio decoder.\n");
							subStream->setQuitLive555(1);
							return;
						}
					}
				//}

				rtsperror("audio decoder initialized.\n");
			}

			env << *rtspClient << "Initiated the \"" << *scs.subsession
				<< "\" subsession (client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum() + 1 << ")\n";
			infoRecorder->logTrace("[RTSP]: %p Initiated the %p subsession (client ports: %d - %d ).\n", rtspClient, scs.subsession, scs.subsession->clientPortNum(), scs.subsession->clientPortNum() + 1);

			// Continue setting up this subsession, by sending a RTSP "SETUP" command:
			rtspClient->sendSetupCommand(*scs.subsession, continueAfterSETUP, False, rtpOverTCP ? True : False/*TCP?*/, False, NULL);
		}
		return;
	} while (0);

	// We've finished setting up all of the subsessions.  Now, send a RTSP "PLAY" command to start the streaming:
	infoRecorder->logTrace("[RTSP]: send play command.\n");
	scs.duration = scs.session->playEndTime() - scs.session->playStartTime();
	rtspClient->sendPlayCommand(*scs.session, continueAfterPLAY);
}

void NATHolePunch(RTPSource *rtpsrc, MediaSubsession *subsession) {
	infoRecorder->logTrace("[Global]: NATHolePunch.\n");
	Groupsock *gs = NULL;
	int s;
	struct sockaddr_in sin;
	unsigned char buf[1] = { 0x00 };

	cg::RTSPConf * rtspConf = cg::RTSPConf::GetRTSPConf();

#ifdef WIN32
	int sinlen = sizeof(sin);
#else
	socklen_t sinlen = sizeof(sin);
#endif
	if (rtspConf->sin.sin_addr.s_addr == 0
		|| rtspConf->sin.sin_addr.s_addr == INADDR_NONE) {
			rtsperror("NAT hole punching: no server address available.\n");
			infoRecorder->logTrace("[Global]: NAT hole punching: no server address available.\n");
			return;
	}
	if (rtpsrc == NULL) {
		rtsperror("NAT hole punching: no RTPSource available.\n");
		infoRecorder->logTrace("[Global]: NAT hole punching: no RTPSource available.\n\n");
		return;
	}
	if (subsession == NULL) {
		rtsperror("NAT hole punching: no subsession available.\n");
		infoRecorder->logTrace("[Global]: NAT hole punching: no subsession available.\n");
		return;
	}
	gs = rtpsrc->RTPgs();
	if (gs == NULL) {
		rtsperror("NAT hole punching: no Groupsock available.\n");
		infoRecorder->logTrace("[Global]: NAT hole punching: no Groupsock available.\n");
		return;
	}
	//
	s = gs->socketNum();
	if (getsockname(s, (struct sockaddr*) &sin, &sinlen) < 0) {
		rtsperror("NAT hole punching: getsockname - %s.\n", strerror(errno));
		infoRecorder->logTrace("[Global]: NAT hole punching: getsockname - %s.\n", strerror(errno));
		return;
	}
	rtsperror("NAT hole punching: fd=%d, local-port=%d/%d server-port=%d\n",
		s, ntohs(sin.sin_port), subsession->clientPortNum(), subsession->serverPortNum);
	infoRecorder->logTrace("[Global]: NAT hole punching: fd=%d, local-port=%d/%d server-port=%d\n",
		s, ntohs(sin.sin_port), subsession->clientPortNum(), subsession->serverPortNum);
	//
	bzero(&sin, sizeof(sin));
	sin.sin_addr = rtspConf->sin.sin_addr;
	sin.sin_port = htons(subsession->serverPortNum);
	// send 5 packets
	// XXX: use const char * for buf pointer to work with Windows
	sendto(s, (const char *)buf, 1, 0, (struct sockaddr*) &sin, sizeof(sin)); usleep(5000);
	sendto(s, (const char *)buf, 1, 0, (struct sockaddr*) &sin, sizeof(sin)); usleep(5000);
	sendto(s, (const char *)buf, 1, 0, (struct sockaddr*) &sin, sizeof(sin)); usleep(5000);
	sendto(s, (const char *)buf, 1, 0, (struct sockaddr*) &sin, sizeof(sin)); usleep(5000);
	sendto(s, (const char *)buf, 1, 0, (struct sockaddr*) &sin, sizeof(sin)); usleep(5000);
	//
	return;
}
void continueAfterSETUP(RTSPClient* rtspClient, int resultCode, char* resultString) {
	infoRecorder->logTrace("[RTSP]: continue after SETUP.\n");
	do {
		UsageEnvironment& env = rtspClient->envir(); // alias
		ourRTSPClient * ourRtspClient = ((ourRTSPClient *)rtspClient);

		StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

		if (resultCode != 0) {
			env << *rtspClient << "Failed to set up the \"" << *scs.subsession << "\" subsession: " << env.getResultMsg() << "\n";
			infoRecorder->logTrace("[RTSP]: Failed to set up the %p subsession:%s.\n", *scs.subsession, env.getResultMsg());
			break;
		}

		env << *rtspClient << "Set up the \"" << *scs.subsession
			<< "\" subsession (client ports " << scs.subsession->clientPortNum() << "-" << scs.subsession->clientPortNum() + 1 << ")\n";
		infoRecorder->logTrace("[RTSP]: Set up the %p subsession (client ports: %d - %d)\n", scs.subsession, scs.subsession->clientPortNum(), scs.subsession->clientPortNum() + 1);

		scs.subsession->sink = DummySink::createNew(env, *scs.subsession, ourRtspClient->getSubStream(), rtspClient->url());

		// perhaps use your own custom "MediaSink" subclass instead
		if (scs.subsession->sink == NULL) {
			env << *rtspClient << "Failed to create a data sink for the \"" << *scs.subsession
				<< "\" subsession: " << env.getResultMsg() << "\n";
			infoRecorder->logTrace("[RTSP]: Failed to create a data sink for the %p sub session: %s.\n", scs.subsession, env.getResultMsg());
			break;
		}

		env << *rtspClient << "Created a data sink for the \"" << *scs.subsession << "\" subsession\n";
		infoRecorder->logTrace("[RTSP]: create a data sink for the %p subsession.\n", scs.subsession);

		scs.subsession->miscPtr = rtspClient; // a hack to let subsession handle functions get the "RTSPClient" from the subsession 

		infoRecorder->logTrace("[RTSP]: start playing.\n");
		scs.subsession->sink->startPlaying(*(scs.subsession->readSource()),
			subsessionAfterPlaying, scs.subsession);
		// Also set a handler to be called if a RTCP "BYE" arrives for this subsession:
		if (scs.subsession->rtcpInstance() != NULL) {
			scs.subsession->rtcpInstance()->setByeHandler(subsessionByeHandler, scs.subsession);
		}
		// NAT hole-punching?
		NATHolePunch(scs.subsession->rtpSource(), scs.subsession);
	} while (0);

	// Set up the next subsession, if any:
	setupNextSubsession(rtspClient);
}

void continueAfterPLAY(RTSPClient* rtspClient, int resultCode, char* resultString) {
	infoRecorder->logTrace("[RTSP]: continue after PLAY, result:%s.\n", resultString);
	do {
		UsageEnvironment& env = rtspClient->envir(); // alias
		StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

		if (resultCode != 0) {
			env << *rtspClient << "Failed to start playing session: " << resultString << "\n";
			infoRecorder->logError("[RTSP]: %p Failed to start playing session: %s.\n", rtspClient, resultString);
			break;
		}

		if (scs.duration > 0) {
			unsigned const delaySlop = 2; // number of seconds extra to delay, after the stream's expected duration.  (This is optional.)
			scs.duration += delaySlop;
			unsigned uSecsToDelay = (unsigned)(scs.duration * 1000000);
			scs.streamTimerTask = env.taskScheduler().scheduleDelayedTask(uSecsToDelay, (TaskFunc*)streamTimerHandler, rtspClient);
		}

		env << *rtspClient << "Started playing session";
		infoRecorder->logTrace("[RTSP]: started playing session, ");
		if (scs.duration > 0) {
			env << " (for up to " << scs.duration << " seconds)";
			infoRecorder->logTrace(" (for up to %d seconds)", scs.duration);
		}
		env << "...\n";
		infoRecorder->logTrace("...\n");

		return;
	} while (0);

	// An unrecoverable error occurred with this stream.
	SubGameStream::shutdownStream(rtspClient);
}
void subsessionAfterPlaying(void* clientData) {
	infoRecorder->logTrace("[RTSP]: subsession after PLAYING.\n");
	MediaSubsession* subsession = (MediaSubsession*)clientData;
	RTSPClient* rtspClient = (RTSPClient*)(subsession->miscPtr);

	// Begin by closing this subsession's stream:
	Medium::close(subsession->sink);
	subsession->sink = NULL;

	// Next, check whether *all* subsessions' streams have now been closed:
	MediaSession& session = subsession->parentSession();
	MediaSubsessionIterator iter(session);
	while ((subsession = iter.next()) != NULL) {
		if (subsession->sink != NULL) {
			infoRecorder->logTrace("[RTSP]: session still alive.\n");
			return; // this subsession is still active
		}
	}

	// All subsessions' streams have now been closed, so shutdown the client:
	infoRecorder->logError("[RTSP]: session is going to shut down.\n");
	SubGameStream::shutdownStream(rtspClient);
}

void subsessionByeHandler(void* clientData) {
	MediaSubsession* subsession = (MediaSubsession*)clientData;
	RTSPClient* rtspClient = (RTSPClient*)subsession->miscPtr;
	UsageEnvironment& env = rtspClient->envir(); // alias

	env << *rtspClient << "Received RTCP \"BYE\" on \"" << *subsession << "\" subsession\n";
	infoRecorder->logTrace("[RTSP]: %p received RTCP 'BYE' on '%d' subsession.\n", rtspClient, subsession);

	// Now act as if the subsession had closed:
	subsessionAfterPlaying(subsession);
}

void streamTimerHandler(void* clientData) {
	ourRTSPClient* rtspClient = (ourRTSPClient*)clientData;
	StreamClientState& scs = rtspClient->scs; // alias

	scs.streamTimerTask = NULL;

	// Shut down the stream:
	SubGameStream::shutdownStream(rtspClient);
}

void SubGameStream::shutdownStream(RTSPClient * rtspClient, int exitCode){
	UsageEnvironment& env = rtspClient->envir(); // alias
	StreamClientState& scs = ((ourRTSPClient*)rtspClient)->scs; // alias

	infoRecorder->logError("[SubGameStream]: shutdown stream with cdoe:%d.\n", exitCode);

	if (RTSPClientCount <= 0)
		return;

	// First, check whether any subsessions have still to be closed:
	if (scs.session != NULL) {
		Boolean someSubsessionsWereActive = False;
		MediaSubsessionIterator iter(*scs.session);
		MediaSubsession* subsession;

		while ((subsession = iter.next()) != NULL) {
			if (subsession->sink != NULL) {
				Medium::close(subsession->sink);
				subsession->sink = NULL;

				if (subsession->rtcpInstance() != NULL) {
					subsession->rtcpInstance()->setByeHandler(NULL, NULL); // in case the server sends a RTCP "BYE" while handling "TEARDOWN"
				}

				someSubsessionsWereActive = True;
			}
		}
		if (someSubsessionsWereActive) {
			// Send a RTSP "TEARDOWN" command, to tell the server to shutdown the stream.
			// Don't bother handling the response to the "TEARDOWN".
			rtspClient->sendTeardownCommand(*scs.session, NULL);
		}
	}

	env << *rtspClient << "Closing the stream.\n";
	Medium::close(rtspClient);
	// Note that this will also cause this stream's "StreamClientState" structure to get reclaimed.

	if (--RTSPClientCount == 0) {
		// The final stream has ended, so exit the application now.
		// (Of course, if you're embedding this code into your own application, you might want to comment this out,
		// and replace it with "eventLoopWatchVariable = 1;", so that we leave the LIVE555 event loop, and continue running "main()".)
		//exit(exitCode);
		rtsperror("rtsp thread: no more rtsp clients\n");

		//TODO : corrent the exit for a client
		//quitLive555 = 1;
	}
}
// the buffer contains the header data
void SubGameStream::playVideo(unsigned char * buffer, int bufSize, struct timeval pts, bool maker){
	DecoderBuffer * pdb = decoderBuffer;

	if(pdb->privBuf == NULL){
		pdb->privBuf = (unsigned char *)malloc(PRIVATE_BUFFER_SIZE);
		if(pdb->privBuf == NULL){
			rtsperror("FATAL: cannot allocate private buffer (%d bytes): %s.\n", PRIVATE_BUFFER_SIZE, strerror(errno));
			quitLive555 = 1;
			return;
		}
	}
	if(bufSize <= 0 || buffer == NULL){
		rtsperror("Empty buffer?\n");
		return;
	}

	// new frame, play the last one
	if(pts.tv_sec != pdb->lastpts.tv_sec || pts.tv_usec != pdb->lastpts.tv_usec){
		if(pdb->privBufLen > 0){

			playVideoPriv(pdb->privBuf, pdb->privBufLen, pdb->lastpts);
		}
		pdb->privBufLen = 0;
	}
	// the current frame
	if(pdb->privBufLen + bufSize <= PRIVATE_BUFFER_SIZE){
		// here we need to be careful, because buffered data contains the frame index and frame tag, so we need to get the real packet data

#ifdef MULTI_SOURCE_SUPPORT
		// original
		bcopy(buffer, &pdb->privBuf[pdb->privBufLen], bufSize);
		pdb->privBufLen += bufSize;

#endif
		pdb->lastpts = pts;
		// find the end flag of the frame, play it
		if(maker && pdb->privBufLen > 0){
			playVideoPriv(pdb->privBuf, pdb->privBufLen, pdb->lastpts);
			pdb->privBufLen = 0;
		}
	}
	else{
		rtsperror("WARNING: video private buffer overflow.\n");
		playVideoPriv(pdb->privBuf, pdb->privBufLen, pdb->lastpts);
		pdb->privBufLen = 0;
	}
	return;
}

static int writeFrameToFile(FILE *file, AVPacket *pkt) {

	size_t ret = fwrite(pkt->data, 1, pkt->size, file);
	cg::core::infoRecorder->logError("[X264Encoder]: write %d data to file.\n", ret);
	return 0;
}

FILE * outfile = NULL;

//the buffer contains the header data, include the frame index
void SubGameStream::playVideoPriv(unsigned char * buffer, int bufSize, struct timeval pts){
	AVPacket avpkt;
	int gotPicture = 0 , len = 0;
#ifndef ANDROID
	union SDL_Event evt;
#endif
	struct pooldata * data = NULL;
	AVPicture * dstFrame = NULL;
	AVFrame * decodedFrame = NULL;

#ifdef USE_TEMPLATE_FRAME_POOL
	TaggedFrame * taggedFrame = NULL;
	DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();
#endif

	av_init_packet(&avpkt);
	unsigned char frameIndex = *(buffer + 4);   // get the frame index
	unsigned char tag = *(buffer + 5);

	if(!displayTimer){
		displayTimer = new cg::core::PTimer();
	}

	unsigned char specialTag = frameIndex & 0xc0;
	if(specialTag){
		delayRecorder->startToDisplay();
		displayTimer->Start();
		infoRecorder->logError("[SubGameStream]: get special tag, frame index:%x, value tag:%d.\n", frameIndex, tag);
	}

	//cg::core::infoRecorder->logError("[SubGameStream]: playVideoPriv, size:%d, frame index:%d, frame tag:%d\n", bufSize, frameIndex, tag);

	//cg::core::infoRecorder->logError("[SubGameStream]: buffer [%X, %X], [%X, %X, %X, %X, %X, %X].\n", buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7]);

	*(buffer + 3) = 0;
	*(buffer + 4) = 0;
	*(buffer + 5) = 1;

	avpkt.size = bufSize - 2;
	avpkt.data = buffer + 2;

	while(avpkt.size > 0){
		// the vframe is the destination frame(decoded), the avpkt is the source frame)
		decodedFrame = this->videoDecoder->decodeVideo(&gotPicture, &len, &avpkt);
		if(!decodedFrame || len < 0){
			infoRecorder->logError("[SubGameStream]: decode failed, return NULL frome VideoDecoder::decodeVideo or get a len <0.\n");
			break;
		}
#if 1
		if(gotPicture){
			// remove the GameStreams part form subStream
			EnterCriticalSection(gameStreams->getSurfaceMutex());   // only one surface 
			if(gameStreams->getSwsctx() == NULL){
#ifdef USE_TEMPLATE_FRAME_POOL
				gameStreams->setWidth(decodedFrame->width);
				gameStreams->setHeight(decodedFrame->height);
				gameStreams->setFormat((PixelFormat)decodedFrame->format);
#else
				gameStreams->setWidth(videoDecoder->getVFrame()->width);
				gameStreams->setHeight(videoDecoder->getVFrame()->height);
				gameStreams->setFormat((PixelFormat)videoDecoder->getVFrame()->format);
#endif

				cg::core::infoRecorder->logError("[SubGameStream]: to create a overlay width:%d, height:%d.\n", gameStreams->getWidth(), gameStreams->getHeight());
				// create the overlay
				LeaveCriticalSection(gameStreams->getSurfaceMutex());

				bzero(&evt, sizeof(evt));
				evt.user.type = SDL_USEREVENT;
				evt.user.timestamp = time(0);
				evt.user.code = SDL_USEREVENT_CREATE_OVERLAY;
				evt.user.data1 = gameStreams;
				evt.user.data2 = (void *)getId();
				SDL_PushEvent(&evt);

				goto skip_frame;
			}
			LeaveCriticalSection(gameStreams->getSurfaceMutex());

			// lock game streams
			FrameKey key;
			key.frameIndex = frameIndex & 0x3F;
			key.tag = tag;

			// get the frame tag
			specialTag = frameIndex & 0xc0;

			taggedFrame = new TaggedFrame;
			taggedFrame->frame = decodedFrame;
			taggedFrame->tag = specialTag;
			taggedFrame->valueTag = tag;

#if 1

#if 0
			gameStreams->storeFrame(key, taggedFrame);
			gameStreams->playVideo();
#else
			gameStreams->formDisplayEvent(taggedFrame);
#endif
			gameStreams->countFrameRate();

#else
			
#endif
			// unlock game streams
		}
		else{
			infoRecorder->logError("[SubGameStreams]: got NO from decoding.\n");
		}
#endif
skip_frame:
		avpkt.size -= len;
		avpkt.data += len;
	}
}

//////////////// GameStreams ////////////////
GameStreams * GameStreams::streams = NULL;

GameStreams::~GameStreams(){
	// delete the critical section
	DeleteCriticalSection(&surfaceMutex);
}

GameStreams::GameStreams(){
	rtspConf = cg::RTSPConf::GetRTSPConf();
	pSection = &section;
	InitializeCriticalSection(pSection);
	totalStreams = 0;
	for(int i = 0; i< MAX_RTSP_STREAM; i++){
		subStreams[i] = NULL;
	}

	videoSessFmt = 0;
	audioSessFmg = 0;
	videoCodecName = NULL;
	audioCodecName = NULL;
	videoFraming = 0;
	audioFraming = 0;

	InitializeCriticalSection(&surfaceMutex);
	surface = NULL;
	renderer = NULL;
	overlay = NULL;

	swsctx = NULL;
	width = 0;
	height = 0;
	vFrame = NULL;

	pipe = NULL;

	InitializeCriticalSection(&audioMutex);
	audioOpened = false;

	currentFrameIndex = -1;     // the index is the index of render
	minFrameIndex = -1;
	maxFrameIndex = -1;

	nextFrameIndex = 0;
	currentFrameTag = -1;    // frame tag represent that the frame for the windows, each render will provide a frame with ceratin tag, when all render generate a frame with the tag, the tag will increase, indicate that all renders accomplished the render for round 'tag'
	minFrameTag = -1;
	maxFrameTag = -1;
	nextFrameTag = 0;

	displayInterval = 20;   // 20 ms
	//frameMatrix = new IndexedFrameMatrix();
	lowBound.frameIndex = 0;
	lowBound.tag = 0;

	running = false;
	newAddEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
	declineEvent  = CreateEvent(NULL, FALSE, FALSE, NULL);
	frameEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	mutex = CreateMutex(NULL, FALSE, NULL);
}

void GameStreams::countFrameRate(){
#ifdef COUNT_FRAME_RATE
	cf_frame ++;
	if(cf_tv0.tv_sec == 0){
		getTimeOfDay(&cf_tv0, NULL);
	}
	if(cf_frame == COUNT_FRAME_RATE){
		getTimeOfDay(&cf_tv1, NULL);
		cf_interval = tvdiff_us(&cf_tv1, &cf_tv0);
		rtsperror("# %u.%06u player frame rate: decoder  @ %.4f fps\n", cf_tv1.tv_sec, cf_tv1.tv_usec, 1000000.0 *cf_frame / cf_interval);
		cf_tv0 = cf_tv1;
		cf_frame = 0;
	}
#endif
}

// the init function for GameStreams
bool GameStreams::init(){

	InitializeCriticalSection(&audioMutex);
	//frameMatrix->init(IMAGE_SOURCE_CHANNEL_MAX , IMAGE_SOURCE_CHANNEL_MAX);
	currentFrameIndex = 0;
	currentFrameTag = 0;
	nextFrameTag = 1;

	return true;
}

bool GameStreams::addSubStream(SubGameStream * subStream){
	subStreams[totalStreams] = subStream;
	maxFrameIndex = totalStreams;
	totalStreams++;
	infoRecorder->logTrace("[GameStream]: add sub streams: total streams:%d, streams:%p.\n", totalStreams, this);
	// 
	return true;
}

// remove the sub stream via given url
// return value is the removed sub stream, but not deleted
SubGameStream * GameStreams::removeStream(char * url){
	SubGameStream * subStream = NULL;
	int found = 0;
	int i=0;
	for(i = 0; i< totalStreams; i++){
		subStream = this->subStreams[i];
		if(!strcmp(url,subStream->getURL())){
			found = 1;
			break;
		}
	}
	if(found){
		for(int j = i; j < totalStreams -1; j++){
			//
			subStreams[j] = subStreams[j+1];
		}
		subStreams[totalStreams - 1] = NULL;
		totalStreams--;

		maxFrameIndex = totalStreams - 1;
	}

	return subStream;
}
#ifdef USE_TEMPLATE_FRAME_POOL
TaggedFrame* GameStreams::getIndexedFrame(FrameKey & lowBound_){
#else
AVFrame * GameStreams::getIndexedFrame(FrameKey & lowBound_){
#endif
	infoRecorder->logTrace("[GameStreams]:getIndexedFrame(), low bound:(frame index:%d, tag:%d).\n", lowBound_.frameIndex, lowBound_.tag);
#ifdef USE_TEMPLATE_FRAME_POOL
	TaggedFrame * ret = NULL;
#else
	AVFrame * ret = NULL; 
#endif
	ret=  framePool.getFrame(lowBound_);
	if(ret == NULL){
		infoRecorder->logTrace("[GameStreams]: get NULL frame, low bound:(frame index:%d, tag:%d).\nUpdate the low bound to zero.\n", lowBound_.frameIndex, lowBound_.tag);
		
		if(lowBound_.tag >= 254){
			lowBound.frameIndex = 0;
			lowBound.tag = 0;
		}
	}else{
		//lowBound = lowBound_;
		infoRecorder->logTrace("[GameStreams]: update low bound to (index:%d, tag:%d).\n", lowBound_.frameIndex, lowBound_.tag);
	}
	return ret;
}

// play video, get the indexed frame in the frame matrix, compare the frame index and the tag, choose to show, if many of them, play it every 20 milliseconds, tag is auto increase
int GameStreams::playVideo(){
	infoRecorder->logTrace("[GameStreams]: play video.\n");
	bool exitDisplay = false;
#ifdef USE_TEMPLATE_FRAME_POOL
	TaggedFrame * frame = NULL;
#else
	AVFrame * frame = NULL;
#endif
	while(!exitDisplay){
		// check the frame index
		if(totalStreams <= 0){
			// no streams invalid
			infoRecorder->logError("[GameStreams]: no stream is valid, total streams: %d, streams:%p\n", totalStreams, this);
			return 0;
		}

		frame = getIndexedFrame(lowBound);
		if(frame){
			formDisplayEvent(frame);
			return 0;
			//Sleep(5);
		}else{
			cg::core::infoRecorder->logError("[GameStream]: get NO frame to show.\n");
			return 0;
		}
	}
	return 0;
}
// form the display event
#ifdef USE_TEMPLATE_FRAME_POOL
int GameStreams::formDisplayEvent(TaggedFrame* taggedFrame){
	AVFrame * frame = taggedFrame->frame;
#else
int GameStreams::formDisplayEvent(AVFrame * frame){
#endif
	cg::core::infoRecorder->logError("[GameStreams]: form display event.\n");
	AVPicture * dstFrame = NULL;
#ifndef ANDROID
	union SDL_Event evt;
#endif
	struct pooldata * data = NULL;

	EnterCriticalSection(&surfaceMutex);
	if(this->swsctx == NULL){
		cg::core::infoRecorder->logError("[GameStreams]: formDisplayEvent(), get NULL swsctx, ERROR.\n");
#ifdef ANDROID

#else
		LeaveCriticalSection(&surfaceMutex);
		bzero(&evt, sizeof(evt));
		evt.user.type = SDL_USEREVENT;
		evt.user.timestamp = time(0);
		evt.user.code = SDL_USEREVENT_CREATE_OVERLAY;
		evt.user.data1 = this;
		evt.user.data2 = (void *)0;
		SDL_PushEvent(&evt);
		//goto skip_frame;
		return 0;
#endif
	}
	LeaveCriticalSection(&surfaceMutex);
	// copy into pool
	data = pipe->allocate_data();
	dstFrame = (AVPicture *)data->ptr;

	sws_scale(swsctx, frame->data, frame->linesize,
		0,frame->height, dstFrame->data, dstFrame->linesize);
	avcodec_free_frame(&frame);
	pipe->store_data(data);

#ifdef ANDROID

#else
	bzero(&evt, sizeof(evt));
	evt.user.type = SDL_USEREVENT;
	evt.user.timestamp = time(0);
	evt.user.code = SDL_USEREVENT_RENDER_IMAGE;
	evt.user.data1=  this;
	unsigned char * ptr = (unsigned char *)&(evt.user.data2);

	*ptr = taggedFrame->tag;
	ptr++;
	*ptr = taggedFrame->valueTag;
	//evt.user.data2 = (void *)taggedFrame->tag;
	SDL_PushEvent(&evt);
#endif  // ANDROID

#ifdef USE_TEMPLATE_FRAME_POOL
	delete taggedFrame;
#endif

	return 1;
}

// each subStreams will call this function to store the recved frame, so lock first
// return value: 1 for alright to trigger frame event
//				 -1 for frame is dropped.
#ifdef USE_TEMPLATE_FRAME_POOL
int GameStreams::storeFrame(FrameKey key,  TaggedFrame * frame){
#else
int GameStreams::storeFrame(FrameKey key, AVFrame * frame){
#endif
	cg::core::infoRecorder->logTrace("[GaemStreams]: storeFrame(), frame index;%d,  tag:%d\n", key.frameIndex, key.tag);
	int ret = 0;
	lock();
	//unsigned short index = frame->tag % MAX_RTSP_STREAM; // get the tag in the matrix
	ret = framePool.addFrame(key, frame);
	cg::core::infoRecorder->logTrace("[GameStreams]: frame pool has:%d.\n", ret);
	unlock();

	SetEvent(frameEvent);
	return ret;
}
SubGameStream * GameStreams::getStream(int index){
	SubGameStream * stream = NULL;
	return subStreams[index];
}

SubGameStream * GameStreams::getStream(char * url){
	SubGameStream * stream = NULL;
	for(int i = 0; i< this->totalStreams; i++){
		if(!strncmp(url, subStreams[i]->getURL(), strlen(subStreams[i]->getURL()))){
			stream = subStreams[i];
			break;
		}
	}
	return stream;
}

// add the renders using the cmd (format: count+url+url...);
bool GameStreams::addRenders(char *cmd){
	bool ret = false;
	char url[50] = {0};

	int count = *(int *)cmd;
	char * p = cmd + sizeof(int) + 1;
	string s(cmd);
	char * urlStart = p;
	char * urlEnd =p;

	cg::core::infoRecorder->logError("[GameStreams]: add render, count:%d, urls:%s\n", count, urlStart);

	while(*urlEnd){
		if(*urlEnd == '+'){
			memset(url, 0, 50);
			strncpy(url, urlStart, urlEnd- urlStart);
			urlStart = urlEnd +1;
			urlEnd = urlStart;
			// craete new stream with given url
			SubGameStream * subStream = new SubGameStream(url);
			if(totalStreams >= MAX_RTSP_STREAM){
				cg::core::infoRecorder->logError("[GameStreams]: too many sub game streams.\n");
				return false;
			}else{
				cg::core::infoRecorder->logError("[GameStreams]: add a new render '%s'.\n", url);
				subStreams[totalStreams++] = subStream;
			}

		}
		urlEnd ++;
	}
	// set the event
	SetEvent(this->newAddEvent);
	currentFrameIndex = 0;
	minFrameIndex = 0;
	maxFrameIndex = totalStreams - 1;

	return true;
}
bool GameStreams::declineRenders(char * cmd){
	bool ret = false;
	char url[50] = {0};
	int count = *(int *)cmd;
	char * p = cmd + sizeof(int) + 1;
	char * urlStart = p;
	char * urlEnd = p;

	if(totalStreams){
		maxFrameIndex = totalStreams - 1;
		minFrameIndex = 0;
	}

	return ret;
}

// the function to create a overlay for the game stream
bool GameStreams::createOverlay(){

	cg::core::infoRecorder->logError("[GameStreams]: to create the overlay for GameStreams.\n");

	unsigned int rendererFlags = SDL_RENDERER_SOFTWARE | SDL_RENDERER_PRESENTVSYNC;
	struct pooldata * data = NULL;
	char windowTitle[64];

	cg::RTSPConf * rtspConf = cg::RTSPConf::GetRTSPConf();

	EnterCriticalSection(&surfaceMutex);
	if(surface != NULL){
		LeaveCriticalSection(&surfaceMutex);
		rtsperror("[GameStreams]:duplicated create window request - image come to fast?\n");
		cg::core::infoRecorder->logError("[GameStreams]: duplicated create window request - image come too fast?");
		return false;
	}

	LeaveCriticalSection(&surfaceMutex);
	//swsctx

	cg::core::infoRecorder->logError("[GameStreams]: to create sws_scale context, width:%d, height;%d.\n", width, height);
	if((swsctx  = sws_getContext(width,height, format, width, height, PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL)) == NULL){
		rtsperror("[GameStreams]: cannot create swsscale context.\n");
		cg::core::infoRecorder->logError("[GameStreams]: cannot create swsscale context, FATAL ERROR.\n");
		return false;
		//exit(-1);
	}
	//pipeline
	if(pipe != NULL){
		cg::core::infoRecorder->logError("[GameStreams]: pipe already exist. FATAL ERROR.\n");
		return false;
	}

	if((pipe = new pipeline(0)) == NULL){
		rtsperror("[GameStreams]: cannot create pipeline.\n");
		cg::core::infoRecorder->logError("[GameStreams]: cannot create pipeline, FATAL ERROR exit ?.\n");
		return false;
	}
	if((data = pipe->datapool_init(POOLSIZE, sizeof(AVPicture))) == NULL){
		rtsperror("[GameStreams]: cannot allocate the data pool.\n");
		cg::core::infoRecorder->logError("[GameStreams]: cannot allocate the data pool,FATAL ERROR, to exit?.\n");
		return false;
		exit(-1);
	}
	for(; data != NULL; data = data->next){
		bzero(data->ptr, sizeof(AVPicture));
		if (avpicture_alloc((AVPicture*)data->ptr, PIX_FMT_YUV420P, width, height) != 0) {
			rtsperror("[GameStreams]: per frame initialization failed.\n");
			cg::core::infoRecorder->logError("[GameStreams]: per frame initialization failed,  FATAL ERROR , to exit?.\n");
			return false;
			exit(-1);
		}
	}

	int wflag = 0; //SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
#ifdef	ANDROID
	wflag = SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS;
#else
	if (rtspConf->confReadBool("fullscreen", 0) != 0){
		wflag |= SDL_WINDOW_FULLSCREEN | SDL_WINDOW_BORDERLESS;
	}
#endif
	snprintf(windowTitle, sizeof(windowTitle), "Player Channel #%d", 0);
	surface = SDL_CreateWindow(windowTitle,
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		width, height, wflag);
	windowSizeX = width;
	windowSizeY = height;
	if (surface == NULL) {
		rtsperror("[GameStreams]: set video mode (create window) failed.\n");
		cg::core::infoRecorder->logError("[GameStreams]: set video mode (create window) failed, FATAL ERROR, to exit?.\n");
		return false;
		exit(-1);
	}

	// move mouse to center
#if 1	// only support SDL2
	//SDL_WarpMouseInWindow(surface, width / 2, height / 2);
#endif

#if 0
	relativeMouseMode = rtspConf->relativeMouse;
	if (relativeMouseMode != 0) {
		SDL_SetRelativeMouseMode(SDL_TRUE);
		showCursor = 0;
		//SDL_ShowCursor(0);
		cg::core::infoRecorder->logError("[GameStreams]: relative mouse mode enabled.\n");
	}
	SDL_ShowCursor(showCursor);
#endif

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
			cg::core::infoRecorder->logError("[ga-client]: renderer#%d - %s (%s%s%s%s)\n",
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
		rtsperror("[GameStreams]: create renderer failed.\n");
		cg::core::infoRecorder->logError("[GameStreams]: create renderer failed, FATAL ERROR, to exit?\n");
		return false;
		exit(-1);
	}
	else{
		cg::core::infoRecorder->logError("[GameStreams]: create render for window, render:%p.\n", renderer);     
	}
	overlay = SDL_CreateTexture(renderer,
		SDL_PIXELFORMAT_YV12,
		SDL_TEXTUREACCESS_STREAMING,//SDL_TEXTUREACCESS_STATIC ,//SDL_TEXTUREACCESS_STREAMING,
		width, height);
#endif
	if (overlay == NULL) {
		rtsperror("[GameStreams]: create overlay (textuer) failed.\n");
		cg::core::infoRecorder->logError("[GameStreams]: create overlay (textuer) failed. to exit?\n");
		return false;
		exit(-1);
	}
	//
	EnterCriticalSection(&surfaceMutex);
#if 1	// only support SDL2
	setWindowId(SDL_GetWindowID(surface));
#endif

	LeaveCriticalSection(&surfaceMutex);
	//
	rtsperror("[GameStreams]: window created successfully (%dx%d).\n", width, height);
	cg::core::infoRecorder->logError("[GameStreams]: window created successfully (%dx%d).\n", width, height);

	// initialize watchdog
	EnterCriticalSection(&watchdogMutex);
	getTimeOfDay(&watchdogTimer, NULL);
	LeaveCriticalSection(&watchdogMutex);
	//
	return true;
}





bool saveTextureToBMP(std::string filepath, SDL_Texture *tex){
	SDL_Surface * infoSurface = NULL;
	SDL_Surface * saveSurface = NULL;

	if(infoSurface == NULL){

	}
	return true;
}

bool saveScreenshotBMP(std::string filepath, SDL_Window* SDLWindow, SDL_Renderer* SDLRenderer) {
	cg::core::infoRecorder->logError("Save scurrent to bmp: SDL_window: %p, renderer: %p.\n", SDLWindow, SDLRenderer);
	SDL_Surface* saveSurface = NULL;
	SDL_Surface* infoSurface = NULL;
	infoSurface = SDL_GetWindowSurface(SDLWindow);
	if (infoSurface == NULL) {
		cg::core::infoRecorder->logError("Failed to create info surface from window in saveScreenshotBMP(string), SDL_GetError() - %d\n", SDL_GetError());
	} else {
		unsigned char * pixels = new (std::nothrow) unsigned char[infoSurface->w * infoSurface->h * infoSurface->format->BytesPerPixel];
		if (pixels == 0) {
			cg::core::infoRecorder->logError("Unable to allocate memory for screenshot pixel data buffer!\n");
			return false;
		} else {
			if (SDL_RenderReadPixels(SDLRenderer, &infoSurface->clip_rect, infoSurface->format->format, pixels, infoSurface->w * infoSurface->format->BytesPerPixel) != 0) {
				cg::core::infoRecorder->logError("Failed to read pixel data from SDL_Renderer object. SDL_GetError() - %d\n",SDL_GetError());
				pixels = NULL;
				return false;
			} else {
				saveSurface = SDL_CreateRGBSurfaceFrom(pixels, infoSurface->w, infoSurface->h, infoSurface->format->BitsPerPixel, infoSurface->w * infoSurface->format->BytesPerPixel, infoSurface->format->Rmask, infoSurface->format->Gmask, infoSurface->format->Bmask, infoSurface->format->Amask);
				if (saveSurface == NULL) {
					cg::core::infoRecorder->logError("Couldn't create SDL_Surface from renderer pixel data. SDL_GetError() - %d.\n", SDL_GetError());
					return false;
				}
				SDL_SaveBMP(saveSurface, filepath.c_str());
				SDL_FreeSurface(saveSurface);
				saveSurface = NULL;
			}
			delete[] pixels;
		}
		SDL_FreeSurface(infoSurface);
		infoSurface = NULL;
	}
	return true;
}

bool GameStreams::renderImage(unsigned char specialTag, unsigned char valueTag){
	struct pooldata *data;
	AVPicture *vframe;
	SDL_Rect rect;
	DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();

	cg::core::infoRecorder->logError("[GameStreams]: render image.\n");

	// get the frame data to display
	if ((data = pipe->load_data()) == NULL) {
		cg::core::infoRecorder->logError("[GameStreams]: get NULL pipeline for channel:%d.\n", 0);
		return false;
	}
	vframe = (AVPicture*)data->ptr;
#ifdef YUV_TEXTURE

	if(SDL_UpdateYUVTexture(overlay, NULL, vframe->data[0], vframe->linesize[0], vframe->data[1], vframe->linesize[1], vframe->data[2], vframe->linesize[2])){
		cg::core::infoRecorder->logError("[GameStreams]: SDL_UpdateYUVTexture failed with:%d, texture not valid.\n");
	} 
#else    // YUV_TEXTURE

#endif    /// YUV_TEXTURE

	pipe->release_data(data);
	rect.x = 0;
	rect.y = 0;
	rect.w = width;
	rect.h = height;
#if 1	// only support SDL2
	int t = 0;

	SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
	SDL_RenderClear(renderer);

	if((t = SDL_RenderCopy(renderer, overlay, NULL, NULL))){
		//if(SDL_SetRenderTarget(renderer, overlay)){
		cg::core::infoRecorder->logError("[GameStreams]: render copy filed with:%d.\n", t);
	}
	//SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
	//SDL_RenderClear(renderer);	
	SDL_SetRenderTarget(renderer, NULL);
	SDL_RenderPresent(renderer);

#endif

	if(specialTag){
		char fileName[1024] = {0};
		sprintf(fileName,"%s/%s-share-c-%d.bmp", name, name, valueTag);
		printf("[Image]: save image %s.\n", fileName);
		saveScreenshotBMP(fileName, surface, renderer);

#if 0
		extern bool responseTickStarted;
		extern BTimer * globalTimer;
		responseTickStarted = false;
		long interval = globalTimer->Stop();

		int displayTime = displayTimer->Stop();
		if(delayRecorder->isSigned()){
			delayRecorder->displayed();
		}
		cg::core::infoRecorder->logError("[Delay]: before + display = total -> %f %f %f\n", delayRecorder->getBeforeDisplay(), delayRecorder->getDisplayDelay(), delayRecorder->getTotalDelay());

		cg::core::infoRecorder->logTrace("[delay]: %f, display: %f.\n", interval * 1000.0 / globalTimer->getFreq(), displayTime * 1000.0 / displayTimer->getFreq());
#endif
	}

	image_rendered = 1;
	return true;
}

// the function to render the image for the gaem stream
bool GameStreams::renderImage(long long special){
	struct pooldata *data;
	AVPicture *vframe;
	SDL_Rect rect;
	DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();

	cg::core::infoRecorder->logTrace("[GameStreams]: render image.\n");

	// get the frame data to display
	if ((data = pipe->load_data()) == NULL) {
		cg::core::infoRecorder->logError("[GameStreams]: get NULL pipeline for channel:%d.\n", 0);
		return false;
	}
	vframe = (AVPicture*)data->ptr;
#ifdef YUV_TEXTURE

	if(SDL_UpdateYUVTexture(overlay, NULL, vframe->data[0], vframe->linesize[0], vframe->data[1], vframe->linesize[1], vframe->data[2], vframe->linesize[2])){
			cg::core::infoRecorder->logError("[GameStreams]: SDL_UpdateYUVTexture failed with:%d, texture not valid.\n");
	} 


#else    // YUV_TEXTURE
	// vframe data is OK
#ifdef SAVE_SURFACE
	// store the YUV data to file
	BYTE * p24 = new BYTE[width * height * 3];
	BYTE * yv12 = new BYTE[width * height * 3 / 2];
	bcopy(vframe->data[0], yv12, width * height);
	bcopy(vframe->data[1], yv12 + ((width * height * 5) >> 2), width * height / 4);
	bcopy(vframe->data[2], yv12 + width * height, width * height / 4);
	YV12ToBGR24_Native(yv12, p24, width, height);
	long size = 0;
	BYTE * bmp = ConvertRGBToBMPBuffer(p24, width, height,&size);
	static int index = 0;
	char name[100] = {0};
	sprintf(name, "%d-result.bmp", index++);
	SaveBMP(bmp, width, height, size, name);

	//delete[] p24;
	delete[] yv12;
	delete[] bmp;

#endif

	//
#if 1	// only support SDL2
	if (SDL_LockTexture(overlay, NULL, (void**)&pixels, &pitch) == 0) {
		bcopy(vframe->data[0], pixels, width * height); 
		bcopy(vframe->data[1], pixels + ((pitch * height * 5) >> 2), width * height / 4);
		bcopy(vframe->data[2], pixels + pitch * height, width * height / 4);
#ifdef SAVE_SURFACE
		if(pitch != width){
			cg::core::infoRecorder->logError("[GameStreams]: pitch != width.\n");
		}
		else{
			YV12ToBGR24_Native(pixels, p24, width, height);
			sprintf(name, "texture-%d-result.bmp", index);
			BYTE * p1 = ConvertRGBToBMPBuffer(p24, width, height, &size);
			SaveBMP(p1, width, height, size, name);
			delete[] p1;
			delete[] p24;

		}
#endif

		SDL_UnlockTexture(overlay);
	}
	else {
		rtsperror("[GameStreams]: lock texture failed - %s\n", SDL_GetError());
		cg::core::infoRecorder->logError("[GameStream]: lock texture failed - %s\n", SDL_GetError());
		return false;
	}
#endif
#endif    /// YUV_TEXTURE

	pipe->release_data(data);
	rect.x = 0;
	rect.y = 0;
	rect.w = width;
	rect.h = height;
#if 1	// only support SDL2
	int t = 0;

	SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
	SDL_RenderClear(renderer);

	if((t = SDL_RenderCopy(renderer, overlay, NULL, NULL))){
		//if(SDL_SetRenderTarget(renderer, overlay)){
		cg::core::infoRecorder->logError("[GameStreams]: render copy filed with:%d.\n", t);
	}
	//SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
	//SDL_RenderClear(renderer);	
	SDL_SetRenderTarget(renderer, NULL);
	SDL_RenderPresent(renderer);
#ifdef SAVE_SURFACE
	sprintf(name,"screen-%d-result.bmp", index);
	saveScreenshotBMP(name, surface, renderer);
#endif

#endif

	if(special){
		extern bool responseTickStarted;
		extern BTimer * globalTimer;
		responseTickStarted = false;
		long interval = globalTimer->Stop();

		int displayTime = displayTimer->Stop();
		if(delayRecorder->isSigned()){
			delayRecorder->displayed();
		}
		cg::core::infoRecorder->logError("[Delay]: before + display = total -> %f %f %f\n", delayRecorder->getBeforeDisplay(), delayRecorder->getDisplayDelay(), delayRecorder->getTotalDelay());

		cg::core::infoRecorder->logTrace("[delay]: %f, display: %f.\n", interval * 1000.0 / globalTimer->getFreq(), displayTime * 1000.0 / displayTimer->getFreq());
	}

	image_rendered = 1;
	return true;
}


//////////////// outRTSPClient //////////////
ourRTSPClient*
	ourRTSPClient::createNew(UsageEnvironment& env, char const* rtspURL,
	int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum) {
		cg::core::infoRecorder->logError("[ourRTSPClient]: construct with rtsp url '%s'.\n", rtspURL);
		return new ourRTSPClient(env, rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum);
}

ourRTSPClient::ourRTSPClient(UsageEnvironment& env, char const* rtspURL,
	int verbosityLevel, char const* applicationName, portNumBits tunnelOverHTTPPortNum)
	: RTSPClient(env, rtspURL, verbosityLevel, applicationName, tunnelOverHTTPPortNum, -1) {
}

ourRTSPClient::~ourRTSPClient() {
}

/////////// StreamClientState /////////////////
StreamClientState::StreamClientState()
	: iter(NULL), session(NULL), subsession(NULL), streamTimerTask(NULL), duration(0.0) {}

StreamClientState::~StreamClientState() {
	delete iter;
	if (session != NULL) {
		// We also need to delete "session", and unschedule "streamTimerTask" (if set)
		UsageEnvironment& env = session->envir(); // alias

		env.taskScheduler().unscheduleDelayedTask(streamTimerTask);
		Medium::close(session);
	}
}

///////////////// DummySink ///////////////
DummySink*
	DummySink::createNew(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId) {
		return new DummySink(env, subsession, streamId);
}

DummySink *
	DummySink::createNew(UsageEnvironment & env, MediaSubsession & subsession,SubGameStream * subStream,char const * streamId /* = NULL */){
		cg::core::infoRecorder->logError("[DummySink]: create DunmmySink with SubGameStream( %p), stream ID:%s.\n", subStream, streamId ? streamId: "NULL");
		DummySink * ret = new DummySink(env, subsession, streamId);
		ret->subStream = subStream;
		return ret;
}

DummySink::DummySink(UsageEnvironment& env, MediaSubsession& subsession, char const* streamId)
	: MediaSink(env), fSubsession(subsession) {
		fStreamId = strDup(streamId);
		fReceiveBuffer = new u_int8_t[MAX_FRAMING_SIZE + DUMMY_SINK_RECEIVE_BUFFER_SIZE];
		// setup framing if necessary
		// H264 framing code
		if (strcmp("H264", fSubsession.codecName()) == 0) {
			videoFraming = 4;

			fReceiveBuffer[MAX_FRAMING_SIZE - videoFraming + 0]
			= fReceiveBuffer[MAX_FRAMING_SIZE - videoFraming + 1]
			= fReceiveBuffer[MAX_FRAMING_SIZE - videoFraming + 2] = 0;
			fReceiveBuffer[MAX_FRAMING_SIZE - videoFraming + 3] = 1;
		}
		return;
}

DummySink::~DummySink() {
	delete[] fReceiveBuffer;
	delete[] fStreamId;
}

void
	DummySink::afterGettingFrame(void* clientData, unsigned frameSize, unsigned numTruncatedBytes, struct timeval presentationTime, unsigned durationInMicroseconds) {
		DummySink* sink = (DummySink*)clientData;
		sink->afterGettingFrame(frameSize, numTruncatedBytes, presentationTime, durationInMicroseconds);
}

void
	DummySink::afterGettingFrame(unsigned frameSize, unsigned numTruncatedBytes,
struct timeval presentationTime, unsigned /*durationInMicroseconds*/) {
#ifndef ANDROID
	extern CRITICAL_SECTION watchdogMutex;
	extern struct timeval watchdogTimer;
#endif
	if (fSubsession.rtpPayloadFormat() == this->subStream->getVideoDecoder()->getVideoSessFmt()) {
		bool marker = false;

		RTPSource *rtpsrc = fSubsession.rtpSource();

		if (rtpsrc != NULL) {
			marker = rtpsrc->curPacketMarkerBit();
		}

		//cg::core::infoRecorder->logError("[DummySink]: after getting frame, get a video frame. frame size:%d\n", frameSize);
		// form the frame
		this->subStream->playVideo(
			fReceiveBuffer + MAX_FRAMING_SIZE - videoFraming,
			frameSize + videoFraming, presentationTime,
			marker);

	}
	else if (fSubsession.rtpPayloadFormat() == gameStreams->getAudioDecoder()->getAudioSessFmg()) {
		gameStreams->getAudioDecoder()->playAudio(fReceiveBuffer + MAX_FRAMING_SIZE - audioFraming,
			frameSize + audioFraming, presentationTime);
	}
#ifndef ANDROID // watchdog is implemented at the Java side
	EnterCriticalSection(&watchdogMutex);
	//pthread_mutex_lock(&watchdogMutex);
	getTimeOfDay(&watchdogTimer, NULL);
	//pthread_mutex_unlock(&watchdogMutex);
	LeaveCriticalSection(&watchdogMutex);
#endif
dropped:
	// Then continue, to request the next frame of data:
	continuePlaying();
}

Boolean
	DummySink::continuePlaying() {
		if (fSource == NULL) return False; // sanity check (should not happen)
		//cg::core::infoRecorder->logError("[DummySink]: getNextFrame.\n");
		// Request the next frame of data from our input source.  "afterGettingFrame()" will get called later, when it arrives:
		fSource->getNextFrame(fReceiveBuffer + MAX_FRAMING_SIZE,
			DUMMY_SINK_RECEIVE_BUFFER_SIZE,
			afterGettingFrame, this,
			onSourceClosure, this);
		return True;
}



//////////////////// game decoder /////////////////////////

int VideoDecoder::init(const char * sprop){
	AVCodec * codec = NULL;
	AVCodecContext * ctx = NULL;
	AVFrame * frame = NULL;
	const char ** names = NULL;

	cg::RTSPConf * rtspConf = cg::RTSPConf::GetRTSPConf();
	cg::core::infoRecorder->logError("[VideoDecoder]: initVDecoder.\n");
	if(this->videoCodecName == NULL){
		rtsperror("[VideoDecoder]: video decoder no codec specified.\n");
		cg::core::infoRecorder->logError("[VideoDecoder]: video decoder no codec specified.\n");
		return -1;
	}
	if((names = ccgClient::LookupDecoders(videoCodecName)) == NULL){
		rtsperror("[VideoDecoder]: video decoder cannot find decoder names for %s\n", videoCodecName);
		cg::core::infoRecorder->logError("[VideoDecoder]: video decoder cannot find decoder names for %s\n", videoCodecName);
		return -1;
	}
	videoCodecId = ccgClient::LookupCodecID(videoCodecName);
	if((codec = AVCodecCommon::AVCodecFindDecoder(names, AV_CODEC_ID_NONE)) == NULL){
		rtsperror("[GameDecoder]: video decoder cannnot find the decoder for %s\n", videoCodecName);
		cg::core::infoRecorder->logError("[GameDecoder]: video decoder cannnot find the decoder for %s\n", videoCodecName);
		return -1;
	}
	rtspConf->video_decoder_codec = codec;   // write to the rtsp config?
	rtsperror("[VideoDecoder]: video decoder use decoder %s\n", names[0]);
	cg::core::infoRecorder->logError("[VideoDecoder]: video decoder use decoder %s\n", names[0]);


	if((ctx = avcodec_alloc_context3(codec)) == NULL){
		rtsperror("[VideoDecoder]: video decoder, cannot allocate context\n");
		cg::core::infoRecorder->logError("[VideoDecoder]: video decoder, cannot allocate context\n");
		return -1;
	}
	if(codec->capabilities & CODEC_CAP_TRUNCATED){
		rtsperror("[VideoDecoder]: video decoder: codec support truncated data\n" );
		cg::core::infoRecorder->logError("[VideoDecoder]: video decoder: codec support truncated data\n");
		ctx->flags |= CODEC_FLAG_TRUNCATED;
	}
	if(sprop != NULL){
		unsigned char * extra = (unsigned char *)_strdup(sprop);
		int extrasize = strlen(sprop);
		extrasize = av_base64_decode(extra, sprop, extrasize);
		if(extrasize > 0){
			ctx->extradata = extra;
			ctx->extradata_size = extrasize;
			rtsperror("[VideoDecoder]: video decoder: sprop configured with '%s', decoded-size=%d\n",  sprop, extrasize);
			cg::core::infoRecorder->logError("[VideoDecoder]: video decoder: sprop configured with '%s', decoded-size=%d\n", sprop, extrasize);
			fprintf(stderr, "SPROP = [");
			for(unsigned char *ptr =  extra; extrasize > 0; extrasize --){
				fprintf(stderr, "%02x", *ptr++);
			}
			fprintf(stderr, " ]\n");
		}
	}
	if(avcodec_open2(ctx, codec, NULL)!=0){
		rtsperror("[VideoDecoder]: video decoder cannot open decoder\n");
		return -1;
	}
	rtsperror("[VideoDecoder]: video decoder codec %s (%s)\n", codec->name, codec->long_name);
	cg::core::infoRecorder->logError("[VideoDecoder]: video decoder codec %s (%s)\n", codec->name, codec->long_name);

	vDecoder = ctx;
	//vFrame = frame;


	inited = true;

	return 0;
}
AVFrame * VideoDecoder::decodeVideo(int * got_picture, int * step_len, AVPacket * pkt){
	AVFrame * ret = NULL;
	if((ret = avcodec_alloc_frame()) == NULL){
		rtsperror("[VideoDecoder]: video decoder: allocate frame failed.]n" );
		cg::core::infoRecorder->logError("[VideoDecoder]: video decoder: allocate frame failed.]n");
		return NULL;
	}
	*step_len = avcodec_decode_video2(vDecoder, ret, got_picture, pkt);
	return ret;
}


int AudioDecoder::initADecoder(){
	AVCodec * codec = NULL; 
	AVCodecContext * ctx = NULL;
	const char **names = NULL;

	if(audioCodecName == NULL){
		rtsperror("[GameDecoder]: no codec specified.\n");
		return -1;
	}
	if((names = ccgClient::LookupDecoders(audioCodecName)) == NULL){
		rtsperror("[GameDecoder]: audio decoder cannot find decoder names for %s\n", audioCodecName);
		return -1;
	}
	audioCodecId = ccgClient::LookupCodecID(audioCodecName);
	if((codec = AVCodecCommon::AVCodecFindDecoder(names, AV_CODEC_ID_NONE)) == NULL){
		rtsperror("[GameDecoder]: audio decoder cannot find the decoder for %s.\n", audioCodecName);
		return -1;
	}
	rtspConf->audio_decoder_codec = codec;
	rtsperror("[GameDecoder]: audio decoder use decoder %s\n", names[0]);

#ifdef ANDROID
	if(rtspConf->builtinAudioDecoder == 0){
#endif
		audioq->packetQueueInit();
#ifdef ANDROID
	}
#endif
	if((aFrame = avcodec_alloc_frame()) == NULL){
		rtsperror("[GameDecoder]: audio decoder allocate frame failed.\n");
		return -1;
	}
	if((ctx = avcodec_alloc_context3(codec)) == NULL){
		rtsperror("[GameDecoder]: audio decoder cannot allocate context.\n");
		return -1;
	}
	if(avcodec_open2(ctx, codec, NULL) != 0){
		rtsperror("[GameDecoder]: audio decoder cannot open decoder.\n");
		return -1;
	}
	rtsperror("[GameDecoder]: audio decoder codec %s (%s)\n", codec->name, codec->long_name);
	aDecoder = ctx;
	return 0;
}

unsigned char * AudioDecoder::audioBufferInit(){
	if(audioBuf == NULL){
		audioBuf = (unsigned char *)malloc(abmaxsize);
		if(audioBuf == NULL){
			return NULL;
		}
	}
	return audioBuf;
}

void AudioDecoder::playAudio(unsigned char * buffer, int bufSize, struct timeval pts){
	cg::core::infoRecorder->logError("[AudioDecoder]: play audio is NULL\n");
}

int AudioDecoder::audioBufferDecode(AVPacket * pkt, unsigned char *dstBuf, int dstLen){
	const unsigned char * srcplanes[SWR_CH_MAX];
	unsigned char * dstplanes[SWR_CH_MAX];
	unsigned char * saveptr = NULL;
	int filled = 0;

	saveptr = pkt->data;
	while(pkt->size > 0){
		int len, gotFrame = 0;
		unsigned char * srcbuf = NULL;
		int dataLen  = 0;

		avcodec_get_frame_defaults(aFrame);
		if((len = avcodec_decode_audio4(aDecoder, aFrame, &gotFrame, pkt)) < 0){
			rtsperror("[GameDecoder]: decode audio failed.\n");
			return -1;
		}
		if(gotFrame == 0){
			pkt->size -= len;
			pkt->data += len;
			continue;
		}

		if(aFrame->format == rtspConf->audio_device_format){
			dataLen = av_samples_get_buffer_size(NULL, aFrame->channels, aFrame->nb_samples, (AVSampleFormat)aFrame->format, 1);
			srcbuf = aFrame->data[0];
		}
		else{
			if(swrctx == NULL){
				if((swrctx = swr_alloc_set_opts(NULL, rtspConf->audio_device_channel_layout, rtspConf->audio_device_format, rtspConf->audio_samplerate, aFrame->channel_layout, (AVSampleFormat)aFrame->format, aFrame->sample_rate, 0, NULL))== NULL){
					rtsperror("[GameDecoder]: audio decoder cannot allocate swrctx.\n");
					return -1;
				}
				if(swr_init(swrctx) < 0){
					rtsperror("[GameDecoder]: audio decoder cannot initialize swrctx.\n");
					return -1;
				}
				maxDecoderSize = av_samples_get_buffer_size(NULL, rtspConf->audio_channels, rtspConf->audio_samplerate * 2, rtspConf->audio_device_format, 1);
				if((convbuf = (unsigned char *)malloc(maxDecoderSize)) == NULL){
					rtsperror("[GameDecoder]: audio decoder cannot allocate conversion buffer.\n");
					return -1;
				}
				rtsperror("[GameDecoder]: audio decoder on-the-fly audio foramt conversion enabled.\n");
				rtsperror("audio decoder: convert from %dch(%x)@%dHz (%s) to %dch(%x)@%dHz (%s).\n",
					(int)aFrame->channels, (int)aFrame->channel_layout, (int)aFrame->sample_rate,
					av_get_sample_fmt_name((AVSampleFormat)aFrame->format),
					(int)rtspConf->audio_channels,
					(int)rtspConf->audio_device_channel_layout,
					(int)rtspConf->audio_samplerate,
					av_get_sample_fmt_name(rtspConf->audio_device_format));
			}
			dataLen = av_samples_get_buffer_size(NULL, rtspConf->audio_channels, aFrame->nb_samples, rtspConf->audio_device_format, 1);

			if(dataLen > maxDecoderSize){
				rtsperror("[GameDecoder]: conversion input too lengthy (%d > %d)\n", dataLen, maxDecoderSize);
				return -1;

			}

			srcplanes[0] = aFrame->data[0];
			if(av_sample_fmt_is_planar((AVSampleFormat)aFrame->format) != 0){
				int i;

				for(i = 1; i< aFrame->channels; i++){
					srcplanes[i] = aFrame->data[i];
				}
				srcplanes[i] = NULL;
			}
			else{
				srcplanes[1] = NULL;
			}
			dstplanes[0]= convbuf;
			dstplanes[1] = NULL;

			swr_convert(swrctx, dstplanes, aFrame->nb_samples, srcplanes, aFrame->nb_samples);
			srcbuf = convbuf;
		}
		if(dataLen > dstLen){
			rtsperror("[GameDecoder]: decoded audio truncated.\n");
			dataLen = dstLen;
		}
		bcopy(srcbuf, dstBuf, dataLen);
		dstBuf+= dataLen;
		dstLen -= dataLen;
		filled += dataLen;

		pkt->size -= len;
		pkt->data += len;
	}
	pkt->data=  saveptr;
	if(pkt->data){
		av_free_packet(pkt);
		return filled;
	}
	return filled;
}

int AudioDecoder::audioBufferFillSDL(void * userdata, unsigned char * stream, int ssize){
	int filled;
	if((filled = audioBufferFill(userdata, stream, ssize)) < 0){
		rtsperror("[GameDecoder]: audio buffer fill failed.\n");
		exit(-1);
	}
	if(image_rendered == 0){
		bzero(stream, ssize);
		return -1;
	}
	bzero(stream + filled, ssize - filled);
	return 0;
}

int AudioDecoder::audioBufferFill(void * userdata, unsigned char * stream, int ssize){
	int filled = 0;
	AVPacket avpkt;
#ifdef ANDROID

#else
	AVCodecContext * aDecoder = (AVCodecContext *)userdata;
#endif
	if(audioBufferInit() == NULL){
		rtsperror("[GameDecoder]: audio decoder cannot allocate audio buffer\n");
#ifdef ANDROID

#else
		exit(-1);
#endif
	}
	while(filled < ssize){
		int dsize = 0, delta = 0;;
		// buffer has enough data
		if (absize - abpos >= ssize - filled) {
			delta = ssize - filled;
			bcopy(audioBuf + abpos, stream, delta);
			abpos += delta;
			filled += delta;
			return ssize;
		}
		else if (absize - abpos > 0) {
			delta = absize - abpos;
			bcopy(audioBuf + abpos, stream, delta);
			stream += delta;
			filled += delta;
			abpos = absize = 0;
		}
		// move data to head, leave more ab buffers
		if (abpos != 0) {
			bcopy(audioBuf + abpos, audioBuf, absize - abpos);
			absize -= abpos;
			abpos = 0;
		}
		// decode more packets
		if (audioq->packetQueueGet(&avpkt, 0) <= 0)
			break;
		if ((dsize = audioBufferDecode(&avpkt, audioBuf + absize, abmaxsize - absize)) < 0)
			break;
		absize += dsize;

	}
	return filled;
}


// each of this thread responsible for a connection
DWORD WINAPI rtspThreadForSubsession(LPVOID param){
	RTSPClient * client = NULL;
	BasicTaskScheduler0 *bs = BasicTaskScheduler::createNew();
	TaskScheduler* scheduler = bs;
	UsageEnvironment* env = BasicUsageEnvironment::createNew(*scheduler);

	cg::core::infoRecorder->logError("[rtspThreadForSubsession]: bs:%p, env:%p.\n", bs, env);

	RTSPConf * rtspConf = RTSPConf::GetRTSPConf(); // get the rtsp config
	//GameStreams * rtspStreams = (GameStreams *)param;
	SubGameStream * subStream = (SubGameStream *)param;
	subStream->setVideoState(RTSP_VIDEOSTATE_NULL);

	if((!subStream->openUrl(env, subStream->getURL()))){
		// error
		rtsperror("[RtspThreadForSubsession]: connect to %s failed.\n", subStream->getURL());
		return -1;
	}
	else{
		client= subStream->getRTSPClient();
		// add the rtspclient to map
		//GameStreams::addMap(client, subStream);
	}

	cg::core::infoRecorder->logError("[RTSPThreadForSubSession]: enter the loop.\n");

	while(subStream->isQuitLive555()){
		bs->SingleStep(1000000);
	}

	SubGameStream::shutdownStream(client);
	rtsperror("[rtspThreadForSubsession]: terminated.\n");
	cg::core::infoRecorder->logError("[rtspThreadForSubsession]: terminated.\n");
#ifndef ANDROID
	exit(0);
#endif
	return 0;
}

// the procedure for handling the display, a gameStreams will wait for the subGameStream's event to load the recved frame, and then display it using a time driven model and a event driven
// if recved frame is the next frame to show, then ,display it immidiately, if not, wait for the right frame, but, 30ms later, diaplay the recved frame and corrent the frame inde
DWORD WINAPI RTSPDisplayThread(LPVOID param){
	HANDLE frameEvent[MAX_RTSP_STREAM + 3]= { NULL };   // the event array
	DWORD threadIds[MAX_RTSP_STREAM] = { 0 };
	HANDLE threadHandle[MAX_RTSP_STREAM] = { NULL };
	int eventCount = 0;

	GameStreams *streams = (GameStreams *)((void **)param)[0];

	// connect to the temp distributor, get the substream count, construct the sub streams
	// get the temp distributor url from rtspConf
	SOCKET disSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(disSocket == INVALID_SOCKET){
		// error socket

		return 0;
	}
	// fill the remote addr information
	sockaddr_in servAddr;
	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(rtspConf->disPort);
	servAddr.sin_addr.S_un.S_addr = inet_addr(rtspConf->disServerName);

	// connect to dis
	if(connect(disSocket, (sockaddr *)&servAddr, sizeof(servAddr)) == -1){
		// connect failed
		cg::core::infoRecorder->logError("[RTSPDisplay]: connect to dis '%s', port:%d failed.\n", rtspConf->disServerName, rtspConf->disPort);
		return 0;
	}

	//recv the urls

	char addCmd[1024] = {0};
	char * p = addCmd;
	int * pCount = (int *)p;
#if 1     // add three render
	// form the cmd
	*pCount = 3;
	p += (sizeof(int));

	*p = '+';
	p++;
	for(int i = 0; i< 3; i++){
		sprintf(p, "%s", rtspConf->disServerName);
		p+= strlen(rtspConf->disServerName);
		*p= '+';
		p++;
	}
#else     // add one render
	*pCount = 1;
	p += (sizeof(int));

	*p = '+';
	p++;
	sprintf(p, "%s+", rtspConf->servername);

#endif
	p = addCmd + sizeof(int) + 1;
	cg::core::infoRecorder->logError("[RTSPPlayThread]: form the add server cmd: '%s'.\n", p);

	// create three add render event
	SDL_Event evt;
	bzero(&evt, sizeof(evt));
	evt.user.type = SDL_USEREVENT;
	evt.user.timestamp = time(0);
	evt.user.code = SDL_USEREVENT_ADD_RENDER;
	evt.user.data1 = streams;
	evt.user.data2 = (void *)addCmd;     /// the data2 is the cmd(format: count+url+url...)
	SDL_PushEvent(&evt);

	frameEvent[0] = streams->getNewAddEvent();
	frameEvent[1] = streams->getDeclineEvent();

	
	cg::core::infoRecorder->logError("[RTSPDisplayThread]: total game streams:%d.\n", streams->getTotalStreams());
	while(streams->checkRunning()){
		//Sleep(100);
		// wait all event, add, decline, dis server event.
		DWORD dw = WaitForMultipleObjects(eventCount + 2, frameEvent, FALSE, 30);
		switch(dw){
		case WAIT_FAILED:
			// bad call to function ( invalid handle?)
			cg::core::infoRecorder->logError("[RTSPDisplayThreadd]: bad call to WaitForMultipleObjects, invalid handle?\n");
			cg::core::infoRecorder->logError("[RTSPPlayThread]: bad call to WaitForMultipleObjects, invalid handle?\n");
			break;
		case WAIT_TIMEOUT:
			// none of the objects became singled within 20 milliseconds, to display the frame in the pipeline
			streams->playVideo();
			break;
		case WAIT_OBJECT_0 + 0: // new add event
			// new renders have connected
			// start all the substreams thread proc
			cg::core::infoRecorder->logError("[RTSPDisplayThread]: total streams to create:%d.\n", streams->getTotalStreams());
			for(int i = 0; i< streams->getTotalStreams(); i++){
				if(threadHandle[i] == NULL){
					threadHandle[i] = chBEGINTHREADEX(NULL, 0, rtspThreadForSubsession, streams->getStream(i), NULL, &threadIds[i]);
					if(threadHandle[i] == NULL){
						cg::core::infoRecorder->logError("[RTSPDisplayThread]: create thread for sub stream %d failed.\n", i);
						cg::core::infoRecorder->logError("[RTSPDisplayThread]: create thread for sub stream %d failed.\n", i);
						return -1;
					}
					else{
						streams->getStream(i)->setThread(threadHandle[i]);
					}
				}
				else{
					// the sub session is already exist
					cg::core::infoRecorder->logError("[RTSPDisplayThread]: the sub session already exist.\n");
				}
			}
			eventCount += streams->getTotalStreams(); // all the streams

			for(int i= 0; i< eventCount; i++){
				frameEvent[i + 2] = streams->getStream(i)->getNotifier();
			}
			break;
		case WAIT_OBJECT_0 + 1: // decline event
			// renders exit

			break;
		case WAIT_OBJECT_0 + 2: // the disclient url event
			// form the add render event ?

			break;
		default:
			// one of the object singled, a frame has arrived
			//streams->playVideo();
			break;
		}
	}
	return 0;
}