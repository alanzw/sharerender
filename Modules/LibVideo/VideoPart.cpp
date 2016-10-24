#include "Config.h"
#include "Commonwin32.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <map>
#include "RtspConf.h"
#include "RtspContext.h"
#include "Pipeline.h"
#include "Encoder.h"

#include "FilterRGB2YUV.h"
#include "VSource.h"

#include "ChannelManager.h"
#include "EncoderManager.h"

#include "CThread.h"
#include "VideoPart.h"

#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"

const char * imagepipefmt = "image-%d";
static const char * imagepipe0 = "image-0";

const char * surfacepipefmt = "surface-%d";
const char * surfacepipe0 = "surface-0";

const char * filterpipefmt = "filter-%d";
static const char * filterpipe0 = "filter-0";

static const char * encoderpipefmt = "encoder-%d";
static const char * encoderpipe0 = "encoder-0";

HANDLE videoInitMutex = NULL;

/// globals 
GlobalManager * GlobalManager::m_manager = NULL;
GlobalManager * globalManager = NULL;

bool encoderRunning = true;
map<DWORD, SOCKET> threadSocketMap;
SOCKET clientSocket = 0;

GlobalManager::GlobalManager(){
	encoderManager = NULL;
	channelManager = NULL;
	filter = NULL;
	rtspConf = NULL;
}
GlobalManager::~GlobalManager(){
	if (encoderManager){
		EncoderManager::ReleaseEncoderManager(encoderManager);
		encoderManager = NULL;
	}
	if (channelManager){
		//ChannelManager::ReleaseChannelManager();
		channelManager = NULL;
	}
	if (rtspConf){
		delete rtspConf;
		rtspConf = NULL;
	}
}

NetParam::NetParam(SOCKET s){
	int remoteLen = sizeof(SOCKADDR_IN);
	this->sock = s;
	getpeername(sock, (SOCKADDR *)&this->remoteAddr, &remoteLen);
	this->remotePort = DWORD(ntohs(remoteAddr.sin_port));
}

static int Init(char * config, NetParam * net){
	srand((int)time(0));
	netStarted = true;
	if (!netStarted){
		WSADATA wd;
		if (WSAStartup(MAKEWORD(2, 2), &wd) != 0)
			return -1;
		else
			netStarted = true;
	}
	av_register_all();
	avcodec_register_all();
	avformat_network_init();

	globalManager = GlobalManager::GetGlobalManager();
	// create the channel manager
	globalManager->init(config);

	// deal with the client addr and port
	// accept the request from logic server
	return 0;
}

////// the rtcpheader definition

struct RTCPHeader {
	unsigned char vps;  // version, padding, RC/SC
#define RTCP_Version(hdr)   (((hdr)->vps) >> 6)
#define RTCP_Padding(hdr)   ((((hdr)->vps) >> 5) & 0x01)
#define RTCP_RC(hdr)        (((hdr)->vps) & 0x1f)
#define RTCP_SC(hdr)        RTCP_RC(hdr)
	unsigned char pt;
	unsigned short length;
}
#ifdef WIN32
;
#else
__attribute__((__packed__));
#endif


static void SkipSpaces(const char **pp) {
	const char *p;
	p = *pp;
	while (*p == ' ' || *p == '\t')
		p++;
	*pp = p;
}
static void GetWord(char *buf, int buf_size, const char **pp) {
	const char *p;
	char *q;

	p = *pp;
	SkipSpaces(&p);
	q = buf;
	while (!isspace(*p) && *p != '\0') {
		if ((q - buf) < buf_size - 1)
			*q++ = *p;
		p++;
	}
	if (buf_size > 0)
		*q = '\0';
	*pp = p;
}
static int handle_rtcp(RTSPContext * ctx, const char *buf, size_t buflen){
	return 0;
}

/// the rtspserver thread proc
static int channelCount = 0;
DWORD WINAPI RtspServerThreadProc(LPVOID arg){

	Channel * ch = (Channel *)arg;
	SOCKET s = ch->getClientSock();
	int sinlen = sizeof(struct sockaddr_in);

	const char *p;
	char buf[8192];
	char cmd[32], url[1024], protocol[32];
	int rlen;
	struct sockaddr_in sin;
	RTSPContext ctx;
	RTSPMessageHeader header1, *header = &header1;
	
	// get the RTSPcontext for the channel
	channelCount++;

	ch->setChannelId(channelCount);

	RTSPConf* rtspConf = ch->getRtspConf();   ///set the context config
	sinlen = sizeof(sin);
	infoRecorder->logError("[RtspServer]: set config:%p.\n", rtspConf);

	getpeername(s, (struct sockaddr*) &sin, &sinlen);
	bzero(&ctx, sizeof(ctx));
	ctx.setConf(rtspConf);
	if (ctx.clientInit(ch->gSources, ch->getMaxWidth(), ch->getMaxHeight()) < 0) {
		infoRecorder->logTrace("server initialization failed.\n");
		return NULL;
	}

	bcopy(&sin, &ctx.client, sizeof(ctx.client));
	ctx.state = SERVER_STATE_IDLE;
	// XXX: hasVideo is used to sync audio/video
	// This value is increased by 1 for each captured frame until it is gerater than zero
	// when this value is greater than zero, audio encoding then starts ...
	//ctx.hasVideo = -(rtspconf->video_fps>>1); // for slow encoders?
	ctx.hasVideo = 0;   // with 'zerolatency'

	/// create the write mutex
	ctx.rtspWriteMutex = CreateMutex(NULL, FALSE, NULL);
	infoRecorder->logTrace("[tid %ld] client connected from %s:%d\n", ccg_gettid(), inet_ntoa(sin.sin_addr), htons(sin.sin_port));
	ctx.fd = s;

	ChannelManager * chm = ChannelManager::GetChannelManager();
	chm->mapRtspContext(ch, &ctx);
	ch->getEncoder()->setRTSPContext(&ctx);  // register the rtsp context to encoder manager
	
	infoRecorder->logTrace("[rtspserver]: start filter thread.\n");
	ch->startFilter();

	do {
		int i, fdmax, active;
		fd_set rfds;
		struct timeval to;
		FD_ZERO(&rfds);
		FD_SET(ctx.fd, &rfds);
		fdmax = ctx.fd;
#ifdef HOLE_PUNCHING
		for (int i = 0; i < 2 * ctx.streamCount; i++){
			FD_SET(ctx.rtpSocket[i], &rfds);
			if (ctx.rtpSocket[i] > fdmax)
				fdmax = ctx.rtpSocket[i];
		}
#endif
		to.tv_sec = 0;
		to.tv_usec = 500000;
		if ((active = select(fdmax + 1, &rfds, NULL, NULL, &to)) < 0) {
			infoRecorder->logTrace("select() failed: %s\n", strerror(errno));
			goto quit;
		}
		if (active == 0) {
			// try again!
			continue;
		}
#ifdef HOLE_PUNCHING
		for (i = 0; i < 2 * ctx.streamCount; i++) {
			struct sockaddr_in xsin;
#ifdef WIN32
			int xsinlen = sizeof(xsin);
#else
			socklen_t xsinlen = sizeof(xsin);
#endif
			if (FD_ISSET(ctx.rtpSocket[i], &rfds) == 0)
				continue;
			recvfrom(ctx.rtpSocket[i], buf, sizeof(buf), 0,
				(struct sockaddr*) &xsin, &xsinlen);
			if (ctx.rtpPortChecked[i] != 0)
				continue;
			// XXX: port should not flip-flop, so check only once
			if (xsin.sin_addr.s_addr != ctx.client.sin_addr.s_addr) {
				infoRecorder->logError("RTP: client address mismatched? %u.%u.%u.%u != %u.%u.%u.%u\n",
					NIPQUAD(ctx.client.sin_addr.s_addr),
					NIPQUAD(xsin.sin_addr.s_addr));
				continue;
			}
			if (xsin.sin_port != ctx.rtpPeerPort[i]) {
				infoRecorder->logError("RTP: client port reconfigured: %u -> %u\n",
					(unsigned int)ntohs(ctx.rtpPeerPort[i]),
					(unsigned int)ntohs(xsin.sin_port));
				ctx.rtpPeerPort[i] = xsin.sin_port;
			}
			else {
				infoRecorder->logError("RTP: client is not under an NAT, port %d confirmed\n",
					(int)ntohs(ctx.rtpPeerPort[i]));
			}
			ctx.rtpPortChecked[i] = 1;
		}
		// is RTSP connection?
		if (FD_ISSET(ctx.fd, &rfds) == 0)
			continue;
#endif
		// read commands
		if ((rlen = ctx.rtspGetNext(buf, sizeof(buf))) < 0) {
			goto quit;
		}
		// Interleaved binary data?
		if (buf[0] == '$') {
			handle_rtcp(&ctx, buf, rlen);
			continue;
		}
		// REQUEST line
		infoRecorder->logTrace("%s", buf);
		p = buf;
		GetWord(cmd, sizeof(cmd), &p);
		GetWord(url, sizeof(url), &p);
		GetWord(protocol, sizeof(protocol), &p);
		// check protocol
		if (strcmp(protocol, "RTSP/1.0") != 0) {
			infoRecorder->logTrace("[rtsp server]: protocol not supported.\n");
			ctx.rtspReplyError(RTSP_STATUS_VERSION);
			goto quit;
		}
		// read headers
		bzero(header, sizeof(*header));
		do {
			int myseq = -1;
			char mysession[sizeof(header->session_id)] = "";
			if ((rlen = ctx.rtspGetNext(buf, sizeof(buf))) < 0)
				goto quit;
			if (buf[0] == '\n' || (buf[0] == '\r' && buf[1] == '\n'))
				break;
			// Special handling to CSeq & Session header
			// ff_rtsp_parse_line cannot handle CSeq & Session properly on Windows
			// any more?
			if (strncasecmp("CSeq: ", buf, 6) == 0) {
				myseq = strtol(buf + 6, NULL, 10);
			}
			if (strncasecmp("Session: ", buf, 9) == 0) {
				strcpy(mysession, buf + 9);
			}
			//
			ff_rtsp_parse_line(header, buf, NULL, NULL);
			//
			if (myseq > 0 && header->seq <= 0) {
				infoRecorder->logError("WARNING: CSeq fixes applied (%d->%d).\n",
					header->seq, myseq);
				header->seq = myseq;
			}
			if (mysession[0] != '\0' && header->session_id[0] == '\0') {
				unsigned i;
				for (i = 0; i < sizeof(header->session_id) - 1; i++) {
					if (mysession[i] == '\0' || isspace(mysession[i]) || mysession[i] == ';')
						break;
					header->session_id[i] = mysession[i];
				}
				header->session_id[i + 1] = '\0';
				infoRecorder->logError("WARNING: Session fixes applied (%s)\n",
					header->session_id);
			}
		} while (1);
		// special handle to session_id
		if (header->session_id != NULL) {
			char *p = header->session_id;
			while (*p != '\0') {
				if (*p == '\r' || *p == '\n') {
					*p = '\0';
					break;
				}
				p++;
			}
		}
		// handle commands
		ctx.seq = header->seq;
		if (!strcmp(cmd, "DESCRIBE"))
			ctx.rtspCmdDescribe(url);
		else if (!strcmp(cmd, "OPTIONS"))
			ctx.rtspCmdOptions(url);
		else if (!strcmp(cmd, "SETUP"))
			ctx.rtspCmdSetup(url, header, ch->getChannelId());
		else if (!strcmp(cmd, "PLAY"))
			ctx.rtspCmdPlay(url, header);
		else if (!strcmp(cmd, "PAUSE"))
			ctx.rtspCmdPause(url, header);
		else if (!strcmp(cmd, "TEARDOWN"))
			ctx.rtspCmdTeardown(url, header, 1);
		else
			ctx.rtspReplyError(RTSP_STATUS_METHOD);
		if (ctx.state == SERVER_STATE_TEARDOWN) {
			break;
		}
	} while (1);
quit:
	ctx.state = SERVER_STATE_TEARDOWN;
	//
	closesocket(ctx.fd);
#ifdef  SHARE_ENCODER
	EncoderManager * encoderManager = globalManager->encoderManager;
	encoderManager->encoderUnregisterClient(&ctx);
	// encoder_unregister_client(&ctx);
#else
	infoRecorder->logTrace("connection closed, checking for worker threads...\n");
	// wait the other thread to finish
	WaitForSingleObject(ctx.vThread, INFINITE);
	//pthread_join(ctx.vthread, (void**)&thread_ret);
#ifdef  ENABLE_AUDIO
	WaitForSingleObject(ctx.aThread, INFINITE);
	//pthread_join(ctx.athread, (void**)&thread_ret);
#endif  /* ENABLE_AUDIO */
#endif  /* SHARE_ENCODER */
	//
	ctx.clientDeinit();
	infoRecorder->logTrace("RTSP client thread terminated.\n");
	//
	return NULL;
}


void DestoryAll(){
	Channel::Release();
	RTSPConf::Release();
	//RTSPContext::Release();
	Filter::Release();
	pipeline::Release();
	ChannelManager::Release();
	EncoderManager::Release();
}

DWORD WINAPI VideoServer(LPVOID param){
	infoRecorder->logTrace("[VideoServer]: entner the video server.\n");

	Channel * ch = (Channel *)param;
	RTSPConf * rtspconf = NULL;
	int chId = 0;
	char imagepipename[64];
	char filterpipename[64];
	char surfacepipename[64];

	// create the mutex
	videoInitMutex = CreateMutex(NULL, FALSE, NULL);
	WaitForSingleObject(videoInitMutex, INFINITE);
	
	/* create the rtsp connection to client   */
	if (!netStarted){
		// start the network service
#if 0
		Init(RENDERPOOL_CONFIG, netParam);
#else
		Init(RENDERPOOL_CONFIG, NULL);
#endif
		netStarted = true;
	}

	rtspconf = globalManager->getGlobalConfig();

	// deal with request
	SOCKET rtspSock, rtspClientSock;
	struct sockaddr_in sin, csin;
	int csInLen = 0;
	if ((rtspSock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0){
		infoRecorder->logError("[video server]: ERROR: create socket failed!\n");
		return -1;
	}
	do{
		BOOL val = 1;
		setsockopt(rtspSock, SOL_SOCKET, SO_REUSEADDR, (const char *)&val, sizeof(val));
	} while (0);

	bzero(&sin, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_port = htons(globalManager->getGlobalConfig()->serverport);
	
	infoRecorder->logError("[video server]: bind port:%d.\n", globalManager->getGlobalConfig()->serverport);
	if (bind(rtspSock, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		infoRecorder->logError("[VIDEO SERVER] ERROR: bind error!\n");
		return -1;
	}

	infoRecorder->logError("[video server]: start to listen port.\n");
	if (listen(rtspSock, 256) < 0){
		infoRecorder->logError("[video server] ERROR: listen error\n");
		return -1;
	}
	ReleaseMutex(videoInitMutex);
	atexit(DestoryAll);

	// accept the client to play
	int request = 0;
	do{
		Log::logscreen("[video server]: wait for client...\n");
		infoRecorder->logError("[video server]: wait for client...\n");
		csInLen = sizeof(csin);
		bzero(&csin, sizeof(csin));
		if ((rtspClientSock = accept(rtspSock, (struct sockaddr *)&csin, &csInLen)) < 0){
			infoRecorder->logError("[VIDEO SERVER] ERROR: accept error!\n");
			return -1;
		}
		Log::logscreen("[video server]: accepted a rtsp client.\n");
		request++; // a new request has been accepted
		// tunning the sending window
		int sndWnd = 8388608; // 8MB
		if (setsockopt(rtspClientSock, SOL_SOCKET, SO_SNDBUF, &sndWnd, sizeof(sndWnd)) == 0){
			infoRecorder->logError("[VIDEO SERVER] INFO: set the tcp sending buffer success\n");
		}
		else{
			infoRecorder->logError("[VIDEO SERVER] ERROR: set the tcp sending buffer failed.\n");
		}

		// start the video thread
		DWORD threadId;
		sprintf(imagepipename, imagepipefmt, chId);
		sprintf(filterpipename, filterpipefmt, chId);
		sprintf(surfacepipename, surfacepipefmt, chId);

		infoRecorder->logTrace("[rtsp server]: image pipenmae:%s, surface pipename:%s, filterpipename:%s.\n", imagepipename, surfacepipename, filterpipename);
		// init the channel, create the encoder

		if (ch == NULL){
			infoRecorder->logTrace("[video part]: get NULL channel in video server.\n");
		}

		ch->setSrcPipeName(imagepipename);
		ch->setSurfacePipeName(surfacepipename);
		ch->setFilterPipeName(filterpipename);

		ENCODER_TYPE type = globalManager->encoderManager->getAvailableType();
		// TODO
		// to form a cuda encoder with image source
		type = ENCODER_TYPE::X264_ENCODER;

		Encoder * encoder = globalManager->encoderManager->getEncoder(type);
		Filter * filter = NULL;
		D3DWrapper * wrapper = NULL;
		SOURCE_TYPE sourceType = SOURCE_TYPE::SOURCE_NONE;

		ch->setEncoderType(type);
		//encoder->setRTSPContext(globalManager->);
		ch->setRtspConf(rtspconf);

		encoder->setSrcPipeName(filterpipename);  // set the encoder source pipe name

		ch->registerEncoder(encoder);
		if (type == ENCODER_TYPE::X264_ENCODER){
			sourceType = SOURCE_TYPE::IMAGE;
			// init the filter
			filter = new Filter();
			// register filter
			Filter::do_register(filterpipename, filter);
			//filter->init(ch->getChannelName(), "filter1"); // init the filter for the x264 encoder
		}
		else if (type == ENCODER_TYPE::CUDA_ENCODER){
			sourceType = SOURCE_TYPE::SURFACE;
		}
		else
		{
			sourceType = SOURCE_TYPE::IMAGE;
			type = ENCODER_TYPE::X264_ENCODER;
			// init the filter
			filter = new Filter();
			Filter::do_register(filterpipename, filter);
			//filter->init(ch->getChannelName(), "filter2");
		}

		ch->waitForDeviceAndWindowHandle();


		if (filter){
			ch->setFilter(filter);
		}
		
		// init the channel, specifically, init the source pipe
		ch->init(type, sourceType);
		ch->setClientSock(rtspClientSock);


		// start the rtsp server thread
		Log::logscreen("[video server]: start the rtsp server thread.\n");
		infoRecorder->logError("[video server]: start the rtsp server thread.\n");
		HANDLE thread = chBEGINTHREADEX(NULL, 0, RtspServerThreadProc, ch, 0, &threadId);
		chId++;
	} while (1);

	return 0;
}