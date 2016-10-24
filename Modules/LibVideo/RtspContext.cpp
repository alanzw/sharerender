#include "Commonwin32.h"
#include "Config.h"
#include "RtspConf.h"

#include "RtspContext.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"


#define RTSP_STREAM_FORMAT "streamid=%d"
#define RTSP_STREAM_FORMAT_MAXLEN 64

//RTSPConf * RTSPContext::rtspConf;
#if 0
AVStream * RTSPContext::sdpVStream[IMAGE_SOURCE_CHANNEL_MAX];
AVCodecContext * RTSPContext::sdpVEncoder[IMAGE_SOURCE_CHANNEL_MAX];
#endif

RTSPContext::~RTSPContext(){
	if (rBuffer){
		free(rBuffer);
		rBuffer = NULL;
	}
	if (rtspConf){
	
	}
	if (rtspWriteMutex){
		CloseHandle(rtspWriteMutex);
		rtspWriteMutex = NULL;
	}
}

// init the client for the rtsp
int RTSPContext::clientInit(int sources, int width, int height){
	int i;
	//this->id = id;
	if (this->initialized){
		return 0;
	}
	else{
		initialized = true;
	}
	this->height = height;
	this->width = width;
	AVOutputFormat * fmt = NULL;
	if ((fmt = av_guess_format("rtp", NULL, NULL)) == NULL){
		infoRecorder->logError("RTP not support.\n");
		return -1;
	}

	if ((sdpFmtCtx = avformat_alloc_context()) == NULL){
		infoRecorder->logError("create avformat context failed.\n");
		return -1;
	}
	sdpFmtCtx->oformat = fmt;
	// video stream, each context has a stream
	//Log::logscreen("[rtspcontext]: rtspconfig:%p.\n", rtspConf);

#ifndef MULTI_CHANNEL
	if ((sdpVStream[id] = AVCodecCommon::AVFormatNewStream(
		sdpFmtCtx, id, rtspConf->video_encoder_codec)) == NULL){
		infoRecorder->logError("# cannot create new video stream (%d:%d)\n",
			id, rtspConf->video_encoder_codec->id);
		return -1;
	}
	if ((sdpVEncoder[id] = AVCodecCommon::AVCodecVEncoderInit(sdpVStream[id]->codec,
		rtspConf->video_encoder_codec, width, height, rtspConf->video_fps, rtspConf->vso)) == NULL){

		infoRecorder->logError("# cannot init video encoder\n");
		return -1;
	}
#else
	infoRecorder->logTrace("[RTSPContext]: init %d channels.\n", sources);
	if (sources <= 0){
		sources = 1;
	}
	for (i = 0; i < sources; i++){
		if ((sdpVStream[i] = AVCodecCommon::AVFormatNewStream(
			sdpFmtCtx, i, rtspConf->video_encoder_codec)) == NULL){
			infoRecorder->logError("# cannot create new video stream (%d:%d)\n",
				i, rtspConf->video_encoder_codec->id);
			return -1;
		}
		if ((sdpVEncoder[i] = AVCodecCommon::AVCodecVEncoderInit(sdpVStream[i]->codec,
			rtspConf->video_encoder_codec, width, height, rtspConf->video_fps, rtspConf->vso)) == NULL){
			infoRecorder->logError("# cannot init video encoder\n");
			return -1;
		}
	}

#endif

	if ((mtu = rtspConf->confReadInt("packet-size")) <= 0){
		this->mtu = RTSP_TCP_MAX_PACKET_SIZE;
	}
	return 0;
}

void RTSPContext::rtspCleanUp(int retCode){
	state = SERVER_STATE_TEARDOWN;
	Sleep(1000);
	return;
}

int RTSPContext::rtspWrite(const void *buf, size_t count){
	return write(this->fd, buf, count);
}

int RTSPContext::rtspPrintf(const char * fmt, ...){
	va_list ap;
	char buf[1024];
	int buflen;

	va_start(ap, fmt);
	buflen = vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	return rtspWrite(buf, buflen);
}


#define MULTI_SOURCE

int RTSPContext::rtspWriteBinData(int streamId, uint8_t * buf, int bufLen, char frameIndex, unsigned char tag){
	int i, pktLen;
	int headerSize = 0;
	char header[5];
	infoRecorder->logTrace("[RTSPContext]: rtspWriteBinData. streamid:%d, size:%d.\n", streamId, bufLen);
	RTSPContext * p = NULL;

	if (bufLen < 4){
		return bufLen;
	}

	// xxx: buffer is the result from avio_open_dyn_buf,
	// multiple RTP pakcets can be placed in a single buffer.
	// Format == 4-bytes (big-endian) packet size + packet-data

	i = 0;
	while (i < bufLen){
		pktLen = (buf[i + 0] << 24);
		pktLen += (buf[i + 1] << 16);
		pktLen += (buf[i + 2] << 8);
		pktLen += (buf[i + 3]);
		if (pktLen == 0){
			i += 4;
			continue;
		}

		header[0] = frameIndex;
		header[1] = tag;
		header[2] = '$';
		header[3] = (streamId << 1) & 0x0ff;
		header[4] = pktLen >> 8;
		header[5] = pktLen & 0x0ff;
		//headerSize = 4;
		// add some code to send the frame index
		//header[5] = frameIndex;
		headerSize = 6;

		WaitForSingleObject(rtspWriteMutex, INFINITE);
		if (rtspWrite(header, headerSize) != headerSize){
			ReleaseMutex(rtspWriteMutex);
			return i;
		}
		if (rtspWrite(&buf[i + headerSize], pktLen) != pktLen){
			return i;
		}
		ReleaseMutex(rtspWriteMutex);
		i += (headerSize + pktLen);
	}
	return i;
}


int RTSPContext::rtspWriteBinData(int streamId, uint8_t * buf, int bufLen){
	int i, pktLen;
	int headerSize = 0;
	char header[5];
	infoRecorder->logTrace("[RTSPContext]: rtspWriteBinData. streamid:%d, size:%d.\n", streamId, bufLen);
	RTSPContext * p = NULL;

	if (bufLen < 4){
		return bufLen;
	}

	// xxx: buffer is the result from avio_open_dyn_buf,
	// multiple RTP pakcets can be placed in a single buffer.
	// Format == 4-bytes (big-endian) packet size + packet-data

	i = 0;
	while (i < bufLen){
		pktLen = (buf[i + 0] << 24);
		pktLen += (buf[i + 1] << 16);
		pktLen += (buf[i + 2] << 8);
		pktLen += (buf[i + 3]);
		if (pktLen == 0){
			i += 4;
			continue;
		}

		header[0] = '$';
		header[1] = (streamId << 1) & 0x0ff;
		header[2] = pktLen >> 8;
		header[3] = pktLen & 0x0ff;
		headerSize = 4;
		// add some code to send the frame index

		WaitForSingleObject(rtspWriteMutex, INFINITE);
		if (rtspWrite(header, headerSize) != headerSize){
			ReleaseMutex(rtspWriteMutex);
			return i;
		}
		if (rtspWrite(&buf[i + headerSize], pktLen) != pktLen){
			return i;
		}
		ReleaseMutex(rtspWriteMutex);
		i += (headerSize + pktLen);
	}
	return i;
}
#ifdef HOLE_PUNCHING
static int rtpOpenInternal(unsigned short * port){
	SOCKET s;
	struct sockaddr_in sin;
	int sinLen;
	bzero(&sin, sizeof(sin));
	if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
		return -1;
	sin.sin_family = AF_INET;
	if (bind(s, (struct sockaddr *)&sin, sizeof(sin)) < 0){
		closesocket(s);
		return -1;
	}
	sinLen = sizeof(sin);
	if (getsockname(s, (struct sockaddr*)&sin, &sinLen) < 0){
		closesocket(s);
		return -1;
	}
	*port = ntohs(sin.sin_port);
	return s;
}

int RTSPContext::rtpOpenPorts(int streamId){

	if (streamId < 0)
		return -1;
	if (streamId + 1 > this->streamCount){
		this->streamCount = streamId + 1;
	}

	streamId *= 2;
	// initialized

	if (rtpSocket[streamId] != 0)
		return 0;

	if ((rtpSocket[streamId] = rtpOpenInternal(&(rtpLocalPort[streamId]))) < 0)
		return -1;
	if ((rtpSocket[streamId + 1] = rtpOpenInternal(&(rtpLocalPort[streamId + 1]))) < 0){
		closesocket(rtpSocket[streamId]);
		return -1;
	}

	infoRecorder->logTrace("RTP: port opened for stream %d, min=%d (fd=%d), max=%d (fd=%d)\n",
		streamId / 2,
		(unsigned int)rtpLocalPort[streamId],
		(int)rtpSocket[streamId],
		(unsigned int)rtpLocalPort[streamId + 1],
		(int)rtpSocket[streamId + 1]);
	return 0;
}

int RTSPContext::rtpClosePorts(int streamId){
	if (rtpSocket[streamId] != 0)
		closesocket(rtpSocket[streamId]);
	if (rtpSocket[streamId + 1] != 0)
		closesocket(rtpSocket[streamId + 1]);

	rtpSocket[streamId] = 0;
	rtpSocket[streamId + 1] = 0;
	return 0;
}

#ifdef MULTI_CLIENTS
// send the rtp data with frame index
int RTSPContext::rtpWriteBinData(int streamId, uint8_t * buf, int bufLen, int frameIndex){
	int i, pktlen;
	struct sockaddr_in sin;
	infoRecorder->logTrace("[RTSPContext]: rtpWriteBinData. streamid:%d, size:%d.\n", streamId, bufLen);
	if (rtpSocket[streamId * 2] == 0){
		infoRecorder->logTrace("[RTSPContext]: rtp write data, NULL rtpsocket");
		return -1;
	}
	if (buf == NULL)
		return 0;
	if (bufLen < 4)
		return bufLen;
	bcopy(&client, &sin, sizeof(sin));
	infoRecorder->logTrace("[RTSPContext]: rtp write data client:%s\n", inet_ntoa(sin.sin_addr));
	sin.sin_port = rtpPeerPort[streamId * 2];
	// XXX: buffer is the reuslt from avio_open_dyn_buf.
	// Multiple RTP packets can be placed in a single buffer.
	// Format == 4-bytes (big-endian) packet size + packet-data
	i = 0;
	while (i < bufLen) {
		pktlen = (buf[i + 0] << 24);
		pktlen += (buf[i + 1] << 16);
		pktlen += (buf[i + 2] << 8);
		pktlen += (buf[i + 3]);
		if (pktlen == 0) {
			i += 4;
			continue;
		}
		infoRecorder->logTrace("[RTSPContext]: sendto... pktlen:%d\n", pktlen);
		sendto(rtpSocket[streamId * 2], (const char*)&buf[i + 4], pktlen, 0,
			(struct sockaddr*) &sin, sizeof(struct sockaddr_in));
		i += (4 + pktlen);
	}
	return i;
}
#endif
int RTSPContext::rtpWriteBinData(int streamId, uint8_t * buf, int bufLen){
	int i, pktlen;
	struct sockaddr_in sin;
	infoRecorder->logTrace("[RTSPContext]: rtpWriteBinData. streamid:%d, size:%d.\n", streamId, bufLen);
	if (rtpSocket[streamId * 2] == 0){
		infoRecorder->logTrace("[RTSPContext]: rtp write data, NULL rtpsocket");
		return -1;
	}
	if (buf == NULL)
		return 0;
	if (bufLen < 4)
		return bufLen;
	bcopy(&client, &sin, sizeof(sin));
	infoRecorder->logTrace("[RTSPContext]: rtp write data client:%s\n", inet_ntoa(sin.sin_addr));
	sin.sin_port = rtpPeerPort[streamId * 2];
	// XXX: buffer is the reuslt from avio_open_dyn_buf.
	// Multiple RTP packets can be placed in a single buffer.
	// Format == 4-bytes (big-endian) packet size + packet-data
	i = 0;
	while (i < bufLen) {
		pktlen = (buf[i + 0] << 24);
		pktlen += (buf[i + 1] << 16);
		pktlen += (buf[i + 2] << 8);
		pktlen += (buf[i + 3]);
		if (pktlen == 0) {
			i += 4;
			continue;
		}
		infoRecorder->logTrace("[RTSPContext]: sendto... pktlen:%d\n", pktlen);
		sendto(rtpSocket[streamId * 2], (const char*)&buf[i + 4], pktlen, 0,
			(struct sockaddr*) &sin, sizeof(struct sockaddr_in));
		i += (4 + pktlen);
	}
	return i;
}
#endif
int RTSPContext::rtspReadInternal(){
	int rlen;
	if ((rlen = read(fd,
		rBuffer + rBufTail,
		rBufSize - rBufTail)) <= 0) {
		return -1;
	}
	rBufTail += rlen;

	return rBufTail - rBufHead;
}

int RTSPContext::rtspReadText(char * buf, size_t count){
	int i;
	size_t textlen;
again:
	for (i = rBufHead; i < rBufTail; i++) {
		if (rBuffer[i] == '\n') {
			textlen = i - rBufHead + 1;
			if (textlen > count - 1) {
				infoRecorder->logTrace("Insufficient string buffer length.\n");
				return -1;
			}
			bcopy(rBuffer + rBufHead, buf, textlen);
			buf[textlen] = '\0';
			rBufHead += textlen;
			if (rBufHead == rBufTail)
				rBufHead = rBufTail = 0;
			return textlen;
		}
	}
	// buffer full?
	if (rBufTail - rBufHead == rBufSize) {
		infoRecorder->logTrace("Buffer full: Extremely long text data encountered?\n");
		return -1;
	}
	// did not found '\n', read more
	bcopy(rBuffer + rBufHead, rBuffer, rBufTail - rBufHead);
	rBufTail = rBufTail - rBufHead;
	rBufHead = 0;
	//
	if (rtspReadInternal() < 0)
		return -1;
	goto again;
	// unreachable, but to meet compiler's requirement
	return -1;
}

int RTSPContext::rtspReadBinary(char *buf, size_t count){
	int reqlength;

	if (rBufTail - rBufHead < 4)
		goto readmore;
again:
	reqlength = (unsigned char)rBuffer[rBufHead + 2];
	reqlength <<= 8;
	reqlength += (unsigned char)rBuffer[rBufHead + 3];
	// data is ready
	if (4 + reqlength <= rBufTail - rBufHead) {
		bcopy(rBuffer + rBufHead, buf, 4 + reqlength);
		rBufHead += (4 + reqlength);
		if (rBufHead == rBufTail)
			rBufHead = rBufTail = 0;
		return 4 + reqlength;
	}
	// second trail?
	if (rBufTail - rBufHead == rBufSize) {
		infoRecorder->logTrace("Buffer full: Extremely long binary data encountered?\n");
		return -1;
	}
readmore:
	bcopy(rBuffer + rBufHead, rBuffer, rBufTail - rBufHead);
	rBufTail = rBufTail - rBufHead;
	rBufHead = 0;
	//
	if (rtspReadInternal() < 0)
		return -1;
	goto again;
	// unreachable, but to meet compiler's requirement
	return -1;
}

int RTSPContext::rtspGetNext(char *buf, size_t count) {
	// initialize if necessary
	if (rBuffer == NULL) {
		rBufSize = 65536;
		if ((rBuffer = (char*)malloc(rBufSize)) == NULL) {
			rBufSize = 0;
			return -1;
		}
		rBufHead = 0;
		rBufTail = 0;
	}
	// buffer is empty, force read
	if (rBufTail == rBufHead) {
		if (rtspReadInternal() < 0)
			return -1;
	}
	// buffer is not empty
	if (rBuffer[rBufHead] != '$') {
		// text data
		return rtspReadText(buf, count);
	}
	// binary data
	return rtspReadBinary(buf, count);
}

void RTSPContext::closeAV(AVFormatContext *fctx, AVStream *st, AVCodecContext *cctx, enum RTSPLowerTransport transport) {
	unsigned i;
	//
	if (cctx) {
		AVCodecCommon::AVCodecClose(cctx);
	}
	if (st && st->codec != NULL) {
		if (st->codec != cctx) {
			AVCodecCommon::AVCodecClose(st->codec);
		}
		st->codec = NULL;
	}
	if (fctx) {
		for (i = 0; i < fctx->nb_streams; i++) {
			if (cctx != fctx->streams[i]->codec) {
				if (fctx->streams[i]->codec)
					AVCodecCommon::AVCodecClose(fctx->streams[i]->codec);
			}
			else {
				cctx = NULL;
			}
			av_freep(&fctx->streams[i]->codec);
			if (st == fctx->streams[i])
				st = NULL;
			av_freep(&fctx->streams[i]);
		}
#ifdef HOLE_PUNCHING
		// do nothing?
#else
		if (transport == RTSP_LOWER_TRANSPORT_UDP && fctx->pb)
			avio_close(fctx->pb);
#endif
		av_free(fctx);
	}
	if (cctx != NULL)
		av_free(cctx);
	if (st != NULL)
		av_free(st);
	return;
}

void RTSPContext::clientDeinit() {
	int i = 0;
#ifndef MULTI_CHANNEL
	closeAV(fmtCtx[id], stream[id], encoder[id], lowerTransport[id]);
#ifdef HOLE_PUNCHING
	if (lowerTransport[id] == RTSP_LOWER_TRANSPORT_UDP)
		rtpClosePorts(id);
#endif
#else
	for (i = 0; i < channels + 1; i++){
		closeAV(fmtCtx[i], stream[i], encoder[i], lowerTransport[i]);
#ifdef HOLE_PUNCHING
		if (lowerTransport[i] == RTSP_LOWER_TRANSPORT_UDP)
			rtpClosePorts(i);
#endif
	}
#endif
	//}
	//close_av(ctx->fmtctx[0], ctx->stream[0], ctx->encoder[0], ctx->lower_transport[0]);
	//close_av(ctx->fmtctx[1], ctx->stream[1], ctx->encoder[1], ctx->lower_transport[1]);
	//
#ifdef HOLE_PUNCHING
	if (sdpFmtCtx->pb)
		avio_close(sdpFmtCtx->pb);
#endif
	closeAV(sdpFmtCtx, NULL, NULL, RTSP_LOWER_TRANSPORT_UDP);
	//
	if (rBuffer) {
		free(rBuffer);
	}
	rBufSize = 0;
	rBufHead = rBufTail = 0;
	//
	return;
}

void RTSPContext::rtspReplyHeader(enum RTSPStatusCode error_number) {
	const char *str;
	time_t ti;
	struct tm rtm;
	char buf2[32];

	switch (error_number) {
	case RTSP_STATUS_OK:
		str = "OK";
		break;
	case RTSP_STATUS_METHOD:
		str = "Method Not Allowed";
		break;
	case RTSP_STATUS_BANDWIDTH:
		str = "Not Enough Bandwidth";
		break;
	case RTSP_STATUS_SESSION:
		str = "Session Not Found";
		break;
	case RTSP_STATUS_STATE:
		str = "Method Not Valid in This State";
		break;
	case RTSP_STATUS_AGGREGATE:
		str = "Aggregate operation not allowed";
		break;
	case RTSP_STATUS_ONLY_AGGREGATE:
		str = "Only aggregate operation allowed";
		break;
	case RTSP_STATUS_TRANSPORT:
		str = "Unsupported transport";
		break;
	case RTSP_STATUS_INTERNAL:
		str = "Internal Server Error";
		break;
	case RTSP_STATUS_SERVICE:
		str = "Service Unavailable";
		break;
	case RTSP_STATUS_VERSION:
		str = "RTSP Version not supported";
		break;
	default:
		str = "Unknown Error";
		break;
	}

	rtspPrintf("RTSP/1.0 %d %s\r\n", error_number, str);
	rtspPrintf("CSeq: %d\r\n", seq);
	/* output GMT time */
	ti = time(NULL);
	gmtime_r(&ti, &rtm);
	strftime(buf2, sizeof(buf2), "%a, %d %b %Y %H:%M:%S", &rtm);
	rtspPrintf("Date: %s GMT\r\n", buf2);
	//
	return;
}

void RTSPContext::rtspReplyError(enum RTSPStatusCode error_number) {
	rtspReplyHeader(error_number);
	rtspPrintf("\r\n");
}

int RTSPContext::prepareSdpDescription(char *buf, int bufsize) {
	buf[0] = '\0';
	av_dict_set(&sdpFmtCtx->metadata, "title", rtspConf->title, 0);
	snprintf(sdpFmtCtx->filename, sizeof(sdpFmtCtx->filename), "rtp://0.0.0.0");
	av_sdp_create(&sdpFmtCtx, 1, buf, bufsize);
	return strlen(buf);
}

void RTSPContext::rtspCmdDescribe(const char *url) {
	struct sockaddr_in myaddr;
	//#ifdef WIN32
#if 1
	int addrlen;
#else
	socklen_t addrlen;
#endif
	char path[4096];
	char content[4096];
	int content_length;
	//
	infoRecorder->logTrace("[RTSPContext]: describe url:%s.\n", url);
	av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);

#ifndef SEPERATE_OBJECT

	if (strcmp(path, rtspConf->object) != 0) {
		infoRecorder->logTrace("[RTSPContext]: service unavaliable, base-object not right, object:%s.\n", rtspConf->object);
		rtspReplyError(RTSP_STATUS_SERVICE);
		return;
	}
#else
	if (strcmp(path, object) != 0) {
		infoRecorder->logTrace("[RTSPContext]: service unavaliable, base-object not right, object:%s.\n", object);
		rtspReplyError(RTSP_STATUS_SERVICE);
		return;
	}
	
#endif
	//
	addrlen = sizeof(myaddr);
	getsockname(fd, (struct sockaddr*) &myaddr, &addrlen);
	content_length = prepareSdpDescription(content, sizeof(content));
	if (content_length < 0) {
		rtspReplyError(RTSP_STATUS_INTERNAL);
		return;
	}
	// state does not change
	rtspReplyHeader(RTSP_STATUS_OK);
	rtspPrintf("Content-Base: %s/\r\n", url);
	rtspPrintf("Content-Type: application/sdp\r\n");
	rtspPrintf("Content-Length: %d\r\n", content_length);
	rtspPrintf("\r\n");
	rtspWrite(content, content_length);
	return;
}

void RTSPContext::rtspCmdOptions(const char *url) {
	// state does not change
	rtspPrintf("RTSP/1.0 %d %s\r\n", RTSP_STATUS_OK, "OK");
	rtspPrintf("CSeq: %d\r\n", seq);
	//rtsp_printf(c, "Public: %s\r\n", "OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY, PAUSE");
	rtspPrintf("Public: %s\r\n", "OPTIONS, DESCRIBE, SETUP, TEARDOWN, PLAY");
	rtspPrintf("\r\n");
	return;
}

RTSPTransportField * RTSPContext::findTransport(RTSPMessageHeader *h, enum RTSPLowerTransport lower_transport) {
	RTSPTransportField *th;
	int i;
	for (i = 0; i < h->nb_transports; i++) {
		th = &h->transports[i];
		if (th->lower_transport == lower_transport)
			return th;
	}
	return NULL;
}

int RTSPContext::rtpNewAVStream(struct sockaddr_in *sin, int streamid, enum AVCodecID codecid) {
	AVOutputFormat *fmt = NULL;
	AVFormatContext *fmtctx = NULL;
	AVStream *stream = NULL;
	AVCodecContext *encoder = NULL;
	uint8_t *dummybuf = NULL;
	//
	infoRecorder->logTrace("[RTSPContext]: rtpNewAVStream streamid:%d.\n", streamid);
	if (streamid > IMAGE_SOURCE_CHANNEL_MAX) {
		infoRecorder->logTrace("invalid stream index (%d > %d)\n",
			streamid, IMAGE_SOURCE_CHANNEL_MAX);
		return -1;
	}
#if 0
	if (codecid != rtspConf->video_encoder_codec->id
		&& codecid != rtspConf->audio_encoder_codec->id) {
		infoRecorder->logTrace("invalid codec (%d)\n", codecid);
		return -1;
	}
#else
	if (codecid != rtspConf->video_encoder_codec->id){
		infoRecorder->logError("[RTSPContext]: invalid codec (%d)\n", codecid);
		return -1;
	}
#endif
	if (this->fmtCtx[streamid] != NULL) {
		infoRecorder->logTrace("duplicated setup to an existing stream (%d)\n",
			streamid);
		return -1;
	}
	if ((fmt = av_guess_format("rtp", NULL, NULL)) == NULL) {
		infoRecorder->logTrace("RTP not supported.\n");
		return -1;
	}
	if ((fmtctx = avformat_alloc_context()) == NULL) {
		infoRecorder->logTrace("create avformat context failed.\n");
		return -1;
	}



	fmtctx->oformat = fmt;
	if (mtu > 0) {
		if (fmtctx->packet_size > 0) {
			fmtctx->packet_size =
				mtu < fmtctx->packet_size ? mtu : fmtctx->packet_size;
		}
		else {
			fmtctx->packet_size = mtu;
		}
		infoRecorder->logTrace("RTP: packet size set to %d (configured: %d)\n",
			fmtctx->packet_size, mtu);
	}
#ifdef HOLE_PUNCHING
	if (ffio_open_dyn_packet_buf(&fmtctx->pb, mtu) < 0) {
		infoRecorder->logTrace("cannot open dynamic packet buffer\n");
		return -1;
	}
	infoRecorder->logTrace("RTP: Dynamic buffer opened, max_packet_size=%d.\n",
		(int)fmtctx->pb->max_packet_size);
	if (lowerTransport[streamid] == RTSP_LOWER_TRANSPORT_UDP) {
		if (rtpOpenPorts(streamid) < 0) {
			infoRecorder->logTrace("RTP: open ports failed - %s\n", strerror(errno));
			return -1;
		}
		infoRecorder->logTrace("[RTSPContext]: rtpLocalPort[0]:%d rtpLocalPort[1]:%d.\n", rtpLocalPort[0], rtpLocalPort[1]);
	}
#else
	if (lowerTransport[streamid] == RTSP_LOWER_TRANSPORT_UDP) {
		snprintf(fmtctx->filename, sizeof(fmtctx->filename),
			"rtp://%s:%d", inet_ntoa(sin->sin_addr), ntohs(sin->sin_port));
		if (avio_open(&fmtctx->pb, fmtctx->filename, AVIO_FLAG_WRITE) < 0) {
			infoRecorder->logTrace("cannot open URL: %s\n", fmtctx->filename);
			return -1;
		}
		infoRecorder->logTrace("RTP/UDP: URL opened [%d]: %s, max_packet_size=%d\n",
			streamid, fmtctx->filename, fmtctx->pb->max_packet_size);
	}
	else if (lowerTransport[streamid] == RTSP_LOWER_TRANSPORT_TCP) {
		// XXX: should we use avio_open_dyn_buf(&fmtctx->pb)?
		if (ffio_open_dyn_packet_buf(&fmtctx->pb, mtu) < 0) {
			infoRecorder->logTrace("cannot open dynamic packet buffer\n");
			return -1;
		}
		infoRecorder->logTrace("RTP/TCP: Dynamic buffer opened, max_packet_size=%d.\n",
			(int)fmtctx->pb->max_packet_size);
	}
#endif
	fmtctx->pb->seekable = 0;
	//
#if 0
	if ((stream = AVCodecCommon::AVFormatNewStream(fmtctx, 0,
		codecid == rtspConf->video_encoder_codec->id ?
		rtspConf->video_encoder_codec : rtspConf->audio_encoder_codec)) == NULL) {
		infoRecorder->logTrace("Cannot create new stream (%d)\n", codecid);
		return -1;
	}
#else
	if ((stream = AVCodecCommon::AVFormatNewStream(fmtctx, 0, rtspConf->video_encoder_codec)) == NULL){
		infoRecorder->logError("[RTSPContext]: cannot create new stream (%d).\n", codecid);
		return -1;
	}
#endif
#ifndef SHARE_ENCODER
	if (codecid == rtspConf->video_encoder_codec->id) {
		encoder = AVCodecCommon::AVCodecVEncoderInit(
			stream->codec,
			rtspConf->video_encoder_codec,
			video_source_maxwidth(streamid),
			video_source_maxheight(streamid),
			rtspConf->video_fps,
			rtspConf->vso);
	}
	else if (codecid == rtspConf->audio_encoder_codec->id) {
		encoder = AVCodecCommon::AVCodecAEncoderInit(
			stream->codec,
			rtspConf->audio_encoder_codec,
			rtspConf->audio_bitrate,
			rtspConf->audio_samplerate,
			rtspConf->audio_channels,
			rtspConf->audio_codec_format,
			rtspConf->audio_codec_channel_layout);
	}
	if (encoder == NULL) {
		infoRecorder->logTrace("Cannot init encoder\n");
		return -1;
	}
#else
	if (codecid == rtspConf->video_encoder_codec->id){
		encoder = AVCodecCommon::AVCodecVEncoderInit(stream->codec,
			rtspConf->video_encoder_codec,
			width, height,
			rtspConf->video_fps, rtspConf->vso);
	}
#endif    /* SHARE_ENCODER */
	//
	if (encoder == NULL){
		infoRecorder->logTrace("[RTSPContext]: Cannot init encoder.\n");
		return -1;
	}
#if 0
	stream->codec->pix_fmt = AV_PIX_FMT_YUV420P;
	stream->codec->flags = CODEC_FLAG_GLOBAL_HEADER;
	stream->codec->width = this->width;
	stream->codec->height = this->height;
	stream->codec->time_base.num = 1;
	stream->codec->time_base.den = rtspConf->video_fps;
	stream->codec->gop_size = rtspConf->video_fps;
	stream->codec->bit_rate = rtspConf->audio_bitrate;
#endif

	this->encoder[streamid] = encoder;
	this->stream[streamid] = stream;
	this->fmtCtx[streamid] = fmtctx;
	infoRecorder->logTrace("[RTSPContext]: strem[%d]:%p, fmtCtx[%d]:%p.\n", streamid, stream, streamid, fmtctx);
	// write header
	if (avformat_write_header(fmtCtx[streamid], NULL) < 0) {
		infoRecorder->logTrace("[RTSPContext]: Cannot write stream id %d, fmtCtx[%d]:%p.\n", streamid, streamid, fmtCtx[streamid]);
		Log::logscreen("[RTSPContext]: cannot write stream id:%d, fmtCtx[%d]:%p.\n", streamid, streamid, fmtCtx[streamid]);
		return -1;
	}
#ifdef HOLE_PUNCHING
	avio_close_dyn_buf(fmtCtx[streamid]->pb, &dummybuf);
	av_free(dummybuf);
#else
	if (lowerTransport[streamid] == RTSP_LOWER_TRANSPORT_TCP) {
		/*int rlen;
		rlen =*/ avio_close_dyn_buf(fmtctx->pb, &dummybuf);
	av_free(dummybuf);
	}
#endif
	return 0;
}
#if 0
RTSPTransportField * RTSPContext::findTransport(RTSPMessageHeader *h, enum RTSPLowerTransport lower_transport) {
	RTSPTransportField *th;
	int i;
	for (i = 0; i < h->nb_transports; i++) {
		th = &h->transports[i];
		if (th->lower_transport == lower_transport)
			return th;
	}
	return NULL;
}
#endif

/*  here may be wrong, about the stream id
	originally, we need to find the stream with path name
	but now, the stream is comes from 
*/


void RTSPContext::rtspCmdSetup(const char *url, RTSPMessageHeader *h, int streamId) {
	infoRecorder->logTrace("[RTSPContext]: rtsp cmd setup.\n");
	int i;
	RTSPTransportField *th;
	struct sockaddr_in destaddr, myaddr;
#ifdef WIN32
	int destaddrlen, myaddrlen;
#else
	socklen_t destaddrlen, myaddrlen;
#endif
	char path[4096];
	char channelname[IMAGE_SOURCE_CHANNEL_MAX + 1][RTSP_STREAM_FORMAT_MAXLEN];
#ifndef SEPERATE_OBJECT
	int baselen = strlen(rtspConf->object);
#else
	int baselen = strlen(object);
#endif
	int streamid;
	int rtp_port, rtcp_port;
	enum RTSPStatusCode errcode;
	//
	av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
	for (i = 0; i < IMAGE_SOURCE_CHANNEL_MAX + 1; i++){
		snprintf(channelname[i], RTSP_STREAM_FORMAT_MAXLEN, RTSP_STREAM_FORMAT, i);// streamId);
	}

#if 1
	//
#ifndef SEPERATE_OBJECT
	if (strncmp(path, rtspConf->object, baselen) != 0) {
		infoRecorder->logError("invalid object (path=%s)\n", path);
		rtspReplyError(RTSP_STATUS_AGGREGATE);
		return;
	}
#else
	if (strncmp(path, object, baselen) != 0) {
		infoRecorder->logError("invalid object (path=%s)\n", path);
		rtspReplyError(RTSP_STATUS_AGGREGATE);
		return;
	}

#endif
	for (i = 0; i < IMAGE_SOURCE_CHANNEL_MAX + 1; i++){
		infoRecorder->logTrace("[RTSPContext]: path:%s, channelname:%s.\n", path + baselen + 1, channelname[i]);
		if (strcmp(path + baselen + 1, channelname[i] )== 0){
			streamid = i;
			break;
		}
	}
	if (i == IMAGE_SOURCE_CHANNEL_MAX + 1){
		//not found
		infoRecorder->logError("invalid service (path=%s)\n", path);
		rtspReplyError(RTSP_STATUS_SERVICE);
		return;
	}
	
#else
	streamid = streamId;
#endif
	//
	if (state != SERVER_STATE_IDLE
		&& state != SERVER_STATE_READY) {
		rtspReplyError(RTSP_STATUS_STATE);
		return;
	}
	// create session id?
	if (sessionId == NULL) {
		if (h->session_id[0] == '\0') {
			snprintf(h->session_id, sizeof(h->session_id), "%04x%04x",
				rand() % 0x0ffff, rand() % 0x0ffff);
			sessionId = _strdup(h->session_id);
			infoRecorder->logError("New session created (id = %s)\n", sessionId);
		}
	}
	// session id must match -- we have only one session
	if (sessionId == NULL
		|| strcmp(sessionId, h->session_id) != 0) {
		infoRecorder->logError("Bad session id %s != %s\n", h->session_id, sessionId);
		errcode = RTSP_STATUS_SESSION;
		goto error_setup;
	}
	// find supported transport
	if ((th = findTransport(h, RTSP_LOWER_TRANSPORT_UDP)) == NULL) {
		th = findTransport(h, RTSP_LOWER_TRANSPORT_TCP);
	}
	if (th == NULL) {
		infoRecorder->logError("Cannot find transport\n");
		errcode = RTSP_STATUS_TRANSPORT;
		goto error_setup;
	}
	//
	destaddrlen = sizeof(destaddr);
	bzero(&destaddr, destaddrlen);
	if (getpeername(fd, (struct sockaddr*) &destaddr, &destaddrlen) < 0) {
		infoRecorder->logError("Cannot get peername\n");
		errcode = RTSP_STATUS_INTERNAL;
		goto error_setup;
	}
	destaddr.sin_port = htons(th->client_port_min);
	//
	myaddrlen = sizeof(myaddr);
	bzero(&myaddr, myaddrlen);
	if (getsockname(fd, (struct sockaddr*) &myaddr, &myaddrlen) < 0) {
		infoRecorder->logError("Cannot get sockname\n");
		errcode = RTSP_STATUS_INTERNAL;
		goto error_setup;
	}
	//
	lowerTransport[streamid] = th->lower_transport;

#if 0
	if (rtpNewAVStream(&destaddr, streamid,
		streamid == streamId/*rtspconf->audio_id*/ ?
		rtspConf->audio_encoder_codec->id : rtspConf->video_encoder_codec->id) < 0) {
		infoRecorder->logTrace("Create AV stream %d failed.\n", streamid);
		errcode = RTSP_STATUS_TRANSPORT;
		goto error_setup;
	}
#else
	infoRecorder->logError("[RTSPContext]: Create AV stream %d.\n", streamid);

	if (rtpNewAVStream(&destaddr, streamid, rtspConf->video_encoder_codec->id) < 0){
		infoRecorder->logError("[RTSPContext]: Create AV stream %d failed.\n", streamid);
		errcode  = RTSP_STATUS_TRANSPORT;
		goto error_setup;
	}
	else{
		infoRecorder->logTrace("[RTSPContext]: after new av stream, rtpLocalPoart[0]%d, rtpLocalPort[1]:%d.\n", rtpLocalPort[0], rtpLocalPort[1]);
	}
#endif
	//
	state = SERVER_STATE_READY;
	rtspReplyHeader(RTSP_STATUS_OK);
	rtspPrintf("Session: %s\r\n", sessionId);
	switch (th->lower_transport) {
	case RTSP_LOWER_TRANSPORT_UDP:
#ifdef HOLE_PUNCHING
		rtp_port = rtpLocalPort[streamid * 2];
		rtcp_port = rtpLocalPort[streamid * 2 + 1];
		rtpPeerPort[streamid * 2] = htons(th->client_port_min);
		rtpPeerPort[streamid * 2 + 1] = htons(th->client_port_max);
#else
		rtp_port = ff_rtp_get_local_rtp_port((URLContext*)fmtCtx[streamId]->pb->opaque);
		rtcp_port = ff_rtp_get_local_rtcp_port((URLContext*)fmtCtx[streamId]->pb->opaque);
#endif
		infoRecorder->logTrace("RTP/UDP: streamid=%d; client=%d-%d; server=%d-%d\n",
			streamid,
			th->client_port_min, th->client_port_max,
			rtp_port, rtcp_port);
		rtspPrintf("Transport: RTP/AVP/UDP;unicast;client_port=%d-%d;server_port=%d-%d\r\n",
			th->client_port_min, th->client_port_max,
			rtp_port, rtcp_port);
		break;
	case RTSP_LOWER_TRANSPORT_TCP:
		infoRecorder->logTrace("RTP/TCP: interleaved=%d-%d\n",
			streamid * 2, streamid * 2 + 1);
		rtspPrintf("Transport: RTP/AVP/TCP;unicast;interleaved=%d-%d\r\n",
			streamid * 2, streamid * 2 + 1, streamid * 2);
		break;
	default:
		// should not happen
		break;
	}
	rtspPrintf("\r\n");
	return;
error_setup:
	if (sessionId != NULL) {
		free(sessionId);
		sessionId = NULL;
	}
	if (encoder[streamid] != NULL) {
		encoder[streamid] = NULL;
	}
	if (stream[streamid] != NULL) {
		stream[streamid] = NULL;
	}
	if (fmtCtx[streamid] != NULL) {
		avformat_free_context(fmtCtx[streamid]);
		fmtCtx[streamid] = NULL;
	}
	rtspReplyError(errcode);
	return;
}
extern int EncoderRegisterClient(RTSPContext * rtsp);
void RTSPContext::rtspCmdPlay(const char *url, RTSPMessageHeader *h) {
	infoRecorder->logTrace("[RTSPContext]: rtsp cmd play.\n");
	char path[4096];
	//
	av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
#ifndef SEPERATE_OBJECT
	infoRecorder->logTrace("[RTSPContext]: path:%s, vs object:%s.\n", path, rtspConf->object);

	if (strncmp(path, rtspConf->object, strlen(rtspConf->object)) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		infoRecorder->logTrace("[RTSPContext]: cmd play RTSP_STATUS_SESSION.\n");
		return;
	}
#else

	infoRecorder->logTrace("[RTSPContext]: path:%s, vs object:%s.\n", path, object);
	if (strncmp(path, object, strlen(object)) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		infoRecorder->logTrace("[RTSPContext]: cmd play RTSP_STATUS_SESSION.\n");
		return;
	}
#endif
	if (strcmp(sessionId, h->session_id) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		infoRecorder->logTrace("[RTSPContext]: cmd play session_id not equal.\n");
		return;
	}
	//
	if (state != SERVER_STATE_READY
		&& state != SERVER_STATE_PAUSE) {
		rtspReplyError(RTSP_STATUS_STATE);
		infoRecorder->logTrace("[RTSPContext]: cmd play server state is not ready or pause.\n");
		return;
	}
	// create threads
#ifndef SHARE_ENCODER
	// no use here
	vThread = chBEGINTHREADEX(NULL, 0, vencoder_thread, &ctx, 0, &(vThreadId));
	/* if (pthread_create(&vthread, NULL, vencoder_thread, ctx) != 0) {
		 infoRecorder->logError("cannot create video thread\n");
		 rtspReplyError(ctx, RTSP_STATUS_INTERNAL);
		 return;
		 }*/
#else
	infoRecorder->logTrace("[RTSPContext]: cmd play to register encoder client.\n");
	if (EncoderRegisterClient(this) < 0){
		infoRecorder->logError("cannot register encoder client.\n");
		rtspReplyError(RTSP_STATUS_INTERNAL);
		return;
	}


#endif  /* SHARE_ENCODER */
	//
	state = SERVER_STATE_PLAYING;
	rtspReplyHeader(RTSP_STATUS_OK);
	rtspPrintf("Session: %s\r\n", sessionId);
	rtspPrintf("\r\n");
	return;
}

void RTSPContext::
rtspCmdPause(const char *url, RTSPMessageHeader *h) {
	infoRecorder->logTrace("[RTSPContext]: rtsp cmd pause.\n");
	char path[4096];
	//
	av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
#ifndef SEPERATE_OBJECT
	infoRecorder->logTrace("[RTSPContext]: path:%s, vs object:%s.\n", path, rtspConf->object);
	if (strncmp(path, rtspConf->object, strlen(rtspConf->object)) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		return;
	}
#else
	infoRecorder->logTrace("[RTSPContext]: path:%s, vs object:%s.\n", path, object);
	if (strncmp(path, object, strlen(object)) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		return;
	}

#endif
	if (strcmp(sessionId, h->session_id) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		return;
	}
	//
	if (state != SERVER_STATE_PLAYING) {
		rtspReplyError(RTSP_STATUS_STATE);
		return;
	}
	//
	state = SERVER_STATE_PAUSE;
	rtspReplyHeader(RTSP_STATUS_OK);
	rtspPrintf("Session: %s\r\n", sessionId);
	rtspPrintf("\r\n");
	return;
}

void RTSPContext::rtspCmdTeardown(const char *url, RTSPMessageHeader *h, int bruteforce) {
	infoRecorder->logTrace("[RTSPContext]: rtsp cmd tear down.\n");
	char path[4096];
	//
	av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);


#ifndef SEPERATE_OBJECT
	infoRecorder->logTrace("[RTSPContext]: path:%s, vs object:%s.\n", path, rtspConf->object);
	if (strncmp(path, rtspConf->object, strlen(rtspConf->object)) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		return;
	}
#else
	infoRecorder->logTrace("[RTSPContext]: path:%s, vs object:%s.\n", path, object);
	if (strncmp(path, object, strlen(object)) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		return;
	}
#endif
	if (strcmp(sessionId, h->session_id) != 0) {
		rtspReplyError(RTSP_STATUS_SESSION);
		return;
	}
	//
	state = SERVER_STATE_TEARDOWN;
	if (bruteforce != 0)
		return;
	// XXX: well, gently response
	rtspReplyHeader(RTSP_STATUS_OK);
	rtspPrintf("Session: %s\r\n", sessionId);
	rtspPrintf("\r\n");
	return;
}




