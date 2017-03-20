//#include "ccg_win32.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "../VideoUtility/videocommon.h"
#include "../VideoUtility/avcodeccommon.h"
#include "../VideoUtility/ccg_config.h"
#include "../VideoUtility/rtspconf.h"
//#include "../LibCore/MemOp.h"

#include "rtspcontext.h"
#include "../LibCore/log.h"
#include "../LibCore/inforecorder.h"

#define RTSP_STREAM_FORMAT "streamid=%d"
#define RTSP_STREAM_FORMAT_MAXLEN 64

using namespace cg;
using namespace cg::core;

// this definition will enable the trace log for RTSP context
#define ENABLE_RTSP_LOG

namespace cg{
	namespace rtsp{

		inline int read(SOCKET fd, void *buf, int count) { return recv(fd, (char *) buf, count, 0); }

		inline int write(SOCKET fd, const void *buf, int count) { return send(fd, (const char*) buf, count, 0);}

		inline int close(SOCKET fd) { return closesocket(fd); }

		////// constructor  ////////////
		RTSPContext::RTSPContext(){
			//infoRecorder->logError("RTSPContext constructor\n");
			memset(object, 0, RTSPCONF_OBJECT_SIZE); 
			tag = 0; 
			state = SERVER_STATE_IDLE;
			hasVideo = 0; 
			rtspWriteMutex = CreateMutex(NULL, FALSE, NULL);
			channels = 0;
			id = 0;
			initialized = false;
			enableGen = false;
			fd = NULL;
			height = 0;
			width = 0;
			rBuffer = NULL;
			rBufSize = 0;
			rBufHead = 0;
			rBufTail = 0;
			sessionId = NULL;
			sdpFmtCtx = NULL;
			seq = 0;
			mtu = 0;
			rtspConf = NULL;
			streamCount = 0;
			for(int i = 0; i < IMAGE_SOURCE_CHANNEL_MAX; i++){
				sdpVStream[i] = NULL;
				sdpVEncoder[i] = NULL;
			}

			sdpAStream = NULL;
			sdpFmtCtx = NULL;
			sdpAEncoder = NULL;

			for(int i = 0; i < RTSP_CHANNEL_MAX; i++){
				fmtCtx[i] = NULL;
				stream[i] = NULL;
				encoder[i] = NULL;
				rtp[i] = NULL;
			}
			for(int i = 0; i < RTSP_CHANNEL_MAXx2; i++){
				rtpSocket[i] = NULL;
				rtpLocalPort[i] = 0;
				rtpPeerPort[i] = 0;
				rtpPortChecked[i] = 0;
			}

		}

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

#ifndef MULTI_CHANNEL
			if ((sdpVStream[id] = AVCodecCommon::AVFormatNewStream(
				sdpFmtCtx, id, rtspConf->video_encoder_codec)) == NULL){
					infoRecorder->logError("# cannot create new video stream (%d:%d)\n", id, rtspConf->video_encoder_codec->id);
					return -1;
			}
			if ((sdpVEncoder[id] = AVCodecCommon::AVCodecVEncoderInit(sdpVStream[id]->codec, rtspConf->video_encoder_codec, width, height, rtspConf->video_fps, rtspConf->vso)) == NULL){
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
						infoRecorder->logError("# cannot create new video stream (%d:%d)\n", i, rtspConf->video_encoder_codec->id);
						return -1;
				}
				if ((sdpVEncoder[i] = AVCodecCommon::AVCodecVEncoderInit(sdpVStream[i]->codec, rtspConf->video_encoder_codec, width, height, rtspConf->video_fps, rtspConf->vso)) == NULL){
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
			int ret = 0;

			va_start(ap, fmt);
			buflen = vsnprintf(buf, sizeof(buf), fmt, ap);
			va_end(ap);
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logError("[RTSPContext]: rtspPrintf: %s.\n", buf);
#endif
			ret = rtspWrite(buf, buflen);
			if(ret <= 0){
				infoRecorder->logError("[RTSPContext]: rtsp write failed with code:%d\n", WSAGetLastError());
			}
			return ret;
		}

		int RTSPContext::rtspWriteBinData(int streamId, unsigned char * buf, int bufLen){
			int i, pktLen;
			char header[4];
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: rtspWriteBinData. streamid:%d, size:%d.\n", streamId, bufLen);
#endif			
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

				WaitForSingleObject(rtspWriteMutex, INFINITE);
				if (rtspWrite(header, 4) != 4){
					ReleaseMutex(rtspWriteMutex);
					return i;
				}
				if (rtspWrite(&buf[i + 4], pktLen) != pktLen){
					return i;
				}
				ReleaseMutex(rtspWriteMutex);
				i += (4 + pktLen);
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


#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("RTP: port opened for stream %d, min=%d (fd=%d), max=%d (fd=%d)\n",
				streamId / 2,
				(unsigned int)rtpLocalPort[streamId],
				(int)rtpSocket[streamId],
				(unsigned int)rtpLocalPort[streamId + 1],
				(int)rtpSocket[streamId + 1]);
#endif
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
		int RTSPContext::rtpWriteBinData(int streamId, uint8_t * buf, int bufLen){
			int i, pktlen;
			struct sockaddr_in sin;
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: rtpWriteBinData. streamid:%d, size:%d.\n", streamId, bufLen);
#endif
			if (rtpSocket[streamId * 2] == 0){
				infoRecorder->logError("[RTSPContext]: rtp write data, NULL rtp socket");
				return -1;
			}
			if (buf == NULL)
				return 0;
			if (bufLen < 4)
				return bufLen;
			bcopy(&client, &sin, sizeof(sin));
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: rtp write data client:%s: %d\n", inet_ntoa(sin.sin_addr), rtpPeerPort[streamId * 2]);
#endif
			sin.sin_port = rtpPeerPort[streamId * 2];
			// XXX: buffer is the result from avio_open_dyn_buf.
			// Multiple RTP packets can be placed in a single buffer.
			// Format == 4-bytes (big-endian) packet size + packet-data
			i = 0;
			while (i < bufLen) {
				pktlen  = (buf[i + 0] << 24);
				pktlen += (buf[i + 1] << 16);
				pktlen += (buf[i + 2] << 8);
				pktlen += (buf[i + 3]);
				if (pktlen == 0) {
					i += 4;
					continue;
				}

#if 0
				/////// here's the code to look inside a rtp h264 packet
				unsigned char nalu = buf[i+ 4 + 12];
				unsigned char forbidden_zero_bit = nalu & ( 1 << 8);
				unsigned char nal_ref_idc = nalu & (3 << 5);
				unsigned char nal_unit_type = nalu & (63);

				infoRecorder->logError("[RTSPContext]: forbidden_zero_bit:%d, NRI:%d, nal_unit_type:%d.\n", forbidden_zero_bit, nal_ref_idc, nal_unit_type);

				if(nal_unit_type == 0){
					infoRecorder->logError("[RTSPContext]: NALU undefined.\n");
				}
				else if(nal_unit_type >= 1 && nal_unit_type <= 23){
					infoRecorder->logError("[RTSPCOntext]: NALU single NAL packet,\n");
				}
				else if(nal_unit_type == 24){
					infoRecorder->logError("[RTSPCONtext]: NALU STAP_A.\n");
				}
				else if(nal_unit_type == 25){
					infoRecorder->logError("[RTSPContext]: NALU STAP-B.\n");
				}

				else if(nal_unit_type == 26){
					infoRecorder->logError("[RTSPContext]: NALU MTAP16.\n");
				}
				else if(nal_unit_type == 27){

					infoRecorder->logError("[RTSPContext]: NALU MTAP24.\n");
				}else if(nal_unit_type == 28){

					infoRecorder->logError("[RTSPContext]: NALU FU-A.\n");
				}else if(nal_unit_type == 29){

					infoRecorder->logError("[RTSPContext]: NALU FU-B.\n");
				}
				else{
					infoRecorder->logError("[RTSPContext]: NALU invalid.\n");
				}
#else


#if 0  // do not write the fragment teype data
				int fragment_type = buf[i+ 4 + 12] & 0x1F;
				int nal_type = buf[i + 4 + 13] & 0x1F;
				int start_bit = buf[i + 4 + 13] & 0x80;
				int end_bit = buf[i + 4 + 13] & 0x40;

				infoRecorder->logError("[RTSPContext]: fragment_type:%d, nal_type:%d.\n", fragment_type, nal_type);
#endif   /// write the fragment type dat
#endif   // write the help data

#if 0   // wretit eh first 100 bytes
				infoRecorder->logError("[RTSPContext]: first 100 bytes: \n");
				for(int j = 0; j < 100; j++){
					infoRecorder->logError(" %X", buf[i + 4 + 12 +j]);
				}
				infoRecorder->logError("\n");
#endif  // write the fist 100 bytes

#ifdef ENABLE_RTSP_LOG
				infoRecorder->logTrace("[RTSPContext]: sendto... pktlen:%d\n", pktlen);
#endif
				sendto(rtpSocket[streamId * 2], (const char*)&buf[i + 4], pktlen, 0, (struct sockaddr*) &sin, sizeof(struct sockaddr_in));
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
					infoRecorder->logError("[RTSPContext]: read error, code:%d.\n", WSAGetLastError());
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
						infoRecorder->logError("Insufficient string buffer length.\n");
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
				infoRecorder->logError("Buffer full: Extremely long text data encountered?\n");
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
				infoRecorder->logError("Buffer full: Extremely long binary data encountered?\n");
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
					infoRecorder->logError("[RTSPContext]: malloc buffer failed.\n");
					rBufSize = 0;
					return -1;
				}
				rBufHead = 0;
				rBufTail = 0;
			}
			// buffer is empty, force read
			if (rBufTail == rBufHead) {
				if (rtspReadInternal() < 0){
					infoRecorder->logError("[RTSPContext]: read internal failed.\n");
					return -1;
				}
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
#ifdef HOLE_PUNCHING
			if (sdpFmtCtx->pb)
				avio_close(sdpFmtCtx->pb);
#endif
			closeAV(sdpFmtCtx, NULL, NULL, RTSP_LOWER_TRANSPORT_UDP);
			//
			if (rBuffer) {
				free(rBuffer);
				rBuffer = NULL;
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
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logError("[RTSPContext]: ReplayError, error num: %d.\n", error_number);
#endif
			rtspReplyHeader(error_number);
			rtspPrintf("\r\n");
		}

		int RTSPContext::prepareSdpDescription(char *buf, int bufsize) {
			buf[0] = '\0';
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: prepareSDPdescription, title:%s", rtspConf->title);
#endif

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
			char path[4096] = {0};
			char content[4096] = {0};
			int content_length = 0;

			strcpy(path, object);

#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: describe url:%s.\n", url);
#endif
			av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
			if (strcmp(path, object) != 0) {
				infoRecorder->logError("[RTSPContext]: service unavaliable, base-object not right, object:%s (path:%s).\n", object, path);
				rtspReplyError(RTSP_STATUS_SERVICE);
				return;
			}
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
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: rtpNewAVStream streamid:%d.\n", streamid);
#endif
			if (streamid > IMAGE_SOURCE_CHANNEL_MAX) {
				infoRecorder->logError("invalid stream index (%d > %d)\n",
					streamid, IMAGE_SOURCE_CHANNEL_MAX);
				return -1;
			}

			if (codecid != rtspConf->video_encoder_codec->id){
				infoRecorder->logError("[RTSPContext]: invalid codec (%d)\n", codecid);
				return -1;
			}
			if (this->fmtCtx[streamid] != NULL) {
				infoRecorder->logError("duplicated setup to an existing stream (%d)\n", streamid);
				return -1;
			}
			if ((fmt = av_guess_format("rtp", NULL, NULL)) == NULL) {
				infoRecorder->logError("RTP not supported.\n");
				return -1;
			}
			if ((fmtctx = avformat_alloc_context()) == NULL) {
				infoRecorder->logError("create avformat context failed.\n");
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
#ifdef ENABLE_RTSP_LOG
				infoRecorder->logError("RTP: packet size set to %d (configured: %d)\n", fmtctx->packet_size, mtu);
#endif
			}
#ifdef HOLE_PUNCHING
			if (ffio_open_dyn_packet_buf(&fmtctx->pb, mtu) < 0) {
				infoRecorder->logError("cannot open dynamic packet buffer\n");
				return -1;
			}
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("RTP: Dynamic buffer opened, max_packet_size=%d.\n", (int)fmtctx->pb->max_packet_size);
#endif
			if (lowerTransport[streamid] == RTSP_LOWER_TRANSPORT_UDP) {
				if (rtpOpenPorts(streamid) < 0) {
					infoRecorder->logError("RTP: open ports failed - %s\n", strerror(errno));
					return -1;
				}
#ifdef ENABLE_RTSP_LOG
				infoRecorder->logError("[RTSPContext]: rtpLocalPort[0]:%d rtpLocalPort[1]:%d.\n", rtpLocalPort[0], rtpLocalPort[1]);
#endif
			}
#else
			if (lowerTransport[streamid] == RTSP_LOWER_TRANSPORT_UDP) {
				snprintf(fmtctx->filename, sizeof(fmtctx->filename),
					"rtp://%s:%d", inet_ntoa(sin->sin_addr), ntohs(sin->sin_port));
				if (avio_open(&fmtctx->pb, fmtctx->filename, AVIO_FLAG_WRITE) < 0) {
					infoRecorder->logError("cannot open URL: %s\n", fmtctx->filename);
					return -1;
				}
				infoRecorder->logError("RTP/UDP: URL opened [%d]: %s, max_packet_size=%d\n",
					streamid, fmtctx->filename, fmtctx->pb->max_packet_size);
			}
			else if (lowerTransport[streamid] == RTSP_LOWER_TRANSPORT_TCP) {
				// XXX: should we use avio_open_dyn_buf(&fmtctx->pb)?
				if (ffio_open_dyn_packet_buf(&fmtctx->pb, mtu) < 0) {
					infoRecorder->logError("cannot open dynamic packet buffer\n");
					return -1;
				}
				infoRecorder->logError("RTP/TCP: Dynamic buffer opened, max_packet_size=%d.\n",
					(int)fmtctx->pb->max_packet_size);
			}
#endif
			fmtctx->pb->seekable = 0;
			//
#if 0
			if ((stream = AVCodecCommon::AVFormatNewStream(fmtctx, 0,
				codecid == rtspConf->video_encoder_codec->id ?
				rtspConf->video_encoder_codec : rtspConf->audio_encoder_codec)) == NULL) {
					infoRecorder->logError("Cannot create new stream (%d)\n", codecid);
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
				infoRecorder->logError("Cannot init encoder\n");
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
				infoRecorder->logError("[RTSPContext]: Cannot init encoder.\n");
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
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logError("[RTSPContext]: stream[%d]:%p, fmtCtx[%d]:%p, stream time_base(num:%d, den:%d).\n", streamid, stream, streamid, fmtctx, stream->time_base.num, stream->time_base.den);
#endif
			// write header
			if (avformat_write_header(fmtCtx[streamid], NULL) < 0) {
				infoRecorder->logError("[RTSPContext]: Cannot write stream id %d, fmtCtx[%d]:%p.\n", streamid, streamid, fmtCtx[streamid]);
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

#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: rtsp cmd setup, url:%s.\n", url);
#endif
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
			int baselen = strlen(object);
			int streamid;
			int rtp_port, rtcp_port;
			enum RTSPStatusCode errcode;
			//
			av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
			for (i = 0; i < IMAGE_SOURCE_CHANNEL_MAX + 1; i++){
				snprintf(channelname[i], RTSP_STREAM_FORMAT_MAXLEN, RTSP_STREAM_FORMAT, i);// streamId);
			}

#if 1
			if (strncmp(path, object, baselen) != 0) {
				infoRecorder->logError("invalid object (path=%s)\n", path);
				rtspReplyError(RTSP_STATUS_AGGREGATE);
				return;
			}
			for (i = 0; i < IMAGE_SOURCE_CHANNEL_MAX + 1; i++){
#ifdef ENABLE_RTSP_LOG
				infoRecorder->logTrace("[RTSPContext]: path:%s, channelname:%s.\n", path + baselen + 1, channelname[i]);
#endif
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
					snprintf(h->session_id, sizeof(h->session_id), "%04x%04x", rand() % 0x0ffff, rand() % 0x0ffff);
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
				infoRecorder->logError("Cannot get peer name\n");
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
					infoRecorder->logError("Create AV stream %d failed.\n", streamid);
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
				infoRecorder->logError("[RTSPContext]: after new av stream, rtpLocalPoart[0]%d, rtpLocalPort[1]:%d.\n", rtpLocalPort[0], rtpLocalPort[1]);
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
				infoRecorder->logError("RTP/UDP: streamid=%d; client=%d-%d; server=%d-%d\n",
					streamid,
					th->client_port_min, th->client_port_max,
					rtp_port, rtcp_port);

				rtspPrintf("Transport: RTP/AVP/UDP;unicast;client_port=%d-%d;server_port=%d-%d\r\n",
					th->client_port_min, th->client_port_max,
					rtp_port, rtcp_port);
				break;
			case RTSP_LOWER_TRANSPORT_TCP:
				infoRecorder->logError("RTP/TCP: interleaved=%d-%d\n",
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
		//extern int EncoderRegisterClient(RTSPContext * rtsp);

		void RTSPContext::rtspCmdPlay(const char *url, RTSPMessageHeader *h) {
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: rtsp cmd play, url:%s.\n", url);
#endif
			char path[4096] = {0};
			//
			av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: path:%s, vs object:%s.\n", path, object);
#endif
			if (strncmp(path, object, strlen(object)) != 0) {
				rtspReplyError(RTSP_STATUS_SESSION);
				infoRecorder->logError("[RTSPContext]: cmd play RTSP_STATUS_SESSION.\n");
				return;
			}
			if (strcmp(sessionId, h->session_id) != 0) {
				rtspReplyError(RTSP_STATUS_SESSION);
				infoRecorder->logError("[RTSPContext]: cmd play session_id not equal.\n");
				return;
			}
			//
			if (state != SERVER_STATE_READY
				&& state != SERVER_STATE_PAUSE) {
					rtspReplyError(RTSP_STATUS_STATE);
					infoRecorder->logError("[RTSPContext]: cmd play server state is not ready or pause.\n");
					return;
			}
			// create threads
#ifndef SHARE_ENCODER
			// no use here

#else

#endif  /* SHARE_ENCODER */
			//
			state = SERVER_STATE_PLAYING;
			rtspReplyHeader(RTSP_STATUS_OK);
			rtspPrintf("Session: %s\r\n", sessionId);
			rtspPrintf("\r\n");
			return;
		}

		void RTSPContext::rtspCmdPause(const char *url, RTSPMessageHeader *h) {
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logTrace("[RTSPContext]: rtsp cmd pause, url: %s.\n", url);
#endif
			char path[4096] = {0};
			//
			av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
#ifdef ENABLE_RTSP_LOG
			infoRecorder->logError("[RTSPContext]: path:%s, vs object:%s.\n", path, object);
#endif
			if (strncmp(path, object, strlen(object)) != 0) {
				rtspReplyError(RTSP_STATUS_SESSION);
				return;
			}
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
			infoRecorder->logError("[RTSPContext]: rtsp cmd tear down.\n");
			char path[4096];
			//
			av_url_split(NULL, 0, NULL, 0, NULL, 0, NULL, path, sizeof(path), url);
			infoRecorder->logError("[RTSPContext]: path:%s, vs object:%s.\n", path, object);
			if (strncmp(path, object, strlen(object)) != 0) {
				rtspReplyError(RTSP_STATUS_SESSION);
				return;
			}
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

		bool RTSPContext::setClientSocket(SOCKET s){
			int sinlen = sizeof(struct sockaddr_in);
			fd = s;
			getpeername(s, (struct sockaddr *)&client, &sinlen);
			return true;
		}

		int handle_rtcp(RTSPContext * ctx, const char * buf, size_t bulen)
		{
			return 0;
		}

		// the rtcp header definition
		struct RTCPHeader{
			unsigned char vps; // version , padding, RC/SC
#define RTCP_Version(hdr)		(((hdr)->vps)>> 6)
#define RTCP_Padding(hdr)		((((hdr)->vps)>> 5) & 0x01)
#define RTCP_RC(hdr)			(((hdr)->vps) & 0x1f)
#define RTCP_SC(hdr)			RTCP_RC(hdr)
			unsigned pt;
			unsigned length;
		};

		void SkipSpace(const char **pp){
			const char *p;
			p = *pp;
			while(*p == ' ' || *p == '\t')
				p++;
			*pp = p;
		}
		void GetWord(char * buf, int buf_size, const char ** pp){
			const char *p;
			char * q;

			p = *pp;
			SkipSpace(&p);
			q = buf;
			while(!isspace(*p) && *p != '\0'){
				if((q - buf) < buf_size - 1)
					*q++ = *p;
				p++;
			}
			if(buf_size > 0)
				*q = '\0';
			*pp = p;
		}

		// the rtsp server thread
		DWORD WINAPI RtspServerThreadProc(LPVOID arg){
			struct timeval to;
			const char * p = NULL; 
			char buf[8192] = {0};
			char cmd[32] = {0}, url[1024] = {0}, protocal[32] = {0};
			int rlen = 0;

			// get the context
			RTSPContext * ctx = (RTSPContext *)arg;
			RTSPMessageHeader header1, *header = &header1;
			RTSPConf * rtspConf  = RTSPConf::GetRTSPConf();
			ctx->setConf(rtspConf);
			// ctx->state == SERVER_STATE_IDLE;   // initially set to IDLE
			// ctx->hasVideo = 0;
			// ctx->rtspWriteMutex = CreateMutex(NULL, FALSE, NULL);

			// XXX: hasVideo is used to sync audio/video
			// This value is increased by 1 for each captured frame until it is greater than zero
			// When this value is greater than zero, audio encoding then starts ...
			// ctx.hasVideo = -(rtspConf->video_fps >> 1); // sor sllow encoders?

			int i = 0, fdmax = 0, active = 0;
			do{
				fd_set rfds;
				FD_ZERO(&rfds);
				FD_SET(ctx->fd, &rfds);
				fdmax = ctx->fd;
#ifdef HOLE_PUNCHING
				for(i = 0; i < 2 * ctx->streamCount; i++){
					FD_SET(ctx->rtpSocket[i], &rfds);
					if(ctx->rtpSocket[i] > fdmax)
						fdmax = ctx->rtpSocket[i];
				}
#endif
				to.tv_sec = 0;
				to.tv_usec = 500000;
				if((active = select(fdmax + 1, &rfds, NULL, NULL, &to)) < 0){
					infoRecorder->logError("[RTSPServer]: select failed: %s, TO QUIT\n",strerror(errno));
					goto quit;
				}
				if(active == 0){
					// try again
					continue;
				}
#ifdef HOLE_PUNCHING
				for(i = 0; i < 2 * ctx->streamCount; i++){
					struct sockaddr_in xsin;
					int xsinlen = sizeof(xsin);

					if(FD_ISSET(ctx->rtpSocket[i], &rfds) == 0){
						continue;
					}
					recvfrom(ctx->rtpSocket[i], buf, sizeof(buf), 0, (struct sockaddr *)&xsin, &xsinlen);
					if(ctx->rtpPortChecked[i] != 0)
						continue;

					// XXX: port should not filp-flop, so check only once
					if(xsin.sin_addr.S_un.S_addr != ctx->client.sin_addr.S_un.S_addr){
						infoRecorder->logError("[RTSPServer]: rtp - client address mismatched? %u.%u.%u.%u != %u.%u.%u.%u\n", NIPQUAD(ctx->client.sin_addr.S_un.S_addr), NIPQUAD(xsin.sin_addr.S_un.S_addr));
						continue;
					}
					if(xsin.sin_port != ctx->rtpPeerPort[i]){
						infoRecorder->logError("[RTSPServer]: rtp - client port reconfigured:%u -> %u.\n", (unsigned int )ntohs(ctx->rtpPeerPort[i]), (unsigned int)ntohs(xsin.sin_port));
						ctx->rtpPeerPort[i] = xsin.sin_port;
					}
					else{
						infoRecorder->logError("[RTSPServer]: rtp - client is not under an NAT, port %d confirmed.\n", (int)ntohs(ctx->rtpPeerPort[i]));
					}
					ctx->rtpPortChecked[i] = 1;
				}
				if(FD_ISSET(ctx->fd, &rfds) == 0){
					continue;
				}
#endif  // HOlLE_PUNCHING
				if((rlen = ctx->rtspGetNext(buf, sizeof(buf)))< 0){
					infoRecorder->logError("[RTSPServer]: GetNext failed. TO QUIT,FD issued, but get NOTHING.\n");
					goto quit;
				}
				// Interleaved binary data?
				if(buf[0] == '$'){
					handle_rtcp(ctx, buf, rlen);
					continue;
				}
				// REQUEST line
#ifdef ENABLE_RTSP_LOG
				infoRecorder->logError("[RTSPServer]: %s.\n", buf);
#endif
				p = buf;
				GetWord(cmd, sizeof(cmd), &p);
				GetWord(url, sizeof(url), &p);
				GetWord(protocal, sizeof(protocal), &p);
				// check protocal
				if(strcmp(protocal, "RTSP/1.0")!= 0){
					infoRecorder->logError("[RTSPServer]: protocol not supported, TO QUIT.\n");
					ctx->rtspReplyError(RTSP_STATUS_VERSION);
					goto quit;
				}
				// read headers
				bzero(header, sizeof(*header));
				do{
					int myseq = -1;
					char mysession[sizeof(header->session_id)] = "";
					if((rlen = ctx->rtspGetNext(buf, sizeof(buf))) < 0){
						infoRecorder->logError("[RTSPServer]: GetNext fialed. TO QUIT, read header failed.\n");
						goto quit;
					}
					if(buf[0] == '\n' || (buf[0] == '\r' && buf[1] == '\n')){
						break;
					}

#ifdef ENABLE_RTSP_LOG
					infoRecorder->logTrace("[RTPSServer]: HEADER: %s.\n", buf);
#endif
					// Special handing to CSeq & Session header
					// ff_rtsp_parser_line cannot handle CSeq & Session properly on windows
					// any more?
					if(strncasecmp("CSeq: ", buf, 6) == 0){
						myseq = strtol(buf + 6, NULL, 10);
					}
					if(strncasecmp("Session: ", buf, 9) == 0){
						strcpy(mysession, buf + 9);
					}
					// 
					ff_rtsp_parse_line(header, buf, NULL, NULL);
					//
					if(myseq > 0 && header->seq <= 0){
						infoRecorder->logError("[RTSPServer]: WARNING, CSeq fixes applied (%d -> %d).\n", header->seq, myseq);
						header->seq = myseq;
					}
					if(mysession[0] != '\0' && header->session_id[0] == '\0'){
						unsigned i;
						for(i = 0; i < sizeof(header->session_id) - 1; i++){
							if(mysession[i] == '\0' || isspace(mysession[i]) || mysession[i] == ';')
								break;
							header->session_id[i] = mysession[i];
						}
						header->session_id[i+1] = '\0';
						infoRecorder->logError("[RTSPServer]: WARANING - Session fixes applied (%s)\n", header->session_id);
					}
				}
				while(1);

				if(header->session_id != NULL){
					char * p = header->session_id;
					while(*p != '\0'){
						if(*p == '\r' || *p == '\n'){
							*p = '\0';
							break;
						}
						p++;
					}
				}

				// handle commands
				ctx->seq = header->seq;
				if(!strcmp(cmd, "DESCRIBE"))
					ctx->rtspCmdDescribe(url);
				else if(!strcmp(cmd, "OPTIONS")){
					ctx->rtspCmdOptions(url);
				}
				else if(!strcmp(cmd, "SETUP")){
					ctx->rtspCmdSetup(url, header, 0);
				}
				else if(!strcmp(cmd, "PLAY")){
					ctx->rtspCmdPlay(url, header);
					// set ctx to encoders
					// need to start the encoding
#ifdef ENABLE_RTSP_LOG
					infoRecorder->logTrace("[RTSPServer]: ctx:%p enable rtsp.\n", ctx);
#endif

					// enable the rtsp when ready.
					ctx->enableGen = true;
				}
				else if(!strcmp(cmd, "PAUSE")){
					infoRecorder->logError("[RTSPServer]: to pause the connection.\n");
					ctx->rtspCmdPause(url, header);
				}
				else if(!strcmp(cmd, "TEARDOWN")){
					infoRecorder->logError("[RTSPServer]: to tear down the connection, set write to net FALSE.\n");
					//gen->setEnableWriteToNet(false);
					ctx->rtspCmdTeardown(url, header, 1);
				}
				else
					ctx->rtspReplyError(RTSP_STATUS_METHOD);
				if(ctx->state == SERVER_STATE_TEARDOWN){
					infoRecorder->logError("[RTSPServer]: STATUS si TEARDOWN.\n");
					//gen->stop();
					//gen->setEnableWriteToNet(false);
					closesocket(ctx->fd);
					break;
				}
			}while(1);
quit:
			ctx->state = SERVER_STATE_TEARDOWN;
			closesocket(ctx->fd);

			ctx->clientDeinit();
			infoRecorder->logError("[RTSPServer]: rtsp client thread terminated.\n");
			return 0;
		}
	}
}
