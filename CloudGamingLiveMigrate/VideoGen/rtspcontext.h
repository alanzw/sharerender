#ifndef __RTSPCONTEXT_H__
#define __RTSPCONTEXT_H__
//#define HAVE_CLOSESOCKET 1
#ifndef __cplusplus__
#define __cplusplus__
#endif
#ifdef __cplusplus__
extern "C" {
#endif
#include "ffmpeg-1.1\rtsp.h"
#include "ffmpeg-1.1\rtspcodes.h"
	int ffio_open_dyn_packet_buf(AVIOContext **, int );
#ifdef __cplusplus__
}
#endif

#include "../VideoUtility/rtspconf.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#define HOLE_PUNCHING  // enable self-implemented hole-punching 
#define RTSP_CHANNEL_MAX 8
#define RTSP_CHANNEL_MAXx2 16

#define IMAGE_SOURCE_CHANNEL_MAX 4 // temp define here
#define SHARE_ENCODER

#define MULTI_CHANNEL

// this definition is a flag to identify the rtsp context support multi-sourced video or not
#define MULTI_SOURCE_SUPPORT

#ifndef NIPQUAD
#define NIPQUAD(x) ((unsigned char *)&(x))[0], \
	((unsigned char *)&(x))[1], \
	((unsigned char *)&(x))[2], \
	((unsigned char *)&(x))[3]
#endif
namespace cg{
	namespace rtsp{
		enum RTSPServerState{
			SERVER_STATE_IDLE = 0,
			SERVER_STATE_READY,
			SERVER_STATE_PLAYING,
			SERVER_STATE_PAUSE,
			SERVER_STATE_TEARDOWN
		};

		struct RTSPContext{
			int channels;
			int id;  // combine the source with rtsp context
			char object[RTSPCONF_OBJECT_SIZE];    // the rtsp service name
			unsigned char tag;

			bool initialized; // 
			bool enableGen;
			SOCKET fd;
			struct sockaddr_in client;

			int state;
			int hasVideo;

			int height, width;
			/// RTSP
			//internal read buffer
			char * rBuffer;
			int rBufHead;
			int rBufTail;
			int rBufSize;

			// for creating SDP
			AVFormatContext *sdpFmtCtx;
			AVStream *sdpVStream[IMAGE_SOURCE_CHANNEL_MAX];
			AVStream *sdpAStream;
			AVCodecContext * sdpVEncoder[IMAGE_SOURCE_CHANNEL_MAX];
			AVCodecContext * sdpAEncoder;
			// for real audio / video encoding
			//
			int seq;
			char *sessionId;
			enum RTSPLowerTransport lowerTransport[RTSP_CHANNEL_MAX];
			AVFormatContext * fmtCtx[RTSP_CHANNEL_MAX];
			AVStream * stream[RTSP_CHANNEL_MAX];
			AVCodecContext * encoder[RTSP_CHANNEL_MAX];
			// streaming 
			int mtu;
			URLContext * rtp[RTSP_CHANNEL_MAX]; // RTP over UDP
			HANDLE rtspWriteMutex; // RTP over RTSP/ TCP

			RTSPConf * rtspConf;
#ifdef HOLE_PUNCHING
			int streamCount;
			SOCKET rtpSocket[RTSP_CHANNEL_MAXx2];

			unsigned short rtpLocalPort[RTSP_CHANNEL_MAXx2];
			unsigned short rtpPeerPort[RTSP_CHANNEL_MAXx2];
			char rtpPortChecked[RTSP_CHANNEL_MAXx2];
#endif

		public:
			void setConf(RTSPConf * conf){ rtspConf = conf; }
			void rtspCleanUp(int retCode);
			int rtspWriteBinData(int streamId, uint8_t *buf, int bufLen);

			// tool functions
			int clientInit(int channels, int width, int height);
			int rtspWrite(const void* buf, size_t count);
			int rtspPrintf(const char* fmt, ...);
#ifdef HOLE_PUNCHING
			int rtpOpenPorts(int streamId);
			int rtpClosePorts(int streamId);
			int rtpWriteBinData(int streamId, uint8_t* buf, int bufLen);
			
#endif

			int rtpNewAVStream(struct sockaddr_in * sin, int streamid, enum AVCodecID codecid);
			int rtspReadInternal();
			int rtspReadText(char* buf, size_t count);
			int rtspReadBinary(char* buf, size_t count);
			int rtspGetNext(char* buf, size_t count);

			void closeAV(AVFormatContext* fctx, AVStream* st, AVCodecContext* cctx,enum RTSPLowerTransport transport );
			void clientDeinit();
			void rtspReplyHeader(enum RTSPStatusCode error_number);
			void rtspReplyError(enum RTSPStatusCode error_number);
			int prepareSdpDescription(char* buf, int bufsize);
			void rtspCmdDescribe(const char* url);
			void rtspCmdOptions(const char* url);
			RTSPTransportField * findTransport(RTSPMessageHeader* h, enum RTSPLowerTransport lower_transport );

			void rtspCmdSetup(const char* url, RTSPMessageHeader* h, int streamId);
			void rtspCmdPlay(const char* url, RTSPMessageHeader* h);
			void rtspCmdPause(const char * rul, RTSPMessageHeader * h);
			void rtspCmdTeardown(const char * rul, RTSPMessageHeader * h, int bruteforce);

			// constructor and destructor
			RTSPContext();
			~RTSPContext();
			RTSPContext & operator=(const RTSPContext &); /// copy constructor

			inline RTSPConf * getRTSPConf(){ 
				if (NULL == rtspConf)
					rtspConf = RTSPConf::GetRTSPConf();
				return rtspConf; 
			}

			inline void setId(int _id){ id = _id;}
			inline int getId(){ return id; }
			bool setClientSocket(SOCKET s);
		};

		DWORD WINAPI RtspServerThreadProc(LPVOID arg);// this is the thread procedure for the rtsp thread
	}
}
#endif
