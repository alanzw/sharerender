#ifndef __RTSPCONF_H__
#define __RTSPCONF_H__
#ifndef __cplusplus__
#define __cplusplus__
#endif
#ifdef __cplusplus__
extern "C" {
#endif // __cplusplus__
#include <libavcodec\avcodec.h>
#ifdef __cplusplus__
}
#endif  // __cplusplus__
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <WinSock2.h>

#include <vector>
#include "ccg_config.h"

#define RTSPCONF_OBJECT_SIZE 128
#define RTSPCONF_TITLE_SIZE 256
#define RTSPCONF_DISPLAY_SIZE 16
#define RTSPCONF_PROTO_SIZE 8
#define RTSPCONF_CODECNAME_SIZE 8


#define DEAULT_CONFIG_NAME "config/server.renderpool.conf"
#define DEFAULT_CLIENT_CONFIG_FILE "config/client.rel.conf"
namespace cg{
	struct RTSPConf : ccgConfig{
		static std::map<std::string, int> initialized;
		static HANDLE rtspConfMutex;
		static std::map<std::string, RTSPConf *> rtspConfMap;
		static std::string myname;

		static int do_register(const char * provider, RTSPConf * conf);
		static void do_unregister(const char * provider);
		static RTSPConf * lookup(const char * provider);
		static const char * name(){
			return myname.c_str();
		}

		char object[RTSPCONF_OBJECT_SIZE];
		char title[RTSPCONF_TITLE_SIZE];
		char display[RTSPCONF_DISPLAY_SIZE];

		char * servername;  // render server name
		char * distributorname;   // the distributor
		char * logic_servername; // the logic server

		struct sockaddr_in dis_sin;
		struct sockaddr_in sin;
		struct sockaddr_in logic_sin;   // sockaddr for the logic connection
		int logic_serverport;
		int dis_port;
		int serverport;

		char proto; // transport layer tcp = 6; udp = 17

		// for controller
		int ctrlenable;
		int ctrlport;
		char ctrlproto;		// transport layer tcp = 6; udp = 17
		int sendmousemotion;

		char *video_encoder_name[RTSPCONF_CODECNAME_SIZE + 1];
		AVCodec *video_encoder_codec;
		char *video_decoder_name[RTSPCONF_CODECNAME_SIZE + 1];
		AVCodec *video_decoder_codec;
		int video_fps;
		int video_renderer_software;	// 0 - use HW renderer, otherwise SW
		//
		char *audio_encoder_name[RTSPCONF_CODECNAME_SIZE + 1];
		AVCodec *audio_encoder_codec;
		char *audio_decoder_name[RTSPCONF_CODECNAME_SIZE + 1];
		AVCodec *audio_decoder_codec;
		int audio_bitrate;
		int audio_samplerate;
		int audio_channels;	// XXX: AVFrame->channels is int64_t, use with care
		AVSampleFormat audio_device_format;
		int64_t audio_device_channel_layout;
		AVSampleFormat audio_codec_format;
		int64_t audio_codec_channel_layout;

		std::vector<std::string> *vso;	// video specific options

		int rtspConfInit();
		int rtspConfParse();
		void rtspConfResolveServer(const char * serverName);

		int rtspConfLoadCodec(const char *key, const char * value,
			const char **names, AVCodec **codec, AVCodec *(*finder)(const char **, enum AVCodecID));
		static std::map<int , RTSPConf *> configMap;

		//static RTSPConf * getRtspConf(int id);
		static RTSPConf * conf;

		void setLogicAddr(struct sockaddr_in * s){
			memcpy(&logic_sin, s, sizeof(struct sockaddr_in));
		}
		void setLogicPort(int p){
			logic_serverport = p;
		}
		void setLogicServerName(char * name){
			logic_servername = _strdup(name);
		}

		static void Release();
		static RTSPConf * GetRTSPConf(char * filename = NULL);

		RTSPConf(char *filename);
		~RTSPConf();
	};
}

#endif
