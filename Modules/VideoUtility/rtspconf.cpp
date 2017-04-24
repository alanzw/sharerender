#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
//#include "ccg_win32.h"
#include "videocommon.h"
#include "ccg_config.h"
#include "rtspconf.h"
#include "vconventer.h"
#include "avcodeccommon.h"
#include "../LibCore/InfoRecorder.h"

#define	RTSP_DEF_OBJECT		"/desktop"
#define	RTSP_DEF_TITLE		"Real-Time Desktop"
#define	RTSP_DEF_DISPLAY	":0"
#define	RTSP_DEF_SERVERPORT	8554
#define	RTSP_DEF_PROTO		IPPROTO_UDP

#define	RTSP_DEF_CONTROL_ENABLED	0
#define	RTSP_DEF_CONTROL_PORT		555
#define	RTSP_DEF_CONTROL_PROTO		IPPROTO_TCP
#define	RTSP_DEF_SEND_MOUSE_MOTION	0

#define	RTSP_DEF_VIDEO_CODEC	CODEC_ID_H264
#define	RTSP_DEF_VIDEO_FPS	24

#define	RTSP_DEF_AUDIO_CODEC	CODEC_ID_MP3
#define	RTSP_DEF_AUDIO_BITRATE	128000
#define	RTSP_DEF_AUDIO_SAMPLERATE 44100
#define	RTSP_DEF_AUDIO_CHANNELS	2
#define	RTSP_DEF_AUDIO_DEVICE_FORMAT	AV_SAMPLE_FMT_S16
#define	RTSP_DEF_AUDIO_DEVICE_CH_LAYOUT	AV_CH_LAYOUT_STEREO
#define	RTSP_DEF_AUDIO_CODEC_FORMAT	AV_SAMPLE_FMT_S16
#define	RTSP_DEF_AUDIO_CODEC_CH_LAYOUT	AV_CH_LAYOUT_STEREO

#define	DELIM	" \t\n\r"

//RTSPConf * RTSPConf::getRtspConf(int id){
//	//get the config using id
//	if(configMap!=NULL){
//		RTSPConf * conf = configMap[id];
//		if(conf->initialized)
//			return conf;
//		else
//			return NULL;
//	}
//	return NULL;
//}


//#define ENABLE_RTSOCONF_LOG

namespace cg{

	std::map<std::string, int>RTSPConf::initialized;
	RTSPConf * RTSPConf::conf = NULL;
	std::map<int, RTSPConf *> RTSPConf::configMap;
	std::map<std::string, RTSPConf *>RTSPConf::rtspConfMap;
	HANDLE RTSPConf::rtspConfMutex = NULL;

	std::string RTSPConf::myname;

	RTSPConf * RTSPConf::GetRTSPConf(char * filename){
#ifdef ENABLE_RTSOCONF_LOG
		cg::core::infoRecorder->logTrace("[RTSPConf]:get rtsp config!\n");
#endif
		if(conf == NULL){
			if (filename == NULL){
				cg::core::infoRecorder->logError("[RTSPConf]: NULL config file name and NO instance.\n");
				return NULL;
			}else{
				// return a new one 
#ifdef ENABLE_RTSOCONF_LOG
				cg::core::infoRecorder->logTrace("[RTSPConf]:create new rtsp config\n");
#endif
				myname = std::string(filename);
				conf = new RTSPConf(filename);
			}
		}
		
		return conf;
	}


	RTSPConf::~RTSPConf(){
		
		if (disServerName){
			free(disServerName);
			disServerName = NULL;
		}
	}

	int RTSPConf::do_register(const char * provider, RTSPConf * conf){
		DWORD ret = WaitForSingleObject(rtspConfMutex, INFINITE);
		if (rtspConfMap.find(provider) != rtspConfMap.end()){
			ReleaseMutex(rtspConfMutex);
			cg::core::infoRecorder->logError("[RTSPConf]: duplicated RTSPConf '%s'.\n", provider);
			return -1;
		}
		rtspConfMap[provider] = conf;
		ReleaseMutex(rtspConfMutex);
		conf->myname = provider;
		#ifdef ENABLE_RTSOCONF_LOG
		cg::core::infoRecorder->logError("[RTSPConf]: new rtsp config '%s' registered.\n", provider);
#endif
		return 0;
	}
	void RTSPConf::do_unregister(const char * provider){
		DWORD ret = WaitForSingleObject(rtspConfMutex, INFINITE);
		rtspConfMap.erase(provider);
		ReleaseMutex(rtspConfMutex);
		#ifdef ENABLE_RTSOCONF_LOG
		cg::core::infoRecorder->logError("[RTSPConf]: rtsp config '%s' unregistered.\n", provider);
#endif
		return;
	}
	RTSPConf * RTSPConf::lookup(const char * provider){
		std::map<std::string, RTSPConf *> ::iterator mi;
		RTSPConf * conf = NULL;
		DWORD ret = WaitForSingleObject(rtspConfMutex, INFINITE);
		if ((mi = rtspConfMap.find(provider)) == rtspConfMap.end()){
			ReleaseMutex(rtspConfMutex);
			return NULL;
		}
		conf = mi->second;
		ReleaseMutex(rtspConfMutex);
		return conf;
	}

	void RTSPConf::Release(){
		if (rtspConfMutex){
			CloseHandle(rtspConfMutex);
			rtspConfMutex = NULL;
		}
		if (conf){
			delete conf;
			conf = NULL;
		}
	}

	RTSPConf::RTSPConf(char * filename):ccgConfig(filename){
		rtspConfInit();
		confLoad(filename);
		rtspConfParse();
	}

	int RTSPConf::rtspConfInit() {
		// set 0

		memset(object, 0, RTSPCONF_OBJECT_SIZE);
		memset(title, 0, RTSPCONF_TITLE_SIZE);
		memset(display, 0, RTSPCONF_DISPLAY_SIZE);
		
		memset(video_encoder_name, 0, RTSPCONF_CODECNAME_SIZE + 1);
		memset(video_decoder_name, 0, RTSPCONF_CODECNAME_SIZE + 1);
		video_encoder_codec = NULL;
		video_decoder_codec = NULL;


		memset(audio_encoder_name, 0, RTSPCONF_CODECNAME_SIZE + 1);
		memset(audio_decoder_name, 0, RTSPCONF_CODECNAME_SIZE + 1);

		audio_encoder_codec = NULL;
		audio_decoder_codec = NULL;

		disServerName = NULL;

		//initialized = 1;
		strncpy(object, RTSP_DEF_OBJECT, strlen(RTSP_DEF_OBJECT));
		strncpy(title, RTSP_DEF_TITLE, RTSPCONF_TITLE_SIZE);
		strncpy(display, RTSP_DEF_DISPLAY, RTSPCONF_DISPLAY_SIZE);
		serverPort = RTSP_DEF_SERVERPORT;
		proto = RTSP_DEF_PROTO;
		// controller
		ctrlEnable = RTSP_DEF_CONTROL_ENABLED;
		ctrlPort = RTSP_DEF_CONTROL_PORT;
		ctrlProto = RTSP_DEF_CONTROL_PROTO;
		sendMouseMotion = RTSP_DEF_SEND_MOUSE_MOTION;
		//
		video_fps = RTSP_DEF_VIDEO_FPS;
		audio_bitrate = RTSP_DEF_AUDIO_BITRATE;
		audio_samplerate = RTSP_DEF_AUDIO_SAMPLERATE;
		audio_channels = RTSP_DEF_AUDIO_CHANNELS;
		audio_device_format = RTSP_DEF_AUDIO_DEVICE_FORMAT;
		audio_device_channel_layout = RTSP_DEF_AUDIO_DEVICE_CH_LAYOUT;
		audio_codec_format = RTSP_DEF_AUDIO_CODEC_FORMAT;
		audio_device_channel_layout = RTSP_DEF_AUDIO_CODEC_CH_LAYOUT;

		//conf->vgo = new vector<string>;
		vso = new std::vector<std::string>;
		if (rtspConfMutex == NULL){
			rtspConfMutex = CreateMutex(NULL, FALSE, NULL);
		}
		//

		AVCodecCommon::Init();

		return 0;
	}

	int RTSPConf::
		rtspConfLoadCodec(const char *key, const char *value,
		const char **names, AVCodec **codec, AVCodec *(*finder)(const char **, enum AVCodecID)) {
			#ifdef ENABLE_RTSOCONF_LOG
			cg::core::infoRecorder->logTrace("[RTSPConf]: rtsp config load codec called.\n");
#endif
			//
			int idx = 0;
			char buf[1024], *saveptr;
			const char *token;
			//
			strncpy(buf, value, sizeof(buf));
			if ((token = strtok_r(buf, DELIM, &saveptr)) == NULL) {
				cg::core::infoRecorder->logError("# RTSP[config]: no codecs specified for %s.\n", key);
				return -1;
			}
			do {
				names[idx] = _strdup(token);
			} while (++idx<RTSPCONF_CODECNAME_SIZE && (token = strtok_r(NULL, DELIM, &saveptr)) != NULL);
			names[idx] = NULL;
			//
			if ((*codec = finder(names, AV_CODEC_ID_NONE)) == NULL) {
				cg::core::infoRecorder->logError("# RTSP[config]: no available %s codecs (%s).\n", key, value);
				return -1;
			}
			//
			#ifdef ENABLE_RTSOCONF_LOG
			cg::core::infoRecorder->logError("# RTSP[config]: %s = %s (%s)\n", key, (*codec)->name,
				(*codec)->long_name == NULL ? "N/A" : (*codec)->long_name);
#endif
			return 0;
	}


#if 0
	int RTSPConf::configParseClient(){
		char *ptr = NULL, buf[1024] = {0};
		int value = 0;

	}

	int RTSPConf::configParseDisServer(){
		char *ptr = NULL, buf[1024] = {0};
		int value = 0;
	}
	int RTSPConf::configParseLogic(){
		char *ptr = NULL, buf[1024] = {0};
		int value = 0;
	}
	int RTSPConf::configParseRender(){
		char *ptr = NULL, buf[1024] = {0};
		int value = 0;
	}
#endif

	int RTSPConf::
		rtspConfParse() {
			char *ptr = NULL, buf[1024] = {0};
			int v;
#ifdef ENABLE_RTSOCONF_LOG
			cg::core::infoRecorder->logError("[RTSPConf]: rtsp config parse called.\n");
#endif
			//read the distributor server name
			if ((ptr = confReadV("dis-server-name", buf, sizeof(buf))) != NULL){
				disServerName = _strdup(ptr);
			}

			//
			if ((ptr = confReadV("base-object", buf, sizeof(buf))) != NULL) {
				strncpy(object, ptr, RTSPCONF_OBJECT_SIZE);
			}
			//
			if ((ptr = confReadV("title", buf, sizeof(buf))) != NULL) {
				strncpy(title, ptr, RTSPCONF_TITLE_SIZE);
			}
			//
			if ((ptr = confReadV("display", buf, sizeof(buf))) != NULL) {
				strncpy(display, ptr, RTSPCONF_DISPLAY_SIZE);
			}
			//
			v = confReadInt("rtsp-port");
			if (v <= 0 || v >= 65536) {
				cg::core::infoRecorder->logError("# RTSP[config]: invalid rtsp server port %d, use default: 8554\n", v);
				serverPort = 8554;
			}
			else{
#ifdef ENABLE_RTSOCONF_LOG
			cg::core::infoRecorder->logError("[RTSPConf]: server port:%d.\n", v);
#endif
			serverPort = v;
			}
			v = confReadInt("loader-port");
			if(v <= 0 || v >= 65536){
				cg::core::infoRecorder->logError("# RTSP[config]: invalied loader port %d, use default: 8556.\n", v);
				loaderPort = 8556;
			}
			else{
				loaderPort = v;
			}
			v = confReadInt("graphic-port");
			if(v <= 0 || v >= 65535){
				cg::core::infoRecorder->logError("# RTSP[config]: invalid graphic port %d, use default: 60000.\n");
				graphicPort = 60000;
			}else{
				graphicPort = v;
			}

			// read the distributor server port
			v = confReadInt("dis-port");
			if (v <= 0 || v >= 65535){
				cg::core::infoRecorder->logError("# RTSP[config]: invalid distributor server port %d, use default: 8557\n", v);
				disPort = 8557;
			}else
				disPort = v;
#ifdef ENABLE_RTSOCONF_LOG
			cg::core::infoRecorder->logError("[RTSPConf]: distributor port:%d.\n", v);
#endif
			// read rtsp proto
			ptr = confReadV("rtsp-proto", buf, sizeof(buf));
			if (ptr == NULL || strcmp(ptr, "tcp") != 0) {
				proto = IPPROTO_UDP;
#ifdef ENABLE_RTSOCONF_LOG
				cg::core::infoRecorder->logError("# RTSP[config]: using 'udp' for RTP flows.\n");
#endif
			}
			else {
				proto = IPPROTO_TCP;
#ifdef ENABLE_RTSOCONF_LOG
				cg::core::infoRecorder->logError("# RTSP[config]: using 'tcp' for RTP flows.\n");
#endif
			}
			// read control enable value
			ctrlEnable = confReadBool("control-enabled", 0);

			if (ctrlEnable != 0) {
				// read the control connection port
				v = confReadInt("control-port");
				if (v <= 0 || v >= 65536) {
					cg::core::infoRecorder->logError("# RTSP[config]: invalid control port %d, use default: 8555\n", v);
					ctrlPort = 8555;
				}else
					ctrlPort = v;
#ifdef ENABLE_RTSOCONF_LOG
				cg::core::infoRecorder->logError("# RTSP[config]: controller port = %d\n", ctrlport);
#endif
				// read the protocol for control connection, default use TCP
				ptr = confReadV("control-proto", buf, sizeof(buf));
				if (ptr == NULL || strcmp(ptr, "tcp") == 0) {
					ctrlProto = IPPROTO_TCP;
#ifdef ENABLE_RTSOCONF_LOG
					cg::core::infoRecorder->logError("# RTSP[config]: controller via 'tcp' protocol.\n");
#endif
				}
				else {
					ctrlProto = IPPROTO_UDP;
#ifdef ENABLE_RTSOCONF_LOG
					cg::core::infoRecorder->logError("# RTSP[config]: controller via 'udp' protocol.\n");
#endif
				}
				//
				sendMouseMotion = confReadBool("control-send-mouse-motion", 1);
				relativeMouse = confReadBool("control-relative-mouse-mode", 0);
			}
			// video-encoder, audio-encoder, video-decoder, and audio-decoder
			if ((ptr = confReadV("video-encoder", buf, sizeof(buf))) != NULL) {
				#ifdef ENABLE_RTSOCONF_LOG
				cg::core::infoRecorder->logError("[RTSPConf]: config parse, video-encoder:%s\n", buf);
#endif
				if (rtspConfLoadCodec("video-encoder", ptr,
					(const char**)video_encoder_name,
					&video_encoder_codec,
					AVCodecCommon::AVCodecFindEncoder) < 0)
					return -1;
			}
			else{
				cg::core::infoRecorder->logError("[RTSPConf]: config parse, NULL video-encoder.\n");
			}
#if 1
			if ((ptr = confReadV("video-decoder", buf, sizeof(buf))) != NULL) {
				if (rtspConfLoadCodec("video-decoder", ptr,
					(const char**)video_decoder_name,
					&video_decoder_codec,
					AVCodecCommon::AVCodecFindDecoder) < 0)
					return -1;
			}
#endif
			
#if 0
			if ((ptr = confReadV("audio-encoder", buf, sizeof(buf))) != NULL) {
				if (rtspConfLoadCodec("audio-encoder", ptr,
				(const char**)audio_encoder_name,
				&audio_encoder_codec,
				AVCodecCommon::AVO) < 0)
				return -1;
			}
			if ((ptr = confReadV("audio-decoder", buf, sizeof(buf))) != NULL) {
				if (rtspconfLoadCodec("audio-decoder", ptr,
					(const char**)audio_decoder_name,
					&audio_decoder_codec,
					ga_avcodec_find_decoder) < 0)
					return -1;
			}
#endif
			//
			v = confReadInt("video-fps");
			if (v <= 0 || v > 120) {
				cg::core::infoRecorder->logError("# RTSP[conf]: video-fps out-of-range %d (valid: 1-120)\n", v);
				//return -1;
				video_fps = 24;
			}
			else
				video_fps = v;
			//
			ptr = confReadV("video-renderer", buf, sizeof(buf));
			if (ptr != NULL && strcmp(ptr, "software") == 0) {
				video_renderer_software = 1;
			}
			else {
				video_renderer_software = 0;
			}
#if 0
			//
			v = confReadInt("audio-bitrate");
			if (v <= 0 || v > 1024000) {
				cg::core::infoRecorder->logError("# RTSP[config]: audio-bitrate out-of-range %d (valid: 1-1024000)\n", v);
				return -1;
			}
			audio_bitrate = v;
			//
			v = confReadInt("audio-samplerate");
			if (v <= 0 || v > 1024000) {
				cg::core::infoRecorder->logError("# RTSP[config]: audio-samplerate out-of-range %d (valid: 1-1024000)\n", v);
				return -1;
			}
			audio_samplerate = v;
			//
			v = confReadInt("audio-channels");
			if (v < 1) {
				cg::core::infoRecorder->logError("# RTSP[config]: audio-channels must be greater than zero (%d).\n", v);
				return -1;
			}
			audio_channels = v;
			//
			if ((ptr = confReadV("audio-device-format", buf, sizeof(buf))) == NULL) {
				cg::core::infoRecorder->logError("# RTSP[config]: no audio device format specified.\n");
				return -1;
			}
			if ((audio_device_format = av_get_sample_fmt(ptr)) == AV_SAMPLE_FMT_NONE) {
				cg::core::infoRecorder->logError("# RTSP[config]: unsupported audio device format '%s'\n", ptr);
				return -1;
			}
			//
			if ((ptr = confReadV("audio-device-channel-layout", buf, sizeof(buf))) == NULL) {
				cg::core::infoRecorder->logError("# RTSP[config]: no audio device channel layout specified.\n");
				return -1;
			}
			if ((audio_device_channel_layout = av_get_channel_layout(ptr)) == 0) {
				cg::core::infoRecorder->logError("# RTSP[config]: unsupported audio device channel layout '%s'\n", ptr);
				return -1;
			}
			//
			if ((ptr = confReadV("audio-codec-format", buf, sizeof(buf))) == NULL) {
				cg::core::infoRecorder->logError("# RTSP[config]: no audio codec format specified.\n");
				return -1;
			}
			if ((audio_codec_format = av_get_sample_fmt(ptr)) == AV_SAMPLE_FMT_NONE) {
				cg::core::infoRecorder->logError("# RTSP[config]: unsupported audio codec format '%s'\n", ptr);
				return -1;
			}
			//
			if ((ptr = confReadV("audio-codec-channel-layout", buf, sizeof(buf))) == NULL) {
				cg::core::infoRecorder->logError("# RTSP[config]: no audio codec channel layout specified.\n");
				return -1;
			}
			if ((audio_codec_channel_layout = av_get_channel_layout(ptr)) == 0) {
				cg::core::infoRecorder->logError("# RTSP[config]: unsupported audio codec channel layout '%s'\n", ptr);
				return -1;
			}
#endif
			// LAST: video-specific parameters
			if (confMapSize("video-specific") > 0) {
				//
				confMapReset("video-specific");
				for (ptr = confMapKey("video-specific", buf, sizeof(buf));
					ptr != NULL;
					ptr = confMapNextKey("video-specific", buf, sizeof(buf))) {
						//
						char *val, valbuf[1024];
						val = confMapValue("video-specific", valbuf, sizeof(valbuf));
						if (val == NULL || *val == '\0')
							continue;
						vso->push_back(ptr);
						vso->push_back(val);
#ifdef ENABLE_RTSOCONF_LOG
						cg::core::infoRecorder->logError("# RTSP[config]: video specific option: %s = %s\n",
							ptr, val);
#endif
				}
			}
			return 0;
	}

	void RTSPConf::
		rtspConfResolveServer(const char *servername) {
			struct in_addr addr;
			struct hostent *hostEnt;
			if ((addr.s_addr = inet_addr(servername)) == INADDR_NONE) {
				if ((hostEnt = gethostbyname(servername)) == NULL) {
					bzero(&sin.sin_addr, sizeof(sin.sin_addr));
					return;
				}
				bcopy(hostEnt->h_addr, (char *)&addr.s_addr, hostEnt->h_length);
			}
			sin.sin_addr = addr;
			return;
	}

}