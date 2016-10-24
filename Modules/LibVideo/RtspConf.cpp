#include "Commonwin32.h"
#include "Config.h"
#include "RtspConf.h"
#include "../LibCore/Log.h"
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



map<string, int>RTSPConf::initialized;
RTSPConf * RTSPConf::conf;
map<int, RTSPConf *> RTSPConf::configMap;
map<string, RTSPConf *>RTSPConf::rtspConfMap;
HANDLE RTSPConf::rtspConfMutex;

RTSPConf::~RTSPConf(){
	if (servername){
		free(servername);
		servername = NULL;
	}
	if (distributorname){
		free(distributorname);
		distributorname = NULL;
	}
	if (logic_servername){
		free(logic_servername);
		logic_servername = NULL;
	}
}

int RTSPConf::do_register(const char * provider, RTSPConf * conf){
	DWORD ret = WaitForSingleObject(rtspConfMutex, INFINITE);
	if (rtspConfMap.find(provider) != rtspConfMap.end()){
		ReleaseMutex(rtspConfMutex);
		infoRecorder->logError("[RTSPConf]: duplicated RTSPConf '%s'.\n", provider);
		return -1;
	}
	rtspConfMap[provider] = conf;
	ReleaseMutex(rtspConfMutex);
	conf->myname = provider;
	infoRecorder->logTrace("[RTSPConf]: new rtsp config '%s' registered.\n", provider);
	return 0;
}
void RTSPConf::do_unregister(const char * provider){
	DWORD ret = WaitForSingleObject(rtspConfMutex, INFINITE);
	rtspConfMap.erase(provider);
	ReleaseMutex(rtspConfMutex);
	infoRecorder->logTrace("[RTSPConf]: rtsp config '%s' unregistered.\n", provider);
	return;
}
RTSPConf * RTSPConf::lookup(const char * provider){
	map<string, RTSPConf *> ::iterator mi;
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
	
	//initialized = 1;
//	configMap[id] = this;
	rtspConfInit();
	confLoad(filename);
	//rtspConfParse();
}

int RTSPConf::rtspConfInit() {
	//initialized = 1;
	strncpy(object, RTSP_DEF_OBJECT, RTSPCONF_OBJECT_SIZE);
	strncpy(title, RTSP_DEF_TITLE, RTSPCONF_TITLE_SIZE);
	strncpy(display, RTSP_DEF_DISPLAY, RTSPCONF_DISPLAY_SIZE);
	serverport = RTSP_DEF_SERVERPORT;
	proto = RTSP_DEF_PROTO;
	// controller
	ctrlenable = RTSP_DEF_CONTROL_ENABLED;
	ctrlport = RTSP_DEF_CONTROL_PORT;
	ctrlproto = RTSP_DEF_CONTROL_PROTO;
	sendmousemotion = RTSP_DEF_SEND_MOUSE_MOTION;
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
	vso = new vector<string>;
	if (rtspConfMutex == NULL){
		rtspConfMutex = CreateMutex(NULL, FALSE, NULL);
	}
	//
	return 0;
}

int RTSPConf::
rtspConfLoadCodec(const char *key, const char *value,
const char **names, AVCodec **codec, AVCodec *(*finder)(const char **, enum AVCodecID)) {
	infoRecorder->logTrace("[RTSPConf]: rtsp config load codec called.\n");
	Log::logscreen("[RTSPConf]: rtsp config load codec called.\n");
	//
	int idx = 0;
	char buf[1024], *saveptr;
	const char *token;
	//
	strncpy(buf, value, sizeof(buf));
	if ((token = strtok_r(buf, DELIM, &saveptr)) == NULL) {
		infoRecorder->logTrace("# RTSP[config]: no codecs specified for %s.\n", key);
		return -1;
	}
	do {
		names[idx] = _strdup(token);
	} while (++idx<RTSPCONF_CODECNAME_SIZE && (token = strtok_r(NULL, DELIM, &saveptr)) != NULL);
	names[idx] = NULL;
	//
	if ((*codec = finder(names, AV_CODEC_ID_NONE)) == NULL) {
		infoRecorder->logTrace("# RTSP[config]: no available %s codecs (%s).\n", key, value);
		return -1;
	}
	//
	infoRecorder->logTrace("# RTSP[config]: %s = %s (%s)\n", key, (*codec)->name,
		(*codec)->long_name == NULL ? "N/A" : (*codec)->long_name);
	return 0;
}

int RTSPConf::
rtspConfParse() {
	char *ptr, buf[1024];
	int v;
	infoRecorder->logTrace("[RTSPConf]: rtsp config parse called.\n");
	//
	//rtspConfInit();
	//read the distributor server name
	if ((ptr = confReadV("distributor-name", buf, sizeof(buf))) != NULL){
		distributorname = _strdup(ptr);
		Log::logscreen("[RTSPConf]: distributor name:%s\n.", distributorname);
	}

	// read the server render server name
	if ((ptr = confReadV("server-name", buf, sizeof(buf))) != NULL) {
		servername = _strdup(ptr);
		Log::logscreen("[RTSPConf]: server name:%s\n.", servername);
	}
	// read the logic server name
	if ((ptr = confReadV("logic-server-name", buf, sizeof(buf))) != NULL) {
		logic_servername = _strdup(ptr);
		Log::logscreen("[RTSPConf]: logic server name:%s\n.", logic_servername);
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
	v = confReadInt("server-port");
	if (v <= 0 || v >= 65536) {
		infoRecorder->logTrace("# RTSP[config]: invalid server port %d\n", v);
		return -1;
	}
	Log::logscreen("[RTSPConf]: server port:%d.\n", v);
	serverport = v;

	// read the distributor server port
	v = confReadInt("distributor-port");
	if (v <= 0 || v >= 65535){
		infoRecorder->logTrace("# RTSP[config]: invalid distributor server port %d\n", v);
		return -1;
	}
	dis_port = v;
	Log::logscreen("[RTSPConf]: distributor port:%d.\n", v);

	v = confReadInt("logic-server-port");
	if (v <= 0 || v >= 65535){
		infoRecorder->logTrace("# RTSP[config]: invalid logic server port %d\n", v);
		return -1;
	}
	Log::logscreen("[RTSPConf]: logic server port:%d.\n", v);
	logic_serverport = v;
	//
	ptr = confReadV("proto", buf, sizeof(buf));
	if (ptr == NULL || strcmp(ptr, "tcp") != 0) {
		proto = IPPROTO_UDP;
		infoRecorder->logTrace("# RTSP[config]: using 'udp' for RTP flows.\n");
	}
	else {
		proto = IPPROTO_TCP;
		infoRecorder->logTrace("# RTSP[config]: using 'tcp' for RTP flows.\n");
	}
	//
	ctrlenable = confReadBool("control-enabled", 0);
	//
	if (ctrlenable != 0) {
		//
		v = confReadInt("control-port");
		if (v <= 0 || v >= 65536) {
			infoRecorder->logTrace("# RTSP[config]: invalid control port %d\n", v);
			return -1;
		}
		ctrlport = v;
		infoRecorder->logTrace("# RTSP[config]: controller port = %d\n", ctrlport);
		//
		ptr = confReadV("control-proto", buf, sizeof(buf));
		if (ptr == NULL || strcmp(ptr, "tcp") != 0) {
			ctrlproto = IPPROTO_UDP;
			infoRecorder->logTrace("# RTSP[config]: controller via 'udp' protocol.\n");
		}
		else {
			ctrlproto = IPPROTO_TCP;
			infoRecorder->logTrace("# RTSP[config]: controller via 'tcp' protocol.\n");
		}
		//
		sendmousemotion = confReadBool("control-send-mouse-motion", 1);
	}
	// video-encoder, audio-encoder, video-decoder, and audio-decoder
	if ((ptr = confReadV("video-encoder", buf, sizeof(buf))) != NULL) {
		infoRecorder->logTrace("[RTSPConf]: config parse, video-encoder:%s\n", buf);
		Log::logscreen("[RTSPConf]: config parse, video-encoder:%s\n", buf);
		if (rtspConfLoadCodec("video-encoder", ptr,
			(const char**)video_encoder_name,
			&video_encoder_codec,
			AVCodecCommon::AVCodecFindEncoder) < 0)
			return -1;
	}
	else{
		infoRecorder->logTrace("[RTSPConf]: config parse, NULL video-encoder.\n");
		Log::logscreen("[RTSPConf]: config parse, NULL video-encoder.\n");
	}
#if 0
	if ((ptr = confReadV("video-decoder", buf, sizeof(buf))) != NULL) {
		if (rtspConfLoadCodec("video-decoder", ptr,
			(const char**)conf->video_decoder_name,
			&conf->video_decoder_codec,
			AVCodecCommon::AVCodecFindEncoder) < 0)
			return -1;
	}
#endif
	if ((ptr = confReadV("audio-encoder", buf, sizeof(buf))) != NULL) {
		/*if (rtspConfLoadCodec("audio-encoder", ptr,
			(const char**)audio_encoder_name,
			&audio_encoder_codec,
			AVCodecCommon::AVO) < 0)
			return -1;*/
	}
#if 0
	if ((ptr = confReadV("audio-decoder", buf, sizeof(buf))) != NULL) {
		if (rtspconf_load_codec("audio-decoder", ptr,
			(const char**)conf->audio_decoder_name,
			&conf->audio_decoder_codec,
			ga_avcodec_find_decoder) < 0)
			return -1;
	}
#endif
	//
	v = confReadInt("video-fps");
	if (v <= 0 || v > 120) {
		infoRecorder->logTrace("# RTSP[conf]: video-fps out-of-range %d (valid: 1-120)\n", v);
		return -1;
	}
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
		infoRecorder->logTrace("# RTSP[config]: audio-bitrate out-of-range %d (valid: 1-1024000)\n", v);
		return -1;
	}
	audio_bitrate = v;
	//
	v = confReadInt("audio-samplerate");
	if (v <= 0 || v > 1024000) {
		infoRecorder->logTrace("# RTSP[config]: audio-samplerate out-of-range %d (valid: 1-1024000)\n", v);
		return -1;
	}
	audio_samplerate = v;
	//
	v = confReadInt("audio-channels");
	if (v < 1) {
		infoRecorder->logTrace("# RTSP[config]: audio-channels must be greater than zero (%d).\n", v);
		return -1;
	}
	audio_channels = v;
	//
	if ((ptr = confReadV("audio-device-format", buf, sizeof(buf))) == NULL) {
		infoRecorder->logTrace("# RTSP[config]: no audio device format specified.\n");
		return -1;
	}
	if ((audio_device_format = av_get_sample_fmt(ptr)) == AV_SAMPLE_FMT_NONE) {
		infoRecorder->logTrace("# RTSP[config]: unsupported audio device format '%s'\n", ptr);
		return -1;
	}
	//
	if ((ptr = confReadV("audio-device-channel-layout", buf, sizeof(buf))) == NULL) {
		infoRecorder->logTrace("# RTSP[config]: no audio device channel layout specified.\n");
		return -1;
	}
	if ((audio_device_channel_layout = av_get_channel_layout(ptr)) == 0) {
		infoRecorder->logTrace("# RTSP[config]: unsupported audio device channel layout '%s'\n", ptr);
		return -1;
	}
	//
	if ((ptr = confReadV("audio-codec-format", buf, sizeof(buf))) == NULL) {
		infoRecorder->logTrace("# RTSP[config]: no audio codec format specified.\n");
		return -1;
	}
	if ((audio_codec_format = av_get_sample_fmt(ptr)) == AV_SAMPLE_FMT_NONE) {
		infoRecorder->logTrace("# RTSP[config]: unsupported audio codec format '%s'\n", ptr);
		return -1;
	}
	//
	if ((ptr = confReadV("audio-codec-channel-layout", buf, sizeof(buf))) == NULL) {
		infoRecorder->logTrace("# RTSP[config]: no audio codec channel layout specified.\n");
		return -1;
	}
	if ((audio_codec_channel_layout = av_get_channel_layout(ptr)) == 0) {
		infoRecorder->logTrace("# RTSP[config]: unsupported audio codec channel layout '%s'\n", ptr);
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
			infoRecorder->logTrace("# RTSP[config]: video specific option: %s = %s\n",
				ptr, val);
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
