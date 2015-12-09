#ifndef __X264ENCODER__
#define __X264ENCODER__

#define ENCODING_MOD_BASE 2

class X264Encoder:public Encoder{
	int outputW;
	int outputH;

	int encoder_width, encoder_height;

	char * srcpipename;
    pipeline * sourcePipe;
	HANDLE avcodecOpenMutex; // avcodec_open / close is not thread-safe
	char * pic_data;

	AVCodec *videoEncoderCodec;
	AVCodec *audioEncoderCodec;
	AVCodecContext * codecContext;

	RTSPContext * rtspContext;
	RTSPConf * rtspConf;

	// temp variables
	HANDLE condMutex, cond;
	struct pooldata * data;
	AVFrame * pic_in;
	unsigned char * pic_in_buf;
	int pic_in_size;
	long long basePts, newPts, pts, ptsSync;
	unsigned char * nalbuf, *nalbuf_a;
	int nalbuf_size, nalign;
	int video_written;
	int rtp_id;

	// distribution of the encoder

	int audio_bitrate;
	int audio_samplerate;
	int audio_channels;	// XXX: AVFrame->channels is int64_t, use with care

	char frameIndex;
public:
	X264Encoder();
	~X264Encoder();
	inline int getOutW(){ return outputW; }
	inline int getOutH(){ return outputH; }
	inline void setOutW(int w){ outputW = w; }
	inline void setOutH(int h){ outputH = h; }
	void InitEncoder(RTSPConf * conf, pipeline * pipe);
	AVCodecContext * InitEncoder(AVCodecContext * ctx, AVCodec *codec, int width, int height, int fps, vector<string> *vso);
	AVCodecContext * InitEncoder(AVCodecContext * ctx, AVCodec *codec, int bitrate, int samplerate, int channels, AVSampleFormat format, uint64_t chlayout);
	
	inline char getFrameIndex(){ return frameIndex; }
	inline void setRTSPContext(RTSPContext * c){ rtspContext = c; }
	inline void setRtspConf(RTSPConf * conf){ rtspConf = conf; }

#if 0
	void CloaseEncoder(AVCodecContext *ctx); 
	AVCodec * FindDecoder(const char **names, enum AVCodecID cid);
	AVCodec * FindEncoder(const char ** names, enum AVCodecID cid);

	AVStream * AVFormatNewStream(AVFormatContext * ctx, int id, AVCodec * codec);
	struct SwsContext * SwscaleInit(PixelFormat format, int inW, int inH, int outW, int outH);
	AVFormatContext * FormatInit(const char * filename);
	AVFormatContext * RtpInit(const char * url);
#endif
    pipeline * getSourcePipe(){ return sourcePipe; }
    bool registerSourcePipe(pipeline * pipe){
		if (pipe == NULL){
			Log::logscreen("[X264Encoder]: NULL pipe specified in register source pipe.\n");
			return false;
		}
        if( sourcePipe == NULL ){
            sourcePipe = pipe;
            return true;
        }
        else{
			Log::logscreen("[X264Encoder]: null source pipe.\n");
            return false;
        }
		return true;
    }	

	virtual void setSrcPipeName(char * name){
		srcpipename = _strdup(name);
	}

	// functions from parent class
	virtual int init(void * arg, pipeline * pipe);    // initilizing both the encoder
	virtual int startEncoding()    // start encoding
	{
		return this->start();
	}
	virtual void idle(){
		
	}

	virtual void setBitrate(int bitrate){
		//this->
	}
	virtual bool setInputBuffer(char * buf){
		
		return true;
	}
	virtual bool setBufferInsideGpu(char * p){

		return true;
	}

	// thread functions
	virtual BOOL stop();
	virtual void run();
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void onThreadStart(LPVOID param = NULL);
	virtual void onQuit();

	int sendPacketAll(int channleId, AVPacket * pkt, int64_t encoderPts);
	int sendPacket(int channelId, RTSPContext * rtsp, AVPacket *pkt, int64_t encoderPts);
};


#endif
