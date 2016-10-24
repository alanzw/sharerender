#ifndef __ENCODERMANAGER_H__
#define __ENCODERMANAGER_H__



using namespace std;

class EncoderManager{
	static int encoders;
	int poolSize; // the pool size for encoder

	HANDLE encoderLock;
	map<RTSPContext *, RTSPContext *> encoderClients;

	DWORD veThreadId[IMAGE_SOURCE_CHANNEL_MAX];
	HANDLE vEThread[IMAGE_SOURCE_CHANNEL_MAX];
	DWORD aeThreadId;

	// for pts sync between encoders
	HANDLE syncMutex;
	bool syncReset;
	struct timeval syncTv;

	map<LPVOID, Encoder *> vEncoder;  // indexed by the rtspcontext pointer
	//map<LPVOID, EncoderThreadProc> vEncoder;
	//map<LPVOID, EncoderThreadProc> aEncoder;

	map<LPVOID, Encoder *> videoEncoderMap;
	list<Encoder *> cudaEncoderPool;   // the encoder pool for cudaEncoder, unused
	list<Encoder *> x264EncoderPool; // the encoder pool for x264endoer, unused

	list<Encoder *> x264EncoderUnderGo;   // the x264 encoders that is working
	list<Encoder *> cudaEncoderUnderGo;   // the cuda encoders that is working

	bool threadLaunched;
	static EncoderManager * manager;

	// constructors
	EncoderManager();
	~EncoderManager();

public:
	// static functions
	int encoderPtsSync(int sampleRate);
	int encoderRunning();
	int encoderRegisterVEncoder(Encoder * encoder, void * arg);
	int encoderRegisterClient(RTSPContext * rtsp);

	int encoderUnregisterClient(RTSPContext * rtstp);
	int encoderSendPacket(const char * prefix, RTSPContext * rtsp, int channelId, AVPacket *pkt, int64_t encoderPts);
	int encoderSendPacketAll(const char * prefix, int channelId, AVPacket *pkt, int64_t encoderPts);

	static EncoderManager* GetEncoderManager();
	static void ReleaseEncoderManager(EncoderManager * manager);

	// common functions
	int checkEncodingDevice();  // check the hardware, whether it support CUDA and NVENC
	Encoder * getEncoder(ENCODER_TYPE type= X264_ENCODER);     // get a free encoder to work
	//Encoder * getEncoder();
	bool idleEncoder(Encoder *); // move the encoder to pool
	int getEncoders();

	ENCODER_TYPE getAvailableType();
	bool assignEncoder(RTSPContext * context, Encoder * encoder);

	static void Release();
};


#endif