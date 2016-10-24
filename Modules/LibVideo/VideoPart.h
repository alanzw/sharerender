#ifndef __VIDEO_PART_H__
#define __VIDEO_PART_H__

/// this is just for the adaptor for video in stream server
using namespace std;

#define SOURCES 1

#define MAX_WIDTH 1024
#define MAX_HEIGHT 1024
#define RENDERPOOL_CONFIG "config\\server.renderpool.conf"

bool netStarted = false;

struct NetParam{
	SOCKET sock;
	SOCKADDR_IN remoteAddr;
	DWORD remotePort;

	NetParam(SOCKET s);
	static DWORD portOff;  // the port offset for all the thread
};

class GlobalManager{
    GlobalManager();
	const char * configFile;

	LPDIRECT3DDEVICE9 d3dDevice;
	int windowW, windowH;
	HANDLE deviceMutex;

public:
    EncoderManager * encoderManager;
    ChannelManager * channelManager;  
    //EncoderConfig * encoderConfig;
    Filter * filter;

	RTSPConf * rtspConf;

    ~GlobalManager();
    static GlobalManager * m_manager;
    static GlobalManager * GetGlobalManager(){
       if(!m_manager)
            return new GlobalManager(); 
       else 
            return m_manager;
    }
    // centeralized scheduling
    bool init(char * config){
        channelManager = ChannelManager::GetChannelManager();
        if(channelManager == NULL){
            // error
			infoRecorder->logTrace("[GlobalManager]: NULL channel manager.\n");
            return false;
        }
        if(channelManager->init(MAX_WIDTH, MAX_HEIGHT, MAX_STRIDE)){
            
        }else{
            // error
            return false;
        }
        encoderManager = EncoderManager::GetEncoderManager();
        if(encoderManager == NULL){
            // error 
			infoRecorder->logTrace("[GlobalManager]: NULL encoder manager.\n");
            return false;
        }
#if 1
        if(config != NULL){
			configFile = _strdup(config);
			if (rtspConf == NULL){
				rtspConf = RTSPConf::GetRTSPConf(config);
			}
			if (rtspConf->rtspConfParse() < 0) {
				infoRecorder->logTrace("[RTSPConf]:parse configuration failed.\n");
				return false;
			}
			return true;
        }
#endif
       // start(); // start the global manager thread
		return true;
    }
#if 0
	virtual BOOL stop();
    virtual void run();
    virtual void onThreadStart(LPVOID param = NULL);
    virtual void onQuit();
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
#endif

	void setGlobalConfig(RTSPConf * conf){ rtspConf = conf; }
	RTSPConf * getGlobalConfig(){ return rtspConf; }

};

HANDLE prsentEvent;  // the present event for graphic part
DWORD WINAPI VideoServer(LPVOID param);   // the main thread for video part


#endif
