#ifndef __VSOURCE_H__
#define __VSOURCE_H__

#include "Wrapper.h"

#include "VideoCommon.h"
#include "../LibVideo/Pipeline.h"


class ChannelBase{
protected:
	HWND windowHandle;   // the game window handle
	int channelId;       // identify the channel
	SOCKET clientSock;   // socket for a client

	char *imagepipename, * filterpipename;
	bool initialized;    // inited or not?
	DWORD channelThreadId;
	HANDLE threadHandle; 


	pipeline *pipe[SOURCES];     // the pipe line for the channel#if 0
	int width[SOURCES], height[SOURCES], stride[SOURCES];

	Filter * filter;  // the filtes is ARGB to YUV420P
	Encoder * encoder;

	SOURCE_TYPE sourceType;   // image or a surface
    ENCODER_TYPE encoderType;

	int gameWidth, gameHeight, encoderWidth, encoderHeight;

	RTSPConf * rtspConf;

	ccgRect *screenRect;

	ccgRect * windowRect;

	ccgImage * cimage;

public:
	// constructor and destructor
	ChannelBase();
	~ChannelBase();

	inline int getChannelId(){ return channelId; }
	inline void setChannelId(int id){ channelId = id; }

	inline void setWindowHandle(HWND hwnd){ windowHandle = hwnd; }
	inline HWND getWindowHandle(){ return windowHandle; }

	void setEncoder(Encoder * encoder);
	inline Encoder * getEncoder(){ return this->encoder; }

	inline void setRtspConf(RTSPConf * conf){ rtspConf = conf; }
	inline RTSPConf * getRtspConf(){ return rtspConf; }

	inline void setSrcPipeName(char * name){ this->imagepipename = _strdup(name); }
	inline void setFilterPipeName(char * name){ this->filterpipename = _strdup(name); }
	inline char * getFilterPipeName(){ return this->filterpipename; }

	inline void setFilter(Filter * f){ filter = f; }
    inline Filter * getFilter(){return filter;}
    inline void setClientSock(SOCKET s){ clientSock = s; }
    inline SOCKET getClientSock(){ return clientSock; }

};

// the class is for the games that does not use D3D. capture the window data through  GDI
class Channel2D : public ChannelBase{
	WindowWrapper * wrapper;    // the window wrapper is only for 2D games

	//DX_VERSION dxVersion;

public:

	Channel2D();
	~Channel2D();

	int doCapture(char * buf, int buflen, struct ccgRect * grect);
	void startThrad();
	static DWORD WINAPI Chanel2DThreadProc(LPVOID param);
	int init(void * arg);
	void deinit(void *arg);
};





class Channel :public CThread , public ChannelBase {
public:

    char tbuf[50];  // temp for tmp using
    static HANDLE initMutex; 
	static HANDLE presentMutex;
    
	char *name[2];
	char *surfacepipename;
	
	pipeline * cudaPipe[SOURCES];  // cuda pipe
	
	int gSources; // the number of source in a channel
	HANDLE presentEvent;  // Presen will notify the presentEvent

	DX_VERSION dxVersion;

	HWND windowHwnd;
    VFrame * source;

    D3DWrapper *wrapper; //the wrapper for the channel

	DXWrapper *dxWrapper;

    int maxWidth, maxHeight, maxStride;
	int game_width, game_height, encoder_width, encoder_height;

	// temp variables
	int frame_interval;
	LARGE_INTEGER initialTv, captureTv, freq;
	struct timeval tv;

	HANDLE deviceEvent, windowHandleEvent;

	IDirect3DDevice9 * device;

	void * dxEntry;

	D3DLOCKED_RECT lockedRect;
	int vsource_initialized;


	int getD3D9Resolution();
	int getD3D10Resolution();
	int getD3D101Resolution();
	int getD3D11Resolution();

public:
	bool setDevice(DX_VERSION version, void * data);

	bool doCapture(IDirect3DDevice9 * device);
	
	
	int getD3DResolution(DX_VERSION version);
	bool startChannelThread(LPVOID param);
#if 0
    bool startThread(LPVOID param);
	static DWORD SourceThreadProc(LPVOID param);
#else
	virtual BOOL stop();
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void onThreadStart(LPVOID param = NULL);
	virtual void onQuit();
    virtual void run();
#endif
    char * getChannelName(){
        sprintf(tbuf, "channel%d", channelId);
        return tbuf;
    }
	char * getChannelPipeName(){
		sprintf(tbuf, "pipe%d", channelId);
		return tbuf;
	}
    Channel();
    Channel(const Channel &);
    Channel & operator=(const Channel &);
	Channel(ENCODER_TYPE type);// { encoderType = type; }
	~Channel();
	static void Release();
    const char * getPipeName();

    bool init(int maxWidth, int maxHeight, int maxStride,SOURCE_TYPE sourceType);
    bool init(ENCODER_TYPE encoderType, SOURCE_TYPE type = SOURCE_TYPE::SOURCE_NONE);
    int setup(const char * pipeformat, struct VsourceConfig *config, int nConfig);
	//int setup(const char * pipeformat, struct VsourceConfig *config);
	int setup(const char * pipeformat);
    unsigned char * getImage();
    IDirect3DSurface9 * getSurface();

	inline ccgRect * getRect(){ return screenRect; }
	inline bool setRect(ccgRect * rect){ screenRect = rect; }

    bool registerEncoder(Encoder * encoder);
    bool registerD3DWrapper(D3DWrapper * wrapper);

    ///////////
	inline void setEncoderType(ENCODER_TYPE type){ encoderType = type; }
	inline int getMaxWidth(){ return maxWidth; }
	inline int getMaxHeight(){ return maxHeight; }
	inline int getMaxStride(){ return maxStride; }

	inline void setMaxWidth(int width){ maxWidth = width; }
	inline void setMaxHeight(int height){ maxHeight = height; }
	inline void setMaxStride(int stride){ maxStride = stride; }
	inline void setWindowHwnd(HWND h){ this->windowHwnd = h; }
	

	inline void setDevice(IDirect3DDevice9 * d){ this->device = d; }
	inline void setPresentEvent(HANDLE e){ this->presentEvent = e; }
	inline HANDLE getPresentEvent(){ return this->presentEvent; }
    
	//setters and getters
    
    VFrame * getFrame();
    void setFrame(VFrame * frame);
    D3DWrapper * getWrapper(D3DWrapper * wrapper);
    void setWrapper(D3DWrapper);
    
    //static DWORD StartChannelThread(LPVOID arg);

	inline void setSurfacePipeName(char *name){this->surfacepipename = _strdup(name); }
	
	void startFilter();

	// reconstruct the code
	void waitForDeviceAndWindowHandle();
};

int ccg_win32_draw_sysytem_cursor(ImageFrame * frame);
#endif
