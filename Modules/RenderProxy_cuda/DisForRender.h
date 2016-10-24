#ifndef __DISFORRENDER_H__
#define __DISFORRENDER_H__

#include "../LibDistrubutor/Distributor.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <WinSock2.h>
#include <map>
#include "../LibCore/CpuWatch.h"
#include "../LibCore/GpuWatch.h"

#include "../LibRender/LibRenderChannel.h"


#include "../LibDistrubutor/Context.h"



// use libevent
///////////// cmd for render proxy ///////////////
extern const char * INFO; // recv info cmd from dis, collect the domain info
extern const char * START_TASK; // recv start game cmd from dis
extern const char * CANCEL_TASK; // recv cancel task cmd from dis

extern const char * ADD_RTSP_CONNECTION;

////////////// const string for DATA ////////////////
extern const char * LOGIC; // dis server get the data with REGISTER
extern const char * RENDER; // dis server get the data with REGISTER
extern const char * CLIENT; // dis server may get the data with REGISTER, not sure


// render channel is to render image and setup the rtsp
class RenderChannel;

// run on server side, inside render domain
class RenderProxy : public DisClient{
	IDENTIFIER renderId;  // render id in DisServer
	//add render task manage


	// for rtsp
	static DWORD WINAPI RTSPThreadProc(LPVOID param);

	//static DWORD WINAPI RenderThreadProc(LPVOID param);
	static RenderProxy * renderProxy;
	RenderProxy(){
		renderId = NULL;

	}
public:

	list<RenderChannel *> serviceMap;
	static RenderProxy * GetProxy(){
		if (renderProxy == NULL)
			renderProxy = new RenderProxy();
		return renderProxy;
	}

	virtual bool start(); // star the render proxy
	virtual bool dealEvent(BaseContext * ctx);
	virtual bool startRTSP(evutil_socket_t sock);

	void startRTSPThread(evutil_socket_t sock);
	//void startRenderThread(RenderChannel * ch);

};


#endif