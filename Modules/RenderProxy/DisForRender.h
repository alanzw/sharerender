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

using namespace cg;
using namespace cg::core;
// render channel is to render image and setup the rtsp
class RenderChannel;

// run on server side, inside render domain
class RenderProxy : public cg::DisClient{
	cg::IDENTIFIER renderId;  // render id in DisServer
	//add render task manage
	cg::RTSPConf * conf;

	// for rtsp
	static DWORD WINAPI RTSPThreadProc(LPVOID param);

	//static DWORD WINAPI RenderThreadProc(LPVOID param);
	static RenderProxy * renderProxy;
	RenderProxy(){
		renderId = NULL;

	}

	int encodeOption;    // the encoder to use, 1 for x264, 2 for nvenc, 3 for cuda

public:

	list<RenderChannel *> serviceMap;
	static RenderProxy * GetProxy(){
		if (renderProxy == NULL)
			renderProxy = new RenderProxy();
		return renderProxy;
	}
	inline void setEncodeOption(int option){ encodeOption = option; }
	inline int getEncodeOption(){ return encodeOption; }
	inline void setRTSPConf(cg::RTSPConf * c){ conf = c; }

	virtual bool start(char * DisUrl); // star the render proxy
	virtual bool dealEvent(cg::BaseContext * ctx);
#if 0
	virtual bool startRTSP(evutil_socket_t sock);

	void startRTSPThread(evutil_socket_t sock);
	//void startRenderThread(RenderChannel * ch);

	// intra-migration between CPU and GPU
	bool regulation();
#endif
};

evutil_socket_t connectToGraphic(char * url, int port);



// global context
extern cg::BaseContext * ctxToDis;    // the context to notify distributor

#endif