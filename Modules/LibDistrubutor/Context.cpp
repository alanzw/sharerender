#include "Context.h"
#include "../VideoGen/generator.h"
#include "../LibCore/InfoRecorder.h"

namespace cg{

#ifdef BACKBUFFER_TEST
	VideoGen * gGenerator = NULL;
#endif

	VideoContext * VideoContext::ctx;

	void RTSPListenerCB(struct evconnlistener * listerner, evutil_socket_t sock, struct sockaddr * sddr, int len, void * _ctx){

		cg::core::infoRecorder->logTrace("[RTSPListenerCB]: callback for listen.\n");
		// we got a new connection ! Set up a bufferevent for it
#if 1
		// use the given context to create the rtsp service
		// the service name must be set to context

		RTSPContext * ctx = (RTSPContext *)_ctx;
		ctx->setClientSocket(sock);
		//
		
		DWORD threadId = 0;
		HANDLE rtspThread = chBEGINTHREADEX(NULL, 0, RtspServerThreadProc, ctx, 0, &threadId);
#endif
	}

	void RTSPAcceptErrorCB(struct evconnlistener * listener, void *ctx){
		struct event_base * base = evconnlistener_get_base(listener);
		int err = EVUTIL_SOCKET_ERROR();
		cg::core::infoRecorder->logError("Get an error %s (%s) on the listener." "Shutting down.\n", err, evutil_socket_error_to_string(err));
		event_base_loopexit(base, NULL);
	}

	struct evconnlistener * listenPort(int port, event_base * base, void * ctx){
		cg::core::infoRecorder->logError("[Global]: listen RTSP port %d, rtsp context: %p.\n", port, ctx);
		struct evconnlistener * listener = NULL;
		sockaddr_in sin;
		int sin_size = sizeof(sin);
		memset(&sin, 0, sin_size);
		sin.sin_family = AF_INET;
		sin.sin_addr.S_un.S_addr = htonl(0);
		sin.sin_port = htons(port);

		listener = evconnlistener_new_bind(base, RTSPListenerCB, ctx, /*LEV_OPT_LEAVE_SOCKETS_BLOCKING |*/ LEV_OPT_CLOSE_ON_FREE | LEV_OPT_REUSEABLE, -1, (sockaddr *)&sin, sin_size);

		if(!listener)
		{
			cg::core::infoRecorder->logError("[Global]: couldn't create rtsp listener.\n");
			return false;
		}
		evconnlistener_set_error_cb(listener, RTSPAcceptErrorCB);

		return listener;
	}

	// video context
	void VideoContext::WaitForNofify(){
		cg::core::infoRecorder->logTrace("[VideoContxt]: to wait the context.\n");
		DWORD ret = WaitForSingleObject(notifier, INFINITE);
		switch(ret){
		case WAIT_OBJECT_0:
			// succ
			infoRecorder->logError("[VideoContext]: wait for video item notification succ.\n");
			break;
		case WAIT_FAILED:
			infoRecorder->logError("[VideoContext]: wait for video item notification failed with:%d.\n", GetLastError());

			break;
		}
	}
}