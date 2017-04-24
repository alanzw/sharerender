/*
* Copyright (c) 2013 Chun-Ying Huang
*
* This file is part of GamingAnywhere (GA).
*
* GA is free software; you can redistribute it and/or modify it
* under the terms of the 3-clause BSD License as published by the
* Free Software Foundation: http://directory.fsf.org/wiki/License:BSD_3Clause
*
* GA is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*
* You should have received a copy of the 3-clause BSD License along with GA;
* if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

/*
* Copyright (c) 2013 Chun-Ying Huang
*
* This file is part of GamingAnywhere (GA).
*
* GA is free software; you can redistribute it and/or modify it
* under the terms of the 3-clause BSD License as published by the
* Free Software Foundation: http://directory.fsf.org/wiki/License:BSD_3Clause
*
* GA is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*
* You should have received a copy of the 3-clause BSD License along with GA;
* if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef __CONTROLLER_H__
#define __CONTROLLER_H__
#include "../LibCore/CThread.h"
#include "../VideoUtility/rtspconf.h"
#include "CtrlConfig.h"


#include "SDL2/SDL_version.h"
#include "SDL2/SDL_keycode.h"
#include "../VideoUtility/videocommon.h"
#include "../LibCore/TimeTool.h"

#define	SDL_EVENT_MSGTYPE_NULL		0
#define	SDL_EVENT_MSGTYPE_KEYBOARD	1
#define	SDL_EVENT_MSGTYPE_MOUSEKEY	2
#define SDL_EVENT_MSGTYPE_MOUSEMOTION	3
#define SDL_EVENT_MSGTYPE_MOUSEWHEEL	4

#ifdef WIN32
#pragma pack(push, 1)
#endif

#define	CTRL_MAX_ID_LENGTH	64
#define	CTRL_CURRENT_VERSION	"GACtrlV01"
#define	CTRL_QUEUE_SIZE		65536	// 64K

#define STREAM_SERVER_CONFIG "config/server.controller.conf"

#define ENABLE_CLIENT_CONTROL


#define USE_CONTROL_CONFIG

namespace cg{
	namespace input{

		struct sdlmsg_s {
			unsigned short msgsize;		// size of this data-structure
			// every message MUST start from a
			// unsigned short message size
			// the size includes the 'msgsize'
			unsigned char msgtype;
			unsigned char which;
			unsigned char padding[60];	// must be large enough to fit
			// all supported type of messages
#if 1
			unsigned char is_pressed;	// for keyboard/mousekey
			unsigned char mousebutton;	// mouse button
			unsigned char mousestate;	// mouse state - key combinations for motion
#if 1	// only support SDL2
			unsigned char unused1;		// padding - 3+1 chars
			unsigned short scancode;	// keyboard scan code
			int sdlkey;			// SDLKey value
			unsigned int unicode;		// unicode or ASCII value
#endif
			unsigned short sdlmod;		// SDLMod value
			unsigned short mousex;		// mouse position (big-endian)
			unsigned short mousey;		// mouse position (big-endian)
			unsigned short mouseRelX;	// mouse relative position (big-endian)
			unsigned short mouseRelY;	// mouse relative position (big-endian)
			unsigned char relativeMouseMode;// relative mouse mode?
			unsigned char padding_[8];	// reserved padding
#endif
		}
#ifdef WIN32
#pragma pack(pop)
#else
		__attribute__((__packed__))
#endif
			;
		typedef struct sdlmsg_s			sdlmsg_t;

		// keyboard event
#ifdef WIN32
#pragma pack(push, 1)
#endif
		struct sdlmsg_keyboard_s {
			unsigned short msgsize;
			unsigned char msgtype;		// SDL_EVENT_MSGTYPE_KEYBOARD
			unsigned char which;
			unsigned char is_pressed;
			unsigned char unused0;
			unsigned short scancode;	// scancode
			int sdlkey;			// SDLKey
			unsigned int unicode;		// unicode or ASCII value
			unsigned short sdlmod;		// SDLMod
		}
#ifdef WIN32
#pragma pack(pop)
#else
		__attribute__((__packed__))
#endif
			;
		typedef struct sdlmsg_keyboard_s	sdlmsg_keyboard_t;

		// mouse event
#ifdef WIN32
#pragma pack(push, 1)
#endif
		struct sdlmsg_mouse_s {
			unsigned short msgsize;
			unsigned char msgtype;		// SDL_EVENT_MSGTYPE_MOUSEKEY
			// SDL_EVENT_MSGTYPE_MOUSEMOTION
			// SDL_EVENT_MSGTYPE_MOUSEWHEEL
			unsigned char which;
			unsigned char is_pressed;	// for mouse button
			unsigned char mousebutton;	// mouse button
			unsigned char mousestate;	// mouse stat
			unsigned char relativeMouseMode;
			unsigned short mousex;
			unsigned short mousey;
			unsigned short mouseRelX;
			unsigned short mouseRelY;
		}
#ifdef WIN32
#pragma pack(pop)
#else
		__attribute__((__packed__))
#endif
			;
		typedef struct sdlmsg_mouse_s		sdlmsg_mouse_t;

#if 1	// only support SDL2
		sdlmsg_t* sdlmsg_keyboard(sdlmsg_t *msg, unsigned char pressed, unsigned short scancode, SDL_Keycode key, unsigned short mod, unsigned int unicode);
		sdlmsg_t* sdlmsg_mousewheel(sdlmsg_t *msg, unsigned short mousex, unsigned short mousey);
#endif  // only support SDL2
		sdlmsg_t* sdlmsg_mousekey(sdlmsg_t *msg, unsigned char pressed, unsigned char button, unsigned short x, unsigned short y);
		sdlmsg_t* sdlmsg_mousemotion(sdlmsg_t *msg, unsigned short mousex, unsigned short mousey, unsigned short relx, unsigned short rely, unsigned char state, int relativeMouseMode);

		/////////////////////////////////////////////////

		typedef void (*msgfunc)(void *, int);

		// handshake message: 
		struct ctrlhandshake {
			unsigned char length;
			char id[CTRL_MAX_ID_LENGTH];
		};

		struct queuemsg {
			unsigned short msgsize;		// a general header for messages
			unsigned char msg[2];		// use '2' to prevent Windows from complaining
		};

		enum MESSAGER_TYPE{
			MESSAGE_SERVER,
			MESSAGE_CLIENT,
		};

		class QueueMessager{
			queuemsg * _msg;
			//HANDLE queueMutex;
			CRITICAL_SECTION  queueSection, *pQueueSection;

			int qhead, qtail, qsize, qunit;
			unsigned char * qbuffer;

		public:
			QueueMessager(){
				_msg = NULL;
				qhead = 0; qtail = 0; qsize = 0; qunit = 0;
				qbuffer = NULL;
			}
			//QueueMessager(int size, int maxunit, MESSAGER_TYPE _type);
			~QueueMessager();

			int initQueue(int size, int maxunit);
			void release();
			struct queuemsg * readMsg();
			void releaseMsg(struct queuemsg *msg);
			int writeMsg(void *msg, int msgsize);
			void clear();
		};

		class CtrlMessagerClient : public QueueMessager, public cg::core::CThread{
			SOCKET ctrlSocket;
			struct sockaddr_in ctrlsin;
			RTSPConf *conf;
			CRITICAL_SECTION wakeupMutex;
			HANDLE wakeup;
			char * ctrlServerUrl;


			int ctrlSocketInit(struct cg::RTSPConf * conf);

		public:
			CtrlMessagerClient();
			~CtrlMessagerClient();

			// from CThread
			virtual BOOL stop();
			virtual BOOL run();
			virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
			virtual BOOL onThreadStart();
			virtual void onQuit();
			// for client

			//int initMessager(int size, int maxunit);
			void sendMsg(void *msg, int msglen);
			int init(cg::RTSPConf * conf,char * url, const char * ctrlid);

		};

		// the callback interface
		class ReplayCallback{
		public:
			virtual void operator()(void * buf, int len) = 0;
			virtual ~ReplayCallback(){};
		};

		class CtrlMessagerServer : public QueueMessager, public cg::core::CThread{

			//double scaleFactorX, scaleFactorY;
			CRITICAL_SECTION reslock;
			CRITICAL_SECTION oreslock;

			SOCKET	ctrlSocket, sock;
			struct	sockaddr_in ctrlsin;
			int		clientaccepted;

			unsigned char buf[8192];
			int		bufhead, buflen;

			HANDLE	wakeupMutex;
			HANDLE	wakeup;
			//msgfunc replay;
			ReplayCallback * replay;
			RTSPConf* conf;

			int		ctrlSocketInit(cg::RTSPConf * conf);

		public:
			CtrlMessagerServer();
			~CtrlMessagerServer();

			// from CThread
			virtual BOOL stop();
			virtual BOOL run();
			virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
			virtual BOOL onThreadStart();
			virtual void onQuit();

			// for server

			inline void setReplay(ReplayCallback * callback){ replay = callback; }

			int		readNext(void *msg, int msglen);
			int		init(RTSPConf* conf, const char * ctrlid);
		};

		class ReplayCallbackImp: public ReplayCallback{
			// attribute
			float scaleFactorX, scaleFactorY;
			float cxsize, cysize;

			RECT*	prect; // the display window rect
			RECT*	pOutputWindowRect;   // the output window rect

			void scaleCoordinate(); // called before use
			int initKeyMap(); // called before use

			void replayNative(sdlmsg_t *msg);
			void replay(sdlmsg_t * msg);
			
		public:
			ReplayCallbackImp(RECT * windowRect, RECT * pOutputRect);
			virtual void operator()(void * buf, int len);
			virtual ~ReplayCallbackImp();
		};
		
		void CreateClientControl(HWND);
	}
}
extern cg::core::BTimer * ctrlTimer;

#endif