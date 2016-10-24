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

#include <stdio.h>
#include "../LibCore/MemOp.h"

#include "../VideoUtility/videocommon.h"
#include "../LibCore/Config.h"
#include "../VideoUtility/rtspconf.h"

#include "CtrlSdl.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"


#include <map>
using namespace std;

#ifdef WIN32
#define	INVALID_KEY	0xffff
typedef WORD	KeySym;		// map to Windows Virtual Keycode
#elif defined __APPLE__
#define	INVALID_KEY	0xffff
typedef	CGKeyCode KeySym;
#elif defined ANDROID
#define	INVALID_KEY	0xffff
typedef short	KeySym;
#else	// X11, has built-in KeySym
#define	INVALID_KEY	0
static Display *display = NULL;
static int screenNumber = 0;
#endif


static void SDLKeyToKeySym_init();
#if 1	// only support SDL2
static map<int, KeySym> keymap;
static KeySym SDLKeyToKeySym(int sdlkey);
#endif

#if 1
// only support SDL2: remap key codes 1.2 -> 2.0
#define	SDLK_KP0	SDLK_KP_0
#define	SDLK_KP1	SDLK_KP_1
#define	SDLK_KP2	SDLK_KP_2
#define	SDLK_KP3	SDLK_KP_3
#define	SDLK_KP4	SDLK_KP_4
#define	SDLK_KP5	SDLK_KP_5
#define	SDLK_KP6	SDLK_KP_6
#define	SDLK_KP7	SDLK_KP_7
#define	SDLK_KP8	SDLK_KP_8
#define	SDLK_KP9	SDLK_KP_9
//
#define SDLK_NUMLOCK	SDLK_NUMLOCKCLEAR
#define SDLK_SCROLLOCK	SDLK_SCROLLLOCK
#define SDLK_RMETA	SDLK_RGUI
#define SDLK_LMETA	SDLK_LGUI
//#define SDLK_LSUPER
//#define SDLK_RSUPER
//#define SDLK_COMPOSE
//#define SDLK_PRINT
#define SDLK_BREAK	SDLK_PRINTSCREEN
#endif


namespace cg{
	namespace input{
		sdlmsg_t *
			sdlmsg_ntoh(sdlmsg_t *msg) {
				sdlmsg_keyboard_t *msgk = (sdlmsg_keyboard_t*) msg;
				sdlmsg_mouse_t *msgm = (sdlmsg_mouse_t*) msg;
				if(msg == NULL)
					return NULL;
				switch(msg->msgtype) {
				case SDL_EVENT_MSGTYPE_KEYBOARD:
					if(msgk->scancode)	msgk->scancode = ntohs(msgk->scancode);
					if(msgk->sdlkey)	msgk->sdlkey = (int) ntohl(msgk->sdlkey);
					if(msgk->unicode)	msgk->unicode = ntohl(msgk->unicode);
					if(msgk->sdlmod)	msgk->sdlmod = ntohs(msgk->sdlmod);
					break;
				case SDL_EVENT_MSGTYPE_MOUSEKEY:
				case SDL_EVENT_MSGTYPE_MOUSEMOTION:
				case SDL_EVENT_MSGTYPE_MOUSEWHEEL:
					if(msgm->mousex)	msgm->mousex = ntohs(msgm->mousex);
					if(msgm->mousey)	msgm->mousey = ntohs(msgm->mousey);
					if(msgm->mouseRelX)	msgm->mouseRelX = ntohs(msgm->mouseRelX);
					if(msgm->mouseRelY)	msgm->mouseRelY = ntohs(msgm->mouseRelY);
					break;
				}
#if 0
#if 1	// only support SDL2
				if(msg->scancode)	msg->scancode = ntohs(msg->scancode);
				if(msg->sdlkey)		msg->sdlkey = (int) ntohl(msg->sdlkey);
				if(msg->unicode)	msg->unicode = ntohl(msg->unicode);
#endif
				if(msg->sdlmod)		msg->sdlmod = ntohs(msg->sdlmod);
				if(msg->mousex)		msg->mousex = ntohs(msg->mousex);
				if(msg->mousey)		msg->mousey = ntohs(msg->mousey);
				if(msg->mouseRelX)	msg->mouseRelX = ntohs(msg->mouseRelX);
				if(msg->mouseRelY)	msg->mouseRelY = ntohs(msg->mouseRelY);
#endif
				return msg;
		}

		sdlmsg_t *
			sdlmsg_keyboard(sdlmsg_t *msg, unsigned char pressed, unsigned short scancode, SDL_Keycode key, unsigned short mod, unsigned int unicode)
		{
			sdlmsg_keyboard_t *msgk = (sdlmsg_keyboard_t*) msg;
			//ga_error("sdl client: key event code=%x key=%x mod=%x pressed=%u\n", scancode, key, mod, pressed);
			bzero(msg, sizeof(sdlmsg_keyboard_t));
			msgk->msgsize = htons(sizeof(sdlmsg_keyboard_t));
			msgk->msgtype = SDL_EVENT_MSGTYPE_KEYBOARD;
			msgk->is_pressed = pressed;
#if 1	// only support SDL2
			msgk->scancode = htons(scancode);
			msgk->sdlkey = htonl(key);
			msgk->unicode = htonl(unicode);
#endif
			msgk->sdlmod = htons(mod);
			return msg;
		}

		sdlmsg_t *
			sdlmsg_mousekey(sdlmsg_t *msg, unsigned char pressed, unsigned char button, unsigned short x, unsigned short y) {
				sdlmsg_mouse_t *msgm = (sdlmsg_mouse_t*) msg;
				//ga_error("sdl client: button event btn=%u pressed=%u\n", button, pressed);
				bzero(msg, sizeof(sdlmsg_mouse_t));
				msgm->msgsize = htons(sizeof(sdlmsg_mouse_t));
				msgm->msgtype = SDL_EVENT_MSGTYPE_MOUSEKEY;
				msgm->is_pressed = pressed;
				msgm->mousex = htons(x);
				msgm->mousey = htons(y);
				msgm->mousebutton = button;
				return msg;
		}

#if 1	// only support SDL2
		sdlmsg_t *
			sdlmsg_mousewheel(sdlmsg_t *msg, unsigned short mousex, unsigned short mousey) {
				sdlmsg_mouse_t *msgm = (sdlmsg_mouse_t*) msg;
				//ga_error("sdl client: motion event x=%u y=%u\n", mousex, mousey);
				bzero(msg, sizeof(sdlmsg_mouse_t));
				msgm->msgsize = htons(sizeof(sdlmsg_mouse_t));
				msgm->msgtype = SDL_EVENT_MSGTYPE_MOUSEWHEEL;
				msgm->mousex = htons(mousex);
				msgm->mousey = htons(mousey);
				return msg;
		}
#endif

		sdlmsg_t *
			sdlmsg_mousemotion(sdlmsg_t *msg, unsigned short mousex, unsigned short mousey, unsigned short relx, unsigned short rely, unsigned char state, int relativeMouseMode) {
				sdlmsg_mouse_t *msgm = (sdlmsg_mouse_t*) msg;
				//ga_error("sdl client: motion event x=%u y=%u\n", mousex, mousey);
				bzero(msg, sizeof(sdlmsg_mouse_t));
				msgm->msgsize = htons(sizeof(sdlmsg_mouse_t));
				msgm->msgtype = SDL_EVENT_MSGTYPE_MOUSEMOTION;
				msgm->mousestate = state;
				msgm->relativeMouseMode = (relativeMouseMode != 0) ? 1 : 0;
				msgm->mousex = htons(mousex);
				msgm->mousey = htons(mousey);
				msgm->mouseRelX = htons(relx);
				msgm->mouseRelY = htons(rely);
				return msg;
		}


		//////////////////////////////////////////////////////////////////////////////

#ifdef WIN32
		static void
			SDLKeyToKeySym_init() {
				unsigned short i;
				//
				keymap[SDLK_BACKSPACE]	= VK_BACK;		//		= 8,
				keymap[SDLK_TAB]	= VK_TAB;		//		= 9,
				keymap[SDLK_CLEAR]	= VK_CLEAR;		//		= 12,
				keymap[SDLK_RETURN]	= VK_RETURN;		//		= 13,
				keymap[SDLK_PAUSE]	= VK_PAUSE;		//		= 19,
				keymap[SDLK_ESCAPE]	= VK_ESCAPE;		//		= 27,
				// Latin 1: starting from space (0x20)
				keymap[SDLK_SPACE]	= VK_SPACE;		//		= 32,
				// (0x20) space, exclam, quotedbl, numbersign, dollar, percent, ampersand,
				// (0x27) quoteright, parentleft, parentright, asterisk, plus, comma,
				// (0x2d) minus, period, slash
				//SDLK_EXCLAIM		= 33,
				keymap[SDLK_QUOTEDBL]	= VK_OEM_7;		//		= 34,
				//SDLK_HASH		= 35,
				//SDLK_DOLLAR		= 36,
				//SDLK_AMPERSAND		= 38,
				keymap[SDLK_QUOTE]	= VK_OEM_7;		//		= 39,
				//SDLK_LEFTPAREN		= 40,
				//SDLK_RIGHTPAREN		= 41,
				//SDLK_ASTERISK		= 42,
				keymap[SDLK_PLUS]	= VK_OEM_PLUS;		//		= 43,
				keymap[SDLK_COMMA]	= VK_OEM_COMMA;		//		= 44,
				keymap[SDLK_MINUS]	= VK_OEM_MINUS;		//		= 45,
				keymap[SDLK_PERIOD]	= VK_OEM_PERIOD;	//		= 46,
				keymap[SDLK_SLASH]	= VK_OEM_2;		//		= 47,
				keymap[SDLK_COLON]	= VK_OEM_1;		//		= 58,
				keymap[SDLK_SEMICOLON]	= VK_OEM_1;		//		= 59,
				keymap[SDLK_LESS]	= VK_OEM_COMMA;		//		= 60,
				keymap[SDLK_EQUALS]	= VK_OEM_PLUS;		//		= 61,
				keymap[SDLK_GREATER]	= VK_OEM_PERIOD;	//		= 62,
				keymap[SDLK_QUESTION]	= VK_OEM_2;		//		= 63,
				//SDLK_AT			= 64,
				/* 
				Skip uppercase letters
				*/
				keymap[SDLK_LEFTBRACKET]= VK_OEM_4;		//		= 91,
				keymap[SDLK_BACKSLASH]	= VK_OEM_5;		//		= 92,
				keymap[SDLK_RIGHTBRACKET]= VK_OEM_6;		//		= 93,
				//SDLK_CARET		= 94,
				keymap[SDLK_UNDERSCORE]	= VK_OEM_MINUS;		//		= 95,
				keymap[SDLK_BACKQUOTE]	= VK_OEM_3;		//		= 96,
				// (0x30-0x39) 0-9
				for(i = 0x30; i <= 0x39; i++) {
					keymap[i] = i;
				}
				// (0x3a) colon, semicolon, less, equal, greater, question, at
				// (0x41-0x5a) A-Z
				// SDL: no upper cases, only lower cases
				// (0x5b) bracketleft, backslash, bracketright, asciicircum/caret,
				// (0x5f) underscore, grave
				// (0x61-7a) a-z
				for(i = 0x61; i <= 0x7a; i++) {
					keymap[i] = i & 0xdf;	// convert to uppercases
				}
				keymap[SDLK_DELETE]	= VK_DELETE;		//		= 127,
				// SDLK_WORLD_0 (0xa0) - SDLK_WORLD_95 (0xff) are ignored
				/** @name Numeric keypad */
				keymap[SDLK_KP0]	= VK_NUMPAD0;	//		= 256,
				keymap[SDLK_KP1]	= VK_NUMPAD1;	//		= 257,
				keymap[SDLK_KP2]	= VK_NUMPAD2;	//		= 258,
				keymap[SDLK_KP3]	= VK_NUMPAD3;	//		= 259,
				keymap[SDLK_KP4]	= VK_NUMPAD4;	//		= 260,
				keymap[SDLK_KP5]	= VK_NUMPAD5;	//		= 261,
				keymap[SDLK_KP6]	= VK_NUMPAD6;	//		= 262,
				keymap[SDLK_KP7]	= VK_NUMPAD7;	//		= 263,
				keymap[SDLK_KP8]	= VK_NUMPAD8;	//		= 264,
				keymap[SDLK_KP9]	= VK_NUMPAD9;	//		= 265,
				keymap[SDLK_KP_PERIOD]	= VK_DECIMAL;	//		= 266,
				keymap[SDLK_KP_DIVIDE]	= VK_DIVIDE;	//		= 267,
				keymap[SDLK_KP_MULTIPLY]= VK_MULTIPLY;	//		= 268,
				keymap[SDLK_KP_MINUS]	= VK_SUBTRACT;	//		= 269,
				keymap[SDLK_KP_PLUS]	= VK_ADD;	//		= 270,
				//keymap[SDLK_KP_ENTER]	= XK_KP_Enter;	//		= 271,
				//keymap[SDLK_KP_EQUALS]	= XK_KP_Equal;	//		= 272,
				/** @name Arrows + Home/End pad */
				keymap[SDLK_UP]		= VK_UP;	//		= 273,
				keymap[SDLK_DOWN]	= VK_DOWN;	//		= 274,
				keymap[SDLK_RIGHT]	= VK_RIGHT;	//		= 275,
				keymap[SDLK_LEFT]	= VK_LEFT;	//		= 276,
				keymap[SDLK_INSERT]	= VK_INSERT;	//		= 277,
				keymap[SDLK_HOME]	= VK_HOME;	//		= 278,
				keymap[SDLK_END]	= VK_END;	//		= 279,
				keymap[SDLK_PAGEUP]	= VK_PRIOR;	//		= 280,
				keymap[SDLK_PAGEDOWN]	= VK_NEXT;	//		= 281,
				/** @name Function keys */
				keymap[SDLK_F1]		= VK_F1;	//		= 282,
				keymap[SDLK_F2]		= VK_F2;	//		= 283,
				keymap[SDLK_F3]		= VK_F3;	//		= 284,
				keymap[SDLK_F4]		= VK_F4;	//		= 285,
				keymap[SDLK_F5]		= VK_F5;	//		= 286,
				keymap[SDLK_F6]		= VK_F6;	//		= 287,
				keymap[SDLK_F7]		= VK_F7;	//		= 288,
				keymap[SDLK_F8]		= VK_F8;	//		= 289,
				keymap[SDLK_F9]		= VK_F9;	//		= 290,
				keymap[SDLK_F10]	= VK_F10;	//		= 291,
				keymap[SDLK_F11]	= VK_F11;	//		= 292,
				keymap[SDLK_F12]	= VK_F12;	//		= 293,
				keymap[SDLK_F13]	= VK_F13;	//		= 294,
				keymap[SDLK_F14]	= VK_F14;	//		= 295,
				keymap[SDLK_F15]	= VK_F15;	//		= 296,
				/** @name Key state modifier keys */
				keymap[SDLK_NUMLOCK]	= VK_NUMLOCK;	//		= 300,
				keymap[SDLK_CAPSLOCK]	= VK_CAPITAL;	//		= 301,
				keymap[SDLK_SCROLLOCK]	= VK_SCROLL;	//		= 302,
				keymap[SDLK_RSHIFT]	= VK_RSHIFT;	//		= 303,
				keymap[SDLK_LSHIFT]	= VK_LSHIFT;	//		= 304,
				keymap[SDLK_RCTRL]	= VK_RCONTROL;	//		= 305,
				keymap[SDLK_LCTRL]	= VK_LCONTROL;	//		= 306,
				keymap[SDLK_RALT]	= VK_RMENU;	//		= 307,
				keymap[SDLK_LALT]	= VK_LMENU;	//		= 308,
				keymap[SDLK_RMETA]	= VK_RWIN;	//		= 309,
				keymap[SDLK_LMETA]	= VK_LWIN;	//		= 310,
				//keymap[SDLK_LSUPER]	= XK_Super_L;	//		= 311,		/**< Left "Windows" key */
				//keymap[SDLK_RSUPER]	= XK_Super_R;	//		= 312,		/**< Right "Windows" key */
				keymap[SDLK_MODE]	= VK_MODECHANGE;//		= 313,		/**< "Alt Gr" key */
				//keymap[SDLK_COMPOSE]	= XK_Multi_key;	//		= 314,		/**< Multi-key compose key */
				/** @name Miscellaneous function keys */
				keymap[SDLK_HELP]	= VK_HELP;	//		= 315,
				//keymap[SDLK_SYSREQ]	= XK_Sys_Req;	//		= 317,
				keymap[SDLK_BREAK]	= VK_CANCEL;	//		= 318,
				keymap[SDLK_MENU]	= VK_MENU;	//		= 319,
#if 0
				SDLK_POWER		= 320,		/**< Power Macintosh power key */
					SDLK_EURO		= 321,		/**< Some european keyboards */
#endif
					//keymap[SDLK_UNDO]	= XK_Undo;	//		= 322,		/**< Atari keyboard has Undo */
					//
					keymap_initialized = true;
				return;
		}
#elif defined __APPLE__
		static void
			SDLKeyToKeySym_init() {
				// read: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/Headers/Events.h
				unsigned short i;
				//
				keymap[SDLK_BACKSPACE]	= kVK_ForwardDelete;	//		= 8,
				keymap[SDLK_TAB]	= kVK_Tab;		//		= 9,
				//keymap[SDLK_CLEAR]	= VK_CLEAR;		//		= 12,
				keymap[SDLK_RETURN]	= kVK_Return;		//		= 13,
				//keymap[SDLK_PAUSE]	= VK_PAUSE;		//		= 19,
				keymap[SDLK_ESCAPE]	= kVK_Escape;		//		= 27,
				// Latin 1: starting from space (0x20)
				keymap[SDLK_SPACE]	= kVK_Space;		//		= 32,
				// (0x20) space, exclam, quotedbl, numbersign, dollar, percent, ampersand,
				// (0x27) quoteright, parentleft, parentright, asterisk, plus, comma,
				// (0x2d) minus, period, slash
				//SDLK_EXCLAIM		= 33,
				keymap[SDLK_QUOTEDBL]	= kVK_ANSI_Quote;	//		= 34,
				//SDLK_HASH		= 35,
				//SDLK_DOLLAR		= 36,
				//SDLK_AMPERSAND		= 38,
				keymap[SDLK_QUOTE]	= kVK_ANSI_Quote;	//		= 39,
				//SDLK_LEFTPAREN		= 40,
				//SDLK_RIGHTPAREN		= 41,
				//SDLK_ASTERISK		= 42,
				keymap[SDLK_PLUS]	= kVK_ANSI_Equal;	//		= 43,
				keymap[SDLK_COMMA]	= kVK_ANSI_Comma;	//		= 44,
				keymap[SDLK_MINUS]	= kVK_ANSI_Minus;	//		= 45,
				keymap[SDLK_PERIOD]	= kVK_ANSI_Period;	//		= 46,
				keymap[SDLK_SLASH]	= kVK_ANSI_Slash;	//		= 47,
				// (0x30-0x39) 0-9
				keymap[SDLK_0]		= kVK_ANSI_0;
				keymap[SDLK_1]		= kVK_ANSI_1;
				keymap[SDLK_2]		= kVK_ANSI_2;
				keymap[SDLK_3]		= kVK_ANSI_3;
				keymap[SDLK_4]		= kVK_ANSI_4;
				keymap[SDLK_5]		= kVK_ANSI_5;
				keymap[SDLK_6]		= kVK_ANSI_6;
				keymap[SDLK_7]		= kVK_ANSI_7;
				keymap[SDLK_8]		= kVK_ANSI_8;
				keymap[SDLK_9]		= kVK_ANSI_9;
				// (0x3a) colon, semicolon, less, equal, greater, question, at
				keymap[SDLK_COLON]	= kVK_ANSI_Semicolon;	//		= 58,
				keymap[SDLK_SEMICOLON]	= kVK_ANSI_Semicolon;	//		= 59,
				keymap[SDLK_LESS]	= kVK_ANSI_Comma;	//		= 60,
				keymap[SDLK_EQUALS]	= kVK_ANSI_Equal;	//		= 61,
				keymap[SDLK_GREATER]	= kVK_ANSI_Period;	//		= 62,
				keymap[SDLK_QUESTION]	= kVK_ANSI_Slash;	//		= 63,
				//SDLK_AT			= 64,
				// (0x41-0x5a) A-Z
				// SDL: no upper cases, only lower cases
				// (0x5b) bracketleft, backslash, bracketright, asciicircum/caret,
				keymap[SDLK_LEFTBRACKET]= kVK_ANSI_LeftBracket;		//		= 91,
				keymap[SDLK_BACKSLASH]	= kVK_ANSI_Backslash;		//		= 92,
				keymap[SDLK_RIGHTBRACKET]= kVK_ANSI_RightBracket;	//		= 93,
				//SDLK_CARET		= 94,
				// (0x5f) underscore, grave
				keymap[SDLK_UNDERSCORE]	= kVK_ANSI_Minus;		//		= 95,
				keymap[SDLK_BACKQUOTE]	= kVK_ANSI_Grave;		//		= 96,
				// (0x61-7a) a-z
				keymap[SDLK_a]		= kVK_ANSI_A;
				keymap[SDLK_b]		= kVK_ANSI_B;
				keymap[SDLK_c]		= kVK_ANSI_C;
				keymap[SDLK_d]		= kVK_ANSI_D;
				keymap[SDLK_e]		= kVK_ANSI_E;
				keymap[SDLK_f]		= kVK_ANSI_F;
				keymap[SDLK_g]		= kVK_ANSI_G;
				keymap[SDLK_h]		= kVK_ANSI_H;
				keymap[SDLK_i]		= kVK_ANSI_I;
				keymap[SDLK_j]		= kVK_ANSI_J;
				keymap[SDLK_k]		= kVK_ANSI_K;
				keymap[SDLK_l]		= kVK_ANSI_L;
				keymap[SDLK_m]		= kVK_ANSI_M;
				keymap[SDLK_n]		= kVK_ANSI_N;
				keymap[SDLK_o]		= kVK_ANSI_O;
				keymap[SDLK_p]		= kVK_ANSI_P;
				keymap[SDLK_q]		= kVK_ANSI_Q;
				keymap[SDLK_r]		= kVK_ANSI_R;
				keymap[SDLK_s]		= kVK_ANSI_S;
				keymap[SDLK_t]		= kVK_ANSI_T;
				keymap[SDLK_u]		= kVK_ANSI_U;
				keymap[SDLK_v]		= kVK_ANSI_V;
				keymap[SDLK_w]		= kVK_ANSI_W;
				keymap[SDLK_x]		= kVK_ANSI_X;
				keymap[SDLK_y]		= kVK_ANSI_Y;
				keymap[SDLK_z]		= kVK_ANSI_Z;
				keymap[SDLK_DELETE]	= kVK_Delete;		//		= 127,
				// SDLK_WORLD_0 (0xa0) - SDLK_WORLD_95 (0xff) are ignored
				/** @name Numeric keypad */
				keymap[SDLK_KP0]	= kVK_ANSI_Keypad0;		//		= 256,
				keymap[SDLK_KP1]	= kVK_ANSI_Keypad1;		//		= 257,
				keymap[SDLK_KP2]	= kVK_ANSI_Keypad2;		//		= 258,
				keymap[SDLK_KP3]	= kVK_ANSI_Keypad3;		//		= 259,
				keymap[SDLK_KP4]	= kVK_ANSI_Keypad4;		//		= 260,
				keymap[SDLK_KP5]	= kVK_ANSI_Keypad5;		//		= 261,
				keymap[SDLK_KP6]	= kVK_ANSI_Keypad6;		//		= 262,
				keymap[SDLK_KP7]	= kVK_ANSI_Keypad7;		//		= 263,
				keymap[SDLK_KP8]	= kVK_ANSI_Keypad8;		//		= 264,
				keymap[SDLK_KP9]	= kVK_ANSI_Keypad9;		//		= 265,
				keymap[SDLK_KP_PERIOD]	= kVK_ANSI_KeypadDecimal;	//		= 266,
				keymap[SDLK_KP_DIVIDE]	= kVK_ANSI_KeypadDivide;	//		= 267,
				keymap[SDLK_KP_MULTIPLY]= kVK_ANSI_KeypadMultiply;	//		= 268,
				keymap[SDLK_KP_MINUS]	= kVK_ANSI_KeypadMinus;		//		= 269,
				keymap[SDLK_KP_PLUS]	= kVK_ANSI_KeypadPlus;		//		= 270,
				keymap[SDLK_KP_ENTER]	= kVK_ANSI_KeypadEnter;		//		= 271,
				keymap[SDLK_KP_EQUALS]	= kVK_ANSI_KeypadEquals;	//		= 272,
				/** @name Arrows + Home/End pad */
				keymap[SDLK_UP]		= kVK_UpArrow;			//		= 273,
				keymap[SDLK_DOWN]	= kVK_DownArrow;		//		= 274,
				keymap[SDLK_RIGHT]	= kVK_RightArrow;		//		= 275,
				keymap[SDLK_LEFT]	= kVK_LeftArrow;		//		= 276,
				//keymap[SDLK_INSERT]	= VK_INSERT;			//		= 277,
				keymap[SDLK_HOME]	= kVK_Home;			//		= 278,
				keymap[SDLK_END]	= kVK_End;			//		= 279,
				keymap[SDLK_PAGEUP]	= kVK_PageUp;			//		= 280,
				keymap[SDLK_PAGEDOWN]	= kVK_PageDown;			//		= 281,
				/** @name Function keys */
				keymap[SDLK_F1]		= kVK_F1;	//		= 282,
				keymap[SDLK_F2]		= kVK_F2;	//		= 283,
				keymap[SDLK_F3]		= kVK_F3;	//		= 284,
				keymap[SDLK_F4]		= kVK_F4;	//		= 285,
				keymap[SDLK_F5]		= kVK_F5;	//		= 286,
				keymap[SDLK_F6]		= kVK_F6;	//		= 287,
				keymap[SDLK_F7]		= kVK_F7;	//		= 288,
				keymap[SDLK_F8]		= kVK_F8;	//		= 289,
				keymap[SDLK_F9]		= kVK_F9;	//		= 290,
				keymap[SDLK_F10]	= kVK_F10;	//		= 291,
				keymap[SDLK_F11]	= kVK_F11;	//		= 292,
				keymap[SDLK_F12]	= kVK_F12;	//		= 293,
				keymap[SDLK_F13]	= kVK_F13;	//		= 294,
				keymap[SDLK_F14]	= kVK_F14;	//		= 295,
				keymap[SDLK_F15]	= kVK_F15;	//		= 296,
				/** @name Key state modifier keys */
				//keymap[SDLK_NUMLOCK]	= VK_NUMLOCK;	//		= 300,
				keymap[SDLK_CAPSLOCK]	= kVK_CapsLock;	//		= 301,
				//keymap[SDLK_SCROLLOCK]	= VK_SCROLL;	//		= 302,
				keymap[SDLK_RSHIFT]	= kVK_RightShift;	//		= 303,
				keymap[SDLK_LSHIFT]	= kVK_Shift;	//		= 304,
				keymap[SDLK_RCTRL]	= kVK_RightControl;	//		= 305,
				keymap[SDLK_LCTRL]	= kVK_Control;	//		= 306,
				keymap[SDLK_RALT]	= kVK_RightOption;	//		= 307,
				keymap[SDLK_LALT]	= kVK_Option;	//		= 308,
				keymap[SDLK_RMETA]	= kVK_Command;	//		= 309,
				keymap[SDLK_LMETA]	= kVK_Command;	//		= 310,
				//keymap[SDLK_LSUPER]	= XK_Super_L;	//		= 311,		/**< Left "Windows" key */
				//keymap[SDLK_RSUPER]	= XK_Super_R;	//		= 312,		/**< Right "Windows" key */
				//keymap[SDLK_MODE]	= VK_MODECHANGE;//		= 313,		/**< "Alt Gr" key */
				//keymap[SDLK_COMPOSE]	= XK_Multi_key;	//		= 314,		/**< Multi-key compose key */
				/** @name Miscellaneous function keys */
				keymap[SDLK_HELP]	= kVK_Help;	//		= 315,
				//keymap[SDLK_PRINT]	= VK_PRINT;	//		= 316,
				//keymap[SDLK_SYSREQ]	= XK_Sys_Req;	//		= 317,
				//keymap[SDLK_BREAK]	= VK_CANCEL;	//		= 318,
				//keymap[SDLK_MENU]	= VK_MENU;	//		= 319,
#if 0
				SDLK_POWER		= 320,		/**< Power Macintosh power key */
					SDLK_EURO		= 321,		/**< Some european keyboards */
#endif
					//keymap[SDLK_UNDO]	= XK_Undo;	//		= 322,		/**< Atari keyboard has Undo */
					//
					keymap_initialized = true;
				return;
		}
#elif defined ANDROID
		static void
			SDLKeyToKeySym_init() {
				// server codes do not support android
				return;
		}
#else // X11
		static void
			SDLKeyToKeySym_init() {
				unsigned short i;
				//
				keymap[SDLK_BACKSPACE]	= XK_BackSpace;		//		= 8,
				keymap[SDLK_TAB]	= XK_Tab;		//		= 9,
				keymap[SDLK_CLEAR]	= XK_Clear;		//		= 12,
				keymap[SDLK_RETURN]	= XK_Return;		//		= 13,
				keymap[SDLK_PAUSE]	= XK_Pause;		//		= 19,
				keymap[SDLK_ESCAPE]	= XK_Escape;		//		= 27,
				// Latin 1: starting from space (0x20)
				// (0x20) space, exclam, quotedbl, numbersign, dollar, percent, ampersand,
				// (0x27) quoteright, parentleft, parentright, asterisk, plus, comma,
				// (0x2d) minus, period, slash
				// (0x30-0x39) 0-9
				// (0x3a) colon, semicolon, less, equal, greater, question, at
				// (0x41-0x5a) A-Z
				// (0x5b) bracketleft, backslash, bracketright, asciicircum/caret,
				// (0x5f) underscore, grave
				// (0x61-7a) a-z
				for(i = 0x20; i <= 0x7a; i++) {
					keymap[i] = i;
				}
				keymap[SDLK_DELETE]	= XK_Delete;		//		= 127,
				// SDLK_WORLD_0 (0xa0) - SDLK_WORLD_95 (0xff) are ignored
				/** @name Numeric keypad */
				keymap[SDLK_KP0]	= XK_KP_0;	//		= 256,
				keymap[SDLK_KP1]	= XK_KP_1;	//		= 257,
				keymap[SDLK_KP2]	= XK_KP_2;	//		= 258,
				keymap[SDLK_KP3]	= XK_KP_3;	//		= 259,
				keymap[SDLK_KP4]	= XK_KP_4;	//		= 260,
				keymap[SDLK_KP5]	= XK_KP_5;	//		= 261,
				keymap[SDLK_KP6]	= XK_KP_6;	//		= 262,
				keymap[SDLK_KP7]	= XK_KP_7;	//		= 263,
				keymap[SDLK_KP8]	= XK_KP_8;	//		= 264,
				keymap[SDLK_KP9]	= XK_KP_9;	//		= 265,
				keymap[SDLK_KP_PERIOD]	= XK_KP_Delete;	//		= 266,
				keymap[SDLK_KP_DIVIDE]	= XK_KP_Divide;	//		= 267,
				keymap[SDLK_KP_MULTIPLY]= XK_KP_Multiply;	//	= 268,
				keymap[SDLK_KP_MINUS]	= XK_KP_Subtract;	//		= 269,
				keymap[SDLK_KP_PLUS]	= XK_KP_Add;	//		= 270,
				keymap[SDLK_KP_ENTER]	= XK_KP_Enter;	//		= 271,
				keymap[SDLK_KP_EQUALS]	= XK_KP_Equal;	//		= 272,
				/** @name Arrows + Home/End pad */
				keymap[SDLK_UP]		= XK_Up;	//		= 273,
				keymap[SDLK_DOWN]	= XK_Down;	//		= 274,
				keymap[SDLK_RIGHT]	= XK_Right;	//		= 275,
				keymap[SDLK_LEFT]	= XK_Left;	//		= 276,
				keymap[SDLK_INSERT]	= XK_Insert;	//		= 277,
				keymap[SDLK_HOME]	= XK_Home;	//		= 278,
				keymap[SDLK_END]	= XK_End;	//		= 279,
				keymap[SDLK_PAGEUP]	= XK_Page_Up;	//		= 280,
				keymap[SDLK_PAGEDOWN]	= XK_Page_Down;	//		= 281,
				/** @name Function keys */
				keymap[SDLK_F1]		= XK_F1;	//		= 282,
				keymap[SDLK_F2]		= XK_F2;	//		= 283,
				keymap[SDLK_F3]		= XK_F3;	//		= 284,
				keymap[SDLK_F4]		= XK_F4;	//		= 285,
				keymap[SDLK_F5]		= XK_F5;	//		= 286,
				keymap[SDLK_F6]		= XK_F6;	//		= 287,
				keymap[SDLK_F7]		= XK_F7;	//		= 288,
				keymap[SDLK_F8]		= XK_F8;	//		= 289,
				keymap[SDLK_F9]		= XK_F9;	//		= 290,
				keymap[SDLK_F10]	= XK_F10;	//		= 291,
				keymap[SDLK_F11]	= XK_F11;	//		= 292,
				keymap[SDLK_F12]	= XK_F12;	//		= 293,
				keymap[SDLK_F13]	= XK_F13;	//		= 294,
				keymap[SDLK_F14]	= XK_F14;	//		= 295,
				keymap[SDLK_F15]	= XK_F15;	//		= 296,
				/** @name Key state modifier keys */
				keymap[SDLK_NUMLOCK]	= XK_Num_Lock;	//		= 300,
				keymap[SDLK_CAPSLOCK]	= XK_Caps_Lock;	//		= 301,
				keymap[SDLK_SCROLLOCK]	= XK_Scroll_Lock;//		= 302,
				keymap[SDLK_RSHIFT]	= XK_Shift_R;	//		= 303,
				keymap[SDLK_LSHIFT]	= XK_Shift_L;	//		= 304,
				keymap[SDLK_RCTRL]	= XK_Control_R;	//		= 305,
				keymap[SDLK_LCTRL]	= XK_Control_L;	//		= 306,
				keymap[SDLK_RALT]	= XK_Alt_R;	//		= 307,
				keymap[SDLK_LALT]	= XK_Alt_L;	//		= 308,
				keymap[SDLK_RMETA]	= XK_Meta_R;	//		= 309,
				keymap[SDLK_LMETA]	= XK_Meta_L;	//		= 310,
				/** @name Miscellaneous function keys */
				keymap[SDLK_HELP]	= XK_Help;	//		= 315,
				keymap[SDLK_SYSREQ]	= XK_Sys_Req;	//		= 317,
				keymap[SDLK_BREAK]	= XK_Break;	//		= 318,
				keymap[SDLK_MENU]	= XK_Menu;	//		= 319,
#if 0
				SDLK_POWER		= 320,		/**< Power Macintosh power key */
					SDLK_EURO		= 321,		/**< Some european keyboards */
#endif
					keymap[SDLK_UNDO]	= XK_Undo;	//		= 322,		/**< Atari keyboard has Undo */
				//
				keymap_initialized = true;
				return;
		}
#endif

		static KeySym
			SDLKeyToKeySym(int sdlkey) {
				map<int, KeySym>::iterator mi;
				if(keymap_initialized == false) {
					SDLKeyToKeySym_init();
				}
				if((mi = keymap.find(sdlkey)) != keymap.end()) {
					return mi->second;
				}
				return INVALID_KEY;
		}

	}
}