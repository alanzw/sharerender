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
#include <stdlib.h>
#include <errno.h>
#include <string.h>

#include "../LibCore/MemOp.h"
#include "../LibCore/InfoRecorder.h"
#include "Controller.h"
#include "../LibCore/TimeTool.h"

//#define USE_INTERCEPTION


#ifdef USE_INTERCEPTION
#include "interception.h"

#pragma comment(lib, "interception.lib")


bool interceptionInited = false;
InterceptionDevice keyboard = INTERCEPTION_KEYBOARD(1);
InterceptionContext context;


void InitInterception(){
	if(!interceptionInited){
		SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
		context = interception_create_context();
		interceptionInited = true;
	}

}

void generateKey(InterceptionDevice device, int up){
	InterceptionKeyStroke kstroke;
	memset(&kstroke, 0, sizeof(InterceptionKeyStroke));

	//InterceptionKeyStroke &kstroke = *(InterceptionKeyStroke *) &stroke;
	kstroke.code = 87;
	kstroke.state = up;  // 0 press, 1 up
	kstroke.information = 0;

	//print_key_info((InterceptionKeyStroke*)&stroke);
	interception_send(context, device, (InterceptionStroke*)&kstroke, 1);
}


#endif

using namespace std;
using namespace cg;
using namespace cg::core;


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
static bool keymap_initialized = false;
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

cg::core::BTimer * ctrlTimer = NULL;

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

		static char *myctrlid = NULL;
		static bool ctrlenabled = true;

		static bool clientControlCreated = false;

		void CreateClientControl(HWND wnd){
#ifdef ENABLE_CLIENT_CONTROL

			if(clientControlCreated)
				return;

			clientControlCreated = true;

			RECT windowRect;
			memset(&windowRect, 0, sizeof(RECT));


#if 0
			cg::input::CtrlConfig * conf = NULL;
			conf = cg::input::CtrlConfig::GetCtrlConfig(STREAM_SERVER_CONFIG);
#else

			cg::RTSPConf *conf = cg::RTSPConf::GetRTSPConf();
#endif
			cg::input::CtrlMessagerServer * ctrlServer = new cg::input::CtrlMessagerServer();
			do{
				if (conf->ctrlEnable){
					if (ctrlServer->initQueue(32768, sizeof(cg::input::sdlmsg_t)) < 0){
						conf->ctrlEnable = 0;
						goto CTRL_CLEAN;
					}
					// create the replay callback

					GetWindowRect(wnd, &windowRect);

					ReplayCallbackImp * replayCallback = new ReplayCallbackImp(&windowRect, &windowRect);
					if(ctrlServer->init(conf, CTRL_CURRENT_VERSION)){
						cg::core::infoRecorder->logError("[SERVER]: cannot start the input thread.\n");
						goto CTRL_CLEAN;	
					}

					ctrlServer->setReplay(replayCallback);

					if (!ctrlServer->start()){
						cg::core::infoRecorder->logError("Cannot create controller thread, controller disable\n");
						conf->ctrlEnable = 0;
						goto CTRL_CLEAN;
					}
				}
				else{
					cg::core::infoRecorder->logError("[MsgServer]: client control is disabled.\n");
					goto CTRL_CLEAN;
				}
				//enableRender = conf->enableRender;
			} while (0);

			return;

CTRL_CLEAN:
			if(ctrlServer){
				delete ctrlServer;
				ctrlServer = NULL;
			}

#endif  // ENABLE_CLIENT_CONTROL
		}
		
		ReplayCallbackImp::ReplayCallbackImp(RECT * windowRect, RECT * pOutputRect): prect(NULL), pOutputWindowRect(NULL), scaleFactorX(0), scaleFactorY(0),cxsize(0), cysize(0)
		{
			if(windowRect){
				prect = new RECT;
				memcpy(prect, windowRect, sizeof(RECT));
			}
			if(pOutputRect){
				pOutputWindowRect = new RECT;
				memcpy(pOutputWindowRect, pOutputRect, sizeof(RECT));
			}
			initKeyMap();
		}

		ReplayCallbackImp::~ReplayCallbackImp(){
			if(prect){
				delete prect;
				prect = NULL;
			}
			if(pOutputWindowRect){
				delete pOutputWindowRect;
				pOutputWindowRect = NULL;
			}
		}
		
		int ReplayCallbackImp::initKeyMap(){
			if(keymap_initialized == false){
				SDLKeyToKeySym_init();
			}
			// set the scaler 

			// get the screen resolution
			cxsize = GetSystemMetrics(SM_CXSCREEN);
			cysize = GetSystemMetrics(SM_CYSCREEN);


			// compute scale factor
			do {
				int baseX, baseY;
				float outputW = 0; int outputH = 0;
				if(pOutputWindowRect == NULL){
					cg::core::infoRecorder->logError("[Callback]: NULL output window rect.\n");
					outputW = cxsize;
					outputH = cysize;
				}
				outputW = pOutputWindowRect->right - pOutputWindowRect->left + 1;
				outputH = pOutputWindowRect->bottom - pOutputWindowRect->top + 1;
				//
				if (prect == NULL) {
					baseX = cxsize;
					baseY = cysize;
				}
				else {
					baseX = prect->right - prect->left + 1;
					baseY = prect->bottom - prect->top + 1;
				}

				float rx, ry;
				rx = 1.0 * baseX / outputW;
				ry = 1.0 * baseY / outputH;

				if(rx <= 0.0)	rx = 1.0;
				if(ry <= 0.0)	ry = 1.0;

				scaleFactorX = rx;
				scaleFactorY = ry;

				cg::core::infoRecorder->logTrace("[Callback]: sdl replayer: mouse coordinate scale factor = (%.3f,%.3f)\n",
					scaleFactorX, scaleFactorY);
			} while (0);


			return 0;
		}

		//void sdlmsg_replay_deinit(void *arg) {
		//#ifdef WIN32
		//#elif defined __APPLE__
		//#elif defined ANDROID
		//#else	// X11
		//	if(display) {
		//		XCloseDisplay(display);
		//		display = NULL;
		//	}
		//#endif
		//	return;
		//}

#ifdef WIN32

		void ReplayCallbackImp::replayNative(sdlmsg_t *msg){
			INPUT				in;
			sdlmsg_keyboard_t *	msgk = (sdlmsg_keyboard_t*) msg;
			sdlmsg_mouse_t *	msgm = (sdlmsg_mouse_t*) msg;
			cg::core::DelayRecorder * delayRecorder = cg::core::DelayRecorder::GetDelayRecorder();
			if(!ctrlTimer){
				ctrlTimer = new PTimer();
			}

			switch(msg->msgtype) {
			case SDL_EVENT_MSGTYPE_KEYBOARD:
				bzero(&in, sizeof(in));
				in.type = INPUT_KEYBOARD;
				if((in.ki.wVk = SDLKeyToKeySym(msgk->sdlkey)) != INVALID_KEY) {
					if(msgk->is_pressed == 0) {
						in.ki.dwFlags |= KEYEVENTF_KEYUP;
					}
					in.ki.wScan = MapVirtualKey(in.ki.wVk, MAPVK_VK_TO_VSC);

					if(in.ki.wVk == VK_F11 && msgk->is_pressed == 0){
						ctrlTimer->Start();
						delayRecorder->setInputArrive();
					}

					cg::core::infoRecorder->logTrace("sdl replayer: vk=%x scan=%x, is press:%d\n", in.ki.wVk, in.ki.wScan, msgk->is_pressed);
#ifndef USE_INTERCEPTION
					SendInput(1, &in, sizeof(in));

#else
					InitInterception();
					generateKey(keyboard, msgk->is_pressed == 0 ? 1 : 0);
#endif
				} else {
					////////////////
					cg::core::infoRecorder->logTrace("sdl replayer: undefined key scan=%u(%04x) key=%u(%04x) mod=%u(%04x) pressed=%d\n",
						msgk->scancode, msgk->scancode,
						msgk->sdlkey, msgk->sdlkey, msgk->sdlmod, msgk->sdlmod,
						msgk->is_pressed);
					////////////////
				}
				break;
			case SDL_EVENT_MSGTYPE_MOUSEKEY:
				cg::core::infoRecorder->logTrace("sdl replayer: button event btn=%u pressed=%d\n", msg->mousebutton, msg->is_pressed);
				bzero(&in, sizeof(in));
				in.type = INPUT_MOUSE;
				if(msgm->mousebutton == 1 && msgm->is_pressed != 0) {
					in.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
				} else if(msgm->mousebutton == 1 && msgm->is_pressed == 0) {
					in.mi.dwFlags = MOUSEEVENTF_LEFTUP;
				} else if(msgm->mousebutton == 2 && msgm->is_pressed != 0) {
					in.mi.dwFlags = MOUSEEVENTF_MIDDLEDOWN;
				} else if(msgm->mousebutton == 2 && msgm->is_pressed == 0) {
					in.mi.dwFlags = MOUSEEVENTF_MIDDLEUP;
				} else if(msgm->mousebutton == 3 && msgm->is_pressed != 0) {
					in.mi.dwFlags = MOUSEEVENTF_RIGHTDOWN;
				} else if(msgm->mousebutton == 3 && msgm->is_pressed == 0) {
					in.mi.dwFlags = MOUSEEVENTF_RIGHTUP;
				} else if(msgm->mousebutton == 4 && msgm->is_pressed != 0) {
					// mouse wheel forward
					in.mi.dwFlags = MOUSEEVENTF_WHEEL;
					in.mi.mouseData = +WHEEL_DELTA;
				} else if(msgm->mousebutton == 5 && msgm->is_pressed != 0) {
					// mouse wheel backward
					in.mi.dwFlags = MOUSEEVENTF_WHEEL;
					in.mi.mouseData = -WHEEL_DELTA;
				}
				SendInput(1, &in, sizeof(in));
				break;
			case SDL_EVENT_MSGTYPE_MOUSEWHEEL:
				if(msgm->mousex != 0) {
					bzero(&in, sizeof(in));
					in.type = INPUT_MOUSE;
					if(((short) msgm->mousex) > 0) {
						// mouse wheel forward
						in.mi.dwFlags = MOUSEEVENTF_WHEEL;
						in.mi.mouseData = +WHEEL_DELTA;
					} else if(((short) msgm->mousex) < 0 ) {
						// mouse wheel backward
						in.mi.dwFlags = MOUSEEVENTF_WHEEL;
						in.mi.mouseData = -WHEEL_DELTA;
					}
					SendInput(1, &in, sizeof(in));
				}
#if 0
				if(msgm->mousey != 0) {
					bzero(&in, sizeof(in));
					in.type = INPUT_MOUSE;
					if(((short) msgm->mousey) > 0) {
						// mouse wheel forward
						in.mi.dwFlags = MOUSEEVENTF_HWHEEL;
						in.mi.mouseData = +WHEEL_DELTA;
					} else if(((short) msgm->mousey) < 0 ) {
						// mouse wheel backward
						in.mi.dwFlags = MOUSEEVENTF_HWHEEL;
						in.mi.mouseData = -WHEEL_DELTA;
					}
					SendInput(1, &in, sizeof(in));
				}
#endif
				break;
			case SDL_EVENT_MSGTYPE_MOUSEMOTION:
				cg::core::infoRecorder->logTrace("sdl replayer: motion event x=%u y=%d, is relative: %s\n", msgm->mousex, msgm->mousey, msgm->relativeMouseMode !=0 ? "true" : "false");
				bzero(&in, sizeof(in));
				in.type = INPUT_MOUSE;
				// mouse x/y has to be mapped to (0,0)-(65535,65535)
				if(msgm->relativeMouseMode == 0) {
					if(prect == NULL) {
						in.mi.dx = (DWORD)
							(65536.0 * scaleFactorX * msgm->mousex) / cxsize;
						in.mi.dy = (DWORD)
							(65536.0 * scaleFactorY * msgm->mousey) / cysize;
					} else {
						in.mi.dx = (DWORD)
							(65536.0 * (prect->left + scaleFactorX * msgm->mousex)) / cxsize;
						in.mi.dy = (DWORD)
							(65536.0 * (prect->top + scaleFactorY * msgm->mousey)) / cysize;
					}
					in.mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE;
				} else {
					in.mi.dx = (short) (scaleFactorX * msgm->mouseRelX);
					in.mi.dy = (short) (scaleFactorY * msgm->mouseRelY);
					in.mi.dwFlags = MOUSEEVENTF_MOVE;
				}
				cg::core::infoRecorder->logTrace("mouse (x, y):(%d, %d).\n", in.mi.dx, in.mi.dy);
				SendInput(1, &in, sizeof(in));
				break;
			default: // do nothing
				break;
			}
			return;
		}

#elif defined __APPLE__
		static void
			sdlmsg_replay_native(sdlmsg_t *msg) {
				// read: CGEventCreateMouseEvent()
				//	 CGEventCreateKeyboardEvent()
				//	 CGEventPost()
				//	 CGEventPostToPSN()
				//	 https://developer.apple.com/library/mac/#documentation/Carbon/Reference/QuartzEventServicesRef/Reference/reference.html
				static CGPoint pt = { 0, 0 };
				CGKeyCode kcode;
				CGEventRef event = NULL;
				sdlmsg_keyboard_t *msgk = (sdlmsg_keyboard_t*) msg;
				sdlmsg_mouse_t *msgm = (sdlmsg_mouse_t*) msg;
				//
				switch(msg->msgtype) {
				case SDL_EVENT_MSGTYPE_KEYBOARD:
					// key codes: /System/Library/Frameworks/Carbon.framework/Versions/A/Frameworks/HIToolbox.framework/Versions/A/Headers/Events.h
					if((kcode = SDLKeyToKeySym(msgk->sdlkey)) != INVALID_KEY) {
						event = CGEventCreateKeyboardEvent(NULL, kcode, msgk->is_pressed ? true : false);
					} else {
						////////////////
						ga_error("sdl replayer: undefined key scan=%u(%04x) key=%u(%04x) mod=%u(%04x) pressed=%d\n",
							msgk->scancode, msgk->scancode,
							msgk->sdlkey, msgk->sdlkey, msgk->sdlmod, msgk->sdlmod,
							msgk->is_pressed);
						////////////////
					}
					break;
				case SDL_EVENT_MSGTYPE_MOUSEKEY:
					//ga_error("sdl replayer: button event btn=%u pressed=%d\n", msg->mousebutton, msg->is_pressed);
					if(msgm->mousebutton == 1 && msgm->is_pressed != 0) {
						event = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseDown, pt, kCGMouseButtonLeft);
					} else if(msgm->mousebutton == 1 && msgm->is_pressed == 0) {
						event = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseUp, pt, kCGMouseButtonLeft);
					} else if(msgm->mousebutton == 2 && msgm->is_pressed != 0) {
						event = CGEventCreateMouseEvent(NULL, kCGEventOtherMouseDown, pt, kCGMouseButtonCenter);
					} else if(msgm->mousebutton == 2 && msgm->is_pressed == 0) {
						event = CGEventCreateMouseEvent(NULL, kCGEventOtherMouseUp, pt, kCGMouseButtonCenter);
					} else if(msgm->mousebutton == 3 && msgm->is_pressed != 0) {
						event = CGEventCreateMouseEvent(NULL, kCGEventRightMouseDown, pt, kCGMouseButtonRight);
					} else if(msgm->mousebutton == 3 && msgm->is_pressed == 0) {
						event = CGEventCreateMouseEvent(NULL, kCGEventRightMouseUp, pt, kCGMouseButtonRight);
					} else if(msgm->mousebutton == 4 && msgm->is_pressed != 0) {
						// mouse wheel forward
						event = CGEventCreateScrollWheelEvent(NULL, kCGScrollEventUnitLine, 1, -5);
					} else if(msgm->mousebutton == 5 && msgm->is_pressed != 0) {
						// mouse wheel backward
						event = CGEventCreateScrollWheelEvent(NULL, kCGScrollEventUnitLine, 1, +5);
					}
					break;
				case SDL_EVENT_MSGTYPE_MOUSEWHEEL:
					do {
						int deltaX, deltaY;
						deltaX = deltaY = 0;
						if(((short) msgm->mousex) > 0) {
							deltaX = -5;	// move wheel forward
						} else if(((short) msgm->mousex) < 0) {
							deltaX = +5;	// move wheel backward
						}
						if(((short) msgm->mousey) > 0) {
							deltaY = +5;	// move wheel right
						} else if(((short) msgm->mousey) < 0) {
							deltaY = -5;	// move wheel left
						}
						if(deltaX == 0 && deltaY == 0) {
							break;
						}
						event = CGEventCreateScrollWheelEvent(NULL, kCGScrollEventUnitLine, 2, deltaX, deltaY);
					} while(0);
					break;
				case SDL_EVENT_MSGTYPE_MOUSEMOTION:
					//ga_error("sdl replayer: motion event x=%u y=%d\n", msg->mousex, msg->mousey);
					if(prect == NULL) {
						pt.x = scaleFactorX * msgm->mousex;
						pt.y = scaleFactorY * msgm->mousey;
					} else {
						pt.x = prect->left + scaleFactorX * msgm->mousex;
						pt.y = prect->top + scaleFactorY * msgm->mousey;
					}
					event = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, pt, 0);
					break;
				default: // do nothing
					break;
				}
				// Post the event
				if(event != NULL) {
					CGEventPost(kCGHIDEventTap, event);
					CFRelease(event);
					event = NULL;
				}
				return;
		}
#elif defined ANDROID
		static void
			sdlmsg_replay_native(sdlmsg_t *msg) {
				// server codes do snot support android
				return;
		}
#else	// X11
		static void
			sdlmsg_replay_native(sdlmsg_t *msg) {
				static KeyCode kcode;
				static KeySym ksym;
				sdlmsg_keyboard_t *msgk = (sdlmsg_keyboard_t*) msg;
				sdlmsg_mouse_t *msgm = (sdlmsg_mouse_t*) msg;
				//
				switch(msg->msgtype) {
				case SDL_EVENT_MSGTYPE_KEYBOARD:
					// read: http://forum.tuts4you.com/topic/23722-gtk-linux-trojan/
					//	http://www.libsdl.org/docs/html/sdlkey.html
					// headers: X11/keysym.h, keysymdef.h, SDL/SDK_keysym.h
					if((ksym = SDLKeyToKeySym(msgk->sdlkey)) != INVALID_KEY) {
						//////////////////
						if((kcode = XKeysymToKeycode(display, ksym)) > 0) {
							XTestGrabControl(display, True);
							XTestFakeKeyEvent(display, kcode,
								msgk->is_pressed ? True : False, CurrentTime);
							XSync(display, True);
							XTestGrabControl(display, False);
						}
#if 0
						ga_error("sdl replayer: received key scan=%u(%04x) key=%u(%04x) mod=%u(%04x) pressed=%d\n",
							msg->scancode, msg->scancode, msg->sdlkey, msg->sdlkey, msg->sdlmod, msg->sdlmod,
							msg->is_pressed);
#endif
						//////////////////
					} else {
						////////////////
						ga_error("sdl replayer: undefined key scan=%u(%04x) key=%u(%04x) mod=%u(%04x) pressed=%d\n",
							msgk->scancode, msgk->scancode,
							msgk->sdlkey, msgk->sdlkey, msgk->sdlmod, msgk->sdlmod,
							msgk->is_pressed);
						////////////////
					}
					break;
				case SDL_EVENT_MSGTYPE_MOUSEKEY:
					//ga_error("sdl replayer: button event btn=%u pressed=%d\n", msg->mousebutton, msg->is_pressed);
					XTestGrabControl(display, True);
					XTestFakeButtonEvent(display, msgm->mousebutton,
						msgm->is_pressed ? True : False, CurrentTime);
					XSync(display, True);
					XTestGrabControl(display, False);
					break;
				case SDL_EVENT_MSGTYPE_MOUSEWHEEL:
					if(msgm->mousex != 0) {
						XTestGrabControl(display, True);
						if(((short) msgm->mousex) > 0) {
							// mouse wheel forward
							XTestFakeButtonEvent(display, 4, True, CurrentTime);
							XSync(display, True);
							XTestFakeButtonEvent(display, 4, False, CurrentTime);
						} else if(((short) msgm->mousex) < 0 ) {
							// mouse wheel backward
							XTestFakeButtonEvent(display, 5, True, CurrentTime);
							XSync(display, True);
							XTestFakeButtonEvent(display, 5, False, CurrentTime);
						}
						XSync(display, True);
						XTestGrabControl(display, False);
					}
					break;
				case SDL_EVENT_MSGTYPE_MOUSEMOTION:
					//ga_error("sdl replayer: motion event x=%u y=%d\n", msg->mousex, msg->mousey);
					XTestGrabControl(display, True);
					if(prect == NULL) {
						XTestFakeMotionEvent(display, screenNumber,
							(int) (scaleFactorX * msgm->mousex),
							(int) (scaleFactorY * msgm->mousey), CurrentTime);
					} else {
						XTestFakeMotionEvent(display, screenNumber,
							(int) (prect->left + scaleFactorX * msgm->mousex),
							(int) (prect->top + scaleFactorY * msgm->mousey), CurrentTime);
					}
					XSync(display, True);
					XTestGrabControl(display, False);
					break;
				default: // do nothing
					break;
				}
				return;
		}
#endif

		void ReplayCallbackImp::operator()(void * buf, int len){
			sdlmsg_t *m = (sdlmsg_t *)buf;
			if(len  != ntohs(m->msgsize)){
				cg::core::infoRecorder->logError("[ReplaCallbackImp]: msg length mismatched. (%d != %d).\n", len, ntohs(m->msgsize));
			}
			replay((sdlmsg_t*)buf);
		}
		void ReplayCallbackImp::replay(sdlmsg_t * msg){
			sdlmsg_ntoh(msg);
			replayNative(msg);
		}
		

		//

#if defined(GA_EMCC) && !defined(INADDR_NONE)
#define INADDR_NONE	((unsigned long int) 0xffffffff)
#endif

#ifdef WIN32
		static unsigned long
#else
		static in_addr_t
#endif
			name_resolve(const char *hostname) {
				struct in_addr addr;
				struct hostent *hostEnt;
				if((addr.s_addr = inet_addr(hostname)) == INADDR_NONE) {
					if((hostEnt = gethostbyname(hostname)) == NULL)
						return INADDR_NONE;
					bcopy(hostEnt->h_addr, (char *) &addr.s_addr, hostEnt->h_length);
				}
				return addr.s_addr;
		}
		// for queue messager
		int QueueMessager::initQueue(int size, int maxunit){

			InitializeCriticalSection(&queueSection);
			pQueueSection = &queueSection;

			qunit = maxunit + sizeof(struct queuemsg);
			qsize = size - (size % qunit);
			qhead = qtail = 0;
			if ((qbuffer = (unsigned char*)malloc(qsize)) == NULL) {
				return -1;
			}
			infoRecorder->logError("[QueueMessager]:controller queue initialized size=%d (%d units)\n",
				qsize, qsize / qunit);
			return 0;
		}
		void QueueMessager::release(){
			//WaitForSingleObject(queueMutex, INFINITE);
			EnterCriticalSection(pQueueSection);
			if (qbuffer != NULL)
				free(qbuffer);
			qbuffer = NULL;
			qhead = qtail = qsize = 0;
			//ReleaseMutex(queueMutex);
			LeaveCriticalSection(pQueueSection);
		}
		struct queuemsg * QueueMessager::readMsg(){
			struct queuemsg *msg;
			//
			//WaitForSingleObject(queueMutex, INFINITE);
			EnterCriticalSection(pQueueSection);
			if (qbuffer == NULL) {
				//ReleaseMutex(queueMutex);
				LeaveCriticalSection(pQueueSection);
				infoRecorder->logError("[QueueMessager]:controller queue buffer released.\n");
				return NULL;
			}
			if (qtail == qhead) {
				// queue is empty
				msg = NULL;
			}
			else {
				msg = (struct queuemsg *) (qbuffer + qhead);
			}
			//ReleaseMutex(queueMutex);
			LeaveCriticalSection(pQueueSection);
			//
			return msg;
		}
		void QueueMessager::releaseMsg(struct queuemsg  * msg){
			struct queuemsg *currmsg;
			//WaitForSingleObject(queueMutex, INFINITE);
			EnterCriticalSection(pQueueSection);
			if (qbuffer == NULL) {
				//ReleaseMutex(queueMutex);
				LeaveCriticalSection(pQueueSection);
				infoRecorder->logError("[QueueMessager]:controller queue buffer released.\n");
				return;
			}
			if (qhead == qtail) {
				// queue is empty
				//ReleaseMutex(queueMutex);
				LeaveCriticalSection(pQueueSection);
				return;
			}
			currmsg = (struct queuemsg *) (qbuffer + qhead);
			if (msg != currmsg) {
				infoRecorder->logError("[QueueMessager]: WARNING - release an incorrect msg?\n");
			}
			qhead += qunit;
			if (qhead == qsize) {
				qhead = 0;
			}
			//ReleaseMutex(queueMutex);
			LeaveCriticalSection(pQueueSection);
			return;
		}
		int QueueMessager::writeMsg(void * msg, int msgsize){
			int nextpos;
			struct queuemsg *qmsg;
			//
			infoRecorder->logTrace("[QueueMessager]: write msg, size:%d.\n", msgsize);
			if ((msgsize + sizeof(struct queuemsg)) > qunit) {
				infoRecorder->logError("[QueueMessager]:controller queue msg size exceeded (%d > %d).\n",
					msgsize + sizeof(struct queuemsg), qunit);
				return 0;
			}
			//WaitForSingleObject(queueMutex, INFINITE);
			EnterCriticalSection(pQueueSection);
			if (qbuffer == NULL) {
				//ReleaseMutex(queueMutex);
				LeaveCriticalSection(pQueueSection);
				infoRecorder->logError("[QueueMessager]:controller queue buffer released.\n");
				return 0;
			}
			//
			nextpos = qtail + qunit;
			if (nextpos == qsize) {
				nextpos = 0;
			}
			//
			if (nextpos == qhead) {
				// queue is full
				msgsize = 0;
			}
			else {
				qmsg = (struct queuemsg*) (qbuffer + qtail);
				qmsg->msgsize = msgsize;
				if (msgsize > 0)
					bcopy(msg, qmsg->msg, msgsize);
				qtail = nextpos;
			}
			//ReleaseMutex(queueMutex);
			LeaveCriticalSection(pQueueSection);
			//
			return msgsize;
		}
		void QueueMessager::clear(){
			//WaitForSingleObject(queueMutex, INFINITE);
			EnterCriticalSection(pQueueSection);
			qhead = qtail = 0;
			//ReleaseMutex(queueMutex);
			LeaveCriticalSection(pQueueSection);
		}
		QueueMessager::~QueueMessager(){
			// release all
			release();
		}

		////////////////////////////////////////////////////////////////////
#if 0
		int CtrlMessagerClient::ctrlSocketInit(CtrlConfig *conf) {
			//
			if (conf->ctrlproto == IPPROTO_TCP) {
				ctrlSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			}
			else if (conf->ctrlproto == IPPROTO_UDP) {
				ctrlSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
			}
			else {
				infoRecorder->logError("[CtrlMessagerClient]:Controller socket-init: not supported protocol.\n");
				return -1;
			}
			if (ctrlSocket < 0) {
				infoRecorder->logError("[CtrlMessagerClient]:Controller socket-init: %s\n", strerror(errno));
				return -1;
			}
			//
			bzero(&ctrlsin, sizeof(struct sockaddr_in));
			ctrlsin.sin_family = AF_INET;
			ctrlsin.sin_port = htons(conf->ctrlport);
			if (conf->ctrl_servername != NULL) {
				ctrlsin.sin_addr.s_addr = name_resolve(conf->ctrl_servername);
				if (ctrlsin.sin_addr.s_addr == INADDR_NONE) {
					infoRecorder->logError("[CtrlMessagerClient]:Name resolution failed: %s\n", conf->ctrl_servername);
					return -1;
				}
			}
			infoRecorder->logError("[CtrlMessagerClient]:controller socket: socket address [%s:%u], sock:%p\n",
				inet_ntoa(ctrlsin.sin_addr), ntohs(ctrlsin.sin_port), ctrlSocket);
			//
			return ctrlSocket;
		}
		

		int CtrlMessagerServer::ctrlSocketInit(CtrlConfig *conf) {
			//
			infoRecorder->logError("[CstrlMessagerServer]: port:%d.\n", conf->ctrlport);
			if (conf->ctrlproto == IPPROTO_TCP) {
				ctrlSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			}
			else if (conf->ctrlproto == IPPROTO_UDP) {
				ctrlSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
			}
			else {
				infoRecorder->logError("[CtrlMessagerServer]:Controller socket-init: not supported protocol. configured protocol:%d\n", conf->ctrlproto);
				return -1;
			}
			if (ctrlSocket < 0) {
				infoRecorder->logError("[CtrlMessagerServer]:Controller socket-init: %s\n", strerror(errno));
			}
			//
			bzero(&ctrlsin, sizeof(struct sockaddr_in));
			ctrlsin.sin_family = AF_INET;
			ctrlsin.sin_port = htons(conf->ctrlport);

#if 0
			if (conf->ctrl_servername != NULL) {
				ctrlsin.sin_addr.s_addr = name_resolve(conf->ctrl_servername);
				if (ctrlsin.sin_addr.s_addr == INADDR_NONE) {
					infoRecorder->logError("Name resolution failed: %s\n", conf->ctrl_servername);
					return -1;
				}
			}
			infoRecorder->logError("controller socket: socket address [%s:%u]\n",
				inet_ntoa(ctrlsin.sin_addr), ntohs(ctrlsin.sin_port));
#endif
			//
			return ctrlSocket;
		}
#endif
		int CtrlMessagerClient::ctrlSocketInit(struct cg::RTSPConf *conf) {
			//
			if(conf->ctrlProto == IPPROTO_TCP) {
				ctrlSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			} else if(conf->ctrlProto == IPPROTO_UDP) {
				ctrlSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
			} else {
				infoRecorder->logError("[CtrlMessagerServer]:Controller socket-init: not supported protocol.\n");
				return -1;
			}
			if (ctrlSocket < 0) {
				infoRecorder->logError("[CtrlMessagerServer]:Controller socket-init: %s\n", strerror(errno));
			}
			//
			bzero(&ctrlsin, sizeof(struct sockaddr_in));
			ctrlsin.sin_family = AF_INET;
			ctrlsin.sin_port = htons(conf->ctrlPort);
			if (ctrlServerUrl != NULL) {
				ctrlsin.sin_addr.s_addr = name_resolve(ctrlServerUrl);
				if(ctrlsin.sin_addr.s_addr == INADDR_NONE) {
					infoRecorder->logError("[CtrlMessagerServer]:Name resolution failed: %s\n", ctrlServerUrl);
					return -1;
				}
			}
			infoRecorder->logError("[CtrlMessagerServer]:controller socket: socket address [%s:%u]\n",
				inet_ntoa(ctrlsin.sin_addr), ntohs(ctrlsin.sin_port));
			//
			return ctrlSocket;
		}
		int CtrlMessagerServer::ctrlSocketInit(struct RTSPConf *conf) {
			//
			if (conf->ctrlProto == IPPROTO_TCP) {
				ctrlSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			}
			else if (conf->ctrlProto == IPPROTO_UDP) {
				ctrlSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
			}
			else {
				infoRecorder->logError("[CtrlMessagerServer]:Controller socket-init: not supported protocol. configured protocol:%d\n", conf->ctrlProto);
				return -1;
			}
			if (ctrlSocket < 0) {
				infoRecorder->logError("[CtrlMessagerServer]:Controller socket-init: %s\n", strerror(errno));
			}
			//
			bzero(&ctrlsin, sizeof(struct sockaddr_in));
			ctrlsin.sin_family = AF_INET;
			ctrlsin.sin_port = htons(conf->ctrlPort);

#if 0
			if (conf->servername != NULL) {
				ctrlsin.sin_addr.s_addr = name_resolve(conf->servername);
				if (ctrlsin.sin_addr.s_addr == INADDR_NONE) {
					infoRecorder->logError("Name resolution failed: %s\n", conf->servername);
					return -1;
				}
			}
			infoRecorder->logError("controller socket: socket address [%s:%u]\n",
				inet_ntoa(ctrlsin.sin_addr), ntohs(ctrlsin.sin_port));
#endif
			//
			return ctrlSocket;
		}

		////////////////////////////////////////////////////////////////////

		int CtrlMessagerClient::init(cg::RTSPConf* rtspConf,char * url, const char * ctrlid){
			
			conf = rtspConf;

			ctrlServerUrl = _strdup(url);

			if (ctrlSocketInit(conf) < 0) {
				conf->ctrlEnable = 0;
				return -1;
			}

			// init the critical section and the event
			InitializeCriticalSection(&wakeupMutex);
			wakeup = CreateEvent(NULL, FALSE, FALSE, NULL);
			if (conf->ctrlProto == IPPROTO_TCP) {
				struct ctrlhandshake hh;
				// connect to the server
				if (connect(ctrlSocket, (struct sockaddr*) &ctrlsin, sizeof(ctrlsin)) < 0) {
					infoRecorder->logError("controller client-connect: %s\n", strerror(errno));
					goto error;
				}
				// send handshake
				hh.length = 1 + strlen(ctrlid) + 1;	// msg total len, id, null-terminated
				if (hh.length > sizeof(hh))
					hh.length = sizeof(hh);
				strncpy(hh.id, ctrlid, sizeof(hh.id));
				if (send(ctrlSocket, (char*)&hh, hh.length, 0) <= 0) {
					infoRecorder->logError("controller client-send(handshake): %s\n", strerror(errno));
					goto error;
				}
				else{
					infoRecorder->logError("controller client-send(handshake): %s sock:%p\n", strerror(errno), ctrlSocket);
				}
			}
			return 0;
error:
			conf->ctrlEnable = 0;
			ctrlenabled = false;
			infoRecorder->logError("controller client: controller disabled.\n");
			closesocket(ctrlSocket);
			ctrlSocket = -1;
			return -1;
		}

		BOOL CtrlMessagerClient::onThreadStart(){
			if (conf == NULL){
				infoRecorder->logError("CtrlMessagerClient:: NULL rtspconfig\n");
				return FALSE;
			}

			infoRecorder->logError("controller client-thread started: tid=%ld.\n", getThreadId());

			return TRUE;
		}
		BOOL CtrlMessagerClient::run(){
			struct queuemsg *qm;
			EnterCriticalSection(&wakeupMutex);
			WaitForSingleObject(wakeup, INFINITE);
			//pthread_cond_wait(&wakeup, &wakeup_mutex);
			LeaveCriticalSection(&wakeupMutex);
			//
			while ((qm = readMsg()) != NULL) {
				int wlen;
				if (qm->msgsize == 0) {
					infoRecorder->logError("controller client: null message received, terminate the thread.\n");
					return FALSE;
				}
#ifdef ANDROID
				if (drop > 0)
					continue;
#endif
				if (conf->ctrlProto == IPPROTO_TCP) {
					if ((wlen = send(ctrlSocket, (char*)qm->msg, qm->msgsize, 0)) < 0) {
						infoRecorder->logError("controller client-send(tcp): %s, sock:%p\n", strerror(errno), ctrlSocket);
#ifdef ANDROID
						drop = 1;
#else
						return FALSE;
#endif
					}
					else{
						unsigned short ms = ntohs(*((unsigned short *)qm->msg));
						infoRecorder->logTrace("[controller client]: send(tcp) size:%d, msgsize:%d.\n", wlen, ms);
					}
				}
				else if (conf->ctrlProto == IPPROTO_UDP) {
					if ((wlen = sendto(ctrlSocket, (char*)qm->msg, qm->msgsize, 0, (struct sockaddr*) &ctrlsin, sizeof(ctrlsin))) < 0) {
						infoRecorder->logError("controller client-send(udp): %s\n", strerror(errno));
#ifdef ANDROID
						drop = 1;
#else
						return FALSE;
#endif
					}
				}
				releaseMsg(qm);
				//ga_error("controller client-debug: send msg (%d bytes)\n", wlen);
				return TRUE;
			}
		}
		BOOL CtrlMessagerClient::stop(){
			infoRecorder->logError("CtrlMessgerClient thread terminated: tid =%d\n", this->getThreadId());
			CThread::stop();
			return TRUE;
		}
		void CtrlMessagerClient::onQuit(){
			closesocket(ctrlSocket);
			ctrlSocket = -1;
			infoRecorder->logError("controller client-thread terminated: tid=%ld.\n", this->getThreadId());
		}
		void CtrlMessagerClient::sendMsg(void *msg, int msglen) {
			if(ctrlenabled == false) {
				infoRecorder->logError("controller client-sendmsg: controller was disabled.\n");
				return;
			}
			if(writeMsg(msg, msglen) != msglen) {
				infoRecorder->logError("controller client-sendmsg: queue full, message dropped.\n");
			} else {
				SetEvent(wakeup);
			}
			return;
		}
		void CtrlMessagerClient::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
			//return CThread::onThreadMsg(msg, wParam, lParam);
			// do nothing
		}
		CtrlMessagerClient::CtrlMessagerClient(){

		}
		CtrlMessagerClient::~CtrlMessagerClient(){
			
		}

		////////////////////////////////////////////////////////////////////
		void CtrlMessagerServer::onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam){
			//return CThread::onThreadMsg(msg, wParam, lParam);
			// do nothing
		}
		void CtrlMessagerServer::onQuit(){

		}
		CtrlMessagerServer::~CtrlMessagerServer(){
			
			if(replay){
				delete replay;
				replay = NULL;
			}
		}

#if 1
		int CtrlMessagerServer::init(struct RTSPConf *conf, const char *ctrlid) {

			this->conf = conf;

			if (ctrlSocketInit(conf) < 0){
				infoRecorder->logError("[CtrlMessagerServer]:init socket failed\n");
				return -1;
			}
			// init the msg queue
			//QueueMessager::initQueue(size, maxunit);
			// init the critical sections
			InitializeCriticalSection(&reslock);
			InitializeCriticalSection(&oreslock);

			clientaccepted = 0;
			wakeup = CreateEvent(NULL, FALSE, FALSE, NULL);

			myctrlid = _strdup(ctrlid);
			// reuse port
			do {
				int val = 1;
				if (setsockopt(ctrlSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&val, sizeof(val)) < 0) {
					infoRecorder->logError("[CtrlMessagerServer]:controller server-bind: %s\n", strerror(errno));
					goto error;
				}
			} while(0);
			// bind for either TCP of UDP
			if (bind(ctrlSocket, (struct sockaddr*) &ctrlsin, sizeof(ctrlsin)) < 0) {
				infoRecorder->logError("[CtrlMessagerServer]:controller server-bind: %s\n", strerror(errno));
				goto error;
			}
			// TCP listen
			if(conf->ctrlProto == IPPROTO_TCP) {
				if (listen(ctrlSocket, 16) < 0) {
					infoRecorder->logError("[CtrlMessagerServer]:controller server-listen: %s\n", strerror(errno));
					goto error;
				}
			}
			return 0;
error:
			closesocket(ctrlSocket);
			ctrlSocket = -1;
			return -1;
		}

#else
		int CtrlMessagerServer::init(CtrlConfig *conf, const char *ctrlid) {
			if (!conf){
				infoRecorder->logError("[CtrlMessagerServer]: NULL CtrlConsfig.\n");
				return -1;
			}
			this->conf = conf;
			if (ctrlSocketInit(conf) < 0){
				infoRecorder->logError("[CtrlMessagerServer]:init socket failed\n");
				return -1;
			}
			// init the msg queue
			//QueueMessager::initQueue(size, maxunit);
			// init the critical sections
			
			InitializeCriticalSection(&reslock);
			InitializeCriticalSection(&oreslock);

			clientaccepted = 0;
			wakeup = CreateEvent(NULL, FALSE, FALSE, NULL);

			myctrlid = _strdup(ctrlid);
			// reuse port
			do {
				int val = 1;
				if (setsockopt(ctrlSocket, SOL_SOCKET, SO_REUSEADDR, (char*)&val, sizeof(val)) < 0) {
					infoRecorder->logError("[CtrlMessagerServer]:controller server-bind: %s\n", strerror(errno));
					goto error;
				}
			} while (0);
			// bind for either TCP of UDP
			if (bind(ctrlSocket, (struct sockaddr*) &ctrlsin, sizeof(ctrlsin)) < 0) {
				infoRecorder->logError("[CtrlMessagerServer]:controller server-bind: %s\n", strerror(errno));
				goto error;
			}
			// TCP listen
			if (conf->ctrlproto == IPPROTO_TCP) {
				if (listen(ctrlSocket, 16) < 0) {
					infoRecorder->logError("[CtrlMessagerServer]:controller server-listen: %s\n", strerror(errno));
					goto error;
				}
			}
			return 0;
error:
			closesocket(ctrlSocket);
			ctrlSocket = -1;
			return -1;
		}
#endif
		
		BOOL CtrlMessagerServer::onThreadStart(){
			struct sockaddr_in csin, xsin;
			clientaccepted = 0;

			if (conf == NULL){
				infoRecorder->logError("[CtrlMessagerServer]: NULL rtsp config\n");
				return FALSE;
			}

			infoRecorder->logError("[CtrlMessagerServer]: controller server started: tid=%ld.\n", this->getThreadId());

restart:
			bzero(&csin, sizeof(csin));
			int csinlen = sizeof(csin);
			csin.sin_family = AF_INET;
			clientaccepted = 0;
			// handle only one client
			if (conf->ctrlProto == IPPROTO_TCP) {
				struct ctrlhandshake *hh = (struct ctrlhandshake*) buf;
				//
				if ((sock = accept(ctrlSocket, (struct sockaddr*) &csin, &csinlen)) < 0) {
					infoRecorder->logError("[CtrlMessagerServer]: controller server-accept: %s.\n", strerror(errno));
					goto restart;
				}
				infoRecorder->logError("[CtrlMessagerServer]: controller server-thread: accepted TCP client from %s.%d\n",
					inet_ntoa(csin.sin_addr), ntohs(csin.sin_port));
				// check initial handshake message
				if ((buflen = recv(sock, (char*)buf, sizeof(buf), 0)) <= 0) {
					infoRecorder->logError("[CtrlMessagerServer]: controller server-thread: %s\n", strerror(errno));
					closesocket(sock);
					goto restart;
				}
				if (hh->length > buflen) {
					infoRecorder->logError("[CtrlMessagerServer]: controller server-thread: bad handshake length (%d > %d)\n",
						hh->length, buflen);
					closesocket(sock);
					goto restart;
				}
				if (memcmp(myctrlid, hh->id, hh->length - 1) != 0) {
					infoRecorder->logError("[CtrlMessagerServer]: controller server-thread: mismatched protocol version (%s != %s), length = %d\n",
						hh->id, myctrlid, hh->length - 1);
					closesocket(sock);
					goto restart;
				}
				infoRecorder->logError("[CtrlMessagerServer]: controller server-thread: receiving events ...\n");
				clientaccepted = 1;
			}

			buflen = 0;


			return TRUE;
		}
		BOOL CtrlMessagerServer::run(){
			infoRecorder->logTrace("[CtrlMessagerServer]: run() called.\n");
			int rlen, msglen;
			struct sockaddr_in csin, xsin;
			int csinlen, xsinlen;
			//
			bufhead = 0;
			//
tcp_readmore:
			if (conf->ctrlProto == IPPROTO_TCP) {
				infoRecorder->logTrace("[CtrlMessagerServer]: run(), TCP recv.\n");
				if ((rlen = recv(sock, (char*)buf + buflen, sizeof(buf)-buflen, 0)) <= 0) {
					infoRecorder->logTrace("[CtrlMessagerServer]: controller server-read: %s\n", strerror(errno));
					closesocket(sock);
					return FALSE;
				}
#if 0
				infoRecorder->logError("controller server-debug: recv msg (%d bytes) | head = %02x %02x\n",
					rlen, buf[0], buf[1]);
#endif
				buflen += rlen;
				infoRecorder->logTrace("[CtrlMessagerServer]: recv %d bytes.\n", rlen);
			}
			else if (conf->ctrlProto == IPPROTO_UDP) {
				infoRecorder->logError("[CtrlMessagerServer]: run(), UDP recv.\n");
				bzero(&xsin, sizeof(xsin));
				xsinlen = sizeof(xsin);
				xsin.sin_family = AF_INET;
				buflen = recvfrom(ctrlSocket, (char*)buf, sizeof(buf), 0, (struct sockaddr*) &xsin, &xsinlen);
				if (clientaccepted == 0) {
					bcopy(&xsin, &csin, sizeof(csin));
					clientaccepted = 1;
				}
				else if (memcmp(&csin, &xsin, sizeof(csin)) != 0) {
					infoRecorder->logError("[CtrlMessagerServer]: controller server-thread: NOTICE - UDP client reconnected?\n");
					bcopy(&xsin, &csin, sizeof(csin));
					//continue;
				}
			}
			else{
				infoRecorder->logError("[CtrlMessagerServer]: invalid network protocol.\n");
				return FALSE;
			}
tcp_again:
			if (buflen < 2) {
				if (conf->ctrlProto == IPPROTO_TCP)
					goto tcp_readmore;
				else
					return TRUE;
			}
			msglen = ntohs(*((unsigned short*)(buf + bufhead)));

			if (msglen == 0) {
				infoRecorder->logError("[CtrlMessagerServer]: controller server: WARNING - invalid message with size equal to zero!\n");
				return TRUE;
			}
			
			// buffer checks for TCP - not sufficient?
			if (conf->ctrlProto == IPPROTO_TCP) {
				if (buflen < msglen) {
					bcopy(buf + bufhead, buf, buflen);
					goto tcp_readmore;
				}
			}
			else if (conf->ctrlProto == IPPROTO_UDP) {
				if (buflen != msglen) {
					infoRecorder->logError("[CtrlMessagerServer]: controller server: UDP msg size matched (expected %d, got %d).\n",
						msglen, buflen);
					return TRUE;
				}
			}
			// replay or queue the event
			if (replay != NULL) {
				(*replay)(buf + bufhead, msglen);
			}
			else if (writeMsg(buf + bufhead, msglen) != msglen) {
				infoRecorder->logError("[CtrlMessagerServer]: controller server: queue full, message dropped.\n");
			}
			else {
				SetEvent(wakeup);
				//pthread_cond_signal(&wakeup);
			}
			// handle buffers for TCP
			if (conf->ctrlProto == IPPROTO_TCP && buflen > msglen) {
				bufhead += msglen;
				buflen -= msglen;
				goto tcp_again;
			}
			buflen = 0;
			return TRUE;
		}
		BOOL CtrlMessagerServer::stop(){
			clientaccepted = 0;
			infoRecorder->logError("[CtrlMessagerServer]: thread terminated: tid =%d\n", this->getThreadId());
			CThread::stop();
			return TRUE;
		}
		int CtrlMessagerServer::readNext(void *msg, int msglen) {
			int ret;
			struct queuemsg *qm;
again:
			if((qm = readMsg()) != NULL) {
				if(qm->msgsize > msglen) {
					ret = -1;
				} else {
					bcopy(qm->msg, msg, qm->msgsize);
					ret = qm->msgsize;
				}
				releaseMsg(qm);
				return ret;
			}
			// nothing available, wait for next input
			WaitForSingleObject(wakeupMutex, INFINITE);
			//pthread_mutex_lock(&wakeup_mutex);
			WaitForSingleObject(wakeup, INFINITE);
			//pthread_cond_wait(&wakeup, &wakeup_mutex);
			ReleaseMutex(wakeupMutex);
			//pthread_mutex_unlock(&wakeup_mutex);
			goto again;
			// never return from here
			return 0;
		}

		CtrlMessagerServer::CtrlMessagerServer(){
			 ctrlSocket = NULL, sock = NULL;
			 clientaccepted = 0;

			 bufhead = 0, buflen = 0;
			 wakeupMutex = NULL;
			 wakeup = NULL;
			 replay = NULL;
			 conf = NULL;
		}
	}
}
