#ifndef __GAME_CLIENT__
#define __GAME_CLIENT__

#include <WinSock2.h>
#include "../LibCore/Utility.h"
#include "../LibCore/Opcode.h"
#include "../LibRender/LibRenderAPI.h"
#include <d3d9.h>
#include <time.h>
#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dx9.lib")

#define COM_METHOD(TYPE, METHOD) TYPE STDMETHODCALLTYPE METHOD

#include "../LibRender/LibRenderVertexbuffer9.h"
#include "../LibRender/LibRenderindexbuffer9.h"
#include "../LibRender/LibRenderTexture9.h"
#include "../LibRender/LibRenderStateblock9.h"
#include "../LibRender/LibRenderCubetexture9.h"
#include "../LibRender/LibRenderSwapchain9.h"
#include "../LibRender/LibRenderSurface9.h"

#if 0
extern IDirect3D9* g_d3d;

extern HWND window_handle;
extern int op_code;
extern int obj_id;
extern CommandClient cc;
#endif

#ifndef WINDOW_CLASS 

#define WINDOW_CLASS "MYWINDOWCLASS"
#endif
#ifndef WINDOW_NAME
#define WINDOW_NAME "game-client"
#endif



#if 0
extern void* device_list[Max_Obj_Cnt];

extern string game_name;
extern bool use_server_style;

HRESULT client_init(CommandClient * cc);
bool init_window(int width,int height, DWORD dwExStyle, DWORD dwStyle);
void init_fptable();

extern DWORD WINAPI InputThread(LPVOID lpParameter);
extern CRITICAL_SECTION main_section;
extern PCRITICAL_SECTION p_main_section;
extern bool main_thread_running;

#endif
void init_fptable();


typedef unsigned (__stdcall * PTHREAD_START)(void *);
#define chBEGINTHREADEX(psa, cbStack, pfnStartAddr, \
	pvParam, fdwCreate, pdwThreadID) \
	((HANDLE) _beginthreadex( \
	(void *)(psa), \
	(unsigned)(cbStack), \
	(PTHREAD_START)(pfnStartAddr), \
	(void *)(pvParam), \
	(unsigned )(fdwCreate), \
	(unsigned *)(pdwThreadID)))


#endif