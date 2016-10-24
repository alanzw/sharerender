#ifndef __TESTRENDERPROXY_H__

#include "../../CloudGamingLiveMigrate/LibRender/LibRenderAPI.h"

#include <d3d9.h>
#include <time.h>

#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dx9.lib")

#define COM_METHOD(TYPE, METHOD) TYPE STDMETHODCALLTYPE METHOD

#include "../../CloudGamingLiveMigrate/LibRender/LibRenderVertexbuffer9.h"
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderIndexbuffer9.h"
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderTexture9.h"
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderStateblock9.h"
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderCubetexture9.h"
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderSwapchain9.h"
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderSurface9.h"

#define WINDOW_CLASS "MYWINDOWCLASS"
#define WINDOW_NAME "Game-Client"

#define Max_Obj_Cnt 20010

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