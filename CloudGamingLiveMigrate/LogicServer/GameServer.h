#ifndef __GAME_SERVER__
#define __GAME_SERVER__

#include "../LibCore/CommonNet.h"
#include "../LibVideo/EventNetwork.h"
#include "../LibCore/Opcode.h"
#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <stdlib.h>
#include <assert.h>

using namespace std;
using namespace cg;
using namespace cg::core;

//#include "CommandServerSet.h"

#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dx9.lib")
#pragma comment(lib, "winmm.lib")

#pragma comment(lib,"dinput8.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "comctl32.lib")


#define COM_METHOD(TYPE, METHOD) TYPE STDMETHODCALLTYPE METHOD

#define STREAM_SERVER_CONFIG "config/server.controller.conf"

#include "detours/detours.h"
#pragma comment(lib, "detours/detours.lib")

extern int need_dump_mesh;
extern bool tex_send[4024];

#ifndef MULTI_CLIENTS
extern CommandServer cs;
extern CommonNet dis;

#else 


#endif

//Ҫ��ȡ������ȫ�ֺ���
extern IDirect3D9* (WINAPI* Direct3DCreate9Next)(UINT SDKVersion);
// hook DirectInput8Create
extern HRESULT (WINAPI * DirectInput8CreateNext)(HINSTANCE hinst,DWORD dwVersion,REFIID riidltf,LPVOID *ppvOut,LPUNKNOWN punkOuter);

extern BOOL
	(WINAPI *
	ShowWindowNext)(
	__in HWND hWnd,
	__in int nCmdShow);
// hook the create window
extern HWND (WINAPI *CreateWindowNext)( 
	__in DWORD dwExStyle,
	__in_opt LPCSTR lpClassName,
	__in_opt LPCSTR lpWindowName,
	__in DWORD dwStyle,
	__in int X,
	__in int Y,
	__in int nWidth,
	__in int nHeight,
	__in_opt HWND hWndParent,
	__in_opt HMENU hMenu,
	__in_opt HINSTANCE hInstance,
	__in_opt LPVOID lpParam);
// hook the create window with unicode
extern HWND (WINAPI *CreateWindowExWNext)(
	__in DWORD dwExStyle,
	__in_opt LPCWSTR lpClassName,
	__in_opt LPCWSTR lpWindowName,
	__in DWORD dwStyle,
	__in int X,
	__in int Y,
	__in int nWidth,
	__in int nHeight,
	__in_opt HWND hWndParent,
	__in_opt HMENU hMenu,
	__in_opt HINSTANCE hInstance,
	__in_opt LPVOID lpParam);

extern void (WINAPI* ExitProcessNext)(UINT uExitCode);

HWND WINAPI CreateWindowCallback(DWORD dwExStyle,LPCSTR lpClassName,LPCSTR lpWindowName, DWORD dwStyle,int x,int y,int nWidth,int nHeight,HWND hWndParent, HMENU hMenu,HINSTANCE hInstance,LPVOID lpParam);
HWND WINAPI CreateWindowExWCallback( DWORD dwExStyle, LPCWSTR lpClassName, LPCWSTR lpWindowName, DWORD dwStyle, int X, int Y, int nWidth, int nHeight, HWND hWndParent, HMENU hMenu, HINSTANCE hInstance, LPVOID lpParam);
HRESULT WINAPI DirectInput8CreateCallback(HINSTANCE hinst,DWORD dwVersion,REFIID riidltf,LPVOID *ppvOut,LPUNKNOWN punkOuter);
IDirect3D9* WINAPI Direct3DCreate9Callback(UINT SDKVersion);

BOOL WINAPI ShowWindowCallback(__in HWND hWnd, __in int nCmdShow);

void WINAPI ExitProcessCallback(UINT uExitCode);

//��loader��������cmd������CmmandStream��InputStream��socket���
void GetSocketsFromCmd();
DWORD GetParentProcessid();
SOCKET GetProcessSocket(SOCKET oldsocket, DWORD source_pid);
void RaiseToDebugP();

void SetKeyboardHook(HINSTANCE hmode, DWORD dwThreadId);

extern bool enableRender;
extern bool F9Pressed;
extern bool f10pressed; // capture screen
extern bool synSign;  // test the latency

extern double time_total;
extern int frame_all_count;

extern int g_frame_index;   // the frame index in a group of frames

#ifndef MULTI_CLIENTS
#endif

#ifdef MULTI_CLIENTS

// for supporting multiple clients
class IdentifierBase{
public:
	unsigned int creationFlag;
	unsigned int updateFlag;

	int curDeviceId;

	bool stable; // the object is changed frequently or not
	bool sync; // indicate that whether the object is a synchronization object

	IdentifierBase():creationFlag(0), updateFlag(0), sync(false), stable(true), curDeviceId(0){}
	IdentifierBase(bool val):creationFlag(0), updateFlag(0), sync(val),stable(true), curDeviceId(0){}

	virtual int sendCreation(void * ctx) = 0;
	virtual int sendUpdate(void * ctx) = 0;

	virtual int checkCreation(void * ctx) = 0;
	virtual int checkUpdate(void * ctx) = 0;

	inline int getDeviceId(){ return curDeviceId; }
	inline void setDeviceID(int id){ curDeviceId = id;}
	void print();
};

class SynEntity: public IdentifierBase{
public:
	SynEntity(bool val): IdentifierBase(val){}
	SynEntity(): IdentifierBase(true){}

	virtual int sendCreation(void * ctx){ return 0; }
	virtual int sendUpdate(void * ctx){ return 0; } 
	virtual int checkUpdate(void * ctx){ return 0; }
	virtual int checkCreation(void * ctx){ return 0; }

};


// IB and VB status for each context.
struct BufferStatus{
	bool isFirst;
	bool isChanged;
};

extern bool sceneBegin;

#endif

#endif
