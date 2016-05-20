#include "CommandServerSet.h"
#include "../LibCore/SmallHash.h"
#include "WrapDirect3d9.h"
#include "WrapDirect3dvertexbuffer9.h"
#include "WrapDirect3dindexbuffer9.h"
#include "WrapDirect3dvertexdeclaration9.h"
#include "WrapDirect3dvertexshader9.h"
#include "WrapDirect3dpixelshader9.h"
#include "WrapDirect3dtexture9.h"
#include "WrapDirect3dstateblock9.h"
#include "WrapDirect3ddevice9.h"
#include "WrapDirect3dcubetexture9.h"
#include "WrapDirect3dswapchain9.h"
#include "WrapDirect3dsurface9.h"
#include "Initializer.h"
#include <assert.h>

struct WindowParam{
	//bool isEx;
	// for no ex
	DWORD dwExStyle;
	DWORD dwStyle;
	int x,y;
	int width, height;
};

int WrapperDirect3D9::ins_count = 0;
int WrapperDirect3DDevice9::ins_count = 0;
int WrapperDirect3DVertexBuffer9::ins_count = 0;
int WrapperDirect3DIndexBuffer9::ins_count = 0;
int WrapperDirect3DVertexDeclaration9::ins_count = 0;
int WrapperDirect3DVertexShader9::ins_count = 0;
int WrapperDirect3DPixelShader9::ins_count = 0;
int WrapperDirect3DTexture9::ins_count = 0;
int WrapperDirect3DStateBlock9::ins_count = 0;
int WrapperDirect3DCubeTexture9::ins_count = 0;
int WrapperDirect3DSwapChain9::ins_count = 0;
int WrapperDirect3DSurface9::ins_count = 0;

HashSet WrapperDirect3D9::m_list;
HashSet WrapperDirect3DIndexBuffer9::m_list;
HashSet WrapperDirect3DVertexBuffer9::m_list;
HashSet WrapperDirect3DPixelShader9::m_list;
HashSet WrapperDirect3DVertexShader9::m_list;
HashSet WrapperDirect3DTexture9::m_list;
HashSet WrapperDirect3DTexture9::m_side_list;
HashSet WrapperDirect3DDevice9::m_list;
HashSet WrapperDirect3DCubeTexture9::m_list;
HashSet WrapperDirect3DSwapChain9::m_list;
HashSet WrapperDirect3DSurface9::m_list;
HashSet WrapperDirect3DVertexDeclaration9::m_list;
HashSet WrapperDirect3DStateBlock9::m_list;


SmallHash<HWND, WindowParam *> windowMap;

// add the hook to show window
BOOL WINAPI
	ShowWindowCallback(
	__in HWND hWnd,
	__in int nCmdShow){
		infoRecorder->logError("[global]: ShowWindow(), hwnd:%p\n", hWnd);

		return ShowWindowNext(hWnd, nCmdShow);
}

#ifdef ENABLE_BACKGROUND_RUNNING

bool isWindowValid(int width, int height){
	if(width >= 50 && width < 10000 && height >= 50 && height < 10000)
		return true;
	else
		return false;
}

map<string, WNDPROC> procMap;
map<wstring, WNDPROC> wProcMap;
map<HWND, WNDPROC> wndMap;
bool actived = false;

LRESULT CALLBACK NewWndProc(
	_In_ HWND hwnd,
	_In_ UINT uMsg,
	_In_ WPARAM wParam,
	_In_ LPARAM lParam){
		LRESULT hr = TRUE;

		//infoRecorder->logTrace("[Global]: WndProc, HWND:%p, Msg:%x.\n", hwnd, uMsg);

		switch(uMsg){
		case WM_ACTIVATE:
		case WM_NCACTIVATE:
			if(actived)
				return FALSE;
			else
				actived = true;
			break;

		case WM_ACTIVATEAPP:
			return FALSE;
		case WM_KILLFOCUS:
		case WM_SETFOCUS:
		case WM_IME_SETCONTEXT:
		case WM_IME_NOTIFY:
			return FALSE;
		default:
			break;
		}
		WNDPROC proc = NULL;
		// how to get the right proc ?????

		map<HWND, WNDPROC>::iterator it = wndMap.find(hwnd);
		if(it != wndMap.end()){
			proc = it->second;
		}
		if(proc){
			return proc(hwnd, uMsg, wParam, lParam);
		}
		else{
			return DefWindowProc(hwnd, uMsg, wParam, lParam);
		}

		return hr;
}

ATOM WINAPI RegisterClassACallback(_In_ const WNDCLASSA *lpwc){
	infoRecorder->logTrace("[Global]: RegisterClassA, proc:%p.\n", lpwc->lpfnWndProc);

	string name(lpwc->lpszClassName);
	if(procMap.find(name) == procMap.end()){
		procMap[name] = lpwc->lpfnWndProc;
	}

	WNDCLASSA wndClass;
	memcpy(&wndClass, lpwc, sizeof(WNDCLASSA));
	wndClass.lpfnWndProc = NewWndProc;

	ATOM atom = RegisterClassANext(&wndClass);
	
	return atom;
}

ATOM WINAPI RegisterClassWCallback(_In_ const WNDCLASSW *lpwc){

	infoRecorder->logTrace("[Global]: RegisterClassW, proc:%p.\n", lpwc->lpfnWndProc);

	wstring classname(lpwc->lpszClassName);
	if(wProcMap.find(classname) == wProcMap.end()){
		wProcMap[classname] = lpwc->lpfnWndProc;
	}

	WNDCLASSW wndClassW;
	memcpy(&wndClassW, lpwc, sizeof(WNDCLASSW));
	wndClassW.lpfnWndProc = NewWndProc;

	ATOM atom = RegisterClassWNext(&wndClassW);
	return atom;

}
ATOM WINAPI RegisterClassExACallback(_In_ const WNDCLASSEXA *lpwcx){
	infoRecorder->logTrace("[Global]: RegisterClassExA, proc:%p.\n", lpwcx->lpfnWndProc);
	string name(lpwcx->lpszClassName);
	if(procMap.find(name) == procMap.end()){
		procMap[name] = lpwcx->lpfnWndProc;
	}

	WNDCLASSEXA wndClassExA;
	memcpy(&wndClassExA, lpwcx, sizeof(WNDCLASSEXA));
	wndClassExA.lpfnWndProc = NewWndProc;

	ATOM atom = RegisterClassExANext(&wndClassExA);
	return atom;
}
ATOM WINAPI RegisterClassExWCallback(_In_ const WNDCLASSEXW *lpwcx){
	infoRecorder->logTrace("[Global]: RegisterClassExW, proc:%p.\n", lpwcx->lpfnWndProc);

	wstring classname(lpwcx->lpszClassName);
	if(wProcMap.find(classname) == wProcMap.end()){
		wProcMap[classname] = lpwcx->lpfnWndProc;
	}

	WNDCLASSEXW wndClassExW;
	memcpy(&wndClassExW, lpwcx, sizeof(WNDCLASSEXW));
	wndClassExW.lpfnWndProc = NewWndProc;

	ATOM atom = RegisterClassExWNext(&wndClassExW);
	return atom;

}

#endif  // ENABLE_BACKGROUND_RUNNING'

HWND WINAPI CreateWindowCallback(
	DWORD dwExStyle,
	LPCSTR lpClassName,
	LPCSTR lpWindowName, 
	DWORD dwStyle,
	int x,
	int y,
	int nWidth,
	int nHeight,
	HWND hWndParent,
	HMENU hMenu,
	HINSTANCE hInstance,
	LPVOID lpParam) {

	
	/*if( nWidth < 800)nWidth = 800;
	if(nHeight < 600) nHeight = 600;*/

	WindowParam * win = new WindowParam();
	win->dwExStyle = dwExStyle;
	win->dwStyle = dwStyle;
	win->x = x;
	win->y = y;
	win->width = nWidth;
	win->height = nHeight;

#if 0
	csSet->beginCommand(CreateWindow_Opcode, 0);
	csSet->writeUInt(dwExStyle);
	csSet->writeUInt(dwStyle);
	csSet->writeInt(x);
	csSet->writeInt(y);
	csSet->writeInt(nWidth);
	csSet->writeInt(nHeight);
	csSet->endCommand();
#endif

	HWND ret = CreateWindowNext(dwExStyle,lpClassName,lpWindowName,dwStyle,x,y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);
	infoRecorder->logTrace("[global]: CreateWindowCallback() called, width:%d, height:%d, window:%p\n", nWidth, nHeight, ret);
	windowMap.addMap(ret, win);

#ifdef ENABLE_BACKGROUND_RUNNING

	WNDPROC proc = NULL;
	if(isWindowValid(nWidth, nHeight)){
		string classname(lpClassName);
		map<string, WNDPROC>::iterator it = procMap.find(classname);
		if(it != procMap.end()){
			proc = it->second;
			wndMap[ret] = proc;
		}
	}
#endif

	return ret;
}

HWND WINAPI CreateWindowExWCallback(
	DWORD dwExStyle, 
	LPCWSTR lpClassName, 
	LPCWSTR lpWindowName, 
	DWORD dwStyle,
	int X, 
	int Y, 
	int nWidth, 
	int nHeight, 
	HWND hWndParent, 
	HMENU hMenu, 
	HINSTANCE hInstance, 
	LPVOID lpParam) {

#if 0
	if(nWidth > 0 && nWidth < 800)nWidth = 800;
	if(nHeight > 0 && nHeight < 600) nHeight = 600;
#endif

#if 0
	csSet->beginCommand(CreateWindow_Opcode, 0);
	csSet->writeUInt(dwExStyle);
	csSet->writeUInt(dwStyle);
	csSet->writeInt(X);
	csSet->writeInt(Y);
	csSet->writeInt(nWidth);
	csSet->writeInt(nHeight);
	csSet->endCommand();
#endif

	HWND ret = CreateWindowExWNext(dwExStyle,lpClassName,lpWindowName,dwStyle,X,Y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);
	infoRecorder->logTrace("[global]: CreateWindowExWCallback() called, x:%d, y:%d, Widht:%d, Height:%d, window:%p\n", X, Y, nWidth, nHeight, ret);
	WindowParam * win = new WindowParam();
	win->dwExStyle = dwExStyle;
	win->dwStyle = dwStyle;
	win->x = X;
	win->y = Y;
	win->width = nWidth;
	win->height = nHeight;
	windowMap.addMap(ret, win);

#ifdef ENABLE_BACKGROUND_RUNNING
	WNDPROC proc = NULL;
	if(isWindowValid(nWidth, nHeight)){
		wstring classname(lpClassName);
		map<wstring, WNDPROC>::iterator it = wProcMap.find(classname);
		if(it != wProcMap.end()){
			proc = it->second;
			wndMap[ret] = proc;
		}
	}
#endif

	return ret;
}

IDirect3D9* WINAPI Direct3DCreate9Callback(UINT SDKVersion) {
	infoRecorder->logTrace("[global]: Direct3DCreate9Callback() called\n");
	IDirect3D9* base_d3d9 = Direct3DCreate9Next(SDKVersion);
	IDirect3D9* pv = NULL;

	if(base_d3d9) {
		pv = dynamic_cast<IDirect3D9*>(WrapperDirect3D9::GetWrapperD3D9(base_d3d9));
#ifndef MULTI_CLIENTS
		cs.begin_command(DirectCreate_Opcode, 0);
		cs.end_command();
#else
		// here, we need to wait for a notification
		infoRecorder->logTrace("[Global]: wait context event in Direct3DCreate9.\n");
#ifndef ENABLE_HOT_PLUG
		csSet->waitCtxEvent();		
#endif
		csSet->beginCommand(DirectCreate_Opcode, 0);
		csSet->endCommand();
#endif
	}else{
		infoRecorder->logTrace("[Global]: create d3d failed.\n");
	}
	return pv;
}

SOCKET GetProcessSocket(SOCKET oldsocket, DWORD source_pid) {
	RaiseToDebugP();
	HANDLE source_handle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, source_pid);
	HANDLE new_handle = 0;
	if(source_handle == NULL) {
		infoRecorder->logTrace("[Global]: GetProcessSocket(), error happen\n");
		return NULL;
	}
	DuplicateHandle(source_handle, (HANDLE)oldsocket, GetCurrentProcess(), &new_handle, 0, FALSE, DUPLICATE_SAME_ACCESS);
	CloseHandle(source_handle);
	infoRecorder->logTrace("[Global]: GetProcessSocket(), pid: %d,old sock: %d, socket: %d\n",source_pid,oldsocket,new_handle);
	return (SOCKET)new_handle;
}

int IdentifierBase::getId(){ return id; }
void IdentifierBase::setId(int _id){ id = _id; }

void ContextAndCache::checkFlags(){
	infoRecorder->logError("[ContextAndCache]: check all flags for ctx:%d.\n", index);
	Initializer * initializer = Initializer::GetInitializer();
	if(isCreated(initializer->creationFlag)){
		infoRecorder->logError("[ContextAndCache]: Initializer created.\n");
	}
	HashSet::iterator it;
	//IdentifierBase *obj = NULL;
	int counter = 0;
	int totalObjects = 0;
	for(it = WrapperDirect3D9::m_list.begin(); it != WrapperDirect3D9::m_list.end(); it++){	
		WrapperDirect3D9 *obj = (WrapperDirect3D9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3D]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3D]: total has %d, checked %d.\n", WrapperDirect3D9::ins_count, counter);

	counter = 0;
	for(it = WrapperDirect3DCubeTexture9::m_list.begin(); it != WrapperDirect3DCubeTexture9::m_list.end(); it++){
		counter++;
		totalObjects++;
		WrapperDirect3DCubeTexture9 *obj = (WrapperDirect3DCubeTexture9 *)it->pData;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DCubeTexture9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DCubeTexture9]: total has %d, checked %d.\n", WrapperDirect3DCubeTexture9::ins_count, counter);
	counter = 0;

	for(it = WrapperDirect3DDevice9::m_list.begin(); it != WrapperDirect3DDevice9::m_list.end(); it++){
		WrapperDirect3DDevice9 *obj = (WrapperDirect3DDevice9 *)it->pData;
		counter ++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DDevice9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag,obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DDevice9]: total has %d, checked %d.\n", WrapperDirect3DDevice9::ins_count, counter);

	counter =0;
	for(it = WrapperDirect3DIndexBuffer9::m_list.begin(); it != WrapperDirect3DIndexBuffer9::m_list.end(); it++){
		WrapperDirect3DIndexBuffer9 *obj = (WrapperDirect3DIndexBuffer9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DIndexBuffer9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DIndexBuffer]: total has %d, checked %d.\n", WrapperDirect3DIndexBuffer9::ins_count, counter);
	counter = 0;
	for(it = WrapperDirect3DPixelShader9::m_list.begin(); it != WrapperDirect3DPixelShader9::m_list.end(); it++){
		WrapperDirect3DPixelShader9 *obj = (WrapperDirect3DPixelShader9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DPixelShader9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DPixelShader9]: total has %d, checked %d.\n", WrapperDirect3DPixelShader9::ins_count, counter);
	counter =0;
	for(it = WrapperDirect3DStateBlock9::m_list.begin(); it != WrapperDirect3DStateBlock9::m_list.end(); it++){
		WrapperDirect3DStateBlock9 *obj = (WrapperDirect3DStateBlock9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DStateBlock9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DStateBlock9]: total has %d, checked %d.\n", WrapperDirect3DStateBlock9::ins_count, counter);
	counter= 0;
	for(it = WrapperDirect3DSurface9::m_list.begin(); it != WrapperDirect3DSurface9::m_list.end(); it++){
		WrapperDirect3DSurface9 *obj = (WrapperDirect3DSurface9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DSurface9]: %d created: %s,\tflag:0x%x,\trefCount:%d, parent tex id:%d(vs p tex:%p, id:%d), level:%d, creation cmd:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount, obj->GetTexId(), obj->getParentTexture(), obj->getParentTexture() ? obj->getParentTexture()->getId(): -1, obj->GetLevel(), obj->creationCommand);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DSurface9]: total has %d, checked %d.\n", WrapperDirect3DSurface9::ins_count, counter);
	counter =0;
	for(it = WrapperDirect3DSwapChain9::m_list.begin(); it != WrapperDirect3DSwapChain9::m_list.end(); it++){
		WrapperDirect3DSwapChain9 *obj = (WrapperDirect3DSwapChain9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DSwapChain9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirecct3DSwapChain9]: total has %d, checked %d.\n", WrapperDirect3DSwapChain9::ins_count, counter);
	counter=0;
	for(it = WrapperDirect3DTexture9::m_list.begin(); it != WrapperDirect3DTexture9::m_list.end(); it ++){
		WrapperDirect3DTexture9 *obj = (WrapperDirect3DTexture9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DTexture9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DTexture9]: total has %d, checked %d.\n", WrapperDirect3DTexture9::ins_count, counter);
	counter =0;
	for(it = WrapperDirect3DVertexBuffer9::m_list.begin(); it != WrapperDirect3DVertexBuffer9::m_list.end(); it++){
		WrapperDirect3DVertexBuffer9 *obj = (WrapperDirect3DVertexBuffer9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DVertexBuffer9]: %d created: %s,\tflag:0x%x, update flag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->updateFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DVertexBuffer9]: total has %d, checked %d.\n", WrapperDirect3DVertexBuffer9::ins_count, counter);
	counter=0;
	for(it = WrapperDirect3DVertexDeclaration9::m_list.begin(); it != WrapperDirect3DVertexDeclaration9::m_list.end(); it++){
		WrapperDirect3DVertexDeclaration9 *obj = (WrapperDirect3DVertexDeclaration9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DVertexDeclaration9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DVertexDeclaration9]: total has %d, checked %d.\n", WrapperDirect3DVertexDeclaration9::ins_count, counter);
	counter=0;
	for(it = WrapperDirect3DVertexShader9::m_list.begin(); it != WrapperDirect3DVertexShader9::m_list.end(); it++){
		WrapperDirect3DVertexShader9 *obj = (WrapperDirect3DVertexShader9 *)it->pData;
		counter++;
		totalObjects++;
		//if(isCreated(obj->creationFlag)){
			infoRecorder->logError("[WrapperDirect3DVertexShader9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
		//}
	}
	infoRecorder->logError("[WrapperDirect3DVertexShader9]: total has %d, checked %d.\n", WrapperDirect3DVertexShader9::ins_count, counter);
	infoRecorder->logError("[ContextAndCache]: totally checked objects %d.\n", totalObjects);
	counter = 0;
}

void ContextAndCache::eraseFlag(){
	infoRecorder->logError("[ContextAndCache]: Clear context flags.\n");
	Initializer * initializer = Initializer::GetInitializer();
	resetCreation(initializer->creationFlag);

	HashSet::iterator it;
	//IdentifierBase * obj = NULL;
	for(it = WrapperDirect3D9::m_list.begin(); it != WrapperDirect3D9::m_list.end(); it++){
		WrapperDirect3D9 *obj = (WrapperDirect3D9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DCubeTexture9::m_list.begin(); it != WrapperDirect3DCubeTexture9::m_list.end(); it++){
		WrapperDirect3DCubeTexture9 *obj = (WrapperDirect3DCubeTexture9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DDevice9::m_list.begin(); it != WrapperDirect3DDevice9::m_list.end(); it++){
		WrapperDirect3DDevice9 *obj = (WrapperDirect3DDevice9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DIndexBuffer9::m_list.begin(); it != WrapperDirect3DIndexBuffer9::m_list.end(); it++){
		WrapperDirect3DIndexBuffer9 *obj = (WrapperDirect3DIndexBuffer9 *)it->pData;
		resetCreation(obj->creationFlag);
		//setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DPixelShader9::m_list.begin(); it != WrapperDirect3DPixelShader9::m_list.end(); it++){
		WrapperDirect3DPixelShader9 *obj = (WrapperDirect3DPixelShader9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DStateBlock9::m_list.begin(); it != WrapperDirect3DStateBlock9::m_list.end(); it++){
		WrapperDirect3DStateBlock9* obj = (WrapperDirect3DStateBlock9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DSurface9::m_list.begin(); it != WrapperDirect3DSurface9::m_list.end(); it++){
		WrapperDirect3DSurface9 *obj = (WrapperDirect3DSurface9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DSwapChain9::m_list.begin(); it != WrapperDirect3DSwapChain9::m_list.end(); it++){
		WrapperDirect3DSwapChain9 *obj = (WrapperDirect3DSwapChain9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DTexture9::m_list.begin(); it != WrapperDirect3DTexture9::m_list.end(); it ++){
		WrapperDirect3DSwapChain9 *obj = (WrapperDirect3DSwapChain9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexBuffer9::m_list.begin(); it != WrapperDirect3DVertexBuffer9::m_list.end(); it++){
		WrapperDirect3DVertexBuffer9 *obj = (WrapperDirect3DVertexBuffer9 *)it->pData;
		resetCreation(obj->creationFlag);
		//setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexDeclaration9::m_list.begin(); it != WrapperDirect3DVertexDeclaration9::m_list.end(); it++){
		WrapperDirect3DVertexDeclaration9 *obj = (WrapperDirect3DVertexDeclaration9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexShader9::m_list.begin(); it != WrapperDirect3DVertexShader9::m_list.end(); it++){
		WrapperDirect3DVertexShader9 *obj = (WrapperDirect3DVertexShader9 *)it->pData;
		resetCreation(obj->creationFlag);
		setChanged(obj->updateFlag);
	}
	// done
}


void printStatics(){
	infoRecorder->logError("[Global]: Summary:\n");

	int ib_buffer_size = 0;
	int vb_buffer_size = 0;
	int vs_buffer_size = 0;
	int ps_buffer_size = 0;
	int tx_buffer_size = 0;
	int vd_buffer_size = 0;
	int sb_buffer_size = 0;

	HashSet::iterator it;
	// get the index buffer size
	for(it = WrapperDirect3DIndexBuffer9::m_list.begin(); it != WrapperDirect3DIndexBuffer9::m_list.end(); it++){
		ib_buffer_size += ((WrapperDirect3DIndexBuffer9 *)(it->pData))->GetLength() * 2;   // com_bufer and buffer
	}
	// get the vertex buffer size
	for(it = WrapperDirect3DVertexBuffer9::m_list.begin(); it!= WrapperDirect3DVertexBuffer9::m_list.end(); it++){
		vb_buffer_size +=((WrapperDirect3DVertexBuffer9 *)(it->pData))->GetLength() * 2;  // com_buffer and buffer
	}
	for(it = WrapperDirect3DVertexShader9::m_list.begin(); it != WrapperDirect3DVertexShader9::m_list.end(); it++){
		vs_buffer_size += ((WrapperDirect3DVertexShader9 *)(it->pData))->shaderLen;
	}
	for(it = WrapperDirect3DPixelShader9::m_list.begin(); it != WrapperDirect3DPixelShader9::m_list.end(); it++){
		ps_buffer_size += ((WrapperDirect3DPixelShader9 *)(it->pData))->shaderSize;
	}
	for(it = WrapperDirect3DTexture9::m_list.begin(); it != WrapperDirect3DTexture9::m_list.end(); it++){
		tx_buffer_size += ((WrapperDirect3DTexture9*)(it->pData))->getBufferSize();
	}

	for(it = WrapperDirect3DVertexDeclaration9::m_list.begin(); it != WrapperDirect3DVertexDeclaration9::m_list.end(); it++){
		vd_buffer_size += ((WrapperDirect3DVertexDeclaration9*)(it->pData))->declSize;
	}

	for(it = WrapperDirect3DStateBlock9::m_list.begin(); it != WrapperDirect3DStateBlock9::m_list.end(); it++){
		sb_buffer_size += ((WrapperDirect3DStateBlock9*)(it->pData))->stateBlock->getSize();
	}
	//output the static data

	infoRecorder->logError("WrapperDirect3D9::ins_count = %d\n", WrapperDirect3D9::ins_count);
	infoRecorder->logError("WrapperDirect3DDevice9::ins_count = %d\n", WrapperDirect3DDevice9::ins_count);
	infoRecorder->logError("WrapperDirect3DVertexBuffer9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DVertexBuffer9::ins_count, vb_buffer_size);
	infoRecorder->logError("WrapperDirect3DIndexBuffer9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DIndexBuffer9::ins_count, ib_buffer_size);
	infoRecorder->logError("WrapperDirect3DVertexDeclaration9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DVertexDeclaration9::ins_count, vd_buffer_size);
	infoRecorder->logError("WrapperDirect3DVertexShader9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DVertexShader9::ins_count, vs_buffer_size);
	infoRecorder->logError("WrapperDirect3DPixelShader9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DPixelShader9::ins_count, ps_buffer_size);
	infoRecorder->logError("WrapperDirect3DTexture9::ins_count = %d, use buffer:%d, total:%d, max:%d\n", WrapperDirect3DTexture9::ins_count, tx_buffer_size, WrapperDirect3DTexture9::totalBuffer, WrapperDirect3DTexture9::maxBufferSize);
	infoRecorder->logError("WrapperDirect3DStateBlock9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DStateBlock9::ins_count, sb_buffer_size);
	infoRecorder->logError("WrapperDirect3DCubeTexture9::ins_count = %d\n", WrapperDirect3DCubeTexture9::ins_count);
	infoRecorder->logError("WrapperDirect3DSwapChain9::ins_count = %d\n", WrapperDirect3DSwapChain9::ins_count);
	infoRecorder->logError("WrapperDirect3DSurface9::ins_count = %d\n", WrapperDirect3DSurface9::ins_count);

}


void WINAPI ExitProcessCallback(UINT uExitCode) {
	printStatics();
	//do the clean job here
	ExitProcessNext(uExitCode);
}

void RaiseToDebugP()
{
	HANDLE hToken;
	HANDLE hProcess = GetCurrentProcess();
	if ( OpenProcessToken(hProcess, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken) )
	{
		TOKEN_PRIVILEGES tkp;
		if ( LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &tkp.Privileges[0].Luid) )
		{
			tkp.PrivilegeCount = 1;
			tkp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
			BOOL bREt = AdjustTokenPrivileges(hToken, FALSE, &tkp, 0, NULL, 0) ;
		}
		CloseHandle(hToken);
	}    
}