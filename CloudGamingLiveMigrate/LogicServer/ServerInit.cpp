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
		infoRecorder->logTrace("[global]: ShowWindow()\n");

		return ShowWindowNext(hWnd, nCmdShow);
}

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

	infoRecorder->logTrace("[global]: CreateWindowCallback() called, width:%d, height:%d\n", nWidth, nHeight);
	/*if( nWidth < 800)nWidth = 800;
	if(nHeight < 600) nHeight = 600;*/
#ifndef MULTI_CLIENTS
	cs.begin_command(CreateWindow_Opcode, 0);
	cs.write_uint(dwExStyle);
	cs.write_uint(dwStyle);
	cs.write_int(x);
	cs.write_int(y);
	cs.write_int(nWidth);
	cs.write_int(nHeight);
	cs.end_command();
#else

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

#endif
	HWND ret = CreateWindowNext(dwExStyle,lpClassName,lpWindowName,dwStyle,x,y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);
	windowMap.addMap(ret, win);
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

	infoRecorder->logTrace("[global]: CreateWindowExWCallback() called, Widht:%d, Height:%d\n", nWidth, nHeight);
	/*if( nWidth < 800)nWidth = 800;
	if(nHeight < 600) nHeight = 600;*/

#ifndef MULTI_CLIENTS
	cs.begin_command(CreateWindow_Opcode, 0);
	cs.write_uint(dwExStyle);
	cs.write_uint(dwStyle);
	cs.write_int(X);
	cs.write_int(Y);
	cs.write_int(nWidth);
	cs.write_int(nHeight);
	cs.end_command();
#else
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

#endif
	HWND ret = CreateWindowExWNext(dwExStyle,lpClassName,lpWindowName,dwStyle,X,Y,nWidth,nHeight,hWndParent,hMenu,hInstance,lpParam);

	WindowParam * win = new WindowParam();
	win->dwExStyle = dwExStyle;
	win->dwStyle = dwStyle;
	win->x = X;
	win->y = Y;
	win->width = nWidth;
	win->height = nHeight;
	windowMap.addMap(ret, win);

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

void GetSocketsFromSharedFileMap(){
	infoRecorder->logTrace("GetSocketsFromSharedFileMap.\n");
	char * cmdLine;
	cmdLine = GetCommandLine();
	int len = strlen(cmdLine);

	if (len >= 100){
		infoRecorder->logTrace("cmdline is too long.\n");
		return;
	}
	else{
		infoRecorder->logTrace("GetSocketsFromCmd(), cmdLine=%s\n", cmdLine);
		string str = cmdLine;
		istringstream in(str);
		DWORD command_socket_handle, input_socket_handle, loader_process_id;
		string appname;

		//string local, port;
		in >> appname >> need_dump_mesh >> command_socket_handle >> input_socket_handle >> loader_process_id;

		if (loader_process_id){

		}
	}

}
DWORD GetParentProcessid(){
	DWORD ret = 0;
	char * cmdLine;
	cmdLine = GetCommandLine();
	int len = strlen(cmdLine);

	if (len < 100) {
		infoRecorder->logTrace("[global]: GetSocketsFromCmd(), cmdLine=%s\n", cmdLine);
		string str = cmdLine;
		istringstream in(str);
		DWORD command_socket_handle, input_socket_handle, loader_process_id, frame_index;
		string appname;

		//string local, port;
		in >> appname >> need_dump_mesh >> command_socket_handle >> input_socket_handle >> loader_process_id >> frame_index;

		ret = loader_process_id;
	}
	else {
		infoRecorder->logTrace("GetSocketsFromCmd(), cmd len >= 100\n");
	}
	return ret;
}

#ifndef MULTI_CLIENTS
void GetSocketsFromCmd() {
	infoRecorder->logTrace("GetSocketsFromCmd() called\n");

	char * cmdLine;
	cmdLine = GetCommandLine();
	int len = strlen(cmdLine);

	if(len < 100) {
		infoRecorder->logTrace("GetSocketsFromCmd(), cmdLine=%s\n", cmdLine);
		string str = cmdLine;
		istringstream in(str);
		DWORD command_socket_handle, input_socket_handle, loader_process_id, frame_index;
		string appname;

		//string local, port;
		in >> appname >> need_dump_mesh >> command_socket_handle >> input_socket_handle >> loader_process_id >> frame_index;

		g_frame_index = frame_index;
		if(command_socket_handle == -1) {
			cs.set_connect_socket(-1);
			infoRecorder->logError("[server init]: error command socket handle.\n");
			//dis.set_connect_socket(-1);
		}
		else {
			cs.set_connect_socket(GetProcessSocket(command_socket_handle, loader_process_id));
			//dis.set_connect_socket(GetProcessSocket(input_socket_handle, loader_process_id));
			infoRecorder->logTrace("[server init]: new command socket handle:%d.\n", cs.get_connect_socket());
		}

	}
	else {
		infoRecorder->logTrace("GetSocketsFromCmd(), cmd len >= 100\n");
	}
}
#endif
SOCKET GetProcessSocket(SOCKET oldsocket, DWORD source_pid) {
	RaiseToDebugP();
	HANDLE source_handle = OpenProcess(PROCESS_ALL_ACCESS, FALSE, source_pid);
	HANDLE new_handle = 0;
	if(source_handle == NULL) {
		infoRecorder->logTrace("[Global]: GetProcessSocket(), error happen\n");
	}
	DuplicateHandle(source_handle, (HANDLE)oldsocket, GetCurrentProcess(), &new_handle, 0, FALSE, DUPLICATE_SAME_ACCESS);
	CloseHandle(source_handle);
	infoRecorder->logTrace("[Global]: GetProcessSocket(), pid: %d,old sock: %d, socket: %d\n",source_pid,oldsocket,new_handle);
	return (SOCKET)new_handle;
}

#if 0
bool WINAPI ClearCtxFlag(ContextAndCache * ctx){
	infoRecorder->logError("[Global]: Clear context flags.\n");
	int index = ctx->getIndex();    // the context index

	HashSet::iterator it;
	IdentifierBase * obj = NULL;
	for(it = WrapperDirect3D9::m_list.begin(); it != WrapperDirect3D9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DCubeTexture9::m_list.begin(); it != WrapperDirect3DCubeTexture9.m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DDevice9::m_list.begin(); it != WrapperDirect3DDevice9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DIndexBuffer9::m_list.begin(); it != WrapperDirect3DIndexBuffer9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DPixelShader9::m_list.begin(); it != WrapperDirect3DPixelShader9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DStateBlock9::m_list.begin(); it != WrapperDirect3DStateBlock9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DSurface9::m_list.begin(); it != WrapperDirect3DSurface9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DSwapChain9::m_list.begin(); it != WrapperDirect3DSwapChain9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DTexture9::m_list.begin(); it != WrapperDirect3DTexture9::m_list.end(); it ++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexBuffer9::m_list.begin(); it != WrapperDirect3DVertexBuffer9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexDeclaration9::m_list.begin(); it != WrapperDirect3DVertexDeclaration9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexShader9::m_list.begin(); it != WrapperDirect3DVertexShader9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		ctx->resetCreation(obj->creationFlag);
		ctx->resetChanged(obj->updateFlag);
	}
	// done

	return true;
}
#else

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
			infoRecorder->logError("[WrapperDirect3DPixelShader9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag,obj->refCount);
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
			infoRecorder->logError("[WrapperDirect3DSurface9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
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
			infoRecorder->logError("[WrapperDirect3DVertexBuffer9]: %d created: %s,\tflag:0x%x,\trefCount:%d.\n", obj->getId(), isCreated(obj->creationFlag) ? "true" : "false", obj->creationFlag, obj->refCount);
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
	IdentifierBase * obj = NULL;
	for(it = WrapperDirect3D9::m_list.begin(); it != WrapperDirect3D9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DCubeTexture9::m_list.begin(); it != WrapperDirect3DCubeTexture9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DDevice9::m_list.begin(); it != WrapperDirect3DDevice9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DIndexBuffer9::m_list.begin(); it != WrapperDirect3DIndexBuffer9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DPixelShader9::m_list.begin(); it != WrapperDirect3DPixelShader9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DStateBlock9::m_list.begin(); it != WrapperDirect3DStateBlock9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DSurface9::m_list.begin(); it != WrapperDirect3DSurface9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DSwapChain9::m_list.begin(); it != WrapperDirect3DSwapChain9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DTexture9::m_list.begin(); it != WrapperDirect3DTexture9::m_list.end(); it ++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexBuffer9::m_list.begin(); it != WrapperDirect3DVertexBuffer9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexDeclaration9::m_list.begin(); it != WrapperDirect3DVertexDeclaration9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	for(it = WrapperDirect3DVertexShader9::m_list.begin(); it != WrapperDirect3DVertexShader9::m_list.end(); it++){
		obj = (IdentifierBase *)it->pData;
		resetCreation(obj->creationFlag);
		resetChanged(obj->updateFlag);
	}
	// done
}
#endif

void WINAPI ExitProcessCallback(UINT uExitCode) {
	infoRecorder->logError("[Global]: Exit process called\n");

	int ib_buffer_size = 0;
	int vb_buffer_size = 0;
	int vs_buffer_size = 0;
	int ps_buffer_size = 0;

	HashSet::iterator it;
	// get the buffer size
	for(it = WrapperDirect3DIndexBuffer9::m_list.begin(); it != WrapperDirect3DIndexBuffer9::m_list.end(); it++){
		ib_buffer_size += ((WrapperDirect3DIndexBuffer9 *)(it->pData))->GetLength() * 2;   // com_bufer and buffer
	}
	for(it = WrapperDirect3DVertexBuffer9::m_list.begin(); it!= WrapperDirect3DVertexBuffer9::m_list.end(); it++){
		vb_buffer_size +=((WrapperDirect3DVertexBuffer9 *)(it->pData))->GetLength() * 2;  // com_buffer and buffer
	}
	for(it = WrapperDirect3DVertexShader9::m_list.begin(); it != WrapperDirect3DVertexShader9::m_list.end(); it++){
		vs_buffer_size += ((WrapperDirect3DVertexShader9 *)(it->pData))->shaderLen;
	}
	for(it = WrapperDirect3DPixelShader9::m_list.begin(); it != WrapperDirect3DPixelShader9::m_list.end(); it++){
		ps_buffer_size += ((WrapperDirect3DPixelShader9 *)(it->pData))->shaderSize;
	}
	//output the static data
	
	infoRecorder->logError("WrapperDirect3D9::ins_count = %d\n", WrapperDirect3D9::ins_count);
	infoRecorder->logError("WrapperDirect3DDevice9::ins_count = %d\n", WrapperDirect3DDevice9::ins_count);
	infoRecorder->logError("WrapperDirect3DVertexBuffer9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DVertexBuffer9::ins_count, vb_buffer_size);
	infoRecorder->logError("WrapperDirect3DIndexBuffer9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DIndexBuffer9::ins_count, ib_buffer_size);
	infoRecorder->logError("WrapperDirect3DVertexDeclaration9::ins_count = %d\n", WrapperDirect3DVertexDeclaration9::ins_count);
	infoRecorder->logError("WrapperDirect3DVertexShader9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DVertexShader9::ins_count, vs_buffer_size);
	infoRecorder->logError("WrapperDirect3DPixelShader9::ins_count = %d, use buffer size:%d\n", WrapperDirect3DPixelShader9::ins_count, ps_buffer_size);
	infoRecorder->logError("WrapperDirect3DTexture9::ins_count = %d\n", WrapperDirect3DTexture9::ins_count);
	infoRecorder->logError("WrapperDirect3DStateBlock9::ins_count = %d\n", WrapperDirect3DStateBlock9::ins_count);
	infoRecorder->logError("WrapperDirect3DCubeTexture9::ins_count = %d\n", WrapperDirect3DCubeTexture9::ins_count);
	infoRecorder->logError("WrapperDirect3DSwapChain9::ins_count = %d\n", WrapperDirect3DSwapChain9::ins_count);
	infoRecorder->logError("WrapperDirect3DSurface9::ins_count = %d\n", WrapperDirect3DSurface9::ins_count);

	//do the clean job here

#ifndef MULTI_CLIENTS
	cs.shut_down();
#else
	
#endif
	Log::close();
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