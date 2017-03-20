#include <WinSock2.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#if 1
#include <map>
#include "../LibCore/InfoRecorder.h"
#endif
#include "LibRenderChannel.h"
#include "LibRenderVertexbuffer9.h"
#include "LibRenderIndexbuffer9.h"
#include "LibRenderTexture9.h"
#include "LibRenderStateblock9.h"
#include "LibRenderCubetexture9.h"
#include "LibRenderSwapchain9.h"
#include "LibRenderSurface9.h"

#include "../LibCore/SmallHash.h"
#include "../LibDistrubutor/Context.h"
#include "../VideoGen/generator.h"

using namespace cg;
using namespace cg::core;

// this definition will log the specific string for each command
#define ENABLE_LOG_SPEC_STRING

#ifdef BACKBUFFER_TEST
extern VideoGen * gGenerator;
#endif

extern SmallHash<HWND, HWND> serverToClient;


extern BTimer * encodeTimer;

// global functions

#define GET_DEVICE() \
	rch->curDevice = (LPDIRECT3DDEVICE9)(device_list[obj_id]);

HRESULT FakeDCreateWindow(RenderChannel * rch) {
#ifdef ENABLE_LOG_SPEC_STRING
	cg::core::infoRecorder->logTrace("FakedCreateWindow() called\n");
#endif
	TCHAR szAppName[]= TEXT("HelloWin");
	TCHAR szClassName[]= TEXT("HelloWinClass");


	DWORD dwExStyle = rch->cc->read_uint();
	DWORD dwStyle = rch->cc->read_uint();
	int x= rch->cc->read_int();
	int y = rch->cc->read_int();
	int nWidth = rch->cc->read_int();
	int nHeight = rch->cc->read_int();

	//static bool window_created = false;

	if (!rch->windowCreated){
		rch->initWindow(nWidth, nHeight, dwExStyle, dwStyle);
		rch->windowCreated = true;
	}

	//window_created = true;
	return D3D_OK;
}

HRESULT FakeDDirectCreate(RenderChannel * rch){
#ifdef ENABLE_LOG_SPEC_STRING
	cg::core::infoRecorder->logTrace("server Direct3DCreate9 called\n");
#endif

	rch->gD3d = Direct3DCreate9(D3D_SDK_VERSION);
	if (rch->gD3d)
		return D3D_OK;
	else{
		cg::core::infoRecorder->logTrace("Direct3DCreate9 failed!\n");
		return D3D_OK;
	}
}

HRESULT FakedCreateDevice(RenderChannel * rch) {
	//printf("FakedCreateDevice called\n");
#ifdef ENABLE_LOG_SPEC_STRING
	cg::core::infoRecorder->logTrace("FakedCreateDevice called\n");
#endif

	return rch->clientInit();
}

HRESULT FakedBeginScene(RenderChannel * rch) {
	//printf("FakedBeginScene called\n");
#ifdef ENABLE_LOG_SPEC_STRING
	cg::core::infoRecorder->logTrace("FakedBeginScene() called\n");
#endif

	rch->getDevice(rch->obj_id);
	return rch->curDevice->BeginScene();
}

HRESULT FakedEndScene(RenderChannel * rch) {
	//printf("FakedEndScene called\n");
#ifdef ENABLE_LOG_SPEC_STRING
	cg::core::infoRecorder->logTrace("FakedEndScene()\n");
#endif
	rch->getDevice(rch->obj_id);
	return rch->curDevice->EndScene();
}

HRESULT FakedClear(RenderChannel * rch) {

	DWORD count = rch->cc->read_uint();
	D3DRECT pRects;

	bool is_null = (bool)rch->cc->read_char();
	if(!is_null) {
		rch->cc->read_byte_arr((char*)(&pRects), sizeof(pRects));
	}

	DWORD Flags = rch->cc->read_uint();
	D3DCOLOR color = rch->cc->read_uint();
	float Z = rch->cc->read_float();
	DWORD stencil = rch->cc->read_uint();

#ifdef ENABLE_LOG_SPEC_STRING
	cg::core::infoRecorder->logTrace("Faked Clear(%d, %p, %d, %d, %f, %d), Color=0x%08x\n",count, is_null ? NULL : &pRects, Flags, color, Z, stencil, color);
#endif

	rch->getDevice(rch->obj_id);
	if(!is_null)
		return rch->curDevice->Clear(count, &pRects, Flags, color, Z, stencil);
	else
		return rch->curDevice->Clear(count, NULL, Flags, color, Z, stencil);
}

float gap = 0;
HANDLE presentEvent = NULL;
HANDLE presentMutex = NULL;
//extern Channel * gChannel;
HRESULT FakedPresent(RenderChannel * rch) {
	static float last_present = 0;
	float now_present = timeGetTime();

	gap = now_present - last_present;
	last_present = now_present;

	float t1 = timeGetTime();

	const RECT* pSourceRect = (RECT*)(rch->cc->read_int());
	const RECT* pDestRect = (RECT*)(rch->cc->read_int());
	HWND hDestWindowOverride = (HWND)(rch->cc->read_int());
	const RGNDATA* pDirtyRegion = (RGNDATA*)(rch->cc->read_int());
	unsigned int tags = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);
	assert(rch->curDevice);

	HRESULT hr = rch->curDevice->Present(NULL, NULL, NULL, NULL);
	rch->onPresent(tags);

	
#ifdef ENABLE_LOG_SPEC_STRING
	float t2 = timeGetTime();
	cg::core::infoRecorder->logTrace("FakedPresent(%p, %p, %p, %p), del_time=%.4f\n",pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion, t2 - t1);
#endif
#if 0
	SetEvent(rch->videoItem->presentEvent);

#ifdef BACKBUFFER_TEST
	if(rch->generator){
		rch->generator->run();
	}
#endif
#endif

	return hr;
}
#ifdef ENABLE_LOG_SPEC_STRING
char * TransformEnumerationToSting(D3DTRANSFORMSTATETYPE trans){
	switch(trans){
	case 2:
		return _strdup("D3DTS_VIEW");
	case 3:
		return _strdup("D3DTS_PROJECTION");
	case 16:
		return _strdup("D3DTS_TEXTURE0");
	case 17:
		return _strdup("D3DTS_TEXTURE1");
	case 18:
		return _strdup("D3DTS_TEXTURE2");
	case 19:
		return _strdup("D3DTS_TEXTURE3");
	case 20:
		return _strdup("D3DTS_TEXTURE4");
	case 21:
		return _strdup("D3DTS_TEXTURE5");
	case 22:
		return _strdup("D3DTS_TEXTURE6");
	case 23:
		return _strdup("D3DTS_TEXTURE7");
	case 0x7fffffff:
		return _strdup("D3DTS_FORCE_DWORD");
	}
	return NULL;
}
#endif

HRESULT FakedSetTransform(RenderChannel * rch) {
	short st = rch->cc->read_short();
	D3DTRANSFORMSTATETYPE state = (D3DTRANSFORMSTATETYPE)st;
	unsigned short mask = rch->cc->read_ushort();

	D3DMATRIX mat;
	int out_len = 0;
	for(int i=0; i<4; ++i) {
		for(int j=0; j<4; ++j) {
			if(mask & (1 << (i * 4 + j))) {
				mat.m[i][j] = rch->cc->read_float();
			}
			else
				mat.m[i][j] = 0.0f;
		}
	}
	rch->getDevice(rch->obj_id);
#ifdef ENABLE_LOG_SPEC_STRING
	char * strState = TransformEnumerationToSting(state);
	if(strState)
		cg::core::infoRecorder->logTrace("FakedSetTransform(%s, %p).\n", strState, &mat);
	else
		cg::core::infoRecorder->logTrace("FakedSetTransform(%d, %p).\n", state, &mat);

	free(strState);
#endif
	return rch->curDevice->SetTransform(state, &mat);
}

#ifdef ENABLE_LOG_SPEC_STRING
char * RenderStateToString(D3DRENDERSTATETYPE state){
	switch(state){
	case 7:
		return _strdup("D3DTS_ZENABLE");
	case 8:
		return _strdup("D3DTS_FILLMODE");
	case 9:
		return _strdup("D3DTS_SHADEMODE");
	case 14:
		return _strdup("D3DTS_ZWRITEENABLE");
	case 15:
		return _strdup("D3DTS_ALPHATESTENABLE");
	case 16:
		return _strdup("D3DTS_LASTPIXEL");
	case 19:
		return _strdup("D3DTS_SRCBLEND");
	case 20:
		return _strdup("D3DTS_DESTBLEND");
	case 22:
		return _strdup("D3DTS_CULLMODE");
	case 23:
		return _strdup("D3DTS_ZFUNC");
	case 24:
		return _strdup("D3DTS_ALPHAREF");
	case 25:
		return _strdup("D3DTS_ALPHAFUNC");
	case 26:
		return _strdup("D3DTS_DITHERENABLE");
	case 27:
		return _strdup("D3DTS_ALPHABLENDENABLE");
	case 28:
		return _strdup("D3DTS_FOGENABLE");
	case 29:
		return _strdup("D3DTS_SPECULARENABLE");
	case 34:
		return _strdup("D3DTS_FOGCOLOR");
	case 35:
		return _strdup("D3DTS_FOGTABLEMODE");
	case 36:
		return _strdup("D3DTS_FOGSTART");
	case 37:
		return _strdup("D3DTS_FOGEND");
	case 38:
		return _strdup("D3DTS_FOGDENSITY");
	case 48:
		return _strdup("D3DTS_RANGEFOGENABLE");
	case 52:
		return _strdup("D3DTS_STENCILENABLE");
	case 53:
		return _strdup("D3DTS_STENCILFALL");
	case 54:
		return _strdup("D3DTS_STENCILZFALL");
	case 55:
		return _strdup("D3DTS_STENCILPASS");
	case 56:
		return _strdup("D3DTS_STENCILFUNC");
	case 57:
		return _strdup("D3DTS_STENCILREF");
	case 58:
		return _strdup("D3DTS_STENCILMASK");
	case 59:
		return _strdup("D3DTS_STENCILWRITEMASK");
	case 60:
		return _strdup("D3DTS_TEXTUREACTOR");
	case 128:
		return _strdup("D3DTS_WRAP0");
	case 129:
		return _strdup("D3DTS_WRAP1");
	case 130:
		return _strdup("D3DTS_WRAP2");
	case 131:
		return _strdup("D3DTS_WRAP3");
	case 132:
		return _strdup("D3DTS_WRAP4");
	case 133:
		return _strdup("D3DTS_WRAP5");
	case 134:
		return _strdup("D3DTS_WRAP6");
	case 135:
		return _strdup("D3DTS_WRAP7");
	case 136:
		return _strdup("D3DTS_CLIPPING");
	case 137:
		return _strdup("D3DTS_LIGHTING");
	case 139:
		return _strdup("D3DTS_AMBIENT");
	case 140:
		return _strdup("D3DTS_FOGVERTEXMODE");
	case 141:
		return _strdup("D3DTS_COLORVERTEX");
	case 142:
		return _strdup("D3DTS_LOCALVIEWER");
	case 143:
		return _strdup("D3DTS_NORMALIZENORMALS");
	case 145:
		return _strdup("D3DTS_DIFFUSEMATERIALSOURCE");
	case 146:
		return _strdup("D3DTS_SPECULARMATERIALSOURCE");
	case 147:
		return _strdup("D3DTS_AMBIENTMATERIALSOURCE");
	case 148:
		return _strdup("D3DTS_EMISSIVEMATERALSORUCE");
	case 151:
		return _strdup("D3DTS_VERTEXBLEND");
	case 152:
		return _strdup("D3DTS_CLIPPLANEENABLE");
	case 154:
		return _strdup("D3DTS_POINTSIZE");
	case 155:
		return _strdup("D3DTS_POINTSIZE_MIN");
	case 156:
		return _strdup("D3DTS_POINTSPRITEENABLE");
	case 157:
		return _strdup("D3DTS_POINTSCALEENABLE");
	case 158:
		return _strdup("D3DTS_POINTSCALE_A");
	case 159:
		return _strdup("D3DTS_POINTSCALE_B");
	case 160:
		return _strdup("D3DTS_POINTSCALE_C");
	case 161:
		return _strdup("D3DTS_MULTISAMPLEANTIALIAS");
	case 162:
		return _strdup("D3DTS_MULTISAAMPLEMASK");
	case 163:
		return _strdup("D3DTS_PATCHEDGESTYPLE");
	case 165:
		return _strdup("D3DTS_DEBUGMONITORTOKEN");
	case 166:
		return _strdup("D3DTS_POINTSIZE_MAX");
	case 167:
		return _strdup("D3DTS_INDEXEDVERTEXBLENDENABLE");
	case 168:
		return _strdup("D3DTS_COLORWRITEENABLE");
	case 170:
		return _strdup("D3DTS_TWEENFACTOR");
	case 171:
		return _strdup("D3DTS_BLENDOP");
	case 172:
		return _strdup("D3DTS_POSITIONDEGREE");
	case 173:
		return _strdup("D3DTS_NORMALDEGREE");
	case 174:
		return _strdup("D3DTS_SCISSORTESTENABLE");
	case 175:
		return _strdup("D3DTS_SLOPESCALEDEPTHBIAS");
	case 176:
		return _strdup("D3DTS_ANTIALIASEDLINEENABLE");
	case 178:
		return _strdup("D3DTS_MINTESSELLATIONLEVEL");
	case 179:
		return _strdup("D3DTS_MAXTESSELLATIONLEVEL");
	case 180:
		return _strdup("D3DTS_ADAPTIVETESS_X");
	case 181:
		return _strdup("D3DTS_ADAPTIVETESS_Y");
	case 182:
		return _strdup("D3DTS_ADAPTIVETESS_Z");
	case 183:
		return _strdup("D3DTS_ADAPTIVETESS_W");
	case 184:
		return _strdup("D3DTS_ENABLEADAPTIVETESSELLATION");
	case 185:
		return _strdup("D3DTS_TWOSIDEDSTENCILMODE");
	case 186:
		return _strdup("D3DTS_CCW_STENCILFAIL");
	case 187:
		return _strdup("D3DTS_CCW_STENCILZFAIL");
	case 188:
		return _strdup("D3DTS_CCW_STENCILPASS");
	case 189:
		return _strdup("D3DTS_CCW_STENCILFUNC");
	case 190:
		return _strdup("D3DTS_COLORWRITEENABLE1");
	case 191:
		return _strdup("D3DTS_COLORWRITEENABLE2");
	case 192:
		return _strdup("D3DTS_COLORWRITEENABLE3");
	case 193:
		return _strdup("D3DTS_BLENDFACTOR");
	case 194:
		return _strdup("D3DTS_SRGBWRITEENABLE");
	case 195:
		return _strdup("D3DTS_DEPTHBIAS");
	case 198:
		return _strdup("D3DTS_WRAP8");
	case 199:
		return _strdup("D3DTS_WRAP9");
	case 200:
		return _strdup("D3DTS_WRAP10");
	case 201:
		return _strdup("D3DTS_WRAP11");
	case 202:
		return _strdup("D3DTS_WRAP12");
	case 203:
		return _strdup("D3DTS_WRAP13");
	case 204:
		return _strdup("D3DTS_WRAP14");
	case 205:
		return _strdup("D3DTS_WRAP15");
	case 206:
		return _strdup("D3DTS_SEPARATEALPHABLENDENABLE");
	case 207:
		return _strdup("D3DTS_SRCBLENALPHA");
	case 208:
		return _strdup("D3DTS_DESTBLENDALPHA");
	case 209:
		return _strdup("D3DTS_BLENDOPALPHA");
	case 0x7fffffff:
		return _strdup("D3DTS_FORCE_DWORD");
	default:
		return _strdup("D3DTS_UNKNOWN_STATE");
	}
}
#endif

HRESULT FakedSetRenderState(RenderChannel * rch) {
	D3DRENDERSTATETYPE State = (D3DRENDERSTATETYPE)(rch->cc->read_uint());
	DWORD Value = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);
#ifdef ENABLE_LOG_SPEC_STRING
	char * strState = RenderStateToString(State);
	cg::core::infoRecorder->logTrace("FakedSetRenderState(%s, %d).\n", strState, Value);
	free(strState);
#endif
	return rch->curDevice->SetRenderState(State, Value);
}

HRESULT FakedSetStreamSource(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedSetStreamSource called\n");
	//UINT StreamNumber,UINT vb_id,UINT OffsetInBytes,UINT Stride

	int StreamNumber, vb_id, OffestInBytes, Stride;


	StreamNumber = rch->cc->read_uint();
	vb_id = rch->cc->read_int();
	OffestInBytes = rch->cc->read_uint();
	Stride = rch->cc->read_uint();

	if(vb_id == -1) {
		return rch->curDevice->SetStreamSource(StreamNumber, NULL, OffestInBytes, Stride);
	}
	else{
		cg::core::infoRecorder->logTrace("FakedSetStreamSource(%d, %p, %d, %d): StreamNum:%d, vb_id:%d, Offset:%d, Stride:%d\n",StreamNumber, vb_id, OffestInBytes, Stride, StreamNumber,vb_id, OffestInBytes , Stride);
	}

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(rch->vb_list[vb_id]);
	IDirect3DVertexBuffer9 * vb = NULL;
	if(svb == NULL){
		cg::core::infoRecorder->logTrace("FakedSetStreamSource: vertex buffer NULL!\n");
	}
	else{
		vb = svb->GetVB();
		svb->stride = Stride;
	}

	return rch->curDevice->SetStreamSource(StreamNumber, vb, OffestInBytes, Stride);
}

HRESULT FakedSetFVF(RenderChannel * rch) {
	//DWORD FVF
	//cg::core::infoRecorder->logTrace("FakedSetFVF called\n");
	DWORD FVF = rch->cc->read_uint();
	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetFVF(%d)\n", FVF);
	return rch->curDevice->SetFVF(FVF);
}

char * PrimitiveTypeToString(D3DPRIMITIVETYPE type){
	switch(type){
	case 1:
		return _strdup("D3DPT_POINTLIST");
	case 2:
		return _strdup("D3DPT_LINELIST");
	case 3:
		return _strdup("D3DPT_LINESTRIP");
	case 4:
		return _strdup("D3DPT_TRIANGLELIST");
	case 5:
		return _strdup("D3DPT_TRIANGLESTRIP");
	case 6:
		return _strdup("D3DPT_TRIANGLELEFAN");
	case 0x7fffffff:
		return _strdup("D3DPT_FROCE_DWORD");
	default:
		return _strdup("D3DPT_KNOWN_TYPE");
	}
}

HRESULT FakedDrawPrimitive(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedDrawPrimitive called\n");

	char type = rch->cc->read_char();
	UINT StartVertex = 0, PrimitiveCount = 0;
	D3DPRIMITIVETYPE PrimitiveType = (D3DPRIMITIVETYPE)type;
	StartVertex = rch->cc->read_uint();
	PrimitiveCount = rch->cc->read_uint();
#ifdef ENABLE_LOG_SPEC_STRING
	char * strType = PrimitiveTypeToString(PrimitiveType);
	cg::core::infoRecorder->logTrace("FakedDrawPrimitive(%s, %d, %d)\n",strType, StartVertex, PrimitiveCount);
	free(strType);

#endif
	rch->getDevice(rch->obj_id);
	return rch->curDevice->DrawPrimitive(PrimitiveType, StartVertex, PrimitiveCount);
}

HRESULT FakedDrawIndexedPrimitive(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedDrawIndexedPrimitive called\n");
	//D3DPRIMITIVETYPE Type,INT BaseVertexIndex,UINT MinVertexIndex,UINT NumVertices,UINT startIndex,UINT primCount
	rch->getDevice(rch->obj_id);

	D3DPRIMITIVETYPE Type = (D3DPRIMITIVETYPE)(rch->cc->read_char());
	int BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount;
	BaseVertexIndex = rch->cc->read_int();
	MinVertexIndex = rch->cc->read_int();
	NumVertices = rch->cc->read_int();
	startIndex = rch->cc->read_int();
	primCount = rch->cc->read_int();
#ifdef ENABLE_LOG_SPEC_STRING
	char * strType = PrimitiveTypeToString(Type);
	cg::core::infoRecorder->logTrace("FakedDrawIndexedPrimitive(%s, %d, %d, %d, %d, %d), BaseVertexIndex=%d, MinVertexIndex=%d, NumVertices=%d, startIndex=%d, primCount=%d\n", strType, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);
	free(strType);

#endif
	return rch->curDevice->DrawIndexedPrimitive(Type, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);
}


#ifdef ENABLE_LOG_SPEC_STRING
char * UsageToString(DWORD usage){
	char tm[100] = {0};
	return NULL;
}
char * FVFToString(DWORD FVF){

	return NULL;
}
char * PoolToString(D3DPOOL pool){
	return NULL;
}
#endif

HRESULT FakedCreateVertexBuffer(RenderChannel * rch) {
	UINT id = rch->cc->read_uint();
	UINT Length = rch->cc->read_uint();
	DWORD Usage = rch->cc->read_uint();
	DWORD FVF = rch->cc->read_uint();
	D3DPOOL Pool = (D3DPOOL)(rch->cc->read_uint());

	LPDIRECT3DVERTEXBUFFER9 vb = NULL;

	cg::core::infoRecorder->logTrace("FakedCreateVertexBuffer(%d, %d, %d, %d, %p, %p)! Length:%d, Usage:%x, FVF:%x, Pool:%d, id:%d\n",Length, Usage, FVF, Pool, &vb, NULL, Length, Usage, FVF, Pool,id);
	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->CreateVertexBuffer(Length, Usage, FVF, Pool, &vb, NULL);
	if(FAILED(hr)){
		infoRecorder->logError("FakedCreateVertexBuffer failed with:%d.\n", hr);
	}
	//cg::core::infoRecorder->logTrace("FakedCreateVertexBuffer created. \n");
	rch->vb_list[id] = new ClientVertexBuffer9(vb, Length);

	//cg::core::infoRecorder->logTrace("FakedCreateVertexBuffer End. id:%d\n", id);
	return hr;
}

HRESULT FakedVertexBufferLock(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedVertexBufferLock called\n");

	int id = rch->obj_id;

	UINT OffestToLock = rch->cc->read_uint();
	UINT SizeToLock = rch->cc->read_uint();
	DWORD Flags = rch->cc->read_uint();

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(rch->vb_list[id]);
	cg::core::infoRecorder->logTrace("FakedVertexBufferLock(%d, %d, %p, %d), id:%d\n",OffestToLock, SizeToLock, NULL, Flags, id);
	return svb->Lock(OffestToLock, SizeToLock, Flags);
}

HRESULT FakedVertexBufferUnlock(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedVertexBufferUnlock called\n");

	int id = rch->obj_id;

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(rch->vb_list[id]);
	if(svb == NULL){
		cg::core::infoRecorder->logError("FakedVertexBufferUnlock() is NULL id:%d\n",id);
	}else{
		cg::core::infoRecorder->logTrace("FakedVertexBufferUnlock() id:%d\n",id);
	}

	return svb->Unlock(rch->cc);
}

HRESULT FakedSetIndices(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedSetIndices called!!\n");
	
	int ib_id = 0;
	
	ib_id = rch->cc->read_short();

	rch->getDevice(rch->obj_id);
	IDirect3DIndexBuffer9 * ib = NULL;

	//cg::core::infoRecorder->logTrace("FakedSetIndices called, ib_id=%d\n", ib_id);

	if(ib_id == -1){
		cg::core::infoRecorder->logTrace("FakedSetIndices called, ib_id=%d\n", ib_id);
		return rch->curDevice->SetIndices(NULL);
	}

	ClientIndexBuffer9* sib = NULL;
	sib = (ClientIndexBuffer9*)(rch->ib_list[ib_id]);

	if(sib == NULL) {
		cg::core::infoRecorder->logTrace("FakedSetIndices, sib is NULL, ib id:%d\n", ib_id);
	}
	
	ib = NULL;
	if(sib == NULL){
		cg::core::infoRecorder->logError("FakedSetIndices, sib is NULL, ERROR\n");
	}else{
		ib = sib->GetIB();
		cg::core::infoRecorder->logTrace("FakedSetIndices(%p), ib id:%d.\n", ib, ib_id);
	}
	
	return rch->curDevice->SetIndices(ib);
}

HRESULT FakedCreateIndexBuffer(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedCreateIndexBuffer called\n");

	UINT id = rch->cc->read_uint();
	UINT Length = rch->cc->read_uint();
	DWORD Usage = rch->cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(rch->cc->read_uint());
	D3DPOOL Pool = (D3DPOOL)(rch->cc->read_uint());
	LPDIRECT3DINDEXBUFFER9 ib = NULL;

	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->CreateIndexBuffer(Length, Usage, Format, Pool, &ib, NULL);

	rch->ib_list[id] = new ClientIndexBuffer9(ib, Length);
	cg::core::infoRecorder->logTrace("FakedCreateIndexBuffer(%d, %d, %d, %d, %p, %p). id:%d\n",Length, Usage, Format, Pool, &ib, NULL, id);
	return hr;
}

HRESULT FakedIndexBufferLock(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedIndexBufferLock called\n");

	int id = rch->obj_id;
	UINT OffestToLock = rch->cc->read_uint();
	UINT SizeToLock = rch->cc->read_uint();
	DWORD Flags = rch->cc->read_uint();

	cg::core::infoRecorder->logTrace("FakedIndexBufferLock(%d, %d, %p, %d) ,id:%d, OffestToLock=%d, SizeToLock=%d, Flags=%d\n",OffestToLock, SizeToLock, NULL, Flags, id, OffestToLock, SizeToLock, Flags);

	ClientIndexBuffer9* sib = NULL;
	sib = (ClientIndexBuffer9*)(rch->ib_list[id]);
	return sib->Lock(OffestToLock, SizeToLock, Flags);
}

HRESULT FakedIndexBufferUnlock(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedIndexBufferUnlock called\n");
	int id = rch->obj_id;
	ClientIndexBuffer9* sib = NULL;
	sib = (ClientIndexBuffer9*)(rch->ib_list[id]);
	cg::core::infoRecorder->logTrace("FakedIndexBufferUnlock IB id:%d\n",id);
	if(sib == NULL){
		cg::core::infoRecorder->logError("FakedIndexBufferUnlock ClientIndexBuffer is NULL ERROR, id=%d\n", id);
		return S_OK;
	}
	return sib->Unlock(rch->cc);
}

HRESULT FakedSetSamplerState(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedSetSamplerState called\n");

	int Sampler, Value;
	Sampler = rch->cc->read_int();
	char type = rch->cc->read_char();
	D3DSAMPLERSTATETYPE Type = (D3DSAMPLERSTATETYPE)type;
	Value = rch->cc->read_int();

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakeSetSampleState(%d, %d, %d).\n", Sampler, Type, Value);
	return rch->curDevice->SetSamplerState(Sampler, Type, Value);
}


char * TypeToString(BYTE type){
	switch(type){
	case D3DDECLTYPE_FLOAT1:
		return _strdup("D3DDECLTYPE_FLOAT1");
	case D3DDECLTYPE_FLOAT2:
		return _strdup("D3DDECLTYPE_FLOAT2");
	case D3DDECLTYPE_FLOAT3:
		return _strdup("D3DDECLTYPE_FLOAT3");
	case D3DDECLTYPE_FLOAT4:
		return _strdup("D3DDECLTYPE_FLOAT4");
	case D3DDECLTYPE_D3DCOLOR:
		return _strdup("D3DDECLTYPE_D3DCOLOR");
	case D3DDECLTYPE_UBYTE4:
		return _strdup("D3DDECLTYPE_UBYTE4");
	case D3DDECLTYPE_SHORT2:
		return _strdup("D3DDECLTYPE_SHORT2");
	case D3DDECLTYPE_SHORT4:
		return _strdup("D3DDECLTYPE_SHORT4");
	case D3DDECLTYPE_UBYTE4N:
		return _strdup("D3DDECLTYPE_UBYTE4N");

	case D3DDECLTYPE_SHORT2N:
		return _strdup("D3DDECLTYPE_SHORT2N");
	case D3DDECLTYPE_SHORT4N:
		return _strdup("D3DDECLTYPE_SHORT4N");
	case D3DDECLTYPE_USHORT2N:
		return _strdup("D3DDECLTYPE_USHORT2N");
	case D3DDECLTYPE_USHORT4N:
		return _strdup("D3DDECLTYPE_USHORT4N");

	case D3DDECLTYPE_UDEC3:
		return _strdup("D3DDECLTYPE_UDEC3");
	case D3DDECLTYPE_DEC3N:
		return _strdup("D3DDECLTYPE_DEC3N");

	case D3DDECLTYPE_FLOAT16_2:
		return _strdup("D3DDECLTYPE_FLOAT16_2");
	case D3DDECLTYPE_FLOAT16_4:
		return _strdup("D3DDECLTYPE_FLOAT16_4");
	case D3DDECLTYPE_UNUSED:
		return _strdup("D3DDECLTYPE_UNUSED");
	default:
		return _strdup("D3DDECLTYPE_UNKNOWN");
	}
}
char * MethodToString(BYTE method){
	switch(method){
	case D3DDECLMETHOD_DEFAULT:
		return _strdup("D3DDECLMETHOD_DEFAULT");
	case D3DDECLMETHOD_PARTIALU:
		return _strdup("D3DDECLMETHOD_PARTIALU");
	case D3DDECLMETHOD_PARTIALV:
		return _strdup("D3DDECLMETHOD_PARTIALV:");
	case D3DDECLMETHOD_CROSSUV:
		return _strdup("D3DDECLMETHOD_CROSSUV");
	case D3DDECLMETHOD_UV:
		return _strdup("D3DDECLMETHOD_UV");
	case D3DDECLMETHOD_LOOKUP:
		return _strdup("D3DDECLMETHOD_LOOKUP");
	case D3DDECLMETHOD_LOOKUPPRESAMPLED:
		return _strdup("D3DDECLMETHOD_LOOKUPPRESAMPLED");
	default:
		return _strdup("D3D_UNKNOWN_ERROR");
	}

}
char * UsageToString(BYTE usage){
	switch(usage){
	case D3DDECLUSAGE_POSITION:
		return _strdup("D3DDECLUSAGE_POSITION");
	case D3DDECLUSAGE_BLENDWEIGHT:
		return _strdup("D3DDECLUSAGE_BLENDWEIGHT");
	case D3DDECLUSAGE_BLENDINDICES:
		return _strdup("D3DDECLUSAGE_BLENDINDICES");
	case D3DDECLUSAGE_NORMAL:
		return _strdup("D3DDECLUSAGE_NORMAL");
	case D3DDECLUSAGE_PSIZE:
		return _strdup("D3DDECLUSAGE_PSIZE");
	case D3DDECLUSAGE_TEXCOORD:
		return _strdup("D3DDECLUSAGE_TEXCOORD");
	case D3DDECLUSAGE_TANGENT:
		return _strdup("D3DDECLUSAGE_TANGENT");
	case D3DDECLUSAGE_BINORMAL:
		return _strdup("D3DDECLUSAGE_BINORMAL");
	case D3DDECLUSAGE_TESSFACTOR:
		return _strdup("D3DDECLUSAGE_TESSFACTOR");
	case D3DDECLUSAGE_POSITIONT:
		return _strdup("D3DDECLUSAGE_POSITIONT:");
	case D3DDECLUSAGE_COLOR:
		return _strdup("D3DDECLUSAGE_COLOR");
	case D3DDECLUSAGE_FOG:
		return _strdup("D3DDECLUSAGE_FOG");
	case D3DDECLUSAGE_DEPTH:
		return _strdup("D3DDECLUSAGE_DEPTH");
	case D3DDECLUSAGE_SAMPLE:
		return _strdup("D3DDECLUSAGE_SAMPLE");
	default:
		return _strdup("D3DECULUSAGE_UNKNOWN");
	}
}

HRESULT FakedCreateVertexDeclaration(RenderChannel * rch) {
	int id = rch->cc->read_int();
	int cnt = rch->cc->read_int();
	D3DVERTEXELEMENT9 dt[100];
	rch->cc->read_byte_arr((char*)dt, cnt * sizeof(D3DVERTEXELEMENT9));

	cg::core::infoRecorder->logTrace("FakedCreateVertexDeclaration(), id=%d, cnt=%d, ", id, cnt);

	LPDIRECT3DVERTEXDECLARATION9 vd = NULL;

	rch->getDevice(rch->obj_id);
	

	// print the vertex declaration
	char * strType = NULL;
	char * strMethod = NULL;
	char * strUsage = NULL;
	for(int i = 0; i < cnt; i++){
		strType = TypeToString(dt[i].Type);
		strMethod = MethodToString(dt[i].Method);
		strUsage = UsageToString(dt[i].Usage);
		//infoRecorder->logError("stream:%d offset:%d type:%s method:%s usage:%s usage index:%d\n", dt[i].Stream, dt[i].Offset, strType, strMethod, strUsage, dt[i].UsageIndex);
		free(strType);
		free(strMethod);
		free(strUsage);
	}
	HRESULT hr = rch->curDevice->CreateVertexDeclaration(dt, &vd);
	rch->vd_list[id] = vd;

	if(SUCCEEDED(hr)) {
		cg::core::infoRecorder->logTrace("succeeded\n");
	}
	else {
		cg::core::infoRecorder->logTrace("failed\n");
	}

	return hr;
}

HRESULT FakedSetVertexDeclaration(RenderChannel * rch) {

	short id = rch->cc->read_short();

	cg::core::infoRecorder->logTrace("FakedSetVertexDeclaration(), id=%d, ", id);

	if(id == -1) return rch->curDevice->SetVertexDeclaration(NULL);

	LPDIRECT3DVERTEXDECLARATION9 vd = NULL;
	vd = (LPDIRECT3DVERTEXDECLARATION9)(rch->vd_list[id]);

	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->SetVertexDeclaration(vd);
	if(SUCCEEDED(hr)) {
		cg::core::infoRecorder->logTrace("succeeded\n");
	}
	else {
		cg::core::infoRecorder->logTrace("failed\n");
	}
	return hr;

	//return rch->curDevice->SetVertexDeclaration(_decl);
}

HRESULT FakedSetSoftwareVertexProcessing(RenderChannel * rch) {
	BOOL bSoftware = rch->cc->read_int();

	cg::core::infoRecorder->logTrace("FakedSetSoftwareVertexProcessing(), bSoftware=%d\n", bSoftware);

	rch->getDevice(rch->obj_id);
	return rch->curDevice->SetSoftwareVertexProcessing(bSoftware);
}

HRESULT FakedSetLight(RenderChannel * rch) {
	//DWORD Index,CONST D3DLIGHT9* pD3DLight9

	DWORD Index = rch->cc->read_uint();
	D3DLIGHT9 light;
	rch->cc->read_byte_arr((char*)(&light), sizeof(D3DLIGHT9));

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetLight(%d, %p)\n", Index, &light);
	return rch->curDevice->SetLight(Index, &light);
}

HRESULT FakedLightEnable(RenderChannel * rch) {
	//DWORD Index,BOOL Enable

	DWORD Index = rch->cc->read_uint();
	BOOL Enable = rch->cc->read_int();

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedLightEnable(%d, %d)\n", Index, Enable);
	return rch->curDevice->LightEnable(Index, Enable);
}

HRESULT FakedCreateVertexShader(RenderChannel * rch) {
	//DWORD* pFunction,IDirect3DVertexShader9** ppShader

	int id = rch->cc->read_int();
	int cnt = rch->cc->read_int();
	DWORD* ptr = new DWORD[cnt];
	rch->cc->read_byte_arr((char*)ptr, cnt * sizeof(DWORD));

	LPDIRECT3DVERTEXSHADER9 base_vs = NULL;

	cg::core::infoRecorder->logTrace("FakedCreateVertexShader(), id=%d, cnt=%d, ", id, cnt);

	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->CreateVertexShader(ptr, &base_vs);

	rch->vs_list[id] = base_vs;

	if(SUCCEEDED(hr)) {
		cg::core::infoRecorder->logTrace("succeeded\n");
	}
	else {
		cg::core::infoRecorder->logTrace("failed\n");
	}

	delete[] ptr;
	ptr = NULL;

	return hr;
}

HRESULT FakedSetVertexShader(RenderChannel * rch) {
	int id = rch->cc->read_int();
	cg::core::infoRecorder->logTrace("FakedSetVertexShader(), id=%d\n", id);

	if(id == -1) return rch->curDevice->SetVertexShader(NULL);

	LPDIRECT3DVERTEXSHADER9 base_vs = NULL;
	base_vs = (LPDIRECT3DVERTEXSHADER9)(rch->vs_list[id]);
	//cg::core::infoRecorder->logTrace("FakedSetVertexShader(), vs=%d\n", rch->vs_list[id]);

	rch->getDevice(rch->obj_id);
	return rch->curDevice->SetVertexShader(base_vs);
}

float vs_data[10000];

HRESULT FakedSetVertexShaderConstantF(RenderChannel * rch) {
	//UINT StartRegister,CONST float* pConstantData,UINT Vector4fCount


	UINT StartRegister = rch->cc->read_ushort();
	UINT Vector4fcount = rch->cc->read_ushort();

	unsigned int i;
	for(i=0; i<Vector4fcount/2; ++i) {
		rch->cc->read_vec(vs_data + (i * 8), 32);
	}

	if(Vector4fcount & 1) {
		rch->cc->read_vec(vs_data + (i * 8), 16);
	}

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetVertexShaderConstantF(%d %p, %d)\n", StartRegister, vs_data, Vector4fcount);
	//HRESULT hr = rch->curDevice->SetVertexShaderConstantF(StartRegister, (float*)(rch->cc->get_cur_ptr((Vector4fcount * 4) * sizeof(float))), Vector4fcount);
	HRESULT hr = rch->curDevice->SetVertexShaderConstantF(StartRegister, vs_data, Vector4fcount);

	return hr;
}

HRESULT FakedCreatePixelShader(RenderChannel * rch) {
	int id = rch->cc->read_int();
	int cnt = rch->cc->read_int();
	DWORD* ptr = new DWORD[cnt];
	rch->cc->read_byte_arr((char*)ptr, cnt * sizeof(DWORD));

	LPDIRECT3DPIXELSHADER9 base_ps = NULL;
	cg::core::infoRecorder->logTrace("FakedCreatePixelShader(), id=%d, cnt=%d, ", id, cnt);

	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->CreatePixelShader(ptr, &base_ps);
	rch->ps_list[id] = base_ps;

	if(SUCCEEDED(hr)) {
		cg::core::infoRecorder->logTrace("succeeded\n");
	}
	else {
		cg::core::infoRecorder->logTrace("failed\n");
	}

	delete[] ptr;
	ptr = NULL;

	return hr;
}

HRESULT FakedSetPixelShader(RenderChannel * rch) {

	int id = rch->cc->read_int();

	if(id == -1){ 
		cg::core::infoRecorder->logTrace("FakedSetPixelShaDER(%p), id:%d.\n", NULL, id);
		return rch->curDevice->SetPixelShader(NULL);
	}

	LPDIRECT3DPIXELSHADER9 base_ps = NULL;
	base_ps = (LPDIRECT3DPIXELSHADER9)(rch->ps_list[id]);

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetPixelShader(%p), id:%d\n", base_ps, id);
	return rch->curDevice->SetPixelShader(base_ps);
}

HRESULT FakedSetPixelShaderConstantF(RenderChannel * rch) {

	UINT StartRegister = rch->cc->read_uint();
	UINT Vector4fcount = rch->cc->read_uint();

	for(UINT i=0; i<Vector4fcount; ++i) {
		rch->cc->read_vec(vs_data + (i * 4));
	}

	rch->getDevice(rch->obj_id);
	//HRESULT hr = rch->curDevice->SetPixelShaderConstantF(StartRegister, (float*)(rch->cc->get_cur_ptr((Vector4fcount * 4) * sizeof(float))), Vector4fcount);
	cg::core::infoRecorder->logTrace("FakedSetPixelShaderConstantF(%d, %p, %d)\n", StartRegister, vs_data, Vector4fcount);
	HRESULT hr = rch->curDevice->SetPixelShaderConstantF(StartRegister, vs_data, Vector4fcount);

	return hr;
}

UINT arr[4];

char up_buf[1000];

HRESULT FakedDrawPrimitiveUP(RenderChannel * rch) {
	
	rch->cc->read_vec((float*)arr);
	D3DPRIMITIVETYPE PrimitiveType = (D3DPRIMITIVETYPE)arr[0];
	UINT PrimitiveCount = arr[1];
	UINT VertexCount = arr[2];
	UINT VertexStreamZeroStride = arr[3];
	
	rch->cc->read_vec((float*)up_buf, VertexCount * VertexStreamZeroStride);

	rch->getDevice(rch->obj_id);

	HRESULT hr = rch->curDevice->DrawPrimitiveUP(PrimitiveType, PrimitiveCount, (void*)up_buf, VertexStreamZeroStride);
	

#ifdef ENABLE_LOG_SPEC_STRING
	char * strType = PrimitiveTypeToString(PrimitiveType);
	cg::core::infoRecorder->logTrace("FakedDrawPrimitiveUP(%s, %d, %p, %d).\n", strType, PrimitiveCount, up_buf, VertexStreamZeroStride);
	free(strType);

#endif

	return hr;
}

HRESULT FakedDrawIndexedPrimitiveUP(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedDrawIndexedPrimitiveUP called\n");
	//D3DPRIMITIVETYPE PrimitiveType,UINT MinVertexIndex,UINT NumVertices,UINT PrimitiveCount,CONST void* pIndexData,D3DFORMAT IndexDataFormat,CONST void* pVertexStreamZeroData,UINT VertexStreamZeroStride

	D3DPRIMITIVETYPE PrimitiveType = (D3DPRIMITIVETYPE)(rch->cc->read_uint());
	UINT MinVertexIndex = rch->cc->read_uint();
	UINT NumVertices = rch->cc->read_uint();
	UINT PrimitiveCount = rch->cc->read_uint();
	D3DFORMAT IndexDataFormat = (D3DFORMAT)(rch->cc->read_uint());
	UINT VertexStreamZeroStride = rch->cc->read_uint();


	int indexSize = 0;

#if 0
	indexSize = NumVertices * 2;
	if(IndexDataFormat == D3DFMT_INDEX32) indexSize = NumVertices * 4;
#else

	indexSize = PrimitiveCount * 2 *3;
	if(IndexDataFormat == D3DFMT_INDEX32) indexSize = PrimitiveCount * 4 * 3;

#endif
	char* indexData = new char[indexSize];
	rch->cc->read_byte_arr(indexData, indexSize);

	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->DrawIndexedPrimitiveUP(PrimitiveType, MinVertexIndex, NumVertices, PrimitiveCount, (void*)indexData, IndexDataFormat, (void*)(rch->cc->get_cur_ptr(NumVertices * VertexStreamZeroStride)), VertexStreamZeroStride);
#ifdef ENABLE_LOG_SPEC_STRING
	char * strType = PrimitiveTypeToString(PrimitiveType);
	cg::core::infoRecorder->logTrace("FakedDrawIndexedPrimitiveUP(%s, %d, %d, %d, %p, %d, %p, %d).\n", strType, MinVertexIndex, NumVertices, PrimitiveCount, indexData, IndexDataFormat, NULL, VertexStreamZeroStride);
	free(strType);
#endif
	delete[] indexData;
	indexData = NULL;

	return hr;
}

HRESULT FakedSetVertexShaderConstantI(RenderChannel * rch) {
	//UINT StartRegister,CONST int* pConstantData,UINT Vector4iCount

	UINT StartRegister = rch->cc->read_uint();
	UINT Vector4icount = rch->cc->read_uint();
	int* ptr = new int[Vector4icount * 4];
	rch->cc->read_byte_arr((char*)ptr, (Vector4icount * 4) * sizeof(int));

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetVertexShaderConstantI(%d, %p, %d).\n", StartRegister, ptr, Vector4icount);
	HRESULT hr = rch->curDevice->SetVertexShaderConstantI(StartRegister, ptr, Vector4icount);

	delete[] ptr;
	ptr = NULL;

	return hr;
}

HRESULT FakedSetVertexShaderConstantB(RenderChannel * rch) {
	//UINT StartRegister,CONST BOOL* pConstantData,UINT  BoolCount

	UINT StartRegister = rch->cc->read_uint();
	UINT BoolCount = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);

	cg::core::infoRecorder->logTrace("FakedSetVertexShaderConstantB(%d, %p, %d).\n", StartRegister, NULL, BoolCount);
	return rch->curDevice->SetVertexShaderConstantB(StartRegister, (BOOL*)(rch->cc->get_cur_ptr(sizeof(BOOL) * BoolCount)), BoolCount);
}

HRESULT FakedSetPixelShaderConstantI(RenderChannel * rch) {
	//UINT StartRegister,CONST int* pConstantData,UINT Vector4iCount

	UINT StartRegister = rch->cc->read_uint();
	UINT Vector4iCount = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);

	cg::core::infoRecorder->logTrace("FakedSetPixelShaderConstantI(%d, %p, %d)\n", StartRegister, NULL, Vector4iCount);

	return rch->curDevice->SetPixelShaderConstantI(StartRegister, (int*)(rch->cc->get_cur_ptr(sizeof(int) * (Vector4iCount * 4))), Vector4iCount);
}

HRESULT FakedSetPixelShaderConstantB(RenderChannel * rch) {
	//UINT StartRegister,CONST BOOL* pConstantData,UINT  BoolCount

	UINT StartRegister = rch->cc->read_uint();
	UINT BoolCount = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetPixelShaderConstantB(%d, %p, %d)\n", StartRegister, NULL, BoolCount);

	return rch->curDevice->SetPixelShaderConstantB(StartRegister, (BOOL*)(rch->cc->get_cur_ptr(sizeof(BOOL) * BoolCount)), BoolCount);
}


//extern SmallHash<HWND, HWND> serverToClient;

HRESULT FakedReset(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedReset(), ");

	D3DPRESENT_PARAMETERS t_d3dpp;
	rch->cc->read_byte_arr((char*)(&t_d3dpp), sizeof(t_d3dpp));
	rch->getDevice(rch->obj_id);

	// correct the window handle
	HWND serverHwnd = t_d3dpp.hDeviceWindow;
	HWND mappedHwnd = NULL;
	if((mappedHwnd = serverToClient.getValue(serverHwnd)) == NULL){
		
		cg::core::infoRecorder->logError("invalid reset window handle.");
	}
	else{
		t_d3dpp.hDeviceWindow = mappedHwnd;
	}
	cg::core::infoRecorder->logError("\n");
	//return D3D_OK;
	return rch->curDevice->Reset(&t_d3dpp);
}

HRESULT FakedSetMaterial(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedSetMaterial()\n");
	D3DMATERIAL9 material;
	rch->cc->read_byte_arr((char*)(&material), sizeof(D3DMATERIAL9));
	rch->getDevice(rch->obj_id);
	return rch->curDevice->SetMaterial(&material);
}
#if 0
extern LPDIRECT3D9 g_d3d;
extern D3DDISPLAYMODE displayMode;
#endif

BOOL IsTextureFormatOk( D3DFORMAT TextureFormat, D3DFORMAT AdapterFormat, IDirect3D9 * d3d)
{
	//cg::core::infoRecorder->logTrace("IsTextureFormatOK.\n");
	HRESULT hr = d3d->CheckDeviceFormat( D3DADAPTER_DEFAULT,
		D3DDEVTYPE_HAL,
		AdapterFormat,
		0,
		D3DRTYPE_TEXTURE,
		TextureFormat);
	return SUCCEEDED( hr );
}


HRESULT FakedCreateTexture(RenderChannel * rch)
{
	//UINT Width,UINT Height,UINT Levels,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool,IDirect3DTexture9** ppTexture,HANDLE* pSharedHandle
	//cg::core::infoRecorder->logTrace("FakedCreateTexture called\n");

	int id = rch->cc->read_int();
	UINT Width = rch->cc->read_uint();
	UINT Height = rch->cc->read_uint();
	UINT Levels = rch->cc->read_uint();
	DWORD Usage = rch->cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(rch->cc->read_uint());
	D3DPOOL Pool = (D3DPOOL)(rch->cc->read_uint());

	LPDIRECT3DTEXTURE9 base_tex = NULL;

	rch->getDevice(rch->obj_id);

	if(IsTextureFormatOk(Format, rch->displayMode.Format, rch->gD3d)){
		// format is ok
	}
	else{
		// form at is not ok.
		switch(Format){
		case D3DFMT_MULTI2_ARGB8:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_MULTI2_ARGB8\n", Format);
			break;
		case D3DFMT_G8R8_G8B8:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_G8R8_G8B8\n", Format);
			break;
		case D3DFMT_R8G8_B8G8:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_R8G8_B8G8\n", Format);
			break;
		case D3DFMT_UYVY:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_UYVY\n", Format);
			break;
		case D3DFMT_YUY2:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_YUY2\n", Format);
			break;
		case D3DFMT_DXT1:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_DXT1\n", Format);
			break;
		case D3DFMT_DXT2:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_DXT2\n", Format);
			break;
		case D3DFMT_DXT3:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_DXT3\n", Format);
			break;
		case D3DFMT_DXT4:
			cg::core::infoRecorder->logTrace("Invalid Format:%d D3DFMT_DXT4\n", Format);
			break;

		}
		cg::core::infoRecorder->logTrace("Invalid Format:%d \n", Format);
		if(Format == 1515474505)
			Format = D3DFMT_D32;
	}
	//cg::core::infoRecorder->logTrace("to call device->createTexture.\n");
	HRESULT hr = rch->curDevice->CreateTexture(Width, Height, Levels, Usage, Format, Pool, &base_tex, NULL);

	cg::core::infoRecorder->logTrace("FakedCreateTexture(%d, %d, %d, %d, %d, %d, %p, %p),", Width, Height, Levels, Usage, Format, Pool, NULL, NULL);

	switch(hr){
	case D3D_OK:
		cg::core::infoRecorder->logTrace(" return D3D_OK\n");
		break;
	case D3DERR_INVALIDCALL:
		cg::core::infoRecorder->logTrace(" return D3DERR_INVALIDCALL, height:%d, width:%d, level:%d, usage:%d, format:%d, pool:%d\n",Height, Width, Levels, Usage, Format,Pool);
		break;
	case D3DERR_OUTOFVIDEOMEMORY:
		cg::core::infoRecorder->logTrace(" return D3DERR_OUTOFVIDEOMEMORY\n");
		break;
	case E_OUTOFMEMORY:
		cg::core::infoRecorder->logTrace(" return E_OUTOFMEMORY\n");
		break;
	default:
		break;
	}
	if(base_tex == NULL) {
		//cg::core::infoRecorder->logTrace("FakedCreateTexture(), CreateTexture failed, id:%d\n",id);
	}
	else{
	}
	rch->tex_list[id] = new ClientTexture9(base_tex);

	return hr;
}

HRESULT FakedSetTexture(RenderChannel * rch) {
	//DWORD Stage,IDirect3DBaseTexture9* pTexture

	int Stage, tex_id;

	Stage = rch->cc->read_uint();
	tex_id = rch->cc->read_int();
	cg::core::infoRecorder->logTrace("FakedSetTexture(%d, %d), tex id:%d, stage:%d.\n",Stage, tex_id, tex_id, Stage);

	rch->getDevice(rch->obj_id);
	if(tex_id == -1) {
		return rch->curDevice->SetTexture(Stage, NULL);
	}

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(rch->tex_list[tex_id]);

	if(tex == NULL) {
		cg::core::infoRecorder->logError("FakedSetTexture(), Texture is NULL, tex id:%d, ERROR\n",tex_id);
	}

#if 0
	if(tex_id == 8){
		char texName[100] = {0};
		sprintf(texName, "texture\\tex_%d.png", tex_id);
		D3DXSaveTextureToFile(texName, D3DXIMAGE_FILEFORMAT::D3DXIFF_PNG, tex->GetTex9(), NULL);
	}
#endif


	return rch->curDevice->SetTexture(Stage, tex->GetTex9());
}

HRESULT FakedSetTextureStageState(RenderChannel * rch) {
	//DWORD Stage,D3DTEXTURESTAGESTATETYPE Type,DWORD Value

	DWORD Stage = rch->cc->read_uint();
	D3DTEXTURESTAGESTATETYPE Type = (D3DTEXTURESTAGESTATETYPE)(rch->cc->read_uint());
	DWORD Value = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetTextureStageState(%d, %d, %d)\n", Stage, Type, Value);
	return rch->curDevice->SetTextureStageState(Stage, Type, Value);
}

#ifdef USE_TEXTURE_GENERATOR
HRESULT FakedTransmitTextureData(RenderChannel * rch){
	int id = rch->obj_id;
	int level = rch->cc->read_int();
	int autoGen = rch->cc->read_int();
	int totalSize = rch->cc->read_int();
	//int packets = rch->cc->read_int();
	//int cur_size = 0;

	ClientTexture9 * tex = NULL;
	tex = (ClientTexture9*)(rch->tex_list[id]);
	if(!tex){
		cg::core::infoRecorder->logError("TransmitTextureData get NULL tex, id:%d, level:%d.\n", id, level);
	}

	HRESULT hr = D3D_OK;
	LPDIRECT3DSURFACE9 desc = NULL;
	D3DLOCKED_RECT rect;
	hr = tex->GetTex9()->GetSurfaceLevel(level, &desc);
	hr = desc->LockRect(&rect, NULL , 0);
#if 0
	memcpy(rect.pBits, rch->cc->get_cur_ptr(totalSize), totalSize);
#else
	rch->cc->recv_packed_byte_arr((char *)rect.pBits, totalSize);
#endif
	desc->UnlockRect();
	desc->Release();
	desc = NULL;
	if(autoGen){
		tex->GenerateMipSubLevels();
	}
	return hr;
}

#else  // USE_TEXTURE_GENERATOR

HRESULT FakedTransmitTextureData(RenderChannel * rch) {

#ifdef SEND_FULL_TEXTURE
	int id = rch->obj_id;
	LPDIRECT3DTEXTURE9 pTex = NULL, pOldTex = NULL;
	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(rch->tex_list[id]);
	pOldTex = tex->m_tex;
	int size = rch->cc->read_int();   // get the size

	char * texData = rch->cc->get_cur_ptr(size);
	HRESULT hr = D3DXCreateTextureFromFileInMemory(rch->curDevice, (void *)texData, size, &pTex);
	if(SUCCEEDED(hr)){
		// succeeded
		if(pOldTex){
			pOldTex->Release();
		}
		tex->m_tex = pTex;
	}
	else{
		cg::core::infoRecorder->logError("FakedTransmitTextureData failed.\n");
		return S_FALSE;
	}
	return D3D_OK;

#else
#ifndef LOCAL_IMG
	
	int id = rch->obj_id;
	//int levelCount = rch->cc->read_int();
	int level = rch->cc->read_int();
	
#ifdef MEM_FILE_TEX
	int Pitch = rch->cc->read_uint();
#endif
	
	int size = rch->cc->read_int();

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(rch->tex_list[id]);
	//cg::core::infoRecorder->logTrace("server TransmitTextureData get texture id:%d level:%d tex:%d\n",id, level,tex);
#ifndef MEM_FILE_TEX

	LPDIRECT3DSURFACE9 des = NULL;
	tex->GetTex9()->GetSurfaceLevel(level,&des);

	D3DXLoadSurfaceFromFileInMemory(des,NULL,NULL,rch->cc->get_cur_ptr(size),size,NULL,D3DX_FILTER_LINEAR,0xff000000,NULL);

#if 0
	char  fname[50];
	sprintf(fname,"surface\\face_%d_leve_%d.png",id,level);
	D3DXSaveSurfaceToFile(fname, D3DXIMAGE_FILEFORMAT::D3DXIFF_PNG, des, NULL, NULL);

	cg::core::infoRecorder->logTrace("FakedTransmitTextureData(id:%d, level:%d, size:%d)\n", id, level, size);
#endif

	return D3D_OK;
#else
	return tex->FillData(level, Pitch, size, (void*)rch->cc->get_cur_ptr(size));
#endif
#else
	
#endif
#endif   // SEND_FULL_TEXTURE
}

#endif // USE_TEXTURE_GENERATOR


HRESULT FakedCreateStateBlock(RenderChannel * rch) {
	
	D3DSTATEBLOCKTYPE Type = (D3DSTATEBLOCKTYPE)(rch->cc->read_uint());
	int id = rch->cc->read_int();

	IDirect3DStateBlock9* base_sb = NULL;

	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->CreateStateBlock(Type, &base_sb);

	rch->sb_list[id] = new ClientStateBlock9(base_sb);
	cg::core::infoRecorder->logTrace("FakedCreateStateBlock(%d, %p)\n", Type, &base_sb);
	return hr;
}

HRESULT FakedBeginStateBlock(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedBeginStateBlock()\n");
	
	rch->getDevice(rch->obj_id);
	return rch->curDevice->BeginStateBlock();
}

HRESULT FakedEndStateBlock(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedEndStateBlock()\n");
	
	int id = rch->cc->read_int();
	IDirect3DStateBlock9* base_sb = NULL;

	rch->getDevice(rch->obj_id);
	HRESULT hr = rch->curDevice->EndStateBlock(&base_sb);
	if(FAILED(hr)){
		cg::core::infoRecorder->logError("FakeEndStateBlock(), create new state block for %d failed with:%d.\n", id, hr);
	}else{
		cg::core::infoRecorder->logTrace("FakeEndStateBlock(), state block %d is created succ.\n", id);
	}
	rch->sb_list[id] = new ClientStateBlock9(base_sb);

	return hr;
}

HRESULT FakedStateBlockCapture(RenderChannel * rch) {
	int id = rch->obj_id;
	ClientStateBlock9* sb = NULL;
	
	sb = (ClientStateBlock9*)(rch->sb_list[id]);
	if(sb && sb->isValid()){
		cg::core::infoRecorder->logTrace("FakedStateBlockCapture(), state block id:%d.\n", id);
		return sb->Capture();
	}
	else{
		cg::core::infoRecorder->logError("FakedStateBlockCapture(), state block:%d is INVALID.\n", id);
		return D3D_OK;
	}
}

HRESULT FakedStateBlockApply(RenderChannel * rch) {
	int id = rch->obj_id;
	ClientStateBlock9* sb = NULL;

	sb = (ClientStateBlock9*)(rch->sb_list[id]);

	if(sb && sb->isValid()){
		cg::core::infoRecorder->logTrace("FakedStateBlockApply(), state block id:%d\n", id);
		//return D3D_OK;
		return sb->Apply();
	}
	else{
		cg::core::infoRecorder->logError("FakeStateBlockCapure(), state block:%d is INVALID.\n", id);
		return D3D_OK;
		}
}

HRESULT FakedDeviceAddRef(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedDeviceAddRef()\n");
	
	rch->getDevice(rch->obj_id);
	return D3D_OK;
	return rch->curDevice->AddRef();
}

HRESULT FakedDeviceRelease(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedDeviceRelease()\n");
	
	rch->getDevice(rch->obj_id);
	return D3D_OK;
	return rch->curDevice->Release();
}

HRESULT FakedSetViewport(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedSetViewport()\n");
	
	rch->getDevice(rch->obj_id);
	D3DVIEWPORT9 viewport;
	rch->cc->read_byte_arr((char*)(&viewport), sizeof(viewport));
	cg::core::infoRecorder->logTrace("FakeSetViewport(), viewport v.x:%d, v.y:%d, y.width:%d, v.Height:%d, v.Maxz:%f, v.Minz:%f\n", viewport.X,viewport.Y,viewport.Width,viewport.Height,viewport.MaxZ,viewport.MinZ);
	return rch->curDevice->SetViewport(&viewport);
}

HRESULT FakedSetNPatchMode(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedSetNPatchMode()\n");
	
	rch->getDevice(rch->obj_id);
	float nSegments = rch->cc->read_float();
	return rch->curDevice->SetNPatchMode(nSegments);
}

HRESULT FakedCreateCubeTexture(RenderChannel * rch) {
	//int id, UINT EdgeLength,UINT Levels,DWORD Usage,D3DFORMAT Format,D3DPOOL Pool
	
	rch->getDevice(rch->obj_id);

	int id = rch->cc->read_int();
	UINT EdgeLength = rch->cc->read_uint();
	UINT Levels = rch->cc->read_uint();
	DWORD Usage = rch->cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(rch->cc->read_uint());
	D3DPOOL Pool = (D3DPOOL)(rch->cc->read_uint());

	IDirect3DCubeTexture9* base_cube_tex = NULL;
	HRESULT hr = rch->curDevice->CreateCubeTexture(EdgeLength, Levels, Usage, Format, Pool, &base_cube_tex, NULL);
	cg::core::infoRecorder->logTrace("FakedCreateCubeTexture(%d ,%d, %d, %d, %d, %p, %p)\n", EdgeLength, Levels, Usage, Format, Pool, &base_cube_tex, NULL);

	if(SUCCEEDED(hr)) {
		rch->ctex_list[id] = new ClientCubeTexture9(base_cube_tex);
	}
	else {
		cg::core::infoRecorder->logTrace("FakedCreateCubeTexture() failed\n");
	}
	return hr;
}

HRESULT FakedSetCubeTexture(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedSetCubeTexture() called\n");
	
	rch->getDevice(rch->obj_id);

	DWORD Stage = rch->cc->read_uint();
	int id = rch->cc->read_int();

	ClientCubeTexture9* cube_tex = NULL;
	cube_tex = (ClientCubeTexture9*)(rch->ctex_list[id]);

	if(cube_tex == NULL) {
		cg::core::infoRecorder->logTrace("FakedSetCubeTexture(), cube_tex is NULL,id:%d\n", id);
	}else{
		cg::core::infoRecorder->logTrace("FakedSetCubeTexture(), cube_tex:%p ,id:%d\n", cube_tex, id);
	}

	return rch->curDevice->SetTexture(Stage, cube_tex->GetCubeTex9());
}

HRESULT FakedGetSwapChain(RenderChannel * rch) {
	//cg::core::infoRecorder->logTrace("FakedGetSwapChain() called\n");
	
	rch->getDevice(rch->obj_id);

	int id = rch->cc->read_int();
	UINT iSwapChain = rch->cc->read_uint();
	
	IDirect3DSwapChain9* base_chain = NULL;
	HRESULT hr = rch->curDevice->GetSwapChain(iSwapChain, &base_chain);

	if(base_chain == NULL) {
		cg::core::infoRecorder->logTrace("FakedGetSwapChain(%d, %p), base_chain is NULL\n", iSwapChain, &base_chain);
	}else{
		cg::core::infoRecorder->logTrace("FakedGetSwapChain(%d, %p), cube_tex is:%p,id:%d\n",iSwapChain, &base_chain,base_chain, id);
	}

	ClientSwapChain9* swap_chain = NULL;
	swap_chain = (ClientSwapChain9*)(rch->chain_list[id]);

	if(swap_chain == NULL) {
		swap_chain = new ClientSwapChain9(base_chain);
		rch->chain_list[id] = swap_chain;
	}

	return hr;
}

HRESULT FakedSwapChainPresent(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedSwapChainPresent() called, TODO\n");
	
	//RECT* pSourceRect,CONST RECT* pDestRect,CONST RGNDATA* pDirtyRegion,DWORD dwFlags
	
	/*
	RECT SourceRect, DestRect;
	RGNDATA DirtyRegion;

	
	int id = obj_id;

	ClientSwapChain9* swap_chain = NULL;
	swap_chain = (ClientSwapChain9*)(chain_list[id]);

	if(swap_chain == NULL) {
		cg::core::infoRecorder->logTrace("FakedSwapChainPresent(), swap_chain is NULL\n");
	}

	extern HWND hWnd;
	*/
	//擦，这里有点问题啊，mark先!
	return rch->curDevice->Present(NULL, NULL, NULL, NULL);
	//return swap_chain->Present(pSourceRect, pDestRect, hWnd, pDirtyRegion, dwFlags);

#ifdef BACKBUFFER_TEST
	if(rch->generator){
		rch->generator->run();
	}

#endif
}

HRESULT FakedSetAutoGenFilterType(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedSetAutoGenFilterType() called\n");
	
	int id = rch->obj_id;
	D3DTEXTUREFILTERTYPE FilterType = (D3DTEXTUREFILTERTYPE)(rch->cc->read_uint());

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(rch->tex_list[id]);

	if(tex == NULL) {
		cg::core::infoRecorder->logTrace("FakedSetAutoGenFilterType(), tex is NULL\n");
	}

	return tex->SetAutoGenFilterType(FilterType);
}

void FakedGenerateMipSubLevels(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedGenerateMipSubLevels() called\n");
	
	int id = rch->obj_id;

	ClientTexture9* tex = NULL;
	tex = (ClientTexture9*)(rch->tex_list[id]);

	return tex->GenerateMipSubLevels();
}

HRESULT FakedSetRenderTarget(RenderChannel * rch) {
	//DWORD RenderTargetIndex,IDirect3DSurface9* pRenderTarget
	
	rch->getDevice(rch->obj_id);
	DWORD RenderTargetIndex = rch->cc->read_uint();
	int sfid = rch->cc->read_int();
	int tex_id = rch->cc->read_int();
	int level = rch->cc->read_int();

	if(sfid == -1){ 
		cg::core::infoRecorder->logTrace("FakedSetRenderTarget(%d, %d)\n", RenderTargetIndex, sfid);
		return rch->curDevice->SetRenderTarget(RenderTargetIndex, NULL);
	}

	ClientSurface9* surface = (ClientSurface9*)(rch->surface_list[sfid]);

	if(!surface){
		cg::core::infoRecorder->logTrace("FakedSetRenderTarget(%d, %d), get NULL surface, id:%d.\n ", RenderTargetIndex, sfid, sfid);
	}else{
		cg::core::infoRecorder->logError("FakedSetRenderTarget(%d, %d), use surface:%d.\n", RenderTargetIndex, sfid,sfid);
		return rch->curDevice->SetRenderTarget(RenderTargetIndex, surface->GetSurface9());
	}


	ClientTexture9* texture = NULL;
	if(tex_id !=-1 && tex_id <10000)
		texture = (ClientTexture9*)(rch->tex_list[tex_id]);


	if(texture!=NULL){
		IDirect3DSurface9 * real_sur=NULL;
		texture->GetTex9()->GetSurfaceLevel(level, &real_sur);
		surface->ReplaceSurface(real_sur);
		/*char  fname[50];
		sprintf(fname,"surface\\Target%d_tex_%d.png",sfid,tex_id);
		D3DXSaveSurfaceToFile(fname,D3DXIFF_PNG,surface->GetSurface9(), NULL, NULL);*/
		cg::core::infoRecorder->logTrace("FakedSetRenderTarget(%d, %d), use texture %d surface\n", RenderTargetIndex, tex_id, sfid);
		return rch->curDevice->SetRenderTarget(RenderTargetIndex,  real_sur);
	}
	else if(surface){
		/*char  fname[50];
		sprintf(fname,"surface\\Target%d_no_tex_%d.png",sfid,tex_id);
		D3DXSaveSurfaceToFile(fname,D3DXIFF_PNG,surface->GetSurface9(), NULL, NULL);*/
		cg::core::infoRecorder->logTrace("FakedSetRenderTarget(%d, %d), use surface\n", RenderTargetIndex, sfid);
		return rch->curDevice->SetRenderTarget(RenderTargetIndex, surface->GetSurface9());
	}
	else{
		cg::core::infoRecorder->logTrace("FakedSetRenderTarget(%d, %d), surface NULL\n", RenderTargetIndex, sfid);
		return rch->curDevice->SetRenderTarget(RenderTargetIndex, NULL);
	}
}

HRESULT FakedSetDepthStencilSurface(RenderChannel * rch) {
	rch->getDevice(rch->obj_id);
	int sfid = rch->cc->read_int();
	cg::core::infoRecorder->logTrace("FakedSetDepthStencilSurface(%d)\n", sfid);
	if(sfid == -1) 
		return rch->curDevice->SetDepthStencilSurface(NULL);

	ClientSurface9* surface = (ClientSurface9*)(rch->surface_list[sfid]);
	return rch->curDevice->SetDepthStencilSurface(surface->GetSurface9());
}

HRESULT FakedTextureGetSurfaceLevel(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedTextureGetSurfaceLevel() called\n");
	
	int id = rch->obj_id;
	int t_id = rch->cc->read_int();
	int sfid = rch->cc->read_int();
	UINT Level = rch->cc->read_int();

	ClientTexture9* tex = (ClientTexture9*)(rch->tex_list[t_id]);
	if(!tex){
		cg::core::infoRecorder->logError("client texture null, id:%d\n", t_id);
	}

	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = tex->GetSurfaceLevel(Level, &base_surface);
	cg::core::infoRecorder->logTrace("FakedTextureGetSurfaceLevel(%d, %p), tex id:%d, surface id:%d.\n", Level, &base_surface, t_id, sfid);
	rch->surface_list[sfid] = new ClientSurface9(base_surface);

	return hr;
}

HRESULT FakedSwapChainGetBackBuffer(RenderChannel * rch) {
	
	int chain_id = rch->obj_id;
	int surface_id = rch->cc->read_int();
	UINT iBackBuffer = rch->cc->read_uint();
	D3DBACKBUFFER_TYPE Type = (D3DBACKBUFFER_TYPE)(rch->cc->read_uint());

	ClientSwapChain9* chain = (ClientSwapChain9*)(rch->chain_list[chain_id]);

	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = chain->GetSwapChain9()->GetBackBuffer(iBackBuffer, Type, &base_surface);
	cg::core::infoRecorder->logTrace("FakedSwapChainGetBackBuffer(%d, %d, %p).\n", iBackBuffer, Type, &base_surface);
	rch->surface_list[surface_id] = new ClientSurface9(base_surface);

	return hr;
}

HRESULT FakedGetDepthStencilSurface(RenderChannel * rch) {
	
	rch->getDevice(rch->obj_id);
	int sfid = rch->cc->read_int();
	cg::core::infoRecorder->logTrace("surface id:%d\n", sfid);
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = rch->curDevice->GetDepthStencilSurface(&base_surface);
	if(hr == D3D_OK){
		cg::core::infoRecorder->logTrace("GetDepthStencilSurface(%p), surface id:%d!\n", &base_surface, sfid);
		rch->surface_list[sfid] = new ClientSurface9(base_surface);
	}else{
		cg::core::infoRecorder->logError("ERROR! GetDepthStencilSurface %d failed!\n", sfid);

	}

	return hr;
}

HRESULT FakedCreateDepthStencilSurface(RenderChannel * rch) {
	
	rch->getDevice(rch->obj_id);
	int id = rch->cc->read_int();
	UINT Width = rch->cc->read_uint();
	UINT Height = rch->cc->read_uint();
	D3DFORMAT Format = (D3DFORMAT)(rch->cc->read_uint());
	D3DMULTISAMPLE_TYPE MultiSample = (D3DMULTISAMPLE_TYPE)(rch->cc->read_uint());
	DWORD MultisampleQuality = rch->cc->read_uint();
	BOOL Discard = rch->cc->read_int();
	
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = rch->curDevice->CreateDepthStencilSurface(Width, Height, Format, MultiSample, MultisampleQuality, Discard, &base_surface, NULL);

	rch->surface_list[id] = new ClientSurface9(base_surface);
	cg::core::infoRecorder->logTrace("FakedCreateDepthStencilSurface(%d, %d, %d, %d, %d, %d, %p, %p), id:%d\n", Width, Height, Format, MultiSample, MultisampleQuality, Discard, &base_surface, NULL, id);

	return hr;
}

HRESULT FakedCubeGetCubeMapSurface(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedCubeGetCubeMapSurface() called\n");
	
	int cube_id = rch->obj_id;
	int surface_id = rch->cc->read_int();
	D3DCUBEMAP_FACES FaceType = (D3DCUBEMAP_FACES)(rch->cc->read_uint());
	UINT Level = rch->cc->read_uint();

	ClientCubeTexture9* cube = (ClientCubeTexture9*)(rch->ctex_list[cube_id]);
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = cube->GetCubeTex9()->GetCubeMapSurface(FaceType, Level, &base_surface);

	rch->surface_list[surface_id] = new ClientSurface9(base_surface);

	return hr;
}

HRESULT FakeTransmitSurface(RenderChannel * rch){
	cg::core::infoRecorder->logTrace("FakeTransmitSurface() called, TODO\n");
	
	int id = rch->obj_id;
	int size = rch->cc->read_int();
	LPDIRECT3DSURFACE9 surface = NULL;

	return D3D_OK;
}


//newly added
HRESULT FakeD3DDeviceGetBackBuffer(RenderChannel * rch){
	int id = rch->obj_id;   // device id

	int surface_id = rch->cc->read_int();
	UINT iSwapChain = rch->cc->read_uint();
	UINT iBackBuffer = rch->cc->read_uint();
	UINT Type = rch->cc->read_uint();
	D3DBACKBUFFER_TYPE type = (D3DBACKBUFFER_TYPE)Type;
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = rch->curDevice->GetBackBuffer(iSwapChain, iBackBuffer, type, &base_surface);

	if(hr== D3D_OK){
		cg::core::infoRecorder->logTrace("FakeDeviceGetBackBuffer(%d, %d, %d, %p), id:%d!\n", iSwapChain, iBackBuffer, Type, &base_surface, surface_id);
		rch->surface_list[surface_id] = new ClientSurface9(base_surface);
	}else{
		cg::core::infoRecorder->logError("FakeDeviceGetBackBuffer(%d, %d, %d, %p), id:%d, ERROR, Failed!\n", iSwapChain, iBackBuffer, Type, &base_surface, surface_id);
	}
	return hr;
}

HRESULT FakeD3DGetDeviceCaps(RenderChannel * rch){
	HRESULT hr;
	cg::core::infoRecorder->logTrace("FakeD3DGetDeviceCaps()!\n");
	D3DDEVTYPE type = (D3DDEVTYPE)rch->cc->read_int();
	D3DCAPS9 d3d_caps;
	hr = rch->gD3d->GetDeviceCaps(D3DADAPTER_DEFAULT, type, &d3d_caps);
	if(SUCCEEDED(hr)){
		// send back the parameters of device
		//rch->cc->send_raw_buffer((char *)&d3d_caps, sizeof(D3DCAPS9));
		//rch->cc->write_byte_arr((char*) &d3d_caps, sizeof(D3DCAPS9));
	}
	else{
		cg::core::infoRecorder->logTrace("FakeD3DGetDeviceCaps() failed!\n");
	}

	return hr;
}
HRESULT FakeD3DDGetRenderTarget(RenderChannel * rch){
	HRESULT hr;
	int sid = -1;
	sid = rch->cc->read_int();
	DWORD RenderTargetIndex = (DWORD)rch->cc->read_uint();
	IDirect3DSurface9 * target = NULL;
	hr = rch->curDevice->GetRenderTarget(RenderTargetIndex, &target);

	if(hr == D3D_OK){
		cg::core::infoRecorder->logTrace("FakeD3DDGetRenderTarget(%d, %d)!\n", RenderTargetIndex, sid);
		
		rch->surface_list[sid] = new ClientSurface9(target);
	}else{
		//DebugBreak();
		cg::core::infoRecorder->logError("FakeD3DDGetRenderTarget(%d, %p), id:%d, ERROR, Failed with:%d.\n", RenderTargetIndex, target, sid, hr);
	}

	return hr;
}
HRESULT FakeD3DDSetScissorRect(RenderChannel * rch){
	HRESULT hr;
	int left = rch->cc->read_int();
	int right = rch->cc->read_int();
	int top = rch->cc->read_int();
	int bottom = rch->cc->read_int();

	rch->getDevice(rch->obj_id);

	RECT re;
	re.left = left;
	re.right= right;
	re.bottom = bottom;
	re.top = top;
	hr = rch->curDevice->SetScissorRect(&re);
	cg::core::infoRecorder->logTrace("FakeD3DDSetScissorRect(), (%d, %d) - (%d, %d)!\n", left, top, right, bottom);

	return hr;
}

HRESULT FakedSetVertexBufferFormat(RenderChannel * rch) {
	cg::core::infoRecorder->logError("FakedSetVertexBufferFormat() called, TODO\n");

	int id = rch->obj_id;

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(rch->vb_list[id]);
	if(svb == NULL){
		cg::core::infoRecorder->logError("FakedSetVertexBufferFormat is NULL id:%d\n",id);
	}else{
		cg::core::infoRecorder->logTrace("FakedSetVertexBufferFormat id:%d\n",id);
	}
	return 0;
	//return svb->SetVertexBufferFormat(rch->cc);
}

HRESULT FakedSetDecimateResult(RenderChannel * rch) {
	cg::core::infoRecorder->logTrace("FakedSetDecimateResult() called, TODO\n");

	int id = rch->obj_id;

	ClientVertexBuffer9* svb = NULL;
	svb = (ClientVertexBuffer9*)(rch->vb_list[id]);
	if(svb == NULL){
		cg::core::infoRecorder->logTrace("FakedSetDecimateResult is NULL id:%d\n",id);
	}else{
		cg::core::infoRecorder->logTrace("FakedSetDecimateResult id:%d\n",id);
	}
	
	return 0;
	//return svb->SetDecimateResult(rch->cc);
}

HRESULT FakedSetGammaRamp(RenderChannel * rch) {

	UINT iSwapChain = rch->cc->read_uint();
	DWORD Flags = rch->cc->read_uint();
	D3DGAMMARAMP pRamp;
	rch->cc->read_byte_arr((char*)&pRamp, sizeof(D3DGAMMARAMP));

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedSetGammaRamp(%d, %d, %p)\n", iSwapChain, Flags, &pRamp);

	rch->curDevice->SetGammaRamp(iSwapChain, Flags, &pRamp);
	return D3D_OK;
}


#ifdef OLD
extern int inputTickEnd;
extern int inputTickStart;



HRESULT FakeNullInstruct(RenderChannel * rch){
	SYSTEMTIME sys, now;
	GetLocalTime(&now);
	inputTickEnd = GetTickCount();
	cg::core::infoRecorder->logError("%d\n", inputTickEnd - inputTickStart);
	return D3D_OK;
}

#else
#if 0
extern int inputTickEnd;
extern int inputTickStart;
extern bool fromserver;
extern SYSTEMTIME start_sys;
extern time_t end_t;
extern time_t start_t;
extern bool synPress;
extern CRITICAL_SECTION syn_sec;
#endif
HRESULT FakeNullInstruct(RenderChannel * rch){
	unsigned char flag = rch->cc->read_uchar();
	unsigned char tag = rch->cc->read_uchar();
#if 0
	SYSTEMTIME sys, now;
	//GetLocalTime(&now);
	GetSystemTime(&now);
	time(&end_t);
	inputTickEnd = GetTickCount();
	if (flag & 1){
		//cg::core::infoRecorder->logError("%d\n", inputTickEnd - inputTickStart);
		bool syn = false;
		EnterCriticalSection(&syn_sec);
		syn = synPress;
		LeaveCriticalSection(&syn_sec);
		if (syn){
			cg::core::infoRecorder->logError("%d\t%4f\n", (now.wHour - start_sys.wHour) * 60 * 60 * 1000 +
				(now.wMinute - start_sys.wMinute) * 1000 * 60 +
				1000 * (now.wSecond - start_sys.wSecond) +
				(now.wMilliseconds - start_sys.wMilliseconds), gap);//, inputTickEnd - inputTickStart);
			EnterCriticalSection(&syn_sec);
			synPress = false;
			LeaveCriticalSection(&syn_sec);

		}
		else{
			//cg::core::infoRecorder->logError("invalid interval! ");
			/*
			cg::core::infoRecorder->logError("%d\n", (now.wHour - start_sys.wHour) * 60 * 60 * 1000 +
			(now.wMinute - start_sys.wMinute) * 1000 * 60 +
			1000 *(now.wSecond - start_sys.wSecond) +
			(now.wMilliseconds - start_sys.wMilliseconds));
			*/
		}


	}
	if (flag & 2){
		// capture the screen
		cg::core::infoRecorder->logError("f10 from server\n");
		INPUT in;
		memset(&in, 0, sizeof(INPUT));
		in.type = INPUT_KEYBOARD;
		in.ki.wVk = VK_F10;

		//in.ki.dwFlags |= KEYEVENTF_KEYUP;

		in.ki.wScan = MapVirtualKey(in.ki.wVk, MAPVK_VK_TO_VSC);
		SendInput(1, &in, sizeof(INPUT));

		in.ki.dwFlags |= KEYEVENTF_KEYUP;
		//in.ki.dwFlags |= KEYEVENTF_KEYDOWN;
		SendInput(1, &in, sizeof(INPUT));

		fromserver = true;
	}
#endif
	DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();
	
	rch->specialTag = flag;
	rch->valueTag = tag;
	if(encodeTimer){
		encodeTimer->Start();
	}
	else{
		encodeTimer = new PTimer();
	}
	if(flag){
		infoRecorder->logError("NullInstruction, get special tag: %d, value tag:%d.\n", flag, tag);
		delayRecorder->startEndcode();

	}
	return D3D_OK;
}

HRESULT FakedD3DCreateRenderTarget(RenderChannel * rch){
	HRESULT hr = D3D_OK;
	infoRecorder->logError("FakedD3DCreateRenderTarget(), TODO!\n");
	UINT sid = rch->cc->read_uint();
	UINT width = rch->cc->read_uint();
	UINT height = rch->cc->read_uint();
	D3DFORMAT fmt = (D3DFORMAT)rch->cc->read_uint();
	D3DMULTISAMPLE_TYPE sampleType = (D3DMULTISAMPLE_TYPE)rch->cc->read_uint();
	DWORD quality = rch->cc->read_uint();
	BOOL lockable = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);

	IDirect3DSurface9 * surface = NULL;
	hr = rch->curDevice->CreateRenderTarget(width, height, fmt, sampleType, quality, lockable, &surface, NULL);
	if(SUCCEEDED(hr)){
		rch->surface_list[sid] = new ClientSurface9(surface);
	}
	else{
		infoRecorder->logError("FakedD3DCreateRenderTarget() failed with:%d.\n", hr);
	}


	return hr;
}

HRESULT FakedD3DDSetStreamSourceFreq(RenderChannel * rch){
	HRESULT hr = D3D_OK;
	UINT StreamNumber = rch->cc->read_uint();
	UINT Setting = rch->cc->read_uint();

	rch->getDevice(rch->obj_id);
	cg::core::infoRecorder->logTrace("FakedD3DDSetStreamSourceFreq(%d, %d)\n", StreamNumber, Setting);
	hr = rch->curDevice->SetStreamSourceFreq(StreamNumber, Setting);

	return hr;
}

HRESULT FakedD3DSurfaceRelease(RenderChannel *rch){
	HRESULT hr = D3D_OK;
	int id = rch->obj_id;  // get the surface id
	ClientSurface9 * surface = (ClientSurface9 *)rch->surface_list[id];

#if 1
	ULONG ref = surface->GetSurface9()->Release();
	cg::core::infoRecorder->logError("FakedD3DSrufaceRelease(), id:%d, ref:%d.\n", id, ref);
	if(ref <=0){
		rch->surface_list[id] = NULL;
	}
#endif


	return hr;
}

#endif

