#include "WrapDirect3dtexture9.h"
#include "WrapDirect3dsurface9.h"
#include "../LibCore/Opcode.h"
#include "CommandServerSet.h"
#include <d3dx9tex.h>
//#define SAVE_IMG
//#define LOCAL_IMG


//#define ENABLE_TEXTURE_LOG

int D3DXTexture = 0;
WrapperDirect3DTexture9::WrapperDirect3DTexture9(IDirect3DTexture9* ptr, int _id): m_tex(ptr), IdentifierBase(_id) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::WrapperDirect3DTexture9(), id=%d, base_tex=%d this=%d\n", id, ptr, this);

	if(ptr == NULL) {
		infoRecorder->logTrace("WrapperDirect3DTexture9::WrapperDirect3DTexture9(), base_tex is NULL\n");
	}
#endif
	m_list.AddMember(ptr, this);

	creationFlag = 0;
	updateFlag = 0x8fffffff;
	stable = true;
}

void WrapperDirect3DTexture9::SayHi(char* str) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("%s\n", str);
#endif
}

IDirect3DTexture9* WrapperDirect3DTexture9::GetTex9() {
	return this->m_tex;
}

WrapperDirect3DTexture9* WrapperDirect3DTexture9::GetWrapperTexture9(IDirect3DTexture9* ptr) {
	//infoRecorder->logError("WrapperDirect3DTexture9::GetWrapperTexture9(), ptr=%u\n", ptr);
	WrapperDirect3DTexture9* ret = (WrapperDirect3DTexture9*)( m_list.GetDataPtr(ptr) );
#ifdef ENABLE_TEXTURE_LOG
	if(ret == NULL) {
		infoRecorder->logTrace("WrapperDirect3DTexture9::GetWrapperTexture9(), ret is NULL\n");
	}
#endif
	return ret;
}

#ifdef MULTI_CLIENTS
int WrapperDirect3DTexture9::sendCreation(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: send creation.\n");
#endif
	if(id == 162){
		//infoRecorder->logError("[WrapperDrirect3DTexture9]: 162 to send creation.\n");
	}

	ContextAndCache * c = (ContextAndCache *)ctx;

	c->beginCommand(CreateTexture_Opcode, getDeviceId());
	c->write_int(getId());
	c->write_uint(Width);
	c->write_uint(Height);
	c->write_uint(Levels);
	c->write_uint(Usage);
	c->write_uint(Format);
	c->write_uint(Pool);
	c->endCommand();

	return 0;
}

int WrapperDirect3DTexture9::checkCreation(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: call check creation, creation flag:%x, id:%d.\n", creationFlag, id);
#endif
	ContextAndCache * cc = (ContextAndCache *)ctx;
	int ret = 0;
	//////////////

	if(!cc->isCreated(creationFlag)){
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: texture %d not created.\n", id);
#endif
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
		sendUpdate(ctx);
		cc->resetChanged(updateFlag);
		ret = 1;
	}else{
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: already created.\n");
#endif
	}
	return ret;
}

int WrapperDirect3DTexture9::checkUpdate(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: check update.\n");
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(c->isChanged(updateFlag)){
		ret = sendUpdate(c);
		c->resetChanged(updateFlag);
		ret = 1;
	}else{
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: not changed!");
#endif
	}

	return ret;

}

int WrapperDirect3DTexture9::sendUpdate(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: send update.\n");
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	SendTextureData(c);

	return ret;
}

#endif

HRESULT WrapperDirect3DTexture9::SendTextureData() {

#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("trying to call GetLevelCount()\n");
#endif
	int levelCount = m_tex->GetLevelCount();

#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DTexture9::SendTextureData(), id=%d, LevelCount=%d\n", this->id, levelCount);
#endif

#ifdef LOCAL_IMG
	for(int level=0; level<levelCount; ++level) {
		//int level = 0;
		LPDIRECT3DSURFACE9 top_surface = NULL;
		m_tex->GetSurfaceLevel(level, &top_surface);

		D3DLOCKED_RECT rect;
		infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), start lock %d level\n", level);
		//top_surface->
		top_surface->LockRect(&rect, 0, 0);

		cs.begin_command(TransmitTextureData_Opcode, id);
		cs.write_int(level);
		cs.end_command();

		//infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), id=%d, height=%d, width=%d, pitch=%d, size=%d\n", this->id, desc.Height, desc.Width, rect.Pitch,  desc.Height * desc.Width * byte_per_pixel);
		char  fname[50];
		sprintf(fname,"surface\\face_%d_leve_%d.png",this->id,level);

		top_surface->UnlockRect();

		infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), quit %d level\n", level);
	}
#else
	double tick_s= 0.0;
	tick_s = GetTickCount();
	HRESULT hr = D3D_OK;

	for(int level=0; level<levelCount; ++level) {
		//int level = 0;

		LPDIRECT3DSURFACE9 top_surface = NULL;
		hr = m_tex->GetSurfaceLevel(level, &top_surface);

		if(FAILED(hr)){
#ifdef ENABLE_TEXTURE_LOG
			infoRecorder->logError("[WrapperDirect3DTexture9]: %d get %d-th surface level failed with: %d.\n",id, level, hr);
#endif
			continue;
		}
		D3DLOCKED_RECT rect;
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), start lock %d level\n", level);
#endif
		//top_surface->
		// here lock failed when debug d3d, the flag and texture creation may error
		//cs.write_int(levelCount);

		//infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), id=%d, height=%d, width=%d, pitch=%d, size=%d\n", this->id, desc.Height, desc.Width, rect.Pitch,  desc.Height * desc.Width * byte_per_pixel);
		//WRITE_DATA(int, desc.Height * desc.Width * byte_per_pixel);
#ifdef MEM_FILE_TEX
		top_surface->LockRect(&rect, 0, D3DLOCK_READONLY);
		D3DSURFACE_DESC desc;
		top_surface->GetDesc(&desc);
		int byte_per_pixel = rect.Pitch / desc.Width;
#ifndef MULTI_CLIENTS
		cs.begin_command(TransmitTextureData_Opcode, id);
		cs.write_int(level);
		cs.write_uint(rect.Pitch);
		cs.write_int(desc.Height * desc.Width * byte_per_pixel);
		cs.write_byte_arr((char *)(rect.pBits), desc.Height * desc.Width * byte_per_pixel);
		cs.end_command();
#else
		csSet->beginCommand(TransmitTextureData_Opcode, id);
		csSet->writeInt(level);
		csSet->writeUInt(rect.Pitch);
		csSet->writeInt(desc.Height * desc.Width * byte_per_pixel);
		csSet->writeByteArr((char *)(rect.pBits), desc.Height *desc.Width * byte_per_pixel);
		csSet->endCommand();
#endif

		top_surface->UnlockRect();
#else
		LPD3DXBUFFER DestBuf = NULL;
		HRESULT hhr = D3DXSaveSurfaceToFileInMemory(&DestBuf,D3DXIFF_PNG,top_surface,NULL,NULL);

#ifdef ENABLE_TEXTURE_LOG
		if(FAILED(hr)){
			if(hr == D3DERR_INVALIDCALL){
				//MessageBox(NULL, "save surface failed INVALIDCALL", "WARNING", MB_OK);
				infoRecorder->logTrace("[WrapperDirect3DTexture9]: D3DXSaveSurfaceToFileInMemory failed INVALIDCALL.\n", hr);
			}
			else{
				//MessageBox(NULL, "save surface failed", "WARNING", MB_OK);
				infoRecorder->logTrace("[WrapperDirect3DTexture9]: D3DXSaveSurfaceToFileInMemory failed with:%d.\n", hr);
			}
		}
#endif

		int size = 0; 

		if(DestBuf){
			size = DestBuf->GetBufferSize();
			csSet->beginCommand(TransmitTextureData_Opcode, id);
			csSet->writeInt(level);
			csSet->writeInt(size);
			csSet->writeByteArr((char *)(DestBuf->GetBufferPointer()), size);
			csSet->endCommand();
			hr = DestBuf->Release();
#ifdef ENABLE_TEXTURE_LOG
			if(hr == 0){
				infoRecorder->logTrace("send surface, to free D3DXBuffer.\n");
			}
#endif
			DestBuf = NULL;
		}else{
#ifdef ENABLE_TEXTURE_LOG
			infoRecorder->logError("[WrapperDirect3DTexture9]: save to memory filed failed, id:%d.\n", id);
#endif
		}
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("send surface id:%d size:%d\n", id, size);
#endif

#endif

#ifdef SAVE_IMG
		char  fname[50];

		sprintf(fname,"surface\\face_%d_leve_%d.png",this->id,level);
		//sprintf(fname,"face_%d_leve_%d.png",this->id,level);
		hr = D3DXSaveSurfaceToFile(fname,D3DXIFF_PNG,top_surface,NULL,NULL);
		if(FAILED(hr)){
			MessageBox(NULL, "SAVE surface failed.", "WARNING", MB_OK);
		}
#endif

		if (top_surface){
			hr = top_surface->Release();
			top_surface = NULL;
		}

		//D3DXSaveSurfaceToFile(fname, D3DXIFF_JPG, top_surface,NULL,NULL);
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), quit %d level\n", level);
#endif
	}
	double tick_a = GetTickCount();
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("\tdeal texture data:%f\n", tick_a - tick_s);
#endif
#endif

	return D3D_OK;
}

HRESULT WrapperDirect3DTexture9::SendTextureData(ContextAndCache *ctx) {
	HRESULT hr = D3D_OK;
	if(ctx->get_connect_socket() == NULL)
		return hr;

#ifndef SEND_FULL_TEXTURE
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("trying to call GetLevelCount()\n");
#endif
	int levelCount = m_tex->GetLevelCount();

#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DTexture9::SendTextureData(), id=%d, LevelCount=%d\n", this->id, levelCount);
#endif

#ifdef LOCAL_IMG
	for(int level=0; level<levelCount; ++level) {
		//int level = 0;
		LPDIRECT3DSURFACE9 top_surface = NULL;
		m_tex->GetSurfaceLevel(level, &top_surface);

		D3DLOCKED_RECT rect;
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), start lock %d level\n", level);
#endif
		//top_surface->
		top_surface->LockRect(&rect, 0, 0);

		cs.begin_command(TransmitTextureData_Opcode, id);
		cs.write_int(level);
		cs.end_command();

		//infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), id=%d, height=%d, width=%d, pitch=%d, size=%d\n", this->id, desc.Height, desc.Width, rect.Pitch,  desc.Height * desc.Width * byte_per_pixel);
		char  fname[50];
		sprintf(fname,"surface\\face_%d_leve_%d.png",this->id,level);

		top_surface->UnlockRect();
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), quit %d level\n", level);
#endif
	}
#else
	double tick_s= 0.0;
	tick_s = GetTickCount();
	int byte_per_pixel = 0;

#ifdef UPDATE_ALL

	for(int level=0; level<levelCount; ++level) {
#else
	int level = 0;
#endif
	LPDIRECT3DSURFACE9 top_surface = NULL;
	hr = m_tex->GetSurfaceLevel(level, &top_surface);

	if(FAILED(hr)){
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: GetSurfaceLevel for %d failed.\n", level);
		//MessageBox(NULL, "GetSurfaceLevel failed.", "WARNING", MB_OK);
		continue;
	}

	
#ifdef MEM_FILE_TEX
	D3DLOCKED_RECT rect;
	hr = top_surface->LockRect(&rect, 0, D3DLOCK_READONLY);
	if(FAILED(hr)){
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: lock rect failed.\n");
	}
	D3DSURFACE_DESC desc;
	hr = top_surface->GetDesc(&desc);
	if(FAILED(hr)){
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: get desc failed.\n");
	}
	byte_per_pixel = rect.Pitch / desc.Width;

	infoRecorder->logTrace("[WrapperDirect3DTexture9]: texture desc, pitch:%d. height:%d, width:%d, byte_per_pixel:%d.\n", rect.Pitch, desc.Height, desc.Width, byte_per_pixel);

#ifndef MULTI_CLIENTS
	cs.begin_command(TransmitTextureData_Opcode, id);
	cs.write_int(level);
	cs.write_uint(rect.Pitch);
	cs.write_int(desc.Height * desc.Width * byte_per_pixel);
	cs.write_byte_arr((char *)(rect.pBits), desc.Height * desc.Width * byte_per_pixel);
	cs.end_command();
#else
	csSet->beginCommand(TransmitTextureData_Opcode, id);
	csSet->writeUInt(level);
	csSet->writeUInt(rect.Pitch);
	csSet->writeInt(desc.Height * rect.Pitch);
	csSet->writeByteArr((char *)(rect.pBits), desc.Height * rect.Pitch);
	csSet->endCommand();
#endif
	top_surface->UnlockRect();
#else
	LPD3DXBUFFER DestBuf = NULL;
	hr = D3DXSaveSurfaceToFileInMemory(&DestBuf,D3DXIFF_PNG,top_surface,NULL,NULL);
	if(FAILED(hr)){
		if(hr == D3DERR_INVALIDCALL){
			infoRecorder->logError("[WrapperDirect3DTexture9]: D3DXSaveSurfaceToFileInMemory failed INVALIDCALL.\n", hr);
		}
		else{
			//MessageBox(NULL, "save surface failed", "WARNING", MB_OK);
			infoRecorder->logError("[WrapperDirect3DTexture9]: D3DXSaveSurfaceToFileInMemory failed with:%d.\n", hr);
		}
		continue;
	}
	if(DestBuf == NULL){
		infoRecorder->logError("[WrapperDirect3DTexture9]: cannot get the file in memory.\n");
		continue;
		//MessageBox(NULL, "File in memory failed.", "WARNING", MB_OK);
	}

	int size = 0;
	size = DestBuf->GetBufferSize();

	ctx->beginCommand(TransmitTextureData_Opcode, id);
	ctx->write_int(level);
	ctx->write_int(size);
	ctx->write_byte_arr((char *)(DestBuf->GetBufferPointer()), size);
	ctx->endCommand();

	if(DestBuf){
		hr = DestBuf->Release();
#ifdef ENABLE_TEXTURE_LOG
		if(hr == 0){
			infoRecorder->logTrace("send surface, to free the D3DXBuffer.\n");
		}
#endif
		DestBuf = NULL;
	}
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("send surface id:%d size:%d\n",this->id, size);
#endif

#endif

#ifdef SAVE_IMG
	char  fname[50] = {0};
	sprintf(fname,"surface\\face_%d_leve_%d.png",this->id,level);
	//sprintf(fname,"face_%d_leve_%d.png",this->id,level);
	hr = D3DXSaveSurfaceToFile(fname,D3DXIFF_PNG,top_surface,NULL,NULL);
	if(FAILED(hr)){
		MessageBox(NULL, "SAVE surface failed.", "WARNING", MB_OK);
	}
#endif

	if (top_surface){
		hr = top_surface->Release();
		top_surface = NULL;
	}
	infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), quit %d level\n", level);

#ifdef UPDATE_ALL
}
#else
#endif
#ifdef ENABLE_TEXTURE_LOG
	double tick_a = GetTickCount();
	infoRecorder->logTrace("\tdeal texture data:%f\n", tick_a - tick_s);
#endif
#endif
#else
	// use the D3DXSaveTextureInMemory
	LPD3DXBUFFER DestBuf = NULL;
	hr = D3DXSaveTextureToFileInMemory(&DestBuf,D3DXIFF_PNG,m_tex,NULL);
	if(FAILED(hr)){
		if(hr == D3DERR_INVALIDCALL){
			MessageBox(NULL, "save texture failed INVALIDCALL", "WARNING", MB_OK);
			infoRecorder->logTrace("[WrapperDirect3DTexture9]: D3DXSaveTextureToFileInMemory failed INVALIDCALL.\n", hr);
		}
		else{
			MessageBox(NULL, "save texture failed", "WARNING", MB_OK);
			infoRecorder->logTrace("[WrapperDirect3DTexture9]: D3DXSaveTextureToFileInMemory failed with:%d.\n", hr);
		}
	}
	if(DestBuf == NULL){
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: cannot get the file in memory.\n");
		MessageBox(NULL, "File in memory failed.", "WARNING", MB_OK);
	}

	int size = DestBuf->GetBufferSize();

	ctx->beginCommand(TransmitTextureData_Opcode, id);
	//ctx->write_int(level);
	ctx->write_int(size);
	ctx->write_byte_arr((char *)(DestBuf->GetBufferPointer()), size);
	ctx->endCommand();

	if(DestBuf){
		hr = DestBuf->Release();
		if(hr == 0){
			infoRecorder->logTrace("send surface, to free the D3DXBuffer.\n");
		}
		DestBuf = NULL;
	}	

#endif

	return D3D_OK;
}

/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DTexture9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::QueryInterface(), ppvObj=%d\n", *ppvObj);
#endif
	HRESULT hr = m_tex->QueryInterface(riid, ppvObj);
	*ppvObj = this;
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::QueryInterface() end, ppvObj=%d\n", *ppvObj);
#endif
	return hr;
}
STDMETHODIMP_(ULONG) WrapperDirect3DTexture9::AddRef(THIS) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::AddRef() called\n");
#endif
	refCount++;
	return m_tex->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3DTexture9::Release(THIS) {
	HRESULT hr = m_tex->Release();
#ifdef LOG_REF_COUNT
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::Release() succeeded, ref:%d\n", hr);
#endif
#endif
	refCount--;
	if(refCount <= 0){
		infoRecorder->logError("[WrapperDirect3DTexture9]: m_tex ref:%d, ref count:%d.\n", refCount, hr);
	}
	return hr;
	//return D3D_OK;
}

/*** IDirect3DBaseTexture9 methods ***/
STDMETHODIMP WrapperDirect3DTexture9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetDevice() TODO\n");
#endif
	return m_tex->GetDevice(ppDevice);
}
STDMETHODIMP WrapperDirect3DTexture9::SetPrivateData(
	THIS_ REFGUID refguid,CONST void* pData,DWORD SizeOfData,DWORD Flags) {
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("WrapperDirect3DTexture9::SetPrivateData() TODO\n");
#endif
		return m_tex->SetPrivateData(refguid, pData, SizeOfData, Flags);
}

STDMETHODIMP WrapperDirect3DTexture9::GetPrivateData(THIS_ REFGUID refguid,void* pData,DWORD* pSizeOfData) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetPrivateData() TODO\n");
#endif
	return m_tex->GetPrivateData(refguid, pData, pSizeOfData);
}
STDMETHODIMP WrapperDirect3DTexture9::FreePrivateData(THIS_ REFGUID refguid) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::FreePrivateData() TODO\n");
#endif
	return m_tex->FreePrivateData(refguid);
}
STDMETHODIMP_(DWORD) WrapperDirect3DTexture9::SetPriority(THIS_ DWORD PriorityNew) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::SetPriority() TODO\n");
#endif
	return m_tex->SetPriority(PriorityNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DTexture9::GetPriority(THIS) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetPriority() called\n");
#endif
	return m_tex->GetPriority();
}

STDMETHODIMP_(void) WrapperDirect3DTexture9::PreLoad(THIS) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::PreLoad() TODO\n");
#endif
	return m_tex->PreLoad();
}

STDMETHODIMP_(D3DRESOURCETYPE) WrapperDirect3DTexture9::GetType(THIS) {
	//infoRecorder->logTrace("WrapperDirect3DTexture9::GetType() TODO\n");
	return m_tex->GetType();
}

STDMETHODIMP_(DWORD) WrapperDirect3DTexture9::SetLOD(THIS_ DWORD LODNew) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::SetLOD() TODO\n");
#endif
	return m_tex->SetLOD(LODNew);
}

STDMETHODIMP_(DWORD) WrapperDirect3DTexture9::GetLOD(THIS) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetLOD() called\n");
#endif
	return m_tex->GetLOD();
}

STDMETHODIMP_(DWORD) WrapperDirect3DTexture9::GetLevelCount(THIS) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetLevelCount() called\n");
#endif
	return m_tex->GetLevelCount();
}

STDMETHODIMP WrapperDirect3DTexture9::SetAutoGenFilterType(THIS_ D3DTEXTUREFILTERTYPE FilterType) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DTexture9::SetAutoGenFilterType() called\n");
#endif

#ifndef MULTI_CLIENTS
	cs.begin_command(TextureSetAutoGenFilterType_Opcode, id);
	cs.write_uint(FilterType);
	cs.end_command();
#else
	csSet->beginCommand(TextureSetAutoGenFilterType_Opcode, id);
	csSet->writeUInt(FilterType);
	csSet->endCommand();
#endif

	return m_tex->SetAutoGenFilterType(FilterType);
}

STDMETHODIMP_(D3DTEXTUREFILTERTYPE) WrapperDirect3DTexture9::GetAutoGenFilterType(THIS) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetAutoGenFilterType() TODO\n");
#endif
	return m_tex->GetAutoGenFilterType();
}

STDMETHODIMP_(void) WrapperDirect3DTexture9::GenerateMipSubLevels(THIS) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DTexture9::GenerateMipSubLevels() called\n");
#endif

#ifndef MULTI_CLIENTS
	cs.begin_command(TextureGenerateMipSubLevels_Opcode, id);
	cs.end_command();
#else
	csSet->beginCommand(TextureGenerateMipSubLevels_Opcode, id);
	csSet->endCommand();
#endif
	return m_tex->GenerateMipSubLevels();
}

STDMETHODIMP WrapperDirect3DTexture9::GetLevelDesc(THIS_ UINT Level,D3DSURFACE_DESC *pDesc) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetLevelDesc() called\n");
#endif
	return m_tex->GetLevelDesc(Level, pDesc);
}

extern int deviceId;

STDMETHODIMP WrapperDirect3DTexture9::GetSurfaceLevel(THIS_ UINT Level,IDirect3DSurface9** ppSurfaceLevel) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetSurfaceLevel() called, Level:%d, tex id:%d\n",Level, id);
#endif
	IDirect3DSurface9* base_surface = NULL;
	HRESULT hr = m_tex->GetSurfaceLevel(Level, &base_surface);//ppSurfaceLevel);

	WrapperDirect3DSurface9* surface = WrapperDirect3DSurface9::GetWrapperSurface9(base_surface);
	if(NULL == surface) {
		// create new wrapper surface to hold the surface
		surface = new WrapperDirect3DSurface9(base_surface, WrapperDirect3DSurface9::ins_count++);

#ifndef MULTI_CLIENTS
		cs.begin_command(TextureGetSurfaceLevel_Opcode, id);
		cs.write_int(this->id);
		cs.write_int(surface->GetID());
		cs.write_uint(Level);
		cs.end_command();
#else
		// TODO : check the texture object is exist or not !
		csSet->checkCreation(this);

		// create new surface
		csSet->beginCommand(TextureGetSurfaceLevel_Opcode, id);
		csSet->writeInt(id);
		csSet->writeInt(surface->getId());
		csSet->writeUInt(Level);
		csSet->endCommand();

		csSet->setCreation(surface->creationFlag);

#endif
		surface->creationCommand = TextureGetSurfaceLevel_Opcode; 
		surface->SetTexId(id);
		surface->SetLevel(Level);
	}
	//csSet->setChangedToAll(updateFlag);
	*ppSurfaceLevel = dynamic_cast<IDirect3DSurface9 *>(surface);
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetSurfaceLevel(), base_surface=%d, ppSurfaceLevel=%d\n", base_surface, *ppSurfaceLevel);
#endif
	return hr;
}

STDMETHODIMP WrapperDirect3DTexture9::LockRect(THIS_ UINT Level,D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9:LockRect(), id:%d, level:%d, rect:%p.\n", id, Level, pRect);
#endif
	tex_send[id] = false;

	//if(!(this->Usage & D3DUSAGE_RENDERTARGET))
	csSet->setChangedToAll(updateFlag);

	return m_tex->LockRect(Level, pLockedRect, pRect, Flags);
}

STDMETHODIMP WrapperDirect3DTexture9::UnlockRect(THIS_ UINT Level) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::UnlockRect(), id:%d, Level=%d\n", id, Level);
#endif
	HRESULT hr = m_tex->UnlockRect(Level);
	return hr;
}

STDMETHODIMP WrapperDirect3DTexture9::AddDirtyRect(THIS_ CONST RECT* pDirtyRect) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::AddDirtyRect() TODO\n");
#endif
	return m_tex->AddDirtyRect(pDirtyRect);
}
