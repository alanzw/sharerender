#include "WrapDirect3dtexture9.h"
#include "WrapDirect3dsurface9.h"
#include "../LibCore/Opcode.h"
#include "CommandServerSet.h"
#include <d3dx9tex.h>
//#define SAVE_IMG
//#define LOCAL_IMG


int WrapperDirect3DTexture9::totalBuffer =0;
int WrapperDirect3DTexture9::maxBufferSize = 0;

WrapperDirect3DTexture9::WrapperDirect3DTexture9(const WrapperDirect3DTexture9 &tex){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("[WrapperDirect3DTexture9]: copy constructor this:%p, id:%d.\n", this, id);
#endif
}

WrapperDirect3DTexture9::WrapperDirect3DTexture9(IDirect3DTexture9* ptr, int _id, int levels): m_tex(ptr), IdentifierBase(_id), Levels(levels) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::WrapperDirect3DTexture9(), id=%d, base_tex=0x%p this=0x%p\n", id, ptr, this);
#endif
	m_list.AddMember(ptr, this);

	creationFlag = 0;
	updateFlag = 0; //0x8fffffff;
	//stable = true;
	stable = false;
	texHelper = NULL;
	bufferSize = 0;

	// allocate the surface array
	surfaceArray = new WrapperDirect3DSurface9 *[levels];
	for(int i = 0; i < levels; i++){
		surfaceArray[i] = NULL;
	}
}

IDirect3DTexture9* WrapperDirect3DTexture9::GetTex9() {
	return this->m_tex;
}

WrapperDirect3DTexture9* WrapperDirect3DTexture9::GetWrapperTexture9(IDirect3DTexture9* ptr) {
	WrapperDirect3DTexture9* ret = (WrapperDirect3DTexture9*)( m_list.GetDataPtr(ptr) );
#ifdef ENABLE_TEXTURE_LOG
	if(ret == NULL) {
		infoRecorder->logError("WrapperDirect3DTexture9::GetWrapperTexture9(), ret is NULL\n");
	}
#endif
	return ret;
}

#ifdef MULTI_CLIENTS
int WrapperDirect3DTexture9::sendCreation(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: send creation.\n");
#endif
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

	return 1;
}

int WrapperDirect3DTexture9::checkCreation(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: call check creation, tex id:%d creation flag:%x\n", id, creationFlag);
#endif
	ContextAndCache * cc = (ContextAndCache *)ctx;
	int ret = 0;
	//////////////

	if(!cc->isCreated(creationFlag)){
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: texture %d not created, compression: %d\n", id, texHelper ? (texHelper->getCompression()): 1);
#endif
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
#if 0
		if(ret = sendUpdate(ctx))
			cc->resetChanged(updateFlag);
#endif
		//ret = 1;
	}else{
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: already created.\n");
#endif
	}
	return ret;
}

int WrapperDirect3DTexture9::checkUpdate(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: check update for %d.\n", id);
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(c->isChanged(updateFlag)){
		ret = sendUpdate(c);
		if(ret)
			c->resetChanged(updateFlag);
		//ret = 1;
	}else{
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: not changed!");
#endif
	}

	return ret;
}

int WrapperDirect3DTexture9::sendUpdate(void *ctx){
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("[WrapperDirect3DTexture9]: send update for %d.\n", id);
#endif
	int ret = 1;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(pTimer){
		pTimer->Start();
	}
	if(E_FAIL == SendTextureData(c)){
		ret = 0;
	}
	if(pTimer){
		unsigned int interval_ = (UINT)pTimer->Stop();
		infoRecorder->logError("[WrapperDriect3DTexture9]: send texture %d use: %f.\n", id, interval_ * 1000.0 / pTimer->getFreq());
	}

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
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), start lock %d level\n", level);
#endif
		//top_surface->
		// here lock failed when debug d3d, the flag and texture creation may error
		//cs.write_int(levelCount);

		//infoRecorder->logTrace("WrapperDirect3DTexture9::SendTextureData(), id=%d, height=%d, width=%d, pitch=%d, size=%d\n", this->id, desc.Height, desc.Width, rect.Pitch,  desc.Height * desc.Width * byte_per_pixel);
		//WRITE_DATA(int, desc.Height * desc.Width * byte_per_pixel);
#ifdef MEM_FILE_TEX
		D3DLOCKED_RECT rect;
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

// return the buffer size for the texture
int WrapperDirect3DTexture9::getBufferSize(){
	//return texHelper->getBufferSize();
	return bufferSize;
}

#ifdef USE_TEXTURE_HELPER
HRESULT WrapperDirect3DTexture9::SendTextureData(ContextAndCache *ctx){
	HRESULT hr = D3D_OK;
	if(NULL == ctx->get_connect_socket())
		return E_FAIL;

#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("[WrapperDirect3DTexture9]: send texture data for %d.\n", id);
#endif
	SurfaceHelper * surHelper = NULL;
	if(!texHelper || Usage & D3DUSAGE_RENDERTARGET){
		infoRecorder->logError("[WrapperDirect3DTexture9]: RenderTarget texture, not to send.\n");
		return hr;
	}
	// update the texture
	if(texHelper->isAutoGenable()){
		surHelper = texHelper->getSurfaceHelper(0);
		if(!surHelper || !surHelper->isAquired()){
			// not aquired yet
			infoRecorder->logError("[WrapperDirect3DTExture9]: surface helper for %d level:%d is not aquired yet.\n", id, 0);
			return E_FAIL;
		}
		if(ctx->isChanged(surHelper->updateFlag)){

			ctx->beginCommand(TransmitTextureData_Opcode, id);
			ctx->write_uint(0);
			ctx->write_uint(texHelper->isAutoGenable() ? 1: 0);
			ctx->write_int(surHelper->getPitchedSize());
			ctx->write_packed_byte_arr((char *)(surHelper->getSurfaceData()), surHelper->getPitchedSize());
			//ctx->endCommand();

#ifdef ENABLE_TEXTURE_LOG
			infoRecorder->logError("[WrapperDirect3DTexture9]; send surface 0, size:%d, update flag:0x%x.\n", surHelper->getPitchedSize(), surHelper->updateFlag);
			ctx->resetChanged(surHelper->updateFlag);
#endif
		}
	}
	else{
		// creation for each surface
		for(UINT i = 0; i < Levels; i++){
			surHelper = texHelper->getSurfaceHelper(i);
			if(!surHelper || !surHelper->isAquired()){
				// not aquired yet
				infoRecorder->logError("[WrapperDirect3DTExture9]: surface helper for %d level:%d is not aquired yet.\n", id, i);
				return E_FAIL;
			}
			if(ctx->isChanged(surHelper->updateFlag)){
#if 0
				csSet->beginCommand(TransmitTextureData_Opcode, id);
				csSet->writeUInt(i);
				csSet->writeUInt(texHelper->isAutoGenable() ? 1: 0);
				csSet->writeInt(surHelper->getPitchedSize());
				csSet->writePackedByteArr((char *)(surHelper->getSurfaceData()), surHelper->getPitchedSize());
				csSet->endCommand();
#else
				ctx->beginCommand(TransmitTextureData_Opcode, id);
				ctx->write_uint(i);
				ctx->write_uint(texHelper->isAutoGenable() ? 1: 0);
				ctx->write_int(surHelper->getPitchedSize());
				ctx->write_packed_byte_arr((char *)(surHelper->getSurfaceData()), surHelper->getPitchedSize());
				//ctx->endCommand();
#endif
#ifdef ENABLE_TEXTURE_LOG
				infoRecorder->logError("[WrapperDirect3DTexture9]; send surface %d, size:%d.\n",i, surHelper->getPitchedSize());
#endif
			}
		}
	}

	return hr;
}

#else // USE_TEXTURE_HELPER

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

#endif // USE_TEXTURE_HELPER
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
	infoRecorder->logTrace("WrapperDirect3DTexture9::AddRef() called for %d\n", id);
#endif
	refCount++;
	return m_tex->AddRef();
}
STDMETHODIMP_(ULONG) WrapperDirect3DTexture9::Release(THIS) {
	ULONG hr = m_tex->Release();
#ifdef LOG_REF_COUNT
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::Release() succeeded, ref:%d\n", hr);
#endif
#endif
	refCount--;
#ifdef USE_TEXTURE_HELPER


#if 1
	if(hr <= 0){
		infoRecorder->logTrace("[WrapperDirect3DTexture9]: m_tex id:%d ref:%d, ref count:%d, buffer size%d.\n",id, refCount, hr, bufferSize);
		for(UINT i = 0; i < Levels; i++){
			if(surfaceArray[i]){
				surfaceArray[i]->releaseData();
				surfaceArray[i] = NULL;
			}
		}
		if(texHelper){
			bufferSize = texHelper->getBufferSize();
			delete texHelper;
			texHelper = NULL;
			totalBuffer -= bufferSize;
		}
		m_list.DeleteMember(m_tex);
	}
#endif


#endif
	return hr;
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
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetType() TODO\n");
#endif
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
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetLevelCount() called, id:%d\n", id);
#endif
	return m_tex->GetLevelCount();
}

STDMETHODIMP WrapperDirect3DTexture9::SetAutoGenFilterType(THIS_ D3DTEXTUREFILTERTYPE FilterType) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DTexture9::SetAutoGenFilterType() called\n");
#endif

	csSet->checkObj(this);

	csSet->beginCommand(TextureSetAutoGenFilterType_Opcode, id);
	csSet->writeUInt(FilterType);
	csSet->endCommand();

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
	csSet->checkObj(this);
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
	infoRecorder->logTrace("WrapperDirect3DTexture9::GetSurfaceLevel() called, Level:%d, tex id:%d, autogen:%s\n",Level, id, texHelper->isAutoGenable() ? "true":"false");
#endif
	IDirect3DSurface9* base_surface = NULL;

	HRESULT hr = m_tex->GetSurfaceLevel(Level, &base_surface);//ppSurfaceLevel);
	//infoRecorder->logError("WrapperDirect3DTexture9::GetSurfaceLevel() called, Level:%d, tex id:%d, autogen:%s, base surface:0x%p\n",Level, id, texHelper->isAutoGenable() ? "true":"false", base_surface);
	WrapperDirect3DSurface9* surface = WrapperDirect3DSurface9::GetWrapperSurface9(base_surface);

	if(NULL == surface ){ //&& NULL == side_surface) {
		// create new wrapper surface to hold the surface
		surface = new WrapperDirect3DSurface9(base_surface, WrapperDirect3DSurface9::ins_count++);
		//m_side_list.AddMember(uid, surface);

		surfaceArray[Level] = surface;

		surface->creationCommand = TextureGetSurfaceLevel_Opcode;
		surface->SetTexId(id);
		surface->SetLevel(Level);

		surface->setParentTexture(this);
#ifdef ENABLE_TEXTURE_LOG
		infoRecorder->logError("[WrapperDirect3DTexture9]: texture %d GetSurfaceLevel, create surface: %d with level:%d.\n", id, surface->getId(), Level);
#endif

		csSet->checkObj(this);

#ifdef USE_TEXTURE_HELPER
		//base_surface->GetDesc(&desc);
		SurfaceHelper * surHelper = NULL;
		if(texHelper){
			if(texHelper->isAutoGenable()){
				if(0 == Level){
					surHelper = texHelper->getSurfaceHelper(Level);
					surface->setSurfaceHelper(surHelper);
				}
			}
			else{
				// no auto gen, buffer each surface
				surHelper = texHelper->getSurfaceHelper(Level);
				surface->setSurfaceHelper(surHelper);
			}
			bufferSize = texHelper->getBufferSize();
			totalBuffer+= bufferSize;
			if(totalBuffer > maxBufferSize){
				maxBufferSize = totalBuffer;
			}
		}
#endif
	}

	*ppSurfaceLevel = dynamic_cast<IDirect3DSurface9 *>(surface);

	return hr;
}

STDMETHODIMP WrapperDirect3DTexture9::LockRect(THIS_ UINT Level, D3DLOCKED_RECT* pLockedRect,CONST RECT* pRect,DWORD Flags) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DTexture9:LockRect(), id:%d, level:%d, rect:%p.\n", id, Level, pRect);
#endif // ENABLE_TEXTURE_LOG
	//tex_send[id] = false;
	//csSet->checkObj(this);
	//csSet->setChangedToAll(updateFlag);
	//updateFlag = 0x8fffffff;

	D3DSURFACE_DESC desc;
	HRESULT hr = m_tex->GetLevelDesc(Level, &desc);

	hr = m_tex->LockRect(Level, pLockedRect, pRect, Flags);

#ifdef USE_TEXTURE_HELPER
	if(!texHelper){
		return hr;
	}

	SurfaceHelper * surHelper = NULL;
	int newAllocate = 0;
	if(texHelper->isAutoGenable()){
		// buffer the top surface
		if(0 == Level){
			surHelper = texHelper->getSurfaceHelper(Level);
			surHelper->setLockFlags(Flags);
			if(!(Flags & D3DLOCK_READONLY)){
				if(!surHelper->isAquired()){
					// not aquired yet, means no memory allocated
					surHelper->setRealSurfacePointer(pLockedRect->pBits);
					pLockedRect->pBits = surHelper->allocateSurfaceBuffer(pLockedRect->Pitch, desc.Height);

					newAllocate += surHelper->getPitchedSize();
					
				}
				else{
					surHelper->setRealSurfacePointer(pLockedRect->pBits);
					pLockedRect->pBits = surHelper->getSurfaceData();
				}
			}
		}
	}
	else{
		surHelper = texHelper->getSurfaceHelper(Level);
		if(!(Flags & D3DLOCK_READONLY)){
			surHelper->setRealSurfacePointer(pLockedRect->pBits);
			if(!surHelper->isAquired()){
				// not aquired yet, means no memory allocated
				pLockedRect->pBits = surHelper->allocateSurfaceBuffer(pLockedRect->Pitch, desc.Height);
				newAllocate = surHelper->getPitchedSize();
			}
			else{
				pLockedRect->pBits = surHelper->getSurfaceData();
			}
		}
	}

	bufferSize = texHelper->getBufferSize();
	if(newAllocate){
		totalBuffer+= newAllocate;
		if(maxBufferSize < totalBuffer){
			maxBufferSize = totalBuffer;
		}
	}

#endif

	return hr;
}

STDMETHODIMP WrapperDirect3DTexture9::UnlockRect(THIS_ UINT Level) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logError("WrapperDirect3DTexture9::UnlockRect(), id:%d, Level=%d\n", id, Level);
#endif

#ifdef USE_TEXTURE_HELPER
	SurfaceHelper* surHelper = NULL;
	if(texHelper){
		if(texHelper->isAutoGenable()){
			if(Level == 0){
				// copy the top data to video memory
				surHelper = texHelper->getSurfaceHelper(Level);
				if(!(surHelper->getLockFlags() & D3DLOCK_READONLY))
					surHelper->copyTextureData();
			}
		}else{
			// when no auto gen, buffer each level
			surHelper = texHelper->getSurfaceHelper(Level);
			if(!(surHelper->getLockFlags() & D3DLOCK_READONLY))
				surHelper->copyTextureData();
		}
		bufferSize = texHelper->getBufferSize();
	}
#endif
	updateFlag = 0x8fffffff;

	HRESULT hr = m_tex->UnlockRect(Level);
	return hr;
}

STDMETHODIMP WrapperDirect3DTexture9::AddDirtyRect(THIS_ CONST RECT* pDirtyRect) {
#ifdef ENABLE_TEXTURE_LOG
	infoRecorder->logTrace("WrapperDirect3DTexture9::AddDirtyRect() TODO\n");
#endif
	return m_tex->AddDirtyRect(pDirtyRect);
}

UINT WrapperDirect3DTexture9::getUID(int tex_id, char level){
	UINT ret = tex_id;
	ret = ret << 8;
	UINT mask = 0xFFFFFF00;
	ret &= mask;
	ret |= level;
	return ret;
}
void WrapperDirect3DTexture9::getTexIdAndLevel(UINT uid, int &_id, short &level){
	UINT tl = uid & 0x000000FF;
	UINT utid = (uid & 0xFFFFFF00);
	int tid = (int)utid;
	tid = tid >> 8;
	level = tl;
	_id = tid;
}
