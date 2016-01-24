#include "CommandServerSet.h"
#include "WrapDirect3d9.h"
#include "WrapDirect3ddevice9.h"
#include "WrapDirect3dvertexbuffer9.h"
#include "WrapDirect3dindexbuffer9.h"
#include "WrapDirect3dvertexdeclaration9.h"
#include "WrapDirect3dvertexshader9.h"
#include "WrapDirect3dpixelshader9.h"
#include "WrapDirect3dtexture9.h"
#include "WrapDirect3dstateblock9.h"
#include "WrapDirect3dcubetexture9.h"
#include "WrapDirect3dswapchain9.h"
#include "WrapDirect3dsurface9.h"
#include "WrapDirect3dvolumetexture9.h"
#include "../LibCore/Opcode.h"
#include "../LibCore/CmdHelper.h"

#include "YMesh.h"

#define DELAY_TO_DRAW
#ifdef MULTI_CLIENTS

//#define ENABLE_DEVICE_LOG

//#define USE_HELPER_SYNC

extern StateBlockRecorder * stateRecorder;

// send the creation command
int WrapperDirect3DDevice9::sendCreation(void * ctx){
#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9}: send creation.\n");
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;
	c->beginCommand(CreateDevice_Opcode, 0);
	c->write_int(getId());
	c->write_uint(adapter);
	c->write_uint(deviceType);
	c->write_uint(behaviorFlags);
	c->write_byte_arr((char *)pPresentParameters, sizeof(D3DPRESENT_PARAMETERS));
	c->endCommand();
	return 0;
}

// check the creation flag, if not created in client, send the creation command
int WrapperDirect3DDevice9::checkCreation(void * ctx){
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]: check creation, id:%d, creation flag:%d.\n", id, creationFlag);
#endif
	int ret = 0;
	ContextAndCache * cc = (ContextAndCache *)ctx;

	if(!cc->isCreated(creationFlag)){
		ret = sendCreation(ctx);
		cc->setCreation(creationFlag);
		ret = 1;
	}
	return ret;
}

// no update
int WrapperDirect3DDevice9::checkUpdate(void * ctx){
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("[WrapeprDirect3DDevice9]: check update. TODO.\n");
#endif
	int ret = 0;
	return ret;
}
int WrapperDirect3DDevice9::sendUpdate(void * ctx){
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("[WrapperDirect3DDevice9]: send update. TODO.\n");
#endif
	return 0;
}

#endif

STDMETHODIMP WrapperDirect3DDevice9::DrawPrimitive(THIS_ D3DPRIMITIVETYPE PrimitiveType,UINT StartVertex,UINT PrimitiveCount) {
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::DrawPrimitive(), vb_0=0x%X, PrimitiveType=%d, startVertex=%d, primitiveCount=%d\n", cur_vbs_[0], PrimitiveType, StartVertex, PrimitiveCount);
#endif

	//根据startvertex和primitivecount来定位ymesh
	//startVertex 不一定为0
	//PrimitiveType 有可能为5
	//setstreamsource 的offest有可能不为0			重要

	// add sync to taskqueue, the task queue must clear all objects in queue
#ifdef USE_HELPER_SYNC
	SynEntity * sync = new SynEntity();
	csSet->pushSync(sync);
#endif

#ifndef MULTI_CLIENTS
		cs.begin_command(DrawPrimitive_Opcode, id);
		cs.write_char(PrimitiveType);
		cs.write_uint(StartVertex);
		cs.write_uint(PrimitiveCount);
		cs.end_command();
#else

		// send command to all clients
		csSet->beginCommand(DrawPrimitive_Opcode, id);
		csSet->writeChar(PrimitiveType);
		csSet->writeUChar(StartVertex);
		csSet->writeUInt(PrimitiveCount);
		csSet->endCommand();
#endif
		HRESULT hr = D3D_OK;
	if(cmdCtrl->isRender()) {
		hr = m_device->DrawPrimitive(PrimitiveType, StartVertex, PrimitiveCount);
		return hr;
	}
	else
		return D3D_OK;

}

STDMETHODIMP WrapperDirect3DDevice9::DrawIndexedPrimitive(THIS_ D3DPRIMITIVETYPE Type,INT BaseVertexIndex,UINT MinVertexIndex,UINT NumVertices,UINT startIndex,UINT primCount) {
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::DrawIndexedPrimitive() device id:%d, type=%d, baseVertexIndex=%d, MinVertexIndex=%d, NumVertex=%d, startIndex=%d, count=%d\n", id, Type, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);
#endif
#ifdef USE_HELPER_SYNC
	SynEntity * sync = new SynEntity();
	csSet->pushSync(sync);
#endif  // USE_HELPER_SYNC

	//the mesh doesn't do the decimation, render yourself.
#ifndef MULTI_CLIENTS
	cs.begin_command(DrawIndexedPrimitive_Opcode, id);
	cs.write_char(Type);
	cs.write_int(BaseVertexIndex);
	cs.write_int(MinVertexIndex);
	cs.write_uint(NumVertices);
	cs.write_uint(startIndex);
	cs.write_uint(primCount);
	cs.end_command();
#else   // MULTI_CLIENTS


#ifdef USE_MESH
	assert(cur_ib_);
	YMesh * yMesh = NULL;
	if(cur_ib_){
		yMesh = cur_ib_->GetYMesh(cur_vbs_[0]->GetId(), BaseVertexIndex, startIndex);
		if(yMesh == NULL){
			// not find
			yMesh = new YMesh(this);
			cur_ib_->PutYMesh(cur_vbs_[0]->GetId(), BaseVertexIndex, startIndex, yMesh);
			assert(cur_vbs_[0]);
			yMesh->SetLeaderVB(cur_vbs_[0]->GetId());
			yMesh->SetIndices(cur_ib_);
			yMesh->SetDeclaration(cur_decl_);
			yMesh->SetStreamSource(cur_vbs_);
			yMesh->SetDrawType(DRAWINDEXEDPRIMITIVE);
			yMesh->SetIndexParams(Type, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);
		}else{
			// find ont
			yMesh->SetIndexParams(Type, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);
		}
	}

	yMesh->Render(BaseVertexIndex, MinVertexIndex, startIndex, primCount);

#else // USE_MESH

	// set index

	// set vertex declaration

	// set stream source

	// call draw
	csSet->beginCommand(DrawIndexedPrimitive_Opcode, id);
	csSet->writeChar(Type);
	csSet->writeInt(BaseVertexIndex);
	csSet->writeInt(MinVertexIndex);
	csSet->writeUInt(NumVertices);
	csSet->writeUInt(startIndex);
	csSet->writeUInt(primCount);
	csSet->endCommand();
#endif // USE_MESH

#endif   // MULTI_CLIENTS
	HRESULT hr = D3D_OK;
	if(cmdCtrl->isRender()){
		hr = m_device->DrawIndexedPrimitive(Type, BaseVertexIndex, MinVertexIndex, NumVertices, startIndex, primCount);
		#ifdef ENABLE_DEVICE_LOG
		if(FAILED(hr)){
			
			infoRecorder->logError("WrapperDirect3DDevice9::DrawIndexedPrimitive failed with:%d.\n", hr);
		}
#endif
		return hr;
	}
	else
		return D3D_OK;

}

UINT arr[4];

STDMETHODIMP WrapperDirect3DDevice9::DrawPrimitiveUP(THIS_ D3DPRIMITIVETYPE PrimitiveType,UINT PrimitiveCount,CONST void* pVertexStreamZeroData,UINT VertexStreamZeroStride) {
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::DrawPrimitiveUP(), type=%d, count=%d\n", PrimitiveType, PrimitiveCount);
#endif
#ifdef USE_HELPER_SYNC
	SynEntity * sync =new SynEntity();
	csSet->pushSync(sync);
#endif

	int VertexCount = 0;
	switch(PrimitiveType){
	case D3DPT_POINTLIST:
		VertexCount = PrimitiveCount;
		break;
	case D3DPT_LINELIST:
		VertexCount = PrimitiveCount * 2;
		break;
	case D3DPT_LINESTRIP:
		VertexCount = PrimitiveCount + 1;
		break;
	case D3DPT_TRIANGLELIST:
		VertexCount = PrimitiveCount * 3;
		break;
	case D3DPT_TRIANGLESTRIP:
	case D3DPT_TRIANGLEFAN:
		VertexCount = PrimitiveCount + 2;
		break;
	default:
		VertexCount = 0;
		break;
	}

#ifndef MULTI_CLIENTS
	cs.begin_command(DrawPrimitiveUP_Opcode, id);
	cs.write_vec(DrawPrimitiveUP_Opcode, (float*)&arr);
	cs.write_vec(DrawPrimitiveUP_Opcode, (float*)pVertexStreamZeroData, VertexCount * VertexStreamZeroStride);
	cs.end_command();
#else
	// send to all connected clients
	arr[0] = PrimitiveType;
	arr[1] = PrimitiveCount;
	arr[2] = VertexCount;
	arr[3] = VertexStreamZeroStride;

	csSet->beginCommand(DrawPrimitiveUP_Opcode, id);
	csSet->writeVec(DrawPrimitiveUP_Opcode, (float *)&arr);
	csSet->writeVec(DrawPrimitiveUP_Opcode, (float *)pVertexStreamZeroData, VertexCount * VertexStreamZeroStride);
	csSet->endCommand();

#if 0
	if(VertexStreamZeroStride == 40){
		// print the vertex
		infoRecorder->logTrace("Vertex count:%d\n", VertexCount);
		for(int i = 0; i< VertexCount; i++){
			infoRecorder->logTrace("vertex: %d is:\n", i);
			for(int j = 0; j < VertexStreamZeroStride >> 2; j++){
				infoRecorder->logTrace("%f\t", ((float *)pVertexStreamZeroData)[i * VertexStreamZeroStride >> 2 + j]);
			}
			infoRecorder->logTrace("\n");
		}
	}
#endif
#endif   // MULTI_CLIENTS
	HRESULT hr = D3D_OK;
	if(cmdCtrl->isRender()){
		hr = m_device->DrawPrimitiveUP(PrimitiveType, PrimitiveCount, pVertexStreamZeroData, VertexStreamZeroStride);
		#ifdef ENABLE_DEVICE_LOG
		if(FAILED(hr)){
			infoRecorder->logError("WrapperDirect3DDevice9::DrawPrimitiveUP failed with:%d.\n", hr);
		}
#endif
		return hr;
	}
	else
		return D3D_OK;
}

STDMETHODIMP WrapperDirect3DDevice9::DrawIndexedPrimitiveUP(THIS_ D3DPRIMITIVETYPE PrimitiveType,UINT MinVertexIndex,UINT NumVertices,UINT PrimitiveCount,CONST void* pIndexData,D3DFORMAT IndexDataFormat,CONST void* pVertexStreamZeroData,UINT VertexStreamZeroStride) {
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::DrawIndexedPrimitiveUP(), type=%d, count=%d\n", PrimitiveType, PrimitiveCount);
#endif
	int IndexSize = 2;
	if(IndexDataFormat == D3DFMT_INDEX32) IndexSize = 4;
#ifdef USE_HELPER_SYNC
	SynEntity * sync = new SynEntity();
	csSet->pushSync(sync);
#endif

#ifndef MULTI_CLIENTS
	cs.begin_command(DrawIndexedPrimitiveUP_Opcode, id);
	cs.write_uint(PrimitiveType);
	cs.write_uint(MinVertexIndex);
	cs.write_uint(NumVertices);
	cs.write_uint(PrimitiveCount);
	cs.write_uint(IndexDataFormat);
	cs.write_uint(VertexStreamZeroStride);
	cs.write_byte_arr((char*)pIndexData, NumVertices * IndexSize);
	cs.write_byte_arr((char*)pVertexStreamZeroData, NumVertices * VertexStreamZeroStride);
	cs.end_command();
#else
	csSet->beginCommand(DrawIndexedPrimitiveUP_Opcode, id);
	csSet->writeUInt(PrimitiveType);
	csSet->writeUInt(MinVertexIndex);
	csSet->writeUInt(NumVertices);
	csSet->writeUInt(PrimitiveCount);
	csSet->writeUInt(IndexDataFormat);
	csSet->writeUInt(VertexStreamZeroStride);
	csSet->writeByteArr((char *)pIndexData, NumVertices * IndexSize);
	csSet->writeByteArr((char *)pVertexStreamZeroData, NumVertices * VertexStreamZeroStride);
	csSet->endCommand();
#endif
	HRESULT hr = D3D_OK;
	if(cmdCtrl->isRender()){
		hr = m_device->DrawIndexedPrimitiveUP(PrimitiveType, MinVertexIndex, NumVertices, PrimitiveCount, pIndexData, IndexDataFormat, pVertexStreamZeroData, VertexStreamZeroStride);
		#ifdef ENABLE_DEVICE_LOG
		if(FAILED(hr)){
			infoRecorder->logError("WrapperDirect3DDevice9::DrawIndexedPrimitiveUP failed with:%d.\n", hr);
		}
#endif
		return hr;
	}
	else
		return D3D_OK;

}

STDMETHODIMP WrapperDirect3DDevice9::CreateVertexDeclaration(THIS_ CONST D3DVERTEXELEMENT9* pVertexElements,IDirect3DVertexDeclaration9** ppDecl) {
#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::CreateVertexDeclaration(), ");
#endif // ENABLE_DEVICE_LOG

	int ve_cnt = 0;
	D3DVERTEXELEMENT9 end = D3DDECL_END();

	while(true) {
		ve_cnt++;
		if(pVertexElements[ve_cnt].Type == end.Type && pVertexElements[ve_cnt].Method == end.Method && pVertexElements[ve_cnt].Offset == end.Offset && pVertexElements[ve_cnt].Stream == end.Stream && pVertexElements[ve_cnt].Usage == end.Usage && pVertexElements[ve_cnt].UsageIndex == end.UsageIndex) break;
	}
	// create the vertex declaration
	//WrapperDirect3DVertexDeclaration9::ins_count++;
	LPDIRECT3DVERTEXDECLARATION9 base_vd = NULL;
	HRESULT hr = m_device->CreateVertexDeclaration(pVertexElements, &base_vd);
	WrapperDirect3DVertexDeclaration9 * vd = NULL;
	if(SUCCEEDED(hr)) {
		vd = new WrapperDirect3DVertexDeclaration9(base_vd, WrapperDirect3DVertexDeclaration9::ins_count++);
		vd->setDeviceID(id);
		*ppDecl = dynamic_cast<IDirect3DVertexDeclaration9*>(vd);
		// store the vertex declaration
#ifdef INITIAL_ALL_RESOURCE
		Initializer::PushObj(vd);
#endif
	}
	else {
		#ifdef ENABLE_DEVICE_LOG
		infoRecorder->logTrace("failed\n");
#endif
		*ppDecl = NULL;
		return hr;
	}

#ifndef MULTI_CLIENTS
	cs.begin_command(CreateVertexDeclaration_Opcode, id);
	cs.write_int(vd->GetID());
	cs.write_int(ve_cnt + 1);
	cs.write_byte_arr((char*)pVertexElements, sizeof(D3DVERTEXELEMENT9) * (ve_cnt + 1));
	cs.end_command();
#else
	vd->setDeviceID(id);

	// send the command to all connected clients
	csSet->beginCommand(CreateVertexDeclaration_Opcode, id);
	csSet->writeInt(vd->getId());
	csSet->writeInt(ve_cnt + 1);
	csSet->writeByteArr((char *)pVertexElements, sizeof(D3DVERTEXELEMENT9) * (ve_cnt + 1));
	csSet->endCommand();
	csSet->setCreation(vd->creationFlag);

	infoRecorder->addCreation();

#endif
	if(vd){
		#ifdef ENABLE_DEVICE_LOG
		infoRecorder->logTrace("numElement:%d and id:%d\n",ve_cnt + 1, vd->GetID());
#endif
		//set the VERTEXELEMENT and numElemts
		vd->numElements = ve_cnt;
		//vd->pDecl = pVertexElements; // check when use, the pointer may be NULL
		vd->pDecl = (D3DVERTEXELEMENT9 *)malloc(sizeof(D3DVERTEXELEMENT9) * (ve_cnt +1));
		memcpy(vd->pDecl, (const void *)pVertexElements, sizeof(D3DVERTEXELEMENT9) * (ve_cnt +1));
		vd->print();
	}

	return hr;
}

STDMETHODIMP WrapperDirect3DDevice9::SetVertexDeclaration(THIS_ IDirect3DVertexDeclaration9* pDecl) {
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetVertexDeclaration(), ");
#endif
	WrapperDirect3DVertexDeclaration9 * decl = (WrapperDirect3DVertexDeclaration9 *)pDecl;
	cur_decl_ = (WrapperDirect3DVertexDeclaration9*)pDecl;

	if(pDecl == NULL) {
		#ifdef ENABLE_DEVICE_LOG
		infoRecorder->logTrace("pDecl is NULL\n");
#endif
#ifndef MULTI_CLIENTS
		cs.begin_command(SetVertexDeclaration_Opcode, id);
		cs.write_short(-1);
		cs.end_command();
#else
		csSet->beginCommand(SetVertexDeclaration_Opcode, id);
		csSet->writeShort(-1);
		csSet->endCommand();
		if(stateRecorder){
			stateRecorder->BeginCommand(SetVertexDeclaration_Opcode, id);
			stateRecorder->WriteShort(-1);
			stateRecorder->EndCommand();
		}
#endif
		return m_device->SetVertexDeclaration(pDecl);
	}

	HRESULT hh= m_device->SetVertexDeclaration(((WrapperDirect3DVertexDeclaration9*)pDecl)->GetVD9());

#ifndef MULTI_CLIENTS
	cs.begin_command(SetVertexDeclaration_Opcode, id);
	cs.write_short(((WrapperDirect3DVertexDeclaration9*)pDecl)->getId());
	cs.end_command();
#else
	// call the VertexDeclaration's check creation
	csSet->checkObj(dynamic_cast<IdentifierBase *>(decl));
	// send the set command to all clients
	csSet->beginCommand(SetVertexDeclaration_Opcode, id);
	csSet->writeShort(((WrapperDirect3DVertexDeclaration9 *)pDecl)->getId());
	csSet->endCommand();
	if(stateRecorder){
		stateRecorder->pushDependency(dynamic_cast<IdentifierBase *>(decl));
		stateRecorder->BeginCommand(SetVertexDeclaration_Opcode, id);
		stateRecorder->WriteShort(decl->getId());
		stateRecorder->EndCommand();
	}
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("with id:%d.\n", decl->getId());
#endif
#endif
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::SetStreamSource(THIS_ UINT StreamNumber,IDirect3DVertexBuffer9* pStreamData,UINT OffsetInBytes,UINT Stride) {
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetStreamSource(), StreamNumber:%d, vb:%d, OffsetInBytes:%d, stride:%d, ", StreamNumber, pStreamData, OffsetInBytes, Stride);
#endif
	assert(StreamNumber < Source_Count);
	if(pStreamData == NULL) {
		cur_vbs_[StreamNumber] = NULL;
#ifndef MULTI_CLIENTS
		cs.begin_command(SetStreamSource_Opcode, id);
		cs.write_uint(StreamNumber);
		cs.write_int(-1);
		cs.write_uint(0);
		cs.write_uint(0);
		cs.end_command();
#else
		csSet->beginCommand(SetStreamSource_Opcode, id);
		csSet->writeUInt(StreamNumber);
		csSet->writeInt(-1);
		csSet->writeUInt(0);
		csSet->writeUInt(0);
		csSet->endCommand();
		if(stateRecorder){
			stateRecorder->BeginCommand(SetStreamSource_Opcode, id);
			stateRecorder->WriteUInt(StreamNumber);
			stateRecorder->WriteInt(-1);
			stateRecorder->WriteUInt(0);
			stateRecorder->WriteUInt(0);
			stateRecorder->EndCommand();
		}
#endif
		#ifdef ENABLE_DEVICE_LOG
		infoRecorder->logTrace("pStreamData is NULL\n");
#endif
		return m_device->SetStreamSource(StreamNumber, pStreamData, OffsetInBytes, Stride);
	}

	WrapperDirect3DVertexBuffer9* wvb = (WrapperDirect3DVertexBuffer9*)pStreamData;
#ifdef MULTI_CLIENTS

#ifndef USE_MESH

	// TODO, not check creation directly????
	csSet->checkObj(dynamic_cast<IdentifierBase *>(wvb));
	// send the command
	csSet->beginCommand(SetStreamSource_Opcode, id);
	csSet->writeUInt(StreamNumber);
	csSet->writeInt(wvb->getId());
	csSet->writeUInt(OffsetInBytes);
	csSet->writeUInt(Stride);
	csSet->endCommand();
	if(stateRecorder){
		stateRecorder->pushDependency(wvb);
		stateRecorder->BeginCommand(SetStreamSource_Opcode, id);
		stateRecorder->WriteUInt(StreamNumber);
		stateRecorder->WriteInt(wvb->getId());
		stateRecorder->WriteUInt(OffsetInBytes);
		stateRecorder->WriteUInt(Stride);
		stateRecorder->EndCommand();
	}
#else
	cur_vbs_[StreamNumber] = wvb;
#endif  // USE_MESH

#endif
	//infoRecorder->logTrace("id:%d\n", wvb->GetId());
	
	wvb->streamNumber = StreamNumber;
	wvb->stride =Stride;
	wvb->offsetInBytes = OffsetInBytes;

	HRESULT hh = m_device->SetStreamSource(StreamNumber, ((WrapperDirect3DVertexBuffer9*)pStreamData)->GetVB9(), OffsetInBytes, Stride);
	return hh;
}

STDMETHODIMP WrapperDirect3DDevice9::SetIndices(THIS_ IDirect3DIndexBuffer9* pIndexData) {
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("WrapperDirect3DDevice9::SetIndices(), ");
#endif

	if(pIndexData == NULL) {
		#ifdef ENABLE_DEVICE_LOG

		infoRecorder->logTrace("pIndexData is NULL\n");
#endif

		cur_ib_ = NULL;
#ifndef MULTI_CLIENTS
		cs.begin_command(SetIndices_Opcode, id);
		cs.write_short(-1);
		cs.end_command();
#else
		csSet->beginCommand(SetIndices_Opcode, id);
		csSet->writeShort(-1);
		csSet->endCommand();

		if(stateRecorder){
			stateRecorder->BeginCommand(SetIndices_Opcode, id);
			stateRecorder->WriteShort(-1);
			stateRecorder->EndCommand();
		}
#endif
		return m_device->SetIndices(pIndexData);
	}

	WrapperDirect3DIndexBuffer9* wib = (WrapperDirect3DIndexBuffer9*)pIndexData;
	#ifdef ENABLE_DEVICE_LOG
	infoRecorder->logTrace("index buffer id:%d\n", wib->getId());
#endif
#ifdef MULTI_CLIENTS
	// send the index
#ifndef USE_MESH

	csSet->checkObj(dynamic_cast<IdentifierBase *>(wib));
	csSet->beginCommand(SetIndices_Opcode, id);
	csSet->writeShort(wib->getId());
	csSet->endCommand();

	if(stateRecorder){
		stateRecorder->pushDependency(wib);
		stateRecorder->BeginCommand(SetIndices_Opcode, id);
		stateRecorder->WriteShort(wib->getId());
		stateRecorder->EndCommand();
	}
#else   // USE_MESH
	cur_ib_ = wib;

#endif   // USE_MESH
	
#endif // MULTI_CLIENTS

	HRESULT hh = m_device->SetIndices(((WrapperDirect3DIndexBuffer9*)pIndexData)->GetIB9());
	return hh;
}
