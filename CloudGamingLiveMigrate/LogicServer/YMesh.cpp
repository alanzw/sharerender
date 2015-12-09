#include "YMesh.h"

// for the Ymesh

std::vector<YMesh *> YMesh::meshList;

YMesh::YMesh(WrapperDirect3DDevice9 * device){
	for(int i = 0; i < MAX_SOURCE_COUNT; ++i){
		vbs_[i] = NULL;
	
	}

	leaderVBId = -1;
	device_ = device;
	decl_ = NULL;
	ib_ = NULL;
	curPrimCount_ = 0;
	memset(selectResultSend_, 0, sizeof selectResultSend_);
	meshList.push_back(this);
	curFrame_ = 1;
}

bool YMesh::IsVBChanged(WrapperDirect3DVertexBuffer9 * curVbs[]){
	bool changed =  false;
	for(int i = 0; i < MAX_SOURCE_COUNT; i++){
		if(vbs_[i] != curVbs[i]){
			changed = true;
			break;
		}
	}
	if(changed){
		infoRecorder->logTrace("[YMesh]: Vertex buffer changed, ib id:%d.\n", ib_->GetID());
	}
	return changed;
}

void YMesh::SetIndexParams(D3DPRIMITIVETYPE type, INT baseVertexIndex, UINT minVertexIndewx, UINT numVertices, UINT startIndex, UINT primCount){
	type_ = type;
	baseVertexIndex_ = baseVertexIndex;
	minVertexIndex_ = minVertexIndewx;
	numVertices_ = numVertices;
	startIndex_ = startIndex;
	primCount_ = primCount;
}

void YMesh::ClearStreamSource(){
	for(int i = 0; i , MAX_SOURCE_COUNT; i++){
		vbs_[i] = NULL;
	}
}

void YMesh::SetStreamSource(WrapperDirect3DVertexBuffer9 * sources[]){
	ClearStreamSource();
	for(int i = 0; i < MAX_SOURCE_COUNT; i++){
		if(sources[i] == NULL) continue;
		vbs_[i] = sources[i];
	}
}

void YMesh::SetDeclaration(WrapperDirect3DVertexDeclaration9 * decl){
	decl_ = decl;
	assert(decl);

}

void YMesh::UpdateIndexBuffer(WrapperDirect3DIndexBuffer9 * ib){
	// update the give index buffer

}

void YMesh::UpdateVertexBuffer(int streamNumber){
	// update the give stream
	WrapperDirect3DVertexBuffer9 * vb = vbs_[streamNumber];
	if(vb->isFirst){
		memset(vb->cache_buffer, 0, vb->Length);
		vb->isFirst = false;
	}
	
	// finally, send the SetStreamSource cmd to client
	csSet->beginCommand(SetStreamSource_Opcode, device_->GetID());
	csSet->writeUInt(vb->streamNumber);
	csSet->writeInt(vb->GetId());
	csSet->writeUInt(vb->offsetInBytes);
	csSet->writeUInt(vb->stride);
	csSet->endCommand();
}

// render this mesh
void YMesh::Render(INT baseVertexIndex, UINT minVertexIndex, int offset, int primCount){
	infoRecorder->logTrace("[YMesh]: render.\n");

	for(int i = 0; i < MAX_SOURCE_COUNT; i++){
		if(vbs_[i] == NULL)
			continue;
		UpdateVertexBuffer(i);
	}

	// if use indexed primitives, update the index buffer
	if(drawMethod_ == DRAWINDEXEDPRIMITIVE){
		UpdateIndexBuffer(ib_);
	}
}