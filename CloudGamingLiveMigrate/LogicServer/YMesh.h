#ifndef __YMESH_H__
#define __YMESH_H__
// define the mesh to store the data to draw
#include "WrapDirect3ddevice9.h"
#include "WrapDirect3dindexbuffer9.h"
#include "WrapDirect3dvertexbuffer9.h"
#include "WrapDirect3dvertexdeclaration9.h"
#include "../LibCore/BitSet.h"
#include <vector>


//#define USE_MESH

enum DRAWMETHOD{
	DRAWPRIMITIVE = 0,
	DRAWINDEXEDPRIMITIVE,
	NONE
};

//class WrapperDirect3DDevice9;

class YMesh{
public:
	YMesh(WrapperDirect3DDevice9 * device);

	void ClearStreamSource();
	void SetStreamSource(WrapperDirect3DVertexBuffer9 * sources[]);
	void SetDeclaration(WrapperDirect3DVertexDeclaration9 * decl);
	void SetIndices(WrapperDirect3DIndexBuffer9 * ib){ ib_ = ib;}
	void SetDrawType(DRAWMETHOD method, D3DPRIMITIVETYPE type){ drawMethod_ = method; primitiveType_ = type; }

	void SetIndexParams(D3DPRIMITIVETYPE type, INT baseVertexIndex, UINT minVertexIndewx, UINT numVertices, UINT startIndex, UINT primCount);
	bool IsVBChanged(WrapperDirect3DVertexBuffer9 * curVbs[]);

	void UpdateIndexBuffer(WrapperDirect3DIndexBuffer9 * ib);
	void UpdateVertexBuffer(int streamNumber);
	void Render(INT baseVertexIndex, UINT minVertexIndex, int offset, int primCount);

	void SetLeaderVB(int id){ leaderVBId = id; }

	D3DPRIMITIVETYPE type_;
	INT baseVertexIndex_;
	UINT minVertexIndex_;
	UINT numVertices_;
	UINT startIndex_;
	UINT primCount_;
private:

	WrapperDirect3DDevice9 * device_;
	WrapperDirect3DVertexDeclaration9 * decl_;
	WrapperDirect3DIndexBuffer9 * ib_;
	WrapperDirect3DVertexBuffer9 * vbs_[MAX_SOURCE_COUNT];


	D3DPRIMITIVETYPE primitiveType_;
	DRAWMETHOD drawMethod_;

	int selectResultSend_[MAX_SOURCE_COUNT];
	int leaderVBId;

	static std::vector<YMesh *> meshList;

	int curPrimCount_;
	int curFrame_;
};



#endif // __YMESH_H__