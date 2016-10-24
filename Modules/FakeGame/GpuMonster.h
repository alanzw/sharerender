#ifndef __GPUMONSTER_H__
#define __GPUMONSTER_H__
//this is for the GPU monster that eats the GPU to given usage

#include "ResourceMonster.h"
#include <Windows.h>
#include <d3d9.h>
#include <d3dx9.h>

class GpuMonster: public ResourceMonster{
	// get a resource monitor

	HWND hWnd;

	// D3D option
	bool enableLight;  // default is false;

	// for D3D Box
	int boxCount;  // hope the count of entity can effect the usage
	

	// D3D context
	LPDIRECT3D9 g_pD3D;
	LPDIRECT3DDEVICE9 g_pd3dDevice;
	LPDIRECT3DVERTEXBUFFER9 g_pVertexBuff;
	LPDIRECT3DINDEXBUFFER9 g_pIndexBuff;
	LPDIRECT3DTEXTURE9 g_pTexture;

	// transform
	float g_RotateAngleY;
	float g_TranslationX;
	float g_TranslationY;
	float g_TranslationZ;

	

	// constructor 
	GpuMonster(){
		g_pD3D = NULL;
		g_pd3dDevice = NULL;
		g_pVertexBuff = NULL;
		g_pIndexBuff = NULL;
		g_pTexture = NULL;
	}
	static GpuMonster * monster;
public:
	static GpuMonster * GetMonster(){ 
		if(!monster){
			monster = new GpuMonster();
		}
		return monster;
	}


	// member function
	BOOL InitInstance(HINSTANCE hInstance, int nCmdShow);
	LRESULT InitD3D(HWND hWnd);
	LRESULT InitMatrices(int move_x, int move_y,int move_z);
	LRESULT CreateVertexBuff();
	LRESULT CreateIndexBuff();
	LRESULT CreateTexture();
	void SetMaterial();
	void SetLight();
	void RenderScene();
	void TransformScene(WPARAM wParam, LPARAM lParam);  // transform
	void Cleanup();

};

struct D3D_VERTEX{
	FLOAT x,y,z;
	D3DVECTOR normal;
	FLOAT u,v;
};

LRESULT CALLBACK WinProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam );

#endif