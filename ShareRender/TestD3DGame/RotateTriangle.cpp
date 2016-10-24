#include "EncoderThread.h"

#pragma comment(lib,"winmm.lib")
#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"d3dx9.lib")

#include <d3d9.h>
#include <d3dx9.h>
#include <strsafe.h>

#pragma warning(disable:4996)

//global variables.
LPDIRECT3D9 g_pD3D=NULL;
LPDIRECT3DDEVICE9 g_pD3DDevice9=NULL;
LPDIRECT3DVERTEXBUFFER9 g_pD3DVB=NULL;
LPDIRECT3DVERTEXBUFFER9 g_pD3DVBQuad = NULL;   // to render to screen

LPDIRECT3DSURFACE9 pRenderSurface = NULL, pBackBuffer = NULL;   // the render surface

LPDIRECT3DTEXTURE9 pTexture = NULL;

int gWidth = 800;
int gHeight = 600;

//custom vertex struct.
struct CUSTOMVERTEX
{
	FLOAT x,y,z;
	DWORD color;
};

struct TEX_VERTEX{
	FLOAT x, y, z;
	FLOAT u, v;

	TEX_VERTEX(FLOAT _x, FLOAT _y, FLOAT _z, FLOAT _u, FLOAT _v): x(_x), y(_y), z(_z), u(_u), v(_v){}
};

//define custom vertex format.
#define D3DFVF_CUSTOMVETEX (D3DFVF_XYZ|D3DFVF_DIFFUSE)

#define D3DFVF_TEXVERTEX (D3DFVF_XYZ | D3DFVF_TEX1)

//init D3D variables
HRESULT InitD3D(HWND hWnd)
{
	//init D3D
	g_pD3D=Direct3DCreate9(D3D_SDK_VERSION);
	if(FAILED(g_pD3D))
	{
		return E_FAIL;
	}

	//init D3D present parameter.
	D3DPRESENT_PARAMETERS d3dpp;
	ZeroMemory(&d3dpp,sizeof(d3dpp));
	d3dpp.Windowed=TRUE;
	d3dpp.SwapEffect=D3DSWAPEFFECT_DISCARD;
	d3dpp.BackBufferFormat = D3DFMT_A8R8G8B8;//D3DFMT_UNKNOWN;
	d3dpp.BackBufferCount = 3;
	d3dpp.hDeviceWindow = hWnd;

	//create d3d device.
	if(FAILED(g_pD3D->CreateDevice(
		D3DADAPTER_DEFAULT,
		D3DDEVTYPE_HAL,
		hWnd,
		D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_MULTITHREADED,    // set the device creation flag to MULTI_THREADED
		&d3dpp,
		&g_pD3DDevice9)))
	{
		return E_FAIL;
	}
	

	onDeviceCreation(g_pD3DDevice9,d3dpp.BackBufferHeight, d3dpp.BackBufferWidth, d3dpp.hDeviceWindow);
	//turn off culling.
	g_pD3DDevice9->SetRenderState(D3DRS_CULLMODE,D3DCULL_NONE);

	//turn off light.
	g_pD3DDevice9->SetRenderState(D3DRS_LIGHTING,FALSE);

	return S_OK;
}

//init geometry
HRESULT InitGeometry()
{
	//initialize three custom vertex.
	CUSTOMVERTEX g_vertices[3]=
	{
		{-1.0f,-1.0f, 0.0f, D3DCOLOR_XRGB(0,255,0)},
		{0.0f, 1.0f, 0.0f, D3DCOLOR_XRGB(255,0,0)},
		{1.0f,-1.0f, 0.0f,  D3DCOLOR_XRGB(0,0,255)}
	};
	TEX_VERTEX *g_quad = NULL;// (TEX_VERTEX **)malloc(sizeof(TEX_VERTEX*) * 6);; 

	

	//create vertex buffer.
	if(FAILED(g_pD3DDevice9->CreateVertexBuffer(
		sizeof(g_vertices),
		0,
		D3DFVF_CUSTOMVETEX,
		D3DPOOL_DEFAULT,
		&g_pD3DVB,
		NULL)))
	{
		return E_FAIL;
	}

	// create the plane vertex buffer
	if(FAILED(g_pD3DDevice9->CreateVertexBuffer(
		sizeof(TEX_VERTEX) * 6,
		0,
		D3DFVF_CUSTOMVETEX,
		D3DPOOL_DEFAULT,
		&g_pD3DVBQuad,
		NULL))){
			return E_FAIL;
	}

	//fill vertex buffer.
	void* pVertices=NULL;
	if(FAILED(g_pD3DVB->Lock(0,sizeof(g_vertices),&pVertices,0)))
	{
		return E_FAIL;
	}
	memcpy(pVertices,g_vertices,sizeof(g_vertices));
	g_pD3DVB->Unlock();

	if(FAILED(g_pD3DVBQuad->Lock(0, 0, (void **)&g_quad, 0))){
		return E_FAIL;
	}

	g_quad[0] = TEX_VERTEX(-1*gWidth/2, -1*gHeight/2, 0.0f, 0.0f, 0.0f);
	g_quad[1] = TEX_VERTEX(-1*gWidth/2,    gHeight/2, 0.0f, 0.0f, 1.0f);
	g_quad[2] = TEX_VERTEX(   gWidth /2 ,  gHeight/2, 0.0f, 1.0f, 1.0f);

	g_quad[3] = TEX_VERTEX(-1*gWidth/2, -1*gHeight/2, 0.0f, 0.0f, 0.0f);
	g_quad[4] = TEX_VERTEX(   gWidth/2,    gHeight/2, 0.0f, 1.0f, 1.0f);
	g_quad[5] = TEX_VERTEX(   gWidth/2, -1*gHeight/2, 0.0f, 1.0f, 0.0f);

	//memcpy(pVertices, g_quad, sizeof(TEX_VERTEX) * 6);
	g_pD3DVBQuad->Unlock();

	//create the render surface
	HRESULT hr = g_pD3DDevice9->CreateRenderTarget(gWidth, gHeight, D3DFORMAT::D3DFMT_A8R8G8B8, D3DMULTISAMPLE_NONE, 0, TRUE, &pRenderSurface, NULL);

	if(FAILED(hr)){
		char error[100] = {0};
		switch(hr){
		case D3DERR_NOTAVAILABLE:
			sprintf(error, "create render target failed. %s", "D3DERR_NOTAVAILABLE");
			break;
		case D3DERR_INVALIDCALL:
			sprintf(error, "create render target failed. %s", "D3DERR_INVALIDCALL");
			break;
		case D3DERR_OUTOFVIDEOMEMORY:
			sprintf(error, "create render target failed. %s", "D3DERR_OUTOFVIDEOMEMORY");
			break;
		case E_OUTOFMEMORY:
			sprintf(error, "create render target failed. %s", "E_OUTOFMEMORY");
			break;
		}
		MessageBox(NULL, error, "ERROR", MB_OK);
	}

	// create the texture
	hr = g_pD3DDevice9->CreateTexture(gWidth, gHeight, 1, D3DUSAGE_RENDERTARGET, D3DFORMAT::D3DFMT_A8R8G8B8, D3DPOOL::D3DPOOL_DEFAULT, &pTexture, NULL);
	if(FAILED(hr)){
		return E_FAIL;
	}


	return S_OK;
}

//clean up d3d variables.
void CleanUp()
{
	if(g_pD3DVB!=NULL)
	{
		g_pD3DVB->Release();
	}

	if(g_pD3DDevice9!=NULL)
	{
		g_pD3DDevice9->Release();
	}

	if(g_pD3D!=NULL)
	{
		g_pD3D->Release();
	}
}

//setup matrix
void SetupMatrix()
{
	//world matrix.
	D3DXMATRIXA16 matWorld;

	//rotation matrix.
	UINT itimes=timeGetTime()%1000;
	FLOAT fAngle=itimes * ( 2.0f * D3DX_PI ) / 1000.0f;
	D3DXMatrixRotationY(&matWorld,fAngle);

	//set world matrix.
	g_pD3DDevice9->SetTransform(D3DTS_WORLD,&matWorld);

	//set view point.
	D3DXVECTOR3 vEyePt(0.0f,3.0f,-5.0f);
	D3DXVECTOR3 vLookAt(0.0f,0.0f,0.0f);
	D3DXVECTOR3 vUp(0.0f,1.0f,0.0f);

	//view matrix.
	D3DXMATRIXA16 matView;

	//set view matrix.
	D3DXMatrixLookAtLH(&matView,&vEyePt,&vLookAt,&vUp);
	g_pD3DDevice9->SetTransform(D3DTS_VIEW,&matView);

	//set projection matrix.
	D3DXMATRIXA16 matProj;
	D3DXMatrixPerspectiveFovLH(&matProj,D3DX_PI/4,1.0f,1.0f,100.0f);
	g_pD3DDevice9->SetTransform(D3DTS_PROJECTION,&matProj);
}

//render the scene.
void Render_bak()
{
	//clear target device.
	//g_pD3DDevice9->Clear(0,NULL,D3DCLEAR_TARGET,D3DCOLOR_XRGB(0,0,0),1.0f,0);

	// clear the surface
	D3DLOCKED_RECT rect;
	pRenderSurface->LockRect(&rect, NULL, D3DLOCK_DISCARD);
	
	memset(rect.pBits, 0, rect.Pitch * gHeight);
	pRenderSurface->UnlockRect();

	g_pD3DDevice9->GetRenderTarget(0, &pBackBuffer);

	g_pD3DDevice9->SetRenderTarget(0, pRenderSurface);
	g_pD3DDevice9->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DXCOLOR(0.0f, 0.0f, 0.0f, 1.0f), 1.0f, 0);

	//g_pD3DDevice9->BeginScene();
	
	

	//draw primitive.
	if(SUCCEEDED(g_pD3DDevice9->BeginScene()))
	{
		SetupMatrix();

		g_pD3DDevice9->SetStreamSource(0,g_pD3DVB,0,sizeof(CUSTOMVERTEX));
		g_pD3DDevice9->SetFVF(D3DFVF_CUSTOMVETEX);
		g_pD3DDevice9->DrawPrimitive(D3DPT_TRIANGLELIST,0,1);

		g_pD3DDevice9->EndScene();
		// save the rendered surface to file
		static int index= 0;
		char name[100]= {0};
		sprintf(name, "rendered-%d.jpg", index++);
		D3DXSaveSurfaceToFile(name, D3DXIMAGE_FILEFORMAT::D3DXIFF_JPG, pRenderSurface, NULL, NULL);

	}
	// render to back buffer
	g_pD3DDevice9->SetRenderTarget(0, pBackBuffer);
	g_pD3DDevice9->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(0,0,0), 1.0f, 0);


	IDirect3DSurface9 * mip = NULL;
	pTexture->GetSurfaceLevel(0, &mip);
	if(FAILED(g_pD3DDevice9->StretchRect(pRenderSurface, NULL, mip, NULL, D3DTEXTUREFILTERTYPE::D3DTEXF_NONE))){
		MessageBox(NULL, "copy surface failed.", "ERROR", MB_OK);
	}

	//draw primitive.
	if(SUCCEEDED(g_pD3DDevice9->BeginScene()))
	{
		
		g_pD3DDevice9->SetTexture(0, pTexture);
		g_pD3DDevice9->SetStreamSource(0,g_pD3DVBQuad,0,sizeof(TEX_VERTEX));
		g_pD3DDevice9->SetFVF(D3DFVF_TEXVERTEX);
		g_pD3DDevice9->DrawPrimitive(D3DPT_TRIANGLELIST,0,2);

		g_pD3DDevice9->EndScene();
		// save the rendered surface to file
		
	}
	
	//present back buffer to display.
	g_pD3DDevice9->Present(NULL,NULL,NULL,NULL);
}

void Render(){
	
	//clear target device.
	g_pD3DDevice9->Clear(0,NULL,D3DCLEAR_TARGET,D3DCOLOR_XRGB(0,0,0),1.0f,0);

	//draw primitive.
	if(SUCCEEDED(g_pD3DDevice9->BeginScene()))
	{
		SetupMatrix();

		g_pD3DDevice9->SetStreamSource(0,g_pD3DVB,0,sizeof(CUSTOMVERTEX));
		g_pD3DDevice9->SetFVF(D3DFVF_CUSTOMVETEX);
		g_pD3DDevice9->DrawPrimitive(D3DPT_TRIANGLELIST,0,1);

		g_pD3DDevice9->EndScene();
	}

	//present back buffer to display.
	g_pD3DDevice9->Present(NULL,NULL,NULL,NULL);
	onPresent();

	Sleep(30);
}

//message loop handler.
LRESULT WINAPI MsgProc(HWND hWnd,UINT msg,WPARAM wParam,LPARAM lParam)
{
	switch(msg)
	{
	case WM_DESTROY:
		CleanUp();
		PostQuitMessage(0);
		return 0;
	}

	return DefWindowProc(hWnd,msg,wParam,lParam);
}

//the application entry point.
INT WINAPI wWinMain(HINSTANCE,HINSTANCE,LPWSTR,INT)
{
	//windclass structure.
	WNDCLASSEX wcex;

	wcex.cbClsExtra=0;
	wcex.cbSize=sizeof(WNDCLASSEX);
	wcex.cbWndExtra=0;
	wcex.hbrBackground=NULL;
	wcex.hCursor=NULL;
	wcex.hIcon=NULL;
	wcex.hIconSm=NULL;
	wcex.hInstance=GetModuleHandle(NULL);
	wcex.lpfnWndProc=MsgProc;
	wcex.lpszClassName="D3D Toturial";
	wcex.lpszMenuName=NULL;
	wcex.style=CS_CLASSDC;

	//register window class.
	RegisterClassEx(&wcex);

	//create window.
	HWND hWnd=CreateWindow(
		"D3D Toturial",
		"D3D Toturial 003",
		WS_POPUP,
		100,
		100,
		gWidth,
		gHeight,
		NULL,
		NULL,
		wcex.hInstance,
		NULL);

	//init d3d.
	if(SUCCEEDED(InitD3D(hWnd)))
	{
		if(SUCCEEDED(InitGeometry()))
		{
			//show window.
			ShowWindow(hWnd,SW_SHOWDEFAULT);
			UpdateWindow(hWnd);

			// Enter the message loop
            MSG msg;
            ZeroMemory( &msg, sizeof( msg ) );
            while( msg.message != WM_QUIT )
            {
                if( PeekMessage( &msg, NULL, 0U, 0U, PM_REMOVE ) )
                {
                    TranslateMessage( &msg );
                    DispatchMessage( &msg );
                }
                else
                    Render();
            }
		}
	}

	//unregister window class.
	UnregisterClass( "D3D Tutorial", wcex.hInstance );
    return 0;
}