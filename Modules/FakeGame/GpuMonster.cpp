#include "GpuMonster.h"

//------------------------------------------------------------------------
// Comment: render a cube with material and texture.
// Created: atyuwen.
// Date: 2008.7.18
//------------------------------------------------------------------------

#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"d3dx9.lib")

GpuMonster * GpuMonster::monster;

//------------------------------------------------------------------------
// D3D vertex format
//------------------------------------------------------------------------
const DWORD VertexFomat = D3DFVF_XYZ | D3DFVF_NORMAL | D3DFVF_TEX1;

//------------------------------------------------------------------------
// create a window instance
//------------------------------------------------------------------------
BOOL GpuMonster::InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	//HWND hWnd; // handle for the window
	WNDCLASS wc; // window class 

	// prepare the window class
	wc.style = CS_VREDRAW | CS_HREDRAW;
	wc.lpfnWndProc = (WNDPROC)WinProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance;
	wc.hIcon = LoadIcon( hInstance, IDI_APPLICATION );
	wc.hCursor = LoadCursor( NULL, IDC_ARROW );
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "MyWindow";
	// register the window class
	RegisterClass( &wc );
	// create the window
	hWnd = CreateWindow("MyWindow",
		"D3D Sample",
		WS_OVERLAPPEDWINDOW,
		100, 100,
		800, 600,
		NULL, NULL,
		hInstance, NULL);
	if(!hWnd)
		return FALSE;
	if(!SUCCEEDED(InitD3D(hWnd))) // init D3D 
		return FALSE;
	CreateVertexBuff(); // create the vertex buffer
	CreateIndexBuff(); // create index buffer
	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);
	return TRUE;
}

//------------------------------------------------------------------------
// 回调函数
//------------------------------------------------------------------------
LRESULT CALLBACK WinProc( HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam )
{
	GpuMonster * monster = GpuMonster::GetMonster();
	switch (message)
	{
	case WM_KEYDOWN: // for keyboard event
		monster->TransformScene(wParam, lParam);
		break;
	case WM_DESTROY: // for exit
		monster->Cleanup();
		PostQuitMessage(0);
		break;
	default:;
	}
	// default window proc
	return DefWindowProc(hWnd, message, wParam, lParam);
}

//------------------------------------------------------------------------
// DirectX初始化函数
//------------------------------------------------------------------------
LRESULT GpuMonster::InitD3D(HWND hWnd)
{
	if( NULL == ( g_pD3D = Direct3DCreate9( D3D_SDK_VERSION ) ) )
		return E_FAIL;
	// init the present parameters
	D3DPRESENT_PARAMETERS d3dpp;
	ZeroMemory( &d3dpp, sizeof(d3dpp) );
	d3dpp.Windowed = TRUE;
	d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;
	d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;
	// set multisample for full screen if supported
	if( SUCCEEDED(g_pD3D->CheckDeviceMultiSampleType( D3DADAPTER_DEFAULT,
		D3DDEVTYPE_HAL , D3DFMT_X8R8G8B8, FALSE,
		D3DMULTISAMPLE_8_SAMPLES, NULL ) ) )
	{
		d3dpp.MultiSampleType = D3DMULTISAMPLE_8_SAMPLES;
	}

	// create d3d device with given window
	if( FAILED( g_pD3D->CreateDevice( D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd,
		D3DCREATE_SOFTWARE_VERTEXPROCESSING,
		&d3dpp, &g_pd3dDevice ) ) )
	{
		return E_FAIL;
	}

	// 
	g_pd3dDevice->SetRenderState( D3DRS_MULTISAMPLEANTIALIAS, TRUE);
	// set material
	SetMaterial();
	// enable light
	SetLight();
	// set texture
	CreateTexture();
	return S_OK;
}

//------------------------------------------------------------------------
// DirectX render the image
//------------------------------------------------------------------------
void GpuMonster::RenderScene()
{
	if( NULL == g_pd3dDevice )
		return;
	// clear to nothing
	g_pd3dDevice->Clear( 0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0,0,0), 1.0f, 0 );
	// begin to draw
	if( SUCCEEDED( g_pd3dDevice->BeginScene()))
	{
		for(int i = -5; i < 5; i++){
			// Rendering of scene objects can happen here
			for(int j = -5; j < 5; j++){
				for(int k = -5; k < 5; k++){
					InitMatrices(i * 10, j * 10, k * 10);
					g_pd3dDevice->SetStreamSource( 0, g_pVertexBuff, 0, sizeof(D3D_VERTEX) );
					g_pd3dDevice->SetFVF( VertexFomat);
					g_pd3dDevice->SetIndices(g_pIndexBuff);
					g_pd3dDevice->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, 8, 0, 12);
				}
			}
		}
		// End the scene
		g_pd3dDevice->EndScene();
	}
	// flush to screen
	g_pd3dDevice->Present( NULL, NULL, NULL, NULL );
}
//------------------------------------------------------------------------
// create the vertex buffer and fill data
//------------------------------------------------------------------------
LRESULT GpuMonster::CreateVertexBuff()
{
	D3D_VERTEX vertices[] =    // 8 points for a box
	{
		{ 0.0f, 0.0f, 0.0f, { -1.0f, -1.0f, -1.0f}, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, { 1.0f, -1.0f, -1.0f}, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 1.0f, { 1.0f, -1.0f, 1.0f}, 1.0f, 1.0f },
		{ 0.0f, 0.0f, 1.0f, { -1.0f, -1.0f, 1.0f}, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 0.0f, { -1.0f, 1.0f, -1.0f}, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, { 1.0f, 1.0f, -1.0f}, 1.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, { 1.0f, 1.0f, 1.0f}, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 1.0f, { -1.0f, 1.0f, 1.0f}, 0.0f, 0.0f },
	};
	if( FAILED( g_pd3dDevice->CreateVertexBuffer( 8*sizeof(D3D_VERTEX),
		0 ,VertexFomat, D3DPOOL_DEFAULT, &g_pVertexBuff, NULL ) ) )
		return E_FAIL;
	VOID* pVertices;           // lock the buffer
	if( FAILED( g_pVertexBuff->Lock( 0, sizeof(vertices), (void**)&pVertices, 0 ) ) )
		return E_FAIL;
	memcpy( pVertices, vertices, sizeof(vertices) ); // write data to buffer
	g_pVertexBuff->Unlock();
	return S_OK;
}

//------------------------------------------------------------------------
// create index buffer and fill the data
//------------------------------------------------------------------------
LRESULT GpuMonster::CreateIndexBuff()
{
	// for a cube, 6 faces, 12 triangles
	WORD indexBuff[] = { 0, 1, 2, 0, 2, 3,
		0, 4, 5, 0, 5, 1,
		4, 7, 6, 4, 6, 5,
		7, 3, 2, 7, 2, 6,
		4, 0, 3, 4, 3, 7,
		6, 2, 1, 6, 1, 5};
	if( FAILED( g_pd3dDevice->CreateIndexBuffer( 36*sizeof(WORD),
		0, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &g_pIndexBuff, NULL )))
		return E_FAIL;
	VOID * pIndices;
	if( FAILED( g_pIndexBuff->Lock( 0, sizeof(indexBuff), (void**)&pIndices, 0)))
		return E_FAIL;

	memcpy( pIndices, indexBuff, sizeof(indexBuff));
	g_pIndexBuff->Unlock();
	return S_OK;
}

//------------------------------------------------------------------------
// init the matrix
//------------------------------------------------------------------------
LRESULT GpuMonster::InitMatrices(int move_x, int move_y,int move_z)
{
	// world matrix
	D3DXMATRIXA16 matWorld, matRotation, matMove;
	D3DXMatrixTranslation( &matWorld, g_TranslationX + move_x/10 , g_TranslationY + move_y / 10, g_TranslationZ + move_z / 10);
	D3DXMatrixRotationY(&matRotation, g_RotateAngleY);
	
	D3DXMatrixMultiply(&matWorld, &matWorld, &matRotation);

	g_pd3dDevice->SetTransform( D3DTS_WORLD, &matWorld );
	// view matrix
	D3DXVECTOR3 vEyePt( 0.0f, 3.0f,-5.0f );
	D3DXVECTOR3 vLookatPt( 0.0f, 0.0f, 0.0f );
	D3DXVECTOR3 vUpVec( 0.0f, 1.0f, 0.0f );
	D3DXMATRIXA16 matView;
	D3DXMatrixLookAtLH( &matView, &vEyePt, &vLookatPt, &vUpVec );
	g_pd3dDevice->SetTransform( D3DTS_VIEW, &matView );
	// perspective matrix
	D3DXMATRIXA16 matProj;
	D3DXMatrixPerspectiveFovLH( &matProj, D3DX_PI/4, 1.0f, 1.0f, 100.0f );
	g_pd3dDevice->SetTransform( D3DTS_PROJECTION, &matProj );
	return S_OK;
}

//------------------------------------------------------------------------
// tranform the scene
//------------------------------------------------------------------------
void GpuMonster::TransformScene(WPARAM wParam, LPARAM lParam)
{
	switch (wParam)
	{
	case VK_SPACE:   // Rotate
		g_RotateAngleY += 2 * D3DX_PI/60.0f;
		break;
	case VK_UP:      // Zoom out
		g_TranslationZ += 0.1f;
		break;
	case VK_DOWN:    // Zoom in
		g_TranslationZ -= 0.1f;
		break;
	default: break;
	}
}
//------------------------------------------------------------------------
// set material 
//------------------------------------------------------------------------
void GpuMonster::SetMaterial()
{
	D3DMATERIAL9 material;
	ZeroMemory(&material, sizeof(D3DMATERIAL9));
	material.Diffuse.r = 0.8f;
	material.Diffuse.g = 0.8f;
	material.Diffuse.b = 0.8f;
	g_pd3dDevice->SetMaterial(&material);
}
//------------------------------------------------------------------------
// set lights
//------------------------------------------------------------------------
void GpuMonster::SetLight()
{
	D3DXVECTOR3 lightDir( 0, 0, 1);
	D3DLIGHT9 light;
	ZeroMemory(&light, sizeof(light));
	light.Type = D3DLIGHT_DIRECTIONAL;
	light.Ambient.r = 0.1f;
	light.Ambient.g = 0.1f;
	light.Ambient.b = 0.1f;
	light.Diffuse.r = 0.8f;
	light.Diffuse.g = 0.8f;
	light.Diffuse.b = 0.8f;
	D3DXVec3Normalize((D3DXVECTOR3*)&light.Direction, &lightDir);

	g_pd3dDevice->SetLight( 0, &light);
	g_pd3dDevice->LightEnable( 0, TRUE);
	g_pd3dDevice->SetRenderState(D3DRS_LIGHTING, TRUE);
}
//------------------------------------------------------------------------
// create the texture from disk
//------------------------------------------------------------------------
LRESULT GpuMonster::CreateTexture()
{
	if( !SUCCEEDED( D3DXCreateTextureFromFile( g_pd3dDevice, "cube.jpg" , &g_pTexture ) ) )
	{
		return E_FAIL;
	}

	g_pd3dDevice->SetTexture( 0, g_pTexture ); // set texture to render pipe
	// 
	g_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLOROP,   D3DTOP_MODULATE );
	g_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG1, D3DTA_TEXTURE );
	g_pd3dDevice->SetTextureStageState( 0, D3DTSS_COLORARG2, D3DTA_DIFFUSE );
	g_pd3dDevice->SetTextureStageState( 0, D3DTSS_ALPHAOP,   D3DTOP_DISABLE );
	g_pd3dDevice->SetSamplerState( 0, D3DSAMP_MAXMIPLEVEL, 4);
	return S_OK;
}
//------------------------------------------------------------------------
// release resources
//------------------------------------------------------------------------
void GpuMonster::Cleanup()
{
	if( g_pTexture != NULL)
		g_pTexture->Release();

	if( g_pIndexBuff != NULL)
		g_pIndexBuff->Release();

	if( g_pVertexBuff != NULL)
		g_pVertexBuff->Release();
	if( g_pd3dDevice != NULL)
		g_pd3dDevice->Release();
	if( g_pD3D != NULL)
		g_pD3D->Release();
}