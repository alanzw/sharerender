#include <d3d9.h>
#include < d3dx9.h>
#include < windows.h>
#include < stdlib.h>
#include < malloc.h>
#include < memory.h>
#include < tchar.h>

// 全局变量:
HINSTANCE hInst;        // 当前实例
TCHAR szTitle[20];        // 标题栏文本
TCHAR szWindowClass[20];      // 主窗口类名
LPDIRECT3D9 g_pD3D = NULL;      // D3D指针
LPDIRECT3DDEVICE9 g_pD3DDevice = NULL;   // D3D设备指针
LPDIRECT3DVERTEXBUFFER9 g_pVertexBuffer = NULL; // 顶点缓存指针
//定义顶点信息的结构体
struct CUSTOMVERTEX
{
    FLOAT x, y, z, rhw;  
    DWORD colour;  
};
//定义自由顶点格式
#define D3DFVF_CUSTOMVERTEX (D3DFVF_XYZRHW|D3DFVF_DIFFUSE)
//定义释放COM对象的宏
#define SafeRelease(pObject) if(pObject != NULL) {pObject->Release(); pObject=NULL;}
// 此代码模块中包含的函数的前向声明:
ATOM    MyRegisterClass(HINSTANCE hInstance);
BOOL    InitInstance(HINSTANCE, int);
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
LRESULT CALLBACK About(HWND, UINT, WPARAM, LPARAM);
//初始化D3D设备
HRESULT InitialiseD3D(HWND hWnd)
{
 //取得D3D9的对象
    g_pD3D = Direct3DCreate9(D3D_SDK_VERSION);
    if(g_pD3D == NULL)
    {
        return E_FAIL;
    }
 //得到当前的显示模式
    D3DDISPLAYMODE d3ddm;
    if(FAILED(g_pD3D->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &d3ddm)))
    {
        return E_FAIL;
    }
 //创建一个D3D设备
    D3DPRESENT_PARAMETERS d3dpp;
    ZeroMemory(&d3dpp, sizeof(d3dpp));
    d3dpp.Windowed = TRUE;//全屏模式还是窗口模式
    d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;//后台缓冲区复制到前台时,清除后台缓冲区内容
    d3dpp.BackBufferFormat = d3ddm.Format;//屏幕的显示模式
	d3dpp.PresentationInterval = 33;
 //创建一个Direct3D设备
    if(FAILED(g_pD3D->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hWnd,D3DCREATE_SOFTWARE_VERTEXPROCESSING, &d3dpp, &g_pD3DDevice)))
    {
        return E_FAIL;
    }
   
    return S_OK;
}
HRESULT InitialiseVertexBuffer()
{
 VOID* pVertices;
 
 //顶点信息数组
 CUSTOMVERTEX cvVertices[] =
 {
  {250.0f, 100.0f, 0.5f, 1.0f, D3DCOLOR_XRGB(255, 0, 0),},
  {400.0f, 350.0f, 0.5f, 1.0f, D3DCOLOR_XRGB(0, 255, 0),},
  {100.0f, 350.0f, 0.5f, 1.0f, D3DCOLOR_XRGB(0, 0, 255),},
 };
 //通过设备创建顶点缓冲
 if(FAILED(g_pD3DDevice->CreateVertexBuffer(3 * sizeof(CUSTOMVERTEX),
                                               0, D3DFVF_CUSTOMVERTEX,
                                               D3DPOOL_DEFAULT, &g_pVertexBuffer,NULL)))
 {
  return E_FAIL;
 }
 //锁定顶点缓冲，并得到一个存放顶点信息的缓冲区的指针
 if(FAILED(g_pVertexBuffer->Lock(0, sizeof(cvVertices), (void**)&pVertices, 0)))
 {
  return E_FAIL;
 }
 //复制顶点信息
 memcpy(pVertices, cvVertices, sizeof(cvVertices));
 //解锁顶点缓冲区
 g_pVertexBuffer->Unlock();
    return S_OK;
}
void Render()
{
    if(g_pD3DDevice == NULL)
    {
        return;
    }
 //清空后备缓冲区为黑色
    g_pD3DDevice->Clear(0, NULL, D3DCLEAR_TARGET, D3DCOLOR_XRGB(0, 0, 0), 1.0f, 0);   
 //开始绘制场景
    g_pD3DDevice->BeginScene();
 //渲染三角形
 g_pD3DDevice->SetStreamSource(0, g_pVertexBuffer, 0, sizeof(CUSTOMVERTEX));
 g_pD3DDevice->SetFVF(D3DFVF_CUSTOMVERTEX);
 g_pD3DDevice->DrawPrimitive(D3DPT_TRIANGLELIST, 0, 1);
 //结束绘制场景
    g_pD3DDevice->EndScene();
   
 //翻页显示
    g_pD3DDevice->Present(NULL, NULL, NULL, NULL);
}
//释放所使用到的所有COM对象
void CleanUp()
{
 SafeRelease(g_pVertexBuffer);
 SafeRelease(g_pD3DDevice);
 SafeRelease(g_pD3D);
}
//游戏循环
void GameLoop()
{
    //进入游戏循环
    MSG msg;
    BOOL fMessage;
    PeekMessage(&msg, NULL, 0U, 0U, PM_NOREMOVE);
   
    while(msg.message != WM_QUIT)
    {
        fMessage = PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE);
        if(fMessage)
        {
            //处理消息
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
   //如果没有消息，则渲染当前的场景
            Render();
        }
    }
}
int APIENTRY _tWinMain(HINSTANCE hInstance,
                     HINSTANCE hPrevInstance,
                     LPTSTR    lpCmdLine,
                     int       nCmdShow)
{
 //注册Windows的窗口类
    WNDCLASSEX wc = {sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L,
                     GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                     "D3DDrawGraphics", NULL};
    RegisterClassEx(&wc);
 //创建一个窗口
    HWND hWnd = CreateWindow("D3DDrawGraphics", "D3D绘制简单图形",
                              WS_OVERLAPPEDWINDOW, 50, 50, 500, 500,
                              GetDesktopWindow(), NULL, wc.hInstance, NULL);
 //初始化Direct3D
    if(SUCCEEDED(InitialiseD3D(hWnd)))
    {
  //显示窗口
        ShowWindow(hWnd, SW_SHOWDEFAULT);
        UpdateWindow(hWnd);
  //初始化顶点缓冲
  if(SUCCEEDED(InitialiseVertexBuffer()))
  {
   //开始游戏: 进入游戏循环
   GameLoop();
  }
    }
   
    CleanUp();
 //撤销窗口类的注册
    UnregisterClass("D3DDrawGraphics", wc.hInstance);
   
    return 0;
}
 

// 注册窗口类
ATOM MyRegisterClass(HINSTANCE hInstance)
{
 WNDCLASSEX wcex;
 wcex.cbSize = sizeof(WNDCLASSEX);
 wcex.style   = CS_HREDRAW | CS_VREDRAW;
 wcex.lpfnWndProc = (WNDPROC)WndProc;
 wcex.cbClsExtra  = 0;
 wcex.cbWndExtra  = 0;
 wcex.hInstance  = hInstance;
 //wcex.hIcon   = LoadIcon(hInstance, (LPCTSTR)IDI_MY);
 wcex.hCursor  = LoadCursor(NULL, IDC_ARROW);
 wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
// wcex.lpszMenuName = (LPCTSTR)IDC_MY;
 wcex.lpszClassName = szWindowClass;
 //wcex.hIconSm  = LoadIcon(wcex.hInstance, (LPCTSTR)IDI_SMALL);
 return RegisterClassEx(&wcex);
}

//   保存实例句柄并创建主窗口
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
   HWND hWnd;
   hInst = hInstance; // 将实例句柄存储在全局变量中
   hWnd = CreateWindow(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, NULL, NULL, hInstance, NULL);
   if (!hWnd)
   {
      return FALSE;
   }
   ShowWindow(hWnd, nCmdShow);
   UpdateWindow(hWnd);
   return TRUE;
}

//  窗口回调函数
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch(message)
    {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        break;
        case WM_KEYUP:
            switch (wParam)
            {
   case VK_ESCAPE:
                    DestroyWindow(hWnd);
                    return 0;
                break;
            }
        break;
    }
    return DefWindowProc(hWnd, message, wParam, lParam);
}