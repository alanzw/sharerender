#include "FakeGame.h"
#pragma comment(lib, "nvapi.lib")
#pragma comment(lib, "d3d9.lib")

//------------------------------------------------------------------------
// main entry of this project
//------------------------------------------------------------------------
#if 1
int WINAPI WinMain( 
	__in HINSTANCE hInstance,
	__in_opt HINSTANCE hPrevInstance,
	__in_opt LPSTR lpCmdLine,
	__in int nShowCmd )
#else

int main()
#endif
{
	MSG msg;

	GpuMonster * monster = GpuMonster::GetMonster();

	// create the main window
	if(!monster->InitInstance(hInstance,nShowCmd))
		return FALSE;
	// enter the message loop
	ZeroMemory( &msg, sizeof(msg) );
	while( msg.message != WM_QUIT )
	{
		if( PeekMessage( &msg, NULL, 0U, 0U, PM_REMOVE ) )
		{
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		else
			monster->RenderScene();
	}
	return msg.wParam;
}