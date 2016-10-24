#include "muliple_game_server.h"


#ifdef OLD
HHOOK kehook = 0;
extern bool enableRender;
LRESULT CALLBACK HookProc(int nCode, WPARAM wParam, LPARAM lParam){
	
	//MessageBox(NULL,"Enter hook", "WARNING", MB_OK);
	if( lParam & 0x80000000) // pressed
	{
		if( wParam == VK_F7) // f7 pressed
		{
			//MessageBox(NULL,"F7 pressed", "WARNING", MB_OK);
			enableRender = false;
			//Log::slog("F7 pressed");
		}
	}
	//
	return CallNextHookEx(kehook,nCode,wParam,lParam); 
}

void SetKeyboardHook(HINSTANCE hmode, DWORD dwThreadId) {
	// set the keyboard hook
	//MessageBox(NULL,"Enter hook", "WARNING", MB_OK);
	//Log::slog("set the keyboard hook!\n");
	kehook = SetWindowsHookEx(WH_KEYBOARD, HookProc, hmode, dwThreadId);
}
#else
HHOOK kehook = 0;
extern bool enableRender;
extern bool synSign;
extern bool f10pressed;
extern CRITICAL_SECTION f9;
extern DWORD tick_start;

LRESULT CALLBACK HookProc(int nCode, WPARAM wParam, LPARAM lParam){

	//MessageBox(NULL,"Enter hook", "WARNING", MB_OK);
	if (lParam & 0x80000000) // released
	{
		if (wParam == VK_F7) // f7 pressed
		{
			//MessageBox(NULL,"F7 pressed", "WARNING", MB_OK);
			enableRender = false;
			//Log::slog("F7 pressed");
		}
		else if (wParam == VK_F11){
			EnterCriticalSection(&f9);
			synSign = true;
			LeaveCriticalSection(&f9);
			tick_start = GetTickCount();
		}
		else if (wParam == VK_F1) // f10 pressed
		{
			EnterCriticalSection(&f9);
			f10pressed = true;
			LeaveCriticalSection(&f9);

			INPUT in;
			memset(&in, 0, sizeof(INPUT));
			in.type = INPUT_KEYBOARD;
			in.ki.wVk = VK_F10;

			in.ki.wScan = MapVirtualKey(in.ki.wVk, MAPVK_VK_TO_VSC);
			SendInput(1, &in, sizeof(INPUT));

			in.ki.dwFlags |= KEYEVENTF_KEYUP;
			//in.ki.dwFlags |= KEYEVENTF_KEYDOWN;
			//SendInput(1, &in, sizeof(INPUT));


		}
	}
	//
	return CallNextHookEx(kehook, nCode, wParam, lParam);
}

void SetKeyboardHook(HINSTANCE hmode, DWORD dwThreadId) {
	// set the keyboard hook
	//MessageBox(NULL,"Enter hook", "WARNING", MB_OK);
	//Log::slog("set the keyboard hook!\n");
	kehook = SetWindowsHookEx(WH_KEYBOARD, HookProc, hmode, dwThreadId);
}

#endif

