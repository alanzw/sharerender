#include "GameServer.h"
#include "../LibCore/CmdHelper.h"
#include "KeyboardHook.h"

namespace cg{
	namespace core{

		KeyCommandHelper * keyCmdHelper = NULL;


		LRESULT CALLBACK HookProc(int nCode, WPARAM wParam, LPARAM lParam){

			KeyCommandHelper * keyHelper = KeyCommandHelper::GetKeyCmdHelper();
			infoRecorder->logError("[Global]: key event, WPARAM: %x, LPARAM:%x.\n", wParam, lParam);
			if (lParam & 0x80000000) // released
			{
				infoRecorder->logError("[Global]: key release, WPARAM: %x, LPARAM:%x.\n", wParam, lParam);
				if (wParam == VK_F7) // f7 pressed
				{
					keyHelper->setRenderStep(0);
#if 0
					if(cmdCtrl){
						cmdCtrl->setFrameStep(cmdCtrl->getFrameStep() == 0 ? 1: 0);  // disable rendering or enable rendering
					}
#endif
				}
				else if (wParam == VK_F11){
					keyHelper->lock();
					keyHelper->setSynSigin(true);
					keyHelper->unlock();
				}
				else if (wParam == VK_F1) // f10 pressed
				{
					keyHelper->lock();
					keyHelper->setF10Pressed(true);
					keyHelper->unlock();

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
				else if(wParam >= 0x31 && wParam <= 0x34){
					// 1, 2, 3, 4
					int renderStep = wParam - 0x30;
					keyHelper->setEnableRender();
					keyHelper->setRenderStep(renderStep);
#if 0
					if(cmdCtrl){
						cmdCtrl->setFrameStep(renderStep);
					}
#endif
				}
				else if(wParam >= 0x36 && wParam <= 0x39){
					// 6, 7, 8, 9
					int sendStep = wParam - 0x35;
					keyHelper->changeSendStep(sendStep);
				}
				else if(wParam == 0x30){
					keyHelper->changeSendStep(0);
				}
			}
			else{
				infoRecorder->logError("[Global]: key pressed, WPARAM: %x, LPARAM:%x.\n", wParam, lParam);
			}
			//
			return CallNextHookEx(keyHelper->getHookHandle(), nCode, wParam, lParam);
		}

		bool KeyCommandHelper::installKeyHook(DWORD threadId){
			// set the keyboard hook
			infoRecorder->logError("set the keyboard hook, module:%p, thread id:%d!\n", NULL, threadId);
			keyHookHandle = SetWindowsHookEx(WH_KEYBOARD, HookProc, NULL, threadId);
			if(!keyHookHandle){
				infoRecorder->logError("[KeyCommandHelper]: set window hook ex failed with:%d.\n", GetLastError());
				return false;
			}
			return true;
		}
		KeyCommandHelper *KeyCommandHelper::keyCmdHelper = NULL;
		KeyCommandHelper::KeyCommandHelper():enableRender(true), synSign(false), synStart(0), f10pressed(false), keyHookHandle(NULL), renderStep(1), sendStep(1), renderStepChanged(false), bufSendingStep(1), sendingStepChanged(false){
			InitializeCriticalSection(&section);
		}
		KeyCommandHelper::~KeyCommandHelper(){
			// release the hook and destroy the critical section
			BOOL ret = TRUE;
			if(keyHookHandle){
				ret= UnhookWindowsHookEx(keyHookHandle);
				if(ret == TRUE)
				keyHookHandle = NULL;
				else{
					infoRecorder->logError("[KeyCommandHelper]: unhook windows hook failed, error code:%d.\n", GetLastError());
				}
			}
			DeleteCriticalSection(&section);
		}
		void KeyCommandHelper::setSynSigin(bool val){
			synSign = val;
			if(synSign){
				synStart = GetTickCount();
			}
		}


		bool KeyCommandHelper::commit(){
			
			// commit at the end of each frame, to change the sending status
			if(sendingStepChanged){
				setSendStep(bufSendingStep);
				sendingStepChanged = false;
			}

			if(sendStep != 0){
				currentSending ++;
				if(currentSending >= sendStep){
					enableSending = true;
					currentSending = 0;
				}
				else
					enableSending = false;
			}
			else
				currentSending = false;

			return currentSending;
		}
		bool KeyCommandHelper::commit(CmdController * _cmdCtrl){
			// set the render step for CmdController
			if(renderStepChanged){
				if(enableRender)
					_cmdCtrl->setFrameStep(renderStep);
				else{
					_cmdCtrl->setFrameStep(0);
				}
				renderStepChanged = false;
			}

			return commit();
		}

	}
}
