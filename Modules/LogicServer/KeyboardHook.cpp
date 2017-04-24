#include "GameServer.h"
#include "../LibCore/CmdHelper.h"
#include "KeyboardHook.h"
#include "../VideoGen/generator.h"
#include "../LibInput/Controller.h"
extern bool toRecorde;

namespace cg{
	namespace core{
		
		//extern BTimer * ctrlTimer;

		float getSysProcessTime(){
			float ret = 0.0;

			int interval = 0;
			if(ctrlTimer) interval = ctrlTimer->Stop();
			return 1000.0 * interval / ctrlTimer->getFreq();
			
		}

		KeyCommandHelper * keyCmdHelper = NULL;
		static int keyCount = 0;

		LRESULT CALLBACK HookProc(int nCode, WPARAM wParam, LPARAM lParam){
			KeyCommandHelper * keyHelper = KeyCommandHelper::GetKeyCmdHelper();
			DelayRecorder * delayRecorder = DelayRecorder::GetDelayRecorder();
#if 0
			if(wParam == VK_F11)
				infoRecorder->logError("[Global]: F11 key, WPARAM: %x, LPARAM:%x.\n", wParam, lParam);
#endif
			if (lParam & 0x80000000) // released
			{
				//infoRecorder->logTrace("[Global]: key release, WPARAM: %x, LPARAM:%x.\n", wParam, lParam);
				if (wParam == VK_F7 || wParam == VK_F6) // f7 pressed
				{
					keyHelper->setRenderStep(0);
					if(infoRecorder){
						infoRecorder->enableLogSecond();
					}
#if 0
					if(cmdCtrl){
						cmdCtrl->setFrameStep(cmdCtrl->getFrameStep() == 0 ? 1: 0);  // disable rendering or enable rendering
					}
#endif
				}
				else if (wParam == VK_F11){
					
					keyCount++;
					infoRecorder->logError("[Global]: VK_F11, key count:%d mode value:%d.\n", keyCount, keyCount %2);
					if(keyCount % 2){
#if 0
						infoRecorder->logTrace("[Global]: wparam:%x, lparam:%x. system process input time: %f.\n", wParam, lParam, getSysProcessTime());
						if(delayRecorder->isInputArrive()){
							delayRecorder->keyTriggered();
						}
#endif
						infoRecorder->logError("[Global]: F11 triggered. to SYN.\n");
						keyHelper->lock();
						keyHelper->setSynSigin(true);
						keyHelper->unlock();
					}
					else{
						infoRecorder->logError("[Global]: key count mode value is:%d.\n", keyCount % 2);
					}
				}
#if 1
				else if(wParam == VK_F12){
					printStatics();
				}
#endif
				else if (wParam == VK_F1) // f10 pressed
				{
					//toRecorde = true;
					//cg::core::InfoRecorder * recorder = cg::core::InfoRecorder::
					if(infoRecorder){
						infoRecorder->enableLogSecond();
					}
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

				}
				else if(wParam >= 0x36 && wParam <= 0x39){
					// 6, 7, 8, 9
					int sendStep = wParam - 0x35;
					keyHelper->changeSendStep(sendStep);
				}
				else if(wParam == 0x30){
					keyHelper->changeSendStep(0);
				}else if(wParam == 0x5a){
					// key Z, for x264 encoder
					if(cg::gGenerator){
						gGenerator->changeEncodeDevice(X264_ENCODER);
					}

				}else if(wParam == 0x58){
					// key X, for CUDA encoder
					if(gGenerator){
						gGenerator->changeEncodeDevice(CUDA_ENCODER);
					}

				}else if(wParam == 0x43){
					// key C, for nvenc encoder
					if(gGenerator){
						gGenerator->changeEncodeDevice(NVENC_ENCODER);
					}
				}
				// control the fps
				else if(wParam == 0x59){
					// key V, for 1 frame per second
					keyHelper->setMaxFps(1);
				}
				else if(wParam == 0x42){
					// key B, for 10 frames per second
					keyHelper->setMaxFps(10);
				}
				else if(wParam == 0x4e){
					// key N, for 20 frames per second
					keyHelper->setMaxFps(20);
				}
				else if(wParam == 0x4d){
					// key M, for 30 frames per second
					keyHelper->setMaxFps(30);
				}
				else if(wParam == VK_OEM_COMMA){
					// key ,, for 40 frames per second
					keyHelper->setMaxFps(40);
				}
				else if(wParam == VK_OEM_PERIOD){
					// key ., for 50 frames per second
					keyHelper->setMaxFps(50);
				}
				else if(wParam == VK_OEM_2){
					// key /, for 60 frames per second
					keyHelper->setMaxFps(60);
				}
			}
			else{
				infoRecorder->logTrace("[Global]: key pressed, WPARAM: %x, LPARAM:%x.\n", wParam, lParam);
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
		KeyCommandHelper::KeyCommandHelper():enableRender(true), synSign(false), synStart(0), f10pressed(false), keyHookHandle(NULL), renderStep(1), sendStep(1), renderStepChanged(false), bufSendingStep(1), sendingStepChanged(false), currentSending(0), maxFps(60), fpsChanged(false), valueTag(0){
			InitializeCriticalSection(&section);
			memset(name, 0, 1024);
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

			return enableSending;
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
			if(fpsChanged){
				_cmdCtrl->setMaxFps(maxFps);
				fpsChanged = false;
			}

			return commit();
		}

	}
}
