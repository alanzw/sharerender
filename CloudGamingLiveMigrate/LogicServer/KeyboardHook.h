#ifndef __HEKBORADHOOK_H__
#define __HEKBORADHOOK_H__

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>


namespace cg{
	namespace core{
		/*hook the key, get the keyboard command, 
		F7		-- enable render or not
		F10		-- capture screen
		F11		-- the syn sign
		1,2,3,4 -- set the render step
		6,7,8,9 -- set the sending step, means how many frames need to send

		*/
		class KeyCommandHelper{
			bool				synSign;
			DWORD				synStart;
			bool				f10pressed;
			HHOOK				keyHookHandle;

			bool			enableRender;
			int					renderStep;
			int					sendStep;

			CRITICAL_SECTION	section;

			static KeyCommandHelper * keyCmdHelper;
			KeyCommandHelper();
		public:
			static KeyCommandHelper * GetKeyCmdHelper(){
				if(!keyCmdHelper){
					keyCmdHelper = new KeyCommandHelper();
				}
				return keyCmdHelper;
			}
			~KeyCommandHelper();

			// install keyboard hook for given thread
			bool installKeyHook(DWORD threadId);

			inline int getRenderStep(){ return renderStep; }
			inline int getSendStep(){ return sendStep; }
			inline bool isF10Pressed(){ return f10pressed; }
			inline bool isSynSigned(){ return synSign; }
			inline DWORD getSynSignedTime(){ return synStart; }
			inline bool isEnableRender(){ return enableRender; }

			inline void setEnableRender(){ 
				bool val = enableRender;
				enableRender = !val; 
			}
			inline void setRenderStep(int val){ renderStep = val; }
			inline void setSendStep(int val){ sendStep = val; }
			inline void setF10Pressed(bool val){ f10pressed = val; }
			void setSynSigin(bool val);

			inline void lock(){ EnterCriticalSection(&section); }
			inline void unlock(){ LeaveCriticalSection(&section); }

		};
		LRESULT CALLBACK HookProc(int nCode, WPARAM wParam, LPARAM lParam);
	}
}

#endif