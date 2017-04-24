#ifndef __CMDHELPER_H__
#define __CMDHELPER_H__
#include "StringTool.h"
#include <d3d9.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
/*
the cmd controller will handle the cmd, enable limit fps, generate video,
monitor statics, enable RTSP service, logic server url£¬ output file path, output file name, frame log option

if spercific the work mode to Listen Mode, we need to check logic url and task id

parameter: -f [FPS] -l [log option] -s [save directory] -o [output] -u [url] -d [monitor data option] -v [rtsp option] -r [render step] -e [encoder option] -i [identifier for task] -m [work mode] -x [graphic mode] -a [socket number] -p [parent process id]

*/


/// when need to test PSNR, we save the picture in server
#define SERVER_SAVE_BMP

namespace cg{
	namespace core{

		enum LOGIC_MODE{
			PURE_LOAD,
			NO_RENDERING,

		};
		/*
		cmd controller is used for DLL to read the cmd line
		*/
		class CmdController{
			bool			connectLogic;
			//char * logicUrl;
			string			logicUrl;
			bool			urlReady;
			int				logicPort;

			bool			rateControl;
			int				maxFps;
			bool			frameLog;		// enable frame log

			bool			enableRTSP;
			bool			generateVideo;
			//char * videoName;
			//char * outputPath;			// the output file path
			string			outputPath;
			string			videoName;
			bool			listenMode;
			int				mode;			// the listen mode: 1 for listen distributor, 2 for listen render proxy

			int				frameStep;		// enable render every [frameStep], if full render, set this to 1, set 2 when need render half, 0 for no render
			int				sendingStep;

			int				encoderOption;	// type of encoder to use
			string			identifier;

			string			sockString;
			string			pidString;

			bool			useInternalSocket;

			int				graphicMode;	// which graphic mode is in use, dx9, dx10, dx11 or 2d

			char *			cmdline;
			string			scmd;
			string			exeName;
			string			objName;

			int				curRender;
			bool			enableToRender;
			bool			fullOffload;  // if work in FULL_OFFLOAD_MODE, no render in logic
			// the context
			IDirect3DDevice9*	d9Device;
			HWND			hwnd;
			HANDLE			taskId;

			static CmdController * cmdCtrl_;
			CmdController(char * cmd);

		public:
			~CmdController();
			static CmdController * GetCmdCtroller(char * cmd = NULL){
				if(!cmdCtrl_){
					if(cmd){
						cmdCtrl_ = new CmdController(cmd);
						cmdCtrl_->parseCmd();
					}
					else{
						// error
						//MessageBox(NULL, "NULL cmdline to create cmd controller", "ERROR", MB_OK);
					}
				}
				return cmdCtrl_;
			}
			static void Release(){
				if(cmdCtrl_){
					delete cmdCtrl_;
					cmdCtrl_ = NULL;
				}
			}
			inline void setDevice(IDirect3DDevice9 *device){ d9Device = device;}
			inline IDirect3DDevice9 * getDevice(){ return d9Device; }
			inline void setHwnd(HWND _hwnd){ hwnd = _hwnd; }
			inline HWND getHwnd(){ return hwnd; }
			inline bool isRender(){ return enableToRender && (!fullOffload); }
			inline HANDLE getTaskId(){ return taskId; }
			inline void setTaskId(HANDLE id){ taskId = id; }
			inline string getObjName(){ return objName; }
			inline void setObjName(string str){ objName = str; }
			inline bool hasRenderConnection(){ return useInternalSocket; }
			inline DWORD getPPid(){ return atoi(pidString.c_str());}
			inline SOCKET getRenderSocket(){ return (SOCKET)atoi(sockString.c_str()); }

			bool commitRender();

			bool			parseCmd();
			string			toString();

			// getter
			inline string	getIdentifier(){ return identifier; }
			inline string	getLogicUrl(){ return logicUrl; }
			inline string	getOutputPath(){ return outputPath; }
			inline string	getVideoName(){ return videoName; }
			inline string	getExeName(){ return exeName; }

			inline bool		enableGenerateVideo(){ return generateVideo; }
			inline bool		isListenMode(){ return listenMode; }
			inline bool		enableFrameLog(){ return frameLog; }
			inline bool		enableRTSPService(){ return enableRTSP; }
			inline bool		enableRateControl(){ return rateControl; }
			inline void		setMaxFps(int val){ rateControl = true; maxFps = val; }
			inline bool		isToConnectLogic(){ return connectLogic; }
			inline bool		isUrlReady(){ return urlReady; }
			inline bool		is2DGame(){ return graphicMode == 3 ? true: false;}

			inline int		getMaxFps(){ return maxFps; }
			inline int		getFrameStep(){ return frameStep; }
			inline int		getSendStep(){ return sendingStep; }
			inline int		getEncoderOption(){ return encoderOption; }
			inline int		getMode(){ return mode; }

			inline void		setGenVideo(){ generateVideo = true; }
			inline void		setEncoderOption(int val){ encoderOption = val; }
			inline void		setFrameStep(int val){ 
				frameStep = val; 
				if(curRender >= frameStep)
					curRender=0;
			}
			inline int		addRenderConnection(){
				if(frameStep != 0){
					int ret = frameStep;
					frameStep++;
					if(curRender >= frameStep)
						curRender=0;
					return ret;
				}else{
					// no render
					return 0;
				}
			}
			inline void		setFullOffload(){ fullOffload = true; }

		};

		extern CmdController * cmdCtrl;
		/*
		cmd parser is used for Loader to construct the cmd
		*/


		

	}
}
#endif // __CMDHELPER_H__