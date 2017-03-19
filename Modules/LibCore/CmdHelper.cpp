#include "CmdHelper.h"
#include <sstream>
#include <Windows.h>
#include "../LibCore/InfoRecorder.h"


using namespace std;

namespace cg{
	namespace core{
		///////////// cmd controller /////////////
		CmdController * CmdController::cmdCtrl_;   // the static variable in cmd
		CmdController * cmdCtrl = NULL;  

		CmdController::CmdController(char *cmd){
			connectLogic = false;

			listenMode = false;
			logicPort = -1;
			maxFps = 60;
			frameStep = 1;
			sendingStep = 1;
			curRender = 0;
			enableToRender = false;

			fullOffload = false;

			urlReady = false;

			rateControl = false;
			maxFps = 60;
			frameLog = false;
			enableRTSP = false;
			generateVideo = false;
			encoderOption = 1;    // default use x264

			useInternalSocket = false;

			cmdline = NULL;
			if(cmd){
				cmdline = cmd;
				scmd = string(cmd);
			}
		}

		CmdController::~CmdController(){

		}

		bool CmdController::parseCmd(){
			vector<string> vt;		// the result
			cg::core::split(cmdline, vt, " ");
			exeName = vt[0];		// the first is exe name, full path name
			string tm = exeName;
			

			char tmName[512] = {0};
			char objName_[512] = {0};
			strcpy(tmName, tm.c_str());
			int len = strlen(tmName);
			char * pEnd = NULL, * pStart = NULL;
			// TODO, get the rtsp service from cmdline
			pEnd = tmName + len -1;
			cg::core::infoRecorder->logTrace("CmdController]: parse cmd, %s.\n", tmName);
			while(*pEnd != '.'){
				pEnd--;
			}
			cg::core::infoRecorder->logTrace("CmdController]: parse cmd, post fix: %s.\n", pEnd);
			if(!strncmp(pEnd, ".exe", 4)){
				// get a postfix
			}
			else{
				pEnd = tmName + len;
			}
			pStart = pEnd - 1;
			cg::core::infoRecorder->logTrace("CmdController]: parse cmd, start: %s.\n", pStart);
			while(*pStart != '/' && *pStart != '\\'){
				pStart--;
			}
			strncpy(objName_, pStart, pEnd - pStart);
			cg::core::infoRecorder->logTrace("CmdController]: parse cmd, obj name: %s.\n", objName_);
			if(objName_[1] > 'a' && objName_[1] < 'z'){
				objName_[1] += ('A' - 'a');
			}
			objName = string(objName_);
			cg::core::infoRecorder->logTrace("[CmdHelper]: obj name:%s.\n", objName_);

			graphicMode = 0;   // default use D3D9

			for(size_t i = 0; i < vt.size(); ++ i){
				// find the options
				if(vt[i] == string("-f") || vt[i] == string("-F")){
					// the FPS option
					rateControl = true;
					maxFps = atoi(vt[i +1].c_str());
				}
				else if(vt[i] == string("-l") || vt[i] == string("-L")){
					// the log option
					if(vt[i+1] == string("frame") || vt[i+1] == string("Frame")){
						frameLog = true;
					}
				}
				else if(vt[i] == string("-s") || vt[i] == string("-S")){
					// the save directory option
					outputPath = string(vt[i+1]);
				}
				else if(vt[i] == string("-o") || vt[i] == string("-O")){
					// the output file option
					videoName = string(vt[i+1]);
					generateVideo = true;
				}
				else if(vt[i] == string("-u") || vt[i] == string("-U")){
					// the url option
					logicUrl = string(vt[i+1]);
				}
				else if(vt[i] == string("-d") || vt[i] == string("-D")){
					// the monitor data option

				}
				else if(vt[i] == string("-v") || vt[i] == string("-V")){
					// the rtsp option 
					if(vt[i+1] == string("true") || vt[i+1] == string("True")){
						enableRTSP = true;
						listenMode = true;
					}
				}
				else if(vt[i] == string("-r")|| vt[i] == string("-R")){
					// the render step
					frameStep = atoi(vt[i+1].c_str());
				}
				else if(vt[i] == string("-e") || vt[i] == string("-E")){
					// the encoder option
					encoderOption = atoi(vt[i+1].c_str());
					//encoderOption = 3;
				}
				else if(vt[i] == string("-i") || vt[i] == string("-I")){
					identifier = string(vt[i+1]);
				}

				else if(vt[i] == string("-m") || vt[i] == string("-M")){
					// work mode, 0 for stand alone mode, 1 for listen mode
					mode = atoi(vt[i+1].c_str());
					if(mode){
						listenMode = true;
					}
				}
				else if(vt[i] == string("-x") || vt[i] == string("-X")){
					// the graphic mode, 0: d9, 1: d10, 2: d11 or 3: 2d
					graphicMode  = atoi(vt[i+1].c_str());
				}else if(vt[i] == string("-a") || vt[i] == string("-A")){
					// the socket number if any
					sockString = string(vt[i+1]);
					//exeName = string(vt[i+2]);
					useInternalSocket = true;
				}
				else if(vt[i] == string("-p") || vt[i] == string("-P")){
					// the parent process id
					pidString = string(vt[i + 1]);
				}
				else if(vt[i] == string("-n") || vt[i] == string("-N")){
					sendingStep = atoi(vt[i+1].c_str());
				}
			}
			return true;
		}

		bool CmdController::commitRender(){

#ifdef SERVER_SAVE_BMP  
			enableToRender = true;
			return true;
#endif


			if(frameStep != 0){
				curRender++;
				if(curRender >= frameStep){
					enableToRender = true;
					curRender = 0;	
				}
				else{
					enableToRender = false;
				}
			}else{
				enableToRender = false;
			
			}
			// rate control
			if(rateControl){

			}
			return enableToRender;
		}

		// to string function to print the content
		string CmdController::toString(){
			stringstream os;
			string ret("");
			os.str(""); // clear
			os << "CmdController1 to string:" << endl;
			os << "cmd: " << cmdline << endl;
			os << "mode: " << mode << endl;
			os << "listen mode: " << (listenMode ? "true" : "false") << endl;
			os << "identifier: " << identifier << endl;
			os << "obj name: " << objName << endl;
			//os >> ret;
			os.clear();

			return os.str();
		}
	}
}