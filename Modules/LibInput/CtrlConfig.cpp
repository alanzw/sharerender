#include "Config.h"
#include <WinSock2.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "CtrlConfig.h"

#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"

#define CTRL_DEF_CONTROL_PROTO IPPROTO_TCP
#define CTRL_DEF_CONTROL_PORT 555
#define CTRL_DEF_SEND_MOUSE_MOTION 0
#define CTRL_DEF_CONTROL_ENABLED 0
#define DELIM " \t\n\r"

namespace cg{
	namespace input{

		int CtrlConfig::initialized = 0;
		CtrlConfig * CtrlConfig::ctrlConf;

		// constructor
		CtrlConfig::CtrlConfig(char * filename):ccgConfig(filename){
			initialized = 1;
			ctrlConfInit();
			confLoad(filename);
			ctrlConfParse();
		}

		int CtrlConfig::ctrlConfInit(){

			ctrlenable = CTRL_DEF_CONTROL_ENABLED;
			ctrlport = CTRL_DEF_CONTROL_PORT;
			ctrlproto = CTRL_DEF_CONTROL_PROTO;
			sendmousemotion = CTRL_DEF_SEND_MOUSE_MOTION;
			enableRender = true;

			return 0;
		}

		int CtrlConfig::ctrlConfParse(){
			char * ptr, buf[1024];
			int v;
			ctrlConfInit();

			enableRender = confReadBool("enable-render", 0);
			ctrlenable = confReadBool("control-enabled", 0);
			//
			if (ctrlenable != 0) {
				//
				v = confReadInt("control-port");
				if (v <= 0 || v >= 65536) {
					cg::core::infoRecorder->logTrace("[CtrlConfig]: invalid control port %d\n", v);
					return -1;
				}
				ctrlport = v;
				cg::core::infoRecorder->logTrace("[CtrlConfig]:: controller port = %d\n", ctrlport);
				//
				ptr = confReadV("control-proto", buf, sizeof(buf));
				if (ptr == NULL || strcmp(ptr, "tcp") != 0) {
					ctrlproto = IPPROTO_UDP;
					cg::core::infoRecorder->logTrace("[CtrlConfig]: controller via 'udp' protocol.\n");
				}
				else {
					ctrlproto = IPPROTO_TCP;
					cg::core::infoRecorder->logTrace("[CtrlConfig]: controller via 'tcp' protocol.\n");
				}
				//
				sendmousemotion = confReadBool("control-send-mouse-motion", 1);

				ptr = confReadV("control-server", buf, sizeof(buf));
				if (ptr != NULL){
					// get the control server name
					ctrl_servername = _strdup(ptr);
				}
			}
			return 1;
		}

		CtrlConfig * CtrlConfig::GetCtrlConfig(char * filename){
			if (ctrlConf == NULL){
				//create new control config
				ctrlConf = new CtrlConfig(filename);
			}
			else{

			}

			return ctrlConf;
		}
	}
}