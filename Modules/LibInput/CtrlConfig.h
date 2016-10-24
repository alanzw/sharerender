#ifndef __CTRLCONFIG__
#define __CTRLCONFIG__

#include "../VideoUtility/ccg_config.h"
//#include "../LibVideo/Config.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <WinSock2.h>

namespace cg{
	namespace input{
		// this is for the controller config
		class CtrlConfig : public cg::ccgConfig{
			static int initialized;
			static CtrlConfig * ctrlConf;
			struct sockaddr_in ctrl_sin;

			CtrlConfig(char * filename);

		public:
			bool enableRender;   // enable screen render
			char * ctrl_servername;
			int ctrlport;
			char ctrlproto; // transport layer tcp = 6; udp = 17

			int ctrlenable;
			int sendmousemotion;

			int ctrlConfInit();
			int ctrlConfParse();
			void ctrlResolveServer(const char * servername);

			static CtrlConfig * GetCtrlConfig(char * filename);
			~CtrlConfig(){
				if (ctrl_servername){
					free(ctrl_servername);
					ctrl_servername = NULL;
				}
			}
		};
	}
}

#endif