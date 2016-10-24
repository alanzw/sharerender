#include "CmdController.h"
#include "../LibCore/InfoRecorder.h"
// for the cmd controller
CmdDealer * CmdDealer::contorller;

// print the detailed information of the dis server
void CmdDealer::print(){
	cg::core::infoRecorder->logTrace("[CmdController]: print(), task map.\n");
	server->printTask();

}