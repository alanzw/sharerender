#ifndef __LOGICEVENT_H__
#define __LOGICEVENT_H__

#include "../LibVideo/EventNetwork.h"

// deal the event from loader

class LogicEvent:public EventNet{

};

DWORD WINAPI LogicEventProc(LPVOID param);

#endif