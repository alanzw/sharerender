#ifndef __LOADEREVENT_H__
#define __LOADEREVENT_H__

#include "../LibVideo/EventNetwork.h"

//#define INCLUDE_DIS


// deal the event from dis server
class LoaderEvent:public EventNet{



};


DWORD WINAPI LoaderEventProc(LPVOID param);

#endif