#ifndef __RENDERDOMAIN_H__
#define __RENDERDOMAIN_H__
#include "CThread.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

// render domain contains the distributor part
class RenderDomain : public CThread{


public:
	// for cthread
	virtual BOOL stop();
	virtual void run();
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void onThreadStart(LPVOID param = NULL);
	virtual void onQuit();

	// constructor and distructor
	RenderDomain();
	~RenderDomain();
};
#endif