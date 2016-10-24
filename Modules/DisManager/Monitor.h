#ifndef __MONITOR_H__
#define __MONITOR_H__

class WatchdogForDis{
	int interval;    // prodic to check
	HANDLE notifier;
public:

	HANDLE getNotifier(){ return notifier; }
};

// the input cmd server recv user cmd from console and execute the cmd
// to control logic server and render server or event the user client
class InputCmdServer{

};

#endif