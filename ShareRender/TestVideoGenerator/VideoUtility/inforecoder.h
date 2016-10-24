#ifndef __INFORECORDER_H__
#define __INFORECORDER_H__

#include <stdio.h>
#include <Windows.h>

#include "lightweightreocder.h"
#include ".\CpuWatch.h"
#include ".\GpuWatch.h"

class InfoRecorder{

	LARGE_INTEGER secondEnd, secondStart, freq, frameStart, frameEnd;
	ULONGLONG timeCount;

	// the cpu and gpu watcher
	CpuWatch *frameCpuWatcher, *secondCpuWatcher;
	GpuWatch *gpuWatcher;
	
	HANDLE processHandle;
	CRITICAL_SECTION recorderMutex;

	HANDLE recoderLock;

	unsigned int frameIndex, secondIndex, frameCountInSecond;

	LightWeightRecorder * frameRecorder;
	LightWeightRecorder * secondRecorder;
	LightWeightRecorder * errorRecorder;

	bool frameStarted, secondStarted;
	void lock();
	void unlock();
public:
	InfoRecorder(char * prefix);
	~InfoRecorder();

	void flush();
	bool onFrameEnd();
	bool onSecondEnd();

	void logError(char * foramt, ...);
	void logFrame(char * format, ...);
	void logSecond(char * format, ...);
};

#endif