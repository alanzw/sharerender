#ifndef __INFORECORDER_H__
#define __INFORECORDER_H__

#include <stdio.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "lightweightreocder.h"
#include "CpuWatch.h"
#include "GpuWatch.h"
namespace cg{
	namespace core{

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
			LightWeightRecorder * traceRecorder;

			bool frameStarted, secondStarted;
			void lock();
			void unlock();

			bool limitFps;
			int maxFps;
			double last_time, elapse_time;
			int frame_cnt;

			// add to record the new created objects in this frame
			int newCreated;

		public:
			InfoRecorder(char * prefix);
			~InfoRecorder();

			void flush();
			bool onFrameEnd(bool recordGpu = true);
			bool onSecondEnd(bool recordGpu = true);
		

			// log error message to file
			void logError(char * foramt, ...);
			void logFrame(char * format, ...);
			void logSecond(char * format, ...);

			// log the trace to file
			void logTrace(char * foramt, ...);
			inline void setLimitFps(int _maxFps){
				limitFps = true;
				maxFps = _maxFps;
			}
			inline void addCreation(){ newCreated++; }
		};

		// the global info recorder
		extern InfoRecorder * infoRecorder;
	}
}

#ifndef TRACE_LOG
#define TRACE_LOG(x) {cg::core::infoRecorder->logTrace(x); printf(x);}

#endif

#ifndef ERROR_LOG
#define ERROR_LOG(x) {cg::core::infoRecorder->logError(x); printf(x);}
#endif

#endif