#ifndef __INFORECORDER_H__
#define __INFORECORDER_H__

#include <stdio.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <string>

#include "lightweightreocder.h"
#include "CpuWatch.h"
#include "GpuWatch.h"


namespace cg{
	namespace core{

		class InfoRecorder{

			LARGE_INTEGER secondEnd, secondStart, freq, frameStart, frameEnd;
			LONGLONG timeCount;

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

			HANDLE namedMutex;
			HANDLE mappingHandle;
			LPVOID mappingAddr;
			bool useMapping;
			int renderStep;
			
			bool toLogSecond;

			// some buffers to store the critical performance data
			float captureTime, convertTime, encodeTime, packetTime;
			short captureCount, convertCount, encodeCount, packetCount;

		public:
			// setters for performance data
			inline void addCaptureTime(float val){ captureTime += val; captureCount++;}
			inline void addConvertTime(float val){ convertTime += val; convertCount++;}
			inline void addEncodeTime(float val){ encodeTime += val; encodeCount++;}
			inline void addPacketTime(float val){ packetTime += val; packetCount++; }
			inline void setRenderStep(int val){ renderStep = val; }
			inline void enableLogSecond(){ toLogSecond = true;}


			InfoRecorder(char * prefix);
			~InfoRecorder();

			bool initMapping(std::string exeName);
			void releaseMapping();
			bool init();

			void flush();
			bool onFrameEnd(bool recordGpu = true);
			bool onSecondEnd(bool recordGpu = true);


		

			bool onFrameEnd(float frameTime, bool recordGpu = true);
			bool onSecondEnd(float aveFrameTime, bool recordGpu = true);
		

			// log error message to file
			void logError(char * foramt, ...);
			void logFrame(char * format, ...);
			void logSecond(char * format, ...);

			void logExtra(char *format, ...);

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

		//// to log the data write in the shared memory map
		class SharedDataLogger{

		public:
			SharedDataLogger(std::string logFilename, std::string shardMappingName);
			~SharedDataLogger();

			void log(char * format, ...);

		private:
			HANDLE namedMutex;
			HANDLE targetProcess;
			LightWeightRecorder * recorder;
		};
	}
}

#ifndef TRACE_LOG
#define TRACE_LOG(x) {cg::core::infoRecorder->logTrace(x); printf(x);}

#endif

#ifndef ERROR_LOG
#define ERROR_LOG(x) {cg::core::infoRecorder->logError(x); printf(x);}
#endif

#endif