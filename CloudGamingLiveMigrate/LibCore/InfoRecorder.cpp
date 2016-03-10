#include "InfoRecorder.h"

//#define DEBUG_
namespace cg{
	namespace core{

		InfoRecorder * infoRecorder = NULL;

#ifdef DEBUG_
#include "Log.h"
#endif

		// constructor and destructor
		InfoRecorder::InfoRecorder(char * prefix): namedMutex(NULL), mappingHandle(NULL), mappingAddr(NULL), useMapping(false), captureTime(0.0f), convertTime(0.0f), encodeTime(0.0f), packetTime(0.0f){
			char recorderName[100];

			// init he frame recorder
			sprintf(recorderName, "%s.frame.log", prefix);
			frameRecorder = new LightWeightRecorder(recorderName);
			memset(recorderName, 0, 100);
			sprintf(recorderName, "%s.second.log", prefix);
			secondRecorder = new LightWeightRecorder(recorderName);
			memset(recorderName, 0, 100);
			sprintf(recorderName, "%s.error.log", prefix);
			errorRecorder = new LightWeightRecorder(recorderName);
			sprintf(recorderName, "%s.trace.log", prefix);
			traceRecorder = new LightWeightRecorder(recorderName);

			// create all the watcher
#if 0
			frameCpuWatcher = new CpuWatch();
			secondCpuWatcher = new CpuWatch();
			gpuWatcher = GpuWatch::GetGpuWatch();
#else
			frameCpuWatcher = NULL;
			secondCpuWatcher = NULL;
			gpuWatcher = NULL;
#endif

			processHandle = NULL;
			processHandle = GetCurrentProcess();
			if (processHandle == NULL){
				MessageBox(NULL, "null process handle, get current process failed.", "Error", MB_OK);
			}


			timeCount = 0;
			QueryPerformanceFrequency(&freq);

			frameIndex = 0;
			secondIndex = 0;
			frameCountInSecond = 0;
			newCreated = 0;

			frameStarted = false;
			secondStarted = false;

			InitializeCriticalSection(&recorderMutex);

			recoderLock = CreateMutex(NULL, FALSE, NULL);

			logFrame("FrameIndex FPS cpu gpu\n");
			logSecond("SecondIndex FPS cpu gpu\n");
		}

		

		InfoRecorder::~InfoRecorder(){

			// flush first
			flush();

			if (frameRecorder){
				delete frameRecorder;
				frameRecorder = NULL;
			}
			if (secondRecorder){
				delete secondRecorder;
				secondRecorder = NULL;
			}
			if (errorRecorder){
				delete errorRecorder;
				errorRecorder = NULL;
			}

			if(traceRecorder){
				delete traceRecorder;
				traceRecorder = NULL;
			}

			if (frameCpuWatcher){
				delete frameCpuWatcher;
				frameCpuWatcher = NULL;
			}
			if (secondCpuWatcher){
				delete secondCpuWatcher;
				secondCpuWatcher = NULL;
			}
			if (gpuWatcher){
				delete gpuWatcher;
				gpuWatcher = NULL;
			}

			DeleteCriticalSection(&recorderMutex);
			if(recoderLock){
				CloseHandle(recoderLock);
				recoderLock = NULL;
			}
			releaseMapping();
		}

		bool InfoRecorder::init(){
			frameCpuWatcher = new CpuWatch();
			secondCpuWatcher = new CpuWatch();
			gpuWatcher = GpuWatch::GetGpuWatch();
			return true;
		}

		bool InfoRecorder::initMapping(std::string exeName){
			useMapping = true;
			std::string mutexName = exeName + std::string("_mutex");
			std::string mappingName = exeName + std::string("_mapping");

			errorRecorder->log("[InfoRecorder]: init mapping, exe name:%s, mutex name:%s, mapping name:%s.\n", exeName.c_str(), mutexName.c_str(), mappingName.c_str());
			//namedMutex = OpenMutex(MUTEX_ALL_ACCESS, FALSE, mutexName.c_str());
			namedMutex = OpenEvent(EVENT_ALL_ACCESS, FALSE, mutexName.c_str());
			if(namedMutex){
				errorRecorder->log("[InfoRecorder]: open mutex success, already exist.\n");
			}
			else{
				//namedMutex = CreateMutex(NULL, FALSE, mutexName.c_str());
				namedMutex = CreateEvent(NULL, FALSE, FALSE, mutexName.c_str());
				errorRecorder->log("[InfoRecorder]: open mutex failed. create new.\n");
			}
			// file mapping
			mappingHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, mappingName.c_str());
			if(mappingHandle){
				// get the file data
				mappingAddr = MapViewOfFile(mappingHandle, FILE_MAP_ALL_ACCESS, 0,0,0);
				errorRecorder->log("[InfoRecorder]: open mapping success, already exist.\n");
			}else{
				errorRecorder->log("[InfoRecorder]: open mapping failed, create new.\n");
				// create mapped file
				mappingHandle = CreateFileMapping((HANDLE)0xFFFFFFFF,NULL, PAGE_READWRITE,0, 64, mappingName.c_str());
				mappingAddr = MapViewOfFile(mappingHandle, FILE_MAP_ALL_ACCESS, 0,0,0);
			}
			if(mappingAddr == NULL){
				errorRecorder->log("[DllMain]: file map:%s get NULL, error:%d.\n", mappingName.c_str(), GetLastError());
			}

			return true;
		}	
		void InfoRecorder::releaseMapping(){
			if(namedMutex){
				CloseHandle(namedMutex);
				namedMutex = NULL;
			}
			if(mappingAddr){
				UnmapViewOfFile(mappingAddr);
				mappingAddr = NULL;
			}
			if(mappingHandle){
				CloseHandle(mappingHandle);
				mappingHandle = NULL;
			}
		}

		void InfoRecorder::lock(){
			//EnterCriticalSection(&recorderMutex);stati
			WaitForSingleObject(recoderLock, 300);
			//LeaveCriticalSection(&recorderMutex);
		}

		void InfoRecorder::unlock(){
			//LeaveCriticalSection(&recorderMutex);
			ReleaseMutex(recoderLock);
			//LeaveCriticalSection(&recorderMutex);
		}

		// functions
		void InfoRecorder::flush(){
			//lock();
			this->frameRecorder->flush(true);
			this->secondRecorder->flush(true);
			this->errorRecorder->flush(true);
			this->traceRecorder->flush(true);
			//unlock();
		}

		// get the cpu usage and gpu usage and the fps
		bool InfoRecorder::onFrameEnd(bool recordGpu){
			logTrace("[InfoRecorder]: on frame end.\n");
			// compute the fps
			float curFps = 0.0f;
			double cpuUsage = 0.0f;
			double gpuUsage = 0.0f;
			// get the frame time
			QueryPerformanceCounter(&frameEnd);

			if (frameStarted){
#if 0
				cpuUsage = frameCpuWatcher->GetProcessCpuUtilization(processHandle);
				gpuUsage = gpuWatcher->GetGpuUsage();
				curFps =  this->freq.QuadPart / ((frameEnd.QuadPart - frameStart.QuadPart));
#endif

				frameCountInSecond++;
				timeCount += (frameEnd.QuadPart - frameStart.QuadPart);
				if (timeCount >= this->freq.QuadPart){
					onSecondEnd(recordGpu);
					timeCount = 0;
				}
#if 0
				// do the log for frame.
				// the log format should be: [index] [fps] [cpuUsage] [gpuUsage]
				frameRecorder->log("%d %f %f %f\n", frameIndex, curFps, cpuUsage, gpuUsage);
#endif
			}
			else{
				frameStarted = true;
			}

			// log the frame index and new created object in this frame
			frameRecorder->log("frame: %d new: %d.\n", frameIndex, newCreated);
			newCreated = 0;

			frameIndex++;
			QueryPerformanceCounter(&frameStart);
			return true;
		}
		// get the cpu and gpu usage and the fps
		bool InfoRecorder::onSecondEnd(bool recordGpu){
			infoRecorder->logTrace("[InfoRecorder]: on second end.\n");
			QueryPerformanceCounter(&secondEnd);

			if (secondStarted){
				// log the second counter
				float cpuUsage = 0.0f;
				float gpuUsage = 0.0f;
				float fpsInSecond = frameCountInSecond * this->freq.QuadPart / (secondEnd.QuadPart - secondStart.QuadPart);

				frameCountInSecond = 0;
				cpuUsage = secondCpuWatcher->GetProcessCpuUtilization(processHandle);
				if(recordGpu)
					gpuUsage = gpuWatcher->GetGpuUsage();

				secondRecorder->log("%d %f %f %f\n", secondIndex, fpsInSecond, cpuUsage, gpuUsage);
				if(useMapping){
					// write to mapping file and notify
					int * p = (int *)mappingAddr;
					*p = secondIndex;
					p++;
					*p = (int)fpsInSecond;
					//ReleaseMutex(namedMutex);
					SetEvent(namedMutex);
				}
			}
			else{
				secondStarted = true;
			}
			secondIndex++;
			QueryPerformanceCounter(&secondStart);
			return true;
		}

		// log the error to file
		void InfoRecorder::logError(char * format, ...){
			char tem[512] = {0};
			lock();
			va_list ap;
			va_start(ap, format);
			int  n = vsprintf(tem, format, ap);
			//errorRecorder->log(format, ap);
			va_end(ap);
#ifdef DEBUG_
			infoRecorder->logError("[INfoRecorder]: tem:'%s', size:%d.\n", tem, n);
#endif
			
			errorRecorder->log(tem, n);
			errorRecorder->flush(true);

			traceRecorder->log(tem, n);
			traceRecorder->flush(true);

			unlock();

		}

		void InfoRecorder::logFrame(char * format, ...){
			char tem[512] = {0};
			va_list ap;
			va_start(ap, format);
			//frameRecorder->log(format, ap);
			int n = vsprintf(tem, format, ap);
			va_end(ap);

			lock();
			frameRecorder->log(tem, n);
			frameRecorder->flush(true);
			unlock();
		}
		void InfoRecorder::logSecond(char * format, ...){
			char tem[512] = {0};
			va_list ap;
			va_start(ap, format);
			//secondRecorder->log(format, ap);
			int n = vsprintf(tem, format, ap);
			va_end(ap);

			lock();
			secondRecorder->log(tem, n);
			secondRecorder->flush(true);
			unlock();
		}

		void InfoRecorder::logTrace(char *format, ...){
#if 0
			char tem[512] = {0};

			va_list ap;
			va_start(ap, format);
			int  n = vsprintf(tem, format, ap);
			//errorRecorder->log(format, ap);
			va_end(ap);
#ifdef DEBUG_
			infoRecorder->logError("[INfoRecorder]: tem:'%s', size:%d.\n", tem, n);
#endif
			lock();
			traceRecorder->log(tem, n);
			traceRecorder->flush(true);
			unlock();
#endif
		}

	}
}