#include <WinSock2.h>
#include "Utility.h"
#include "Log.h"
#include <time.h>
#include <stdlib.h>

namespace cg{
	namespace core{

		char Log::fname_[100];
		ofstream Log::fs_;
		int Log::is_init_ = false;

		time_t tv;
		HANDLE logMutex = NULL;
		void Log::init(const char* fname) {
			if(is_init_) return;
			is_init_ = true;

			strcpy(fname_, fname);
			fs_.open(fname_, ios::out);
			log("Log::init(), file name=%s\n", fname_);
			logMutex = CreateMutex(NULL, FALSE, NULL);
		}

		void Log::close() {
			log("Log::close() called\n");
			fs_.close();
		}

		void Log::log(const char* text, ...) {
#ifdef ENABLE_LOG
			WaitForSingleObject(logMutex, INFINITE);
			if(!is_init_) init("fuck.log");
			char buffer[MAX_CHAR];
			char timestr[30];

			tv = time(0);
			tm* ltime = localtime(&tv);
			strftime(timestr, sizeof(timestr), "%H:%M:%S", ltime);

			va_list ap;
			va_start(ap, text);
			vsprintf(buffer, text, ap);
			va_end(ap);
			DWORD thread = GetCurrentThreadId();
			fs_ << timestr << ": " << " [tid: "<< thread << "]" << buffer;
			fs_.flush();
			ReleaseMutex(logMutex);
#endif
		}

		void Log::slog(const char* text, ...) {
			WaitForSingleObject(logMutex, INFINITE);
			if(!is_init_) init("sfuck.log");
			char buffer[MAX_CHAR];

			char timestr[30];

			tv = time(0);
			tm* ltime = localtime(&tv);
			strftime(timestr, sizeof(timestr), "%H:%M:%S", ltime);

			va_list ap;
			va_start(ap, text);
			vsprintf(buffer, text, ap);
			va_end(ap);
			DWORD thread = GetCurrentThreadId();
			fs_ << timestr << ": " <<" [tid: "<<thread<<"]"<< buffer;
			fs_.flush();
			ReleaseMutex(logMutex);
		}


		void Log::logscreen(const char * text, ...){
			WaitForSingleObject(logMutex, INFINITE);
			char buffer[MAX_CHAR];
			char timestr[30];

			tv = time(0);
			tm  * ltime = localtime(&tv);
			strftime(timestr, sizeof(timestr), "%H:%M:%S", ltime);

			va_list ap;
			va_start(ap, text);
			vsprintf(buffer, text, ap);
			va_end(ap);

			cerr << timestr << ": " << buffer;
			cerr.flush();
			ReleaseMutex(logMutex);
		}

	}
}