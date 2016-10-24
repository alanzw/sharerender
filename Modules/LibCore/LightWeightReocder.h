#ifndef __LIGHTWEIGHTRECORDER_H__
#define __LIGHTWEIGHTRECORDER_H__
// this is for the high performance logger
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_LOG_BUFFER_SIZE (1024 * 10)
namespace cg{
	namespace core{
		class LightWeightRecorder{
			int bufferSize;

			int curLogSize;
			char * curPoint;
			char * logBuffer;
			char * temBuffer;
			FILE * file;


		public:
			LightWeightRecorder(char * prefix, int bufferSize = MAX_LOG_BUFFER_SIZE);
			~LightWeightRecorder();

			void log(char * format, ...);
			void log(char * text, int len);
			bool init(char * prefix);
			bool close();
			bool flush(bool force);
		};
	}
}

#endif