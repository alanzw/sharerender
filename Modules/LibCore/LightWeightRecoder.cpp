#include "lightweightreocder.h"
// this is for the high performance logger.
//#define DEBUG_

#ifdef DEBUG_
#include "log.h"

#endif

namespace cg{
	namespace core{
		// constructor and destructor

		LightWeightRecorder::LightWeightRecorder(char * prefix, int bufferSize){
			this->bufferSize = bufferSize;
			logBuffer = NULL;
			curPoint = NULL;
			file = NULL;
			curLogSize = 0;

			temBuffer = NULL;

			if (!init(prefix)){
				// error
			}
		}

		LightWeightRecorder::~LightWeightRecorder(){
			if (file){
				flush(true);
				close();
				file = NULL;
			}
			if (logBuffer){

				free(logBuffer);
				logBuffer = NULL;
				bufferSize = 0;
				curPoint = NULL;
				curLogSize = 0;
			}

			if (temBuffer){
				free(temBuffer);
				temBuffer = NULL;
			}

		}

		bool LightWeightRecorder::init(char * prefix){
			logBuffer = (char *)malloc(sizeof(char)* bufferSize);
			curPoint = logBuffer;

			temBuffer = (char *)malloc(sizeof(char)* 512);
			memset(logBuffer, 0, bufferSize);
			memset(temBuffer, 0, 512);

			// format the log name
			char filename[128];
			sprintf(filename, "%s.log", prefix);
			file = fopen(filename, "w");
			if (file == NULL){
				// error
				return false;
			}

			return true;

		}
		void LightWeightRecorder::log(char * text, int len){
			//int n = strlen(text);
#ifdef DEBUG_
			infoRecorder->logError("[LightRecorder]: %s. size:%d, curPoint:%p, curSize:%d, logBuffer:%p\n", text, len, curPoint, curLogSize, logBuffer);
#endif

			if(this->curLogSize + len >= MAX_LOG_BUFFER_SIZE){
#ifdef DEBUG_
				infoRecorder->logError("[Error]: should not be here.\n");
#endif
				flush(true);
			}
			// copy the log
			strcpy(this->curPoint, text);
			this->curLogSize += len;
			this->curPoint += len;
		}

		void LightWeightRecorder::log(char * format, ...){
			va_list ap;
			va_start(ap, format);
			int n = vsprintf(temBuffer, format, ap);
			va_end(ap);

			if (this->curLogSize + n >= MAX_LOG_BUFFER_SIZE){
				flush(true);
			}
			// copy the log
			strcpy(this->curPoint, temBuffer);

			this->curLogSize += n;
			this->curPoint += n;

			flush(true);
		}
		bool LightWeightRecorder::close(){
			if (file){
				fclose(file);
				file = NULL;
			}
			return true;
		}
		bool LightWeightRecorder::flush(bool force){
			if (force && curLogSize > 0){
				fprintf(file, "%s", this->logBuffer);
				memset(logBuffer, 0, MAX_LOG_BUFFER_SIZE);
				curLogSize = 0;
				curPoint = logBuffer;
				fflush(file);
			}
			return true;
		}

	}
}