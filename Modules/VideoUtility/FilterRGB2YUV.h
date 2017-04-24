#ifndef __FILTERRGB2YUV_H__
#define __FILTERRGB2YUV_H__
#include "avcodeccommon.h"
#include "../LibCore/TimeTool.h"

#define POOLSIZE 8


#define SHARE_CONVENTER

namespace cg{
	class Filter : public core::CThread{
		static std::map<void *, bool> initialized;
		static HANDLE filterMutex;
		static std::map<std::string, Filter*> filterMap;

		int outputW;
		int outputH;
		static HANDLE initMutex;
		HANDLE thread;
		DWORD threadId;

		pipeline * srcPipe, * dstPipe;   // must be set ahread of time
		std::string myname;

		bool running;
		bool inited;

		PixelFormat outputFormat;
		struct SwsContext * swsCtx;


		CRITICAL_SECTION fmtSection;

		HANDLE cond, condMutex;
		UINT convertTime;
		cg::core::PTimer * pTimer;

	public:

		float getConvertTime(){

			return (float)1000.0 * convertTime / pTimer->getFreq();
		}

		virtual ~Filter();
		Filter();
		static void Release();

		int init(char * sourcePipeName, char * filterPipeName);
		int init(pipeline * source, char * filterPipeName);

		int init(int ih, int iw, int oh, int ow);

		int getW(){ return outputW; }
		int getH(){ return outputH; }
		void setW(int w){outputW = w;}
		void setH(int h){outputH = h;}

		bool registerSourcePipe(pipeline * pipe){
			if(srcPipe == NULL){
				srcPipe = pipe;

				// create the dst pipeline for filter according to srcPipe
				if(!initDestinationPipeline()){
					return false;
				}
				return true;
			}
			else{
				return false;
			}
		}

		bool initDestinationPipeline();

		bool registerFilterPipe(pipeline * p){
			if(dstPipe == NULL){
				dstPipe = p;
				return true;
			}
			else
				return false;
		}
		inline pipeline * getSrcPipe(){ return srcPipe; }
		inline pipeline * getFilterPipe(){ return dstPipe; }
		inline bool isInited(){ return inited; }

		PixelFormat getOutputFormat();
		void setOutputFormat(PixelFormat fmt);

		// manage
		static int do_register(const char * provider, Filter * filter);
		static void do_unregister(const char * provider);
		static Filter * lookup(const char * provider);
		const char * name();
		inline bool isRunning(){ return running; }
		DWORD StartFilterThread(LPVOID arg);


		virtual BOOL stop();
		virtual BOOL run();
		virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
		virtual void onQuit();
		virtual BOOL onThreadStart();


		void registerEvent(){ srcPipe->client_register(ccg_gettid(), cond); }

	};

	/// the thread proc for the filter
	DWORD WINAPI FilterThreadProc(LPVOID param);

}
#endif
