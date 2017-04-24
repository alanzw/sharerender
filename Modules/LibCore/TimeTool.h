#ifndef __TIMETOOL_H__
#define __TIMETOOL_H__
// the time tool used in the whole solution
//#include <time.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <WinSock2.h>

#define DELTA_EPOCH_IN_USEC	11644473600000000Ui64

typedef unsigned __int64 u_int64_t;
namespace cg{
	namespace core{

		struct timespec{
			time_t tv_sec;
			long tv_nsec;
		};

		long long pcdiff_us(LARGE_INTEGER t1, LARGE_INTEGER t2, LARGE_INTEGER freq);
		long long tvdiff_us(struct ::timeval *tv1, struct ::timeval *tv2);
		long long usleep(long long interval, struct ::timeval *ptv);
		int getTimeOfDay(struct ::timeval *tvv, void *tz);
		int usleep(long long waitTime);

		inline unsigned __int64 GetCycleCount(void){
			_asm _emit 0x0F
			_asm _emit 0x31
		}

		inline void machine_pause (__int32 delay ) {
			_asm 
			{
				mov eax, delay
TIMEL1: 
				pause
					add eax, -1
					jne TIMEL1  
			}
			return;
		}

		class BTimer{
		public:
			virtual void Start(void) = 0;
			virtual unsigned __int64 Stop(void) = 0;
			virtual unsigned __int64 getOverhead(void) = 0;
			virtual unsigned int getFreq(void) = 0;
		};
		class PTimer: public BTimer{
			LARGE_INTEGER m_startcycle;
			unsigned __int64 m_overhead;
			LARGE_INTEGER freq;
			bool startTimer;   // true when started and never start twice
		public:
			PTimer(): startTimer(false){
				m_overhead = 0;
				QueryPerformanceFrequency(&freq);
				Start();
				m_overhead = Stop();
			}
			virtual void Start(void){
				startTimer = false;
				if(!startTimer){
					QueryPerformanceCounter(&m_startcycle);
					startTimer = true;
				}
			}
			virtual unsigned __int64 Stop(void){
				LARGE_INTEGER m_endcycle;
				QueryPerformanceCounter(&m_endcycle);
				unsigned __int64 ret = m_endcycle.QuadPart - m_startcycle.QuadPart - m_overhead;
				m_startcycle = m_endcycle;
				if(startTimer)
					startTimer = false;

				return ret;
			}
			virtual unsigned __int64 getOverhead(){ return m_overhead; }
			virtual unsigned int getFreq(){ return (int)freq.QuadPart; }
		};

		class KTimer:public BTimer{
			unsigned __int64 m_startcycle;
			unsigned __int64 m_overhead;
			LARGE_INTEGER freq;
		public:
			KTimer(void){
				m_overhead = 0;
				QueryPerformanceFrequency(&freq);
				Start();
				m_overhead = Stop();
			}
			virtual void Start(void){
				m_startcycle = GetCycleCount();
			}
			virtual unsigned __int64 Stop(void){
				return GetCycleCount() - m_startcycle - m_overhead;
			}
			virtual unsigned __int64 getOverhead(){ return m_overhead; }
			virtual unsigned int getFreq(){ return (int)freq.QuadPart; }
		};
		class DelayRecorder{
			PTimer * timer;

			// for server
			float systemProcessDelay;
			float encodingDelay;
			float renderDelay;  // hook F11 to present
			bool inputArrived;

			// for client
			bool sign;   // if F11 is pressed
			float totalDelay;
			float beforeDisplay;
			float displayDelay;  // set from outside


			DelayRecorder(): systemProcessDelay(0.0), encodingDelay(0.0), inputArrived(false), sign(false), totalDelay(0.0), displayDelay(0.0){
				timer = new PTimer();
			}

			static DelayRecorder * delayRecorder;
		public:
			~DelayRecorder(){
				if(timer){
					delete timer;
					timer = NULL;
				}
			}
			inline float getRenderDelay(){ return renderDelay; }
			inline float getSystemProcessDelay(){ return systemProcessDelay; }
			inline float getEncodingDelay(){ return encodingDelay; }
			inline float getTotalDelay(){ return totalDelay; }
			inline float getDisplayDelay(){ return displayDelay; }
			inline float getBeforeDisplay(){ return beforeDisplay; }

			inline bool isInputArrive(){ return inputArrived; }
			inline bool isSigned(){ return sign; }

			static DelayRecorder * GetDelayRecorder(){
				if(!delayRecorder){
					delayRecorder = new DelayRecorder();
				}
				return delayRecorder;
			}

			// for server
			void setInputArrive(){
				inputArrived = true;
				timer->Start();
			}

			// for server
			void keyTriggered(){
				int systemInterval = timer->Stop();
				systemProcessDelay = (1000.0 * systemInterval)/ timer->getFreq();
			}
			// for logic server
			void renderEnd(){
				int renderInterval = timer->Stop();
				renderDelay = 1000.0 * renderInterval / timer->getFreq();
				inputArrived = false;
			}

			// for render server only
			void startEndcode(){
				sign = true;
				timer->Start();
			}

			// for server when encoding
			void encodeEnd(){
				int encodeInterval = timer->Stop();
				encodingDelay = 1000.0 * encodeInterval / timer->getFreq();
				inputArrived = false;
			}
			// for client
			void setDisplayDelay(float val){ displayDelay = val; }

			// for client, start to count when get F11 in clinet ctrl
			void startDelayCount(){ sign = true; timer->Start(); }
			// for client, to query time before display
			void startToDisplay(){
				int delayInterval = timer->Stop();
				beforeDisplay = delayInterval * 1000.0 / timer->getFreq();
			}

			void displayed(){
				int delayInterval = timer->Stop();
				displayDelay = delayInterval * 1000.0 / timer->getFreq();

				totalDelay = beforeDisplay + displayDelay;

				sign = false;
			}

		};

	}
}
#endif