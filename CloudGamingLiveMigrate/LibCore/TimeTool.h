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

	}
}
#endif