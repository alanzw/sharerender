#ifndef __TIMETOOL_H__
#define __TIMETOOL_H__
// the time tool used in the whole solution
//#include <time.h>
#include <Windows.h>

struct timespec{
	time_t tv_sec;
	long tv_nsec;
};

long long pcdiff_us(LARGE_INTEGER t1, LARGE_INTEGER t2, LARGE_INTEGER freq);
long long tvdiff_us(struct timeval *tv1, struct timeval *tv2);
long long usleep(long long interval, struct timeval *ptv);
int getTimeOfDay(struct timeval *tvv, void *tz);
int usleep(long long waitTime);

#endif