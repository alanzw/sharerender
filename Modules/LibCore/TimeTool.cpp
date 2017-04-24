#include "TimeTool.h"


cg::core::DelayRecorder * cg::core::DelayRecorder::delayRecorder = NULL;

static u_int64_t
	filetime_to_unix_epoch(const FILETIME *ft) {
		u_int64_t res = (u_int64_t)ft->dwHighDateTime << 32;
		res |= ft->dwLowDateTime;
		res /= 10;                   /* from 100 nano-sec periods to usec */
		res -= DELTA_EPOCH_IN_USEC;  /* from Win epoch to Unix epoch */
		return (res);
}

int
	cg::core::getTimeOfDay(struct ::timeval *tvv, void *tz) {
		FILETIME  ft;
		u_int64_t tim;
		if (!tvv) {
			//errno = EINVAL;
			return (-1);
		}
		GetSystemTimeAsFileTime(&ft);
		tim = filetime_to_unix_epoch(&ft);
		tvv->tv_sec = (long)(tim / 1000000L);
		tvv->tv_usec = (long)(tim % 1000000L);
		return (0);
}

int
	cg::core::usleep(long long waitTime) {
#if 0
		LARGE_INTEGER t1, t2, freq;
#else
		struct ::timeval t1, t2;
#endif
		long long ms, elapsed;
		if (waitTime <= 0)
			return 0;
#if 0
		QueryPerformanceCounter(&t1);
		QueryPerformanceFrequency(&freq);
		if (freq.QuadPart == 0) {
			// not supported
			Sleep(waitTime / 1000);
			return 0;
		}
#else
		getTimeOfDay(&t1, NULL);
#endif
		// Sleep() may be fine
		ms = waitTime / 1000;
		waitTime %= 1000;
		if (ms > 0) {
			Sleep(ms);
		}
		// Sleep for the rest
		if (waitTime > 0) do {
#if 0
			QueryPerformanceCounter(&t2);

			elapsed = 1000000.0 * (t2.QuadPart - t1.QuadPart) / freq.QuadPart;
#else
			getTimeOfDay(&t2, NULL);
			elapsed = tvdiff_us(&t2, &t1);
#endif
		} while (elapsed < waitTime);
		//
		return 0;
}

long long cg::core::tvdiff_us(struct ::timeval *tv1, struct ::timeval *tv2){
	struct timeval delta;
	delta.tv_sec = tv1->tv_sec - tv2->tv_sec;
	delta.tv_usec = tv1->tv_usec - tv2->tv_usec;
	if (delta.tv_usec < 0){
		delta.tv_sec--;
		delta.tv_usec += 1000000;

	}
	return 1000000LL * delta.tv_sec + delta.tv_usec;
}

long long	/* return microsecond */
	cg::core::pcdiff_us(LARGE_INTEGER t1, LARGE_INTEGER t2, LARGE_INTEGER freq) {
		return 1000000LL * (t1.QuadPart - t2.QuadPart) / (1+freq.QuadPart);
}

long long cg::core::usleep(long long interval, struct ::timeval *ptv) {
	long long delta;
	struct ::timeval tv;
	if(ptv != NULL) {
		getTimeOfDay(&tv, NULL);
		delta = tvdiff_us(&tv, ptv);
		if(delta >= interval) {
			usleep(1);
			return -1;
		}
		interval -= delta;
	}
	usleep(interval);
	return 0LL;
}
