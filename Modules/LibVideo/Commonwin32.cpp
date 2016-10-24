#include "Commonwin32.h"
#include "VideoCommon.h"
#include "../LibCore/Log.h"

#define DELTA_EPOCH_IN_USEC	11644473600000000Ui64

typedef unsigned __int64 u_int64_t;

static u_int64_t
filetime_to_unix_epoch(const FILETIME *ft) {
	u_int64_t res = (u_int64_t)ft->dwHighDateTime << 32;
	res |= ft->dwLowDateTime;
	res /= 10;                   /* from 100 nano-sec periods to usec */
	res -= DELTA_EPOCH_IN_USEC;  /* from Win epoch to Unix epoch */
	return (res);
}
int gettimeofday(struct timeval * tv, void * tz){
	FILETIME  ft;
	u_int64_t tim;
	if (!tv) {
		//errno = EINVAL;
		return (-1);
	}
	GetSystemTimeAsFileTime(&ft);
	tim = filetime_to_unix_epoch(&ft);
	tv->tv_sec = (long)(tim / 1000000L);
	tv->tv_usec = (long)(tim % 1000000L);
	return (0);
}
int
getTimeOfDay(struct timeval *tvv, void *tz) {
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
usleep(long long waitTime) {
#if 0
	LARGE_INTEGER t1, t2, freq;
#else
	struct timeval t1, t2;
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

long long tvdiff_us(struct timeval *tv1, struct timeval *tv2){
	struct timeval delta;
	delta.tv_sec = tv1->tv_sec - tv2->tv_sec;
	delta.tv_usec = tv1->tv_usec - tv2->tv_usec;
	if (delta.tv_usec < 0){
		delta.tv_sec--;
		delta.tv_usec += 1000000;

	}
	return 1000000LL * delta.tv_sec + delta.tv_usec;
}

void
ccg_win32_fill_bitmap_info(BITMAPINFO *pinfo, int w, int h, int bitsPerPixel) {
	ZeroMemory(pinfo, sizeof(BITMAPINFO));
	pinfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	pinfo->bmiHeader.biBitCount = bitsPerPixel;
	pinfo->bmiHeader.biCompression = BI_RGB;
	pinfo->bmiHeader.biWidth = w;
	pinfo->bmiHeader.biHeight = h;
	pinfo->bmiHeader.biPlanes = 1; // must be 1
	pinfo->bmiHeader.biSizeImage = pinfo->bmiHeader.biHeight
		* pinfo->bmiHeader.biWidth
		* pinfo->bmiHeader.biBitCount / 8;
	return;
}

long long	/* return microsecond */
pcdiff_us(LARGE_INTEGER t1, LARGE_INTEGER t2, LARGE_INTEGER freq) {
	return 1000000LL * (t1.QuadPart - t2.QuadPart) / freq.QuadPart;
}

int
read(SOCKET fd, void *buf, int count) {
	return recv(fd, (char *)buf, count, 0);
}

int
write(SOCKET fd, const void *buf, int count) {
	return send(fd, (const char*)buf, count, 0);
}

int
closeSock(SOCKET fd) {
	return closesocket(fd);
}


int ve_malloc(int size, void **ptr, int * alignment){
	if ((*ptr = malloc(size + 16)) == NULL)
		return -1;

	*alignment = 16 - (((unsigned)* ptr) & 0x0f);
	return 0;
}

struct ccgRect * ccg_fillrect(struct ccgRect * rect, int left, int top, int right, int bottom){
	if (rect == NULL){
		return NULL;
	}
#define SWAP(a, b) do {int tmp = a; a = b; b = tmp; }while(0);
	if (left > right)
		SWAP(left, right);
	if (top > bottom)
		SWAP(top, bottom);
#undef SWAP
	rect->left = left;
	rect->top = top;
	rect->right = right;
	rect->bottom = bottom;

	rect->width = rect->right - rect->left + 1;
	rect->height = rect->bottom - rect->top + 1;
	rect->linesize = rect->width * RGBA_SIZE;
	rect->size = rect->width * rect->height * RGBA_SIZE;

	if (rect->width <= 0 || rect->height <= 0){
		return NULL;
	}
	return rect;
}

long ccg_gettid(){
	return GetCurrentThreadId();
}
void ccg_dump_codecs(){
	int n, count;
	char buf[8192], *ptr;
	AVCodec * c = NULL;
	n = snprintf(buf, sizeof(buf), "Registered codecs: ");
	ptr = &buf[n];
	count = 0;
	for (c = av_codec_next(NULL); c != NULL; c = av_codec_next(c)){
		n = snprintf(ptr, sizeof(buf)-(ptr - buf), "%s ", c->name);
		ptr += n;
		count++;
	}
	snprintf(ptr, sizeof(buf)-(ptr - buf), "(%d)\n", count);
	
	return;
}

int ccg_init(const char * config, const char * url){
	srand(time(0));
	

	return 0;
}
void ccg_deinit(){}

void ccg_openlog(){}

void ccg_closelog(){}

long ccg_atoi(const char * str){
	char buf[64];
	long val;
	strncpy(buf, str, sizeof(buf));
	val = strtol(buf, NULL, 0);
	return val;
}

int ccg_crop_window(struct ccgRect * rect, struct ccgRect **prect){
	return 0;
}

void ccg_backtrace(){}

void ccg_dummyfunc(){
	swr_alloc_set_opts(NULL, 0, (AVSampleFormat)0, 0, 0, (AVSampleFormat)0, 0, 0, NULL);
}

long long ccg_usleep(long long interval, struct timeval * ptv){
	long long delta;
	struct timeval tv;
	if (ptv != NULL){
		gettimeofday(&tv, NULL);
		delta = tvdiff_us(&tv, ptv);
		if (delta >= interval){
			usleep(1);
			return -1;
		}
		interval -= delta;
	}
	usleep(interval);
	infoRecorder->logTrace("usleep :%d us.\n");
	return 0LL;
}