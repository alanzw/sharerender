#ifndef __CCG_WIN32__
#define __CCG_WIN32__

#include <WinSock2.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <process.h>
#include <list>
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <d3d9.h>
#include <stdio.h>

// ccg headers

#include "AVCodecCommon.h"
#include "../LibCore/CThread.h"


using namespace std;

#define RGBA_SIZE 4 /* in bytes*/
#define EXPROT __declspec(dllexport)

#define	VK_PRETENDED_LCONTROL	0x98	// L_CONTROL key: 162
#define	VK_PRETENDED_LALT	0x99	// L_ALT key: 164

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
//#include <libavformat\avformat.h>
//#include <libavutil\avutil.h>
#ifdef __cplusplus
}
#endif

//#pragma comment(lib, "libliveMedia.lib")
//#pragma comment(lib, "libgroupsock.lib")
////#pragma commnet(lib, "libgroupsock.d.lib")
//#pragma comment(lib, "libBasicUsageEnvironment.lib")
//#pragma comment(lib, "libUsageEnvironment.lib")
//#pragma comment(lib, "SDL2.lib")
//#pragma comment(lib, "SDL2main.lib")
//#pragma comment(lib, "SDL2_ttf.lib")
//#pragma comment(lib, "swscale.lib")
//#pragma comment(lib, "swresample.lib")
//#pragma comment(lib, "postproc.lib")
//#pragma comment(lib, "avdevice.lib")
//#pragma comment(lib, "avfilter.lib")
//#pragma comment(lib, "avformat.lib")
//#pragma comment(lib, "avcodec.lib")
//#pragma comment(lib, "avutil.lib")
////#pragma comment(lib, ".lib")
//#pragma comment(lib, "d3d9.lib")
//#pragma comment(lib, "d3dx9.lib")

struct WIN32IMAGE {
	int width;
	int height;
	int bytes_per_line;
};


//int read(SOCKET fd, void *buf, int count);
//int write(SOCKET fd, const void *buf, int count);
//int close(SOCKET fd);

char *dlerror();
void ccg_win32_fill_bitmap_info(BITMAPINFO *pinfo, int w, int h, int bitsPerPixel);
long long pcdiff_us(LARGE_INTEGER t1, LARGE_INTEGER t2, LARGE_INTEGER freq);
int read(SOCKET fd, void *buf, int count);
int write(SOCKET fd, const void *buf, int count);
int closeSock(SOCKET fd);

long long tvdiff_us(struct timeval *tv1, struct timeval *tv2);
long long ccg_usleep(long long interval, struct timeval *ptv);
int getTimeOfDay(struct timeval *tvv, void *tz);
int gettimeofday(struct timeval * tv, void * tz);
int usleep(long long waitTime);

//int	ga_malloc(int size, void **ptr, int *alignment);
long	ccg_gettid();
void	ccg_dump_codecs();
int		ccg_init(const char *config, const char *url);
void	ccg_deinit();
void	ccg_openlog();
void	ccg_closelog();
long	ccg_atoi(const char *str);
struct ccgRect * ccg_fillrect(struct ccgRect *rect, int left, int top, int right, int bottom);
int		ccg_crop_window(struct ccgRect *rect, struct ccgRect **prect);
void	ccg_backtrace();
void	ccg_dummyfunc();

#if 0
const char * ccg_lookup_mime(const char *key);
const char ** ccg_lookup_ffmpeg_decoders(const char *key);
enum AVCodecID ccg_lookup_codec_id(const char *key);
#endif

typedef unsigned(__stdcall * PTHREAD_START)(void *);
#define chBEGINTHREADEX(psa, cbStack, pfnStartAddr, \
	pvParam, fdwCreate, pdwThreadID) \
	((HANDLE)_beginthreadex(\
	(void *)(psa), \
	(unsigned)(cbStack), \
	(PTHREAD_START)(pfnStartAddr), \
	(void *)(pvParam), \
	(unsigned)(fdwCreate), \
	(unsigned *)(pdwThreadID)))

#ifndef bzero
#define bzero(m, n) ZeroMemory(m, n)
#endif
#ifndef bcopy
#define	bcopy(s,d,n)		CopyMemory(d, s, n)
#endif
#ifndef strcasecmp
#define strcasecmp(a, b) _stricmp(a, b)
#endif
#ifndef strncasecmp
#define strncasecmp(a, b, n) _strnicmp(a, b, n)
#endif

#ifndef strtok_r
#define strtok_r(s,d,v) strtok_s(s,d,v)
#endif

#ifndef gmtime_r
#define gmtime_r(pt, ptm) gmtime_s(ptm, pt)
#endif
// snprintf
#ifndef snprintf
#define	snprintf(b,n,f,...)	_snprintf_s(b,n,_TRUNCATE,f,__VA_ARGS__)
#endif
int ve_malloc(int size, void **ptr, int *alignment);

#endif