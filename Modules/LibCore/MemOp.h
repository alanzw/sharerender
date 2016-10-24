#ifndef __MEMOP_H__
#define __MEMOP_H__

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
// define the memroy operations

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


#endif