#ifndef __TIMEHELPER_H__
#define __TIMEHELPER_H__

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

class TimeHelper{
	LARGE_INTEGER freq;
	LARGE_INTEGER start, end;
	
public:
	void init(){
		QueryPerformanceFrequency(&freq);
	}

	void startCount(){
		QueryPerformanceCounter(&start);
	}
	void endCount(){
		QueryPerformanceCounter(&end);
	}

	double getMS(){
		return (end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;
	}
	double getUS(){
		return (end.QuadPart - start.QuadPart) * 1000000.0/ freq.QuadPart;
	}
	double getS(){
		return (end.QuadPart - start.QuadPart) * 1.0 / freq.QuadPart;
	}


};

#endif