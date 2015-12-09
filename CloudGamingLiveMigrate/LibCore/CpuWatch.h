//观测CPU信息的类
#pragma once
#include "WatcherDefines.h"

namespace cg{
	namespace core{

		class CpuWatch{
		public:
			CpuWatch(void);
			CpuWatch(char *procName);	//构造函数
			~CpuWatch(void);
			//时间转换函数
			static int64_t FileTimeToDouble(FILETIME &t);

			//进程CPU利用率相关
			double GetProcessCpuUtilization();	//获取进程CPU利用率
			double GetProcessCpuUtilization(HANDLE proc);

			//系统CPU利用率相关
			double GetSysCpuUtilization();	//获取系统CPU利用率
			unsigned int GetProcessorNum();	//获取CPU个数
			int64_t GetCpuFreq();	//获取CPU频率
			//void GetSysCpuUtilizationLoop(double &result,int sleepTime);	//周期性地获取系统CPU利用率

			//进程操作相关
			HANDLE getProcH(char *procName);	//根据进程名获取进程号
			HANDLE GetProcH();	
			inline void SetHandle(HANDLE handle){ procHandle = handle; }
			void SetProcName(char *input);			//设置进程名

		private:
			int64_t last_sys_kernel;
			int64_t last_sys_user;
			int64_t last_sys_idle;

			int64_t last_time;
			int64_t last_sys_time;
			static unsigned int processor_num;	//CPU个数
			char processName[MAXSIZE];	//进程名
			HANDLE procHandle;
		};

	}
}