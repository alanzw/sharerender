#pragma once
#ifndef UTILITY_H
#define UTILITY_H
#include "WatcherDefines.h"



////基类
class GpuInterface
{
public:
	virtual int GetGpuUsage(){	return -1;	};
	virtual int GetGpuTemp(){	return -1;	};
	virtual bool InitInterface(){return true;}
	~GpuInterface(){}
};
namespace cg{
	namespace core{

		//获取显卡利用率和温度信息的类
		class GpuWatch
		{
		public:
			
			~GpuWatch(void);
			//初始化显卡
			bool InitGpuInterface(void);
			int GetGpuUsage();	//获取GPU利用率
			int GetGpuTemp();	//获取GPU温度
			void GetGpuInformation(char *buf,int size);	//获取GPU描述信息
			int gpuUtilization , gpuTemp , graNum;	//gpu利用率和温度,显卡数目
			/*static GpuInterface* pInterface;*/
			std::string graInfo;	//显卡信息
			void ChangeToLower(std::string &str);
			int type;

			static GpuWatch * GetGpuWatch(){
				if(!gpuWatch){
					gpuWatch = new GpuWatch();
				}
				return gpuWatch;
			}


		private:
			bool isInit;	//显卡是否已经初始化
			GpuInterface * gpuInterface;
			GpuWatch(void);

			static GpuWatch * gpuWatch;
		};


		//AMD显卡相关的类,继承GpuInterface类
		class AMDInterface: public GpuInterface
		{
		public:
			AMDInterface(void);	//构造函数
			virtual ~AMDInterface();
			static void* _stdcall ADLMainMemoryAlloc(int size);	//申请空间
			static void _stdcall ADLMainMemoryFree(void **buf);	//释放空间

			virtual int GetGpuUsage();	//获取GPU负载
			virtual int GetGpuTemp();	//获取GPU温度
			virtual bool InitInterface();

			//定义函数指针
			static ADL_MAIN_CONTROL_CREATE  AdlMainControlCreate;
			static ADL_MAIN_CONTROL_REFRESH AdlMainControlRefresh;
			static ADL_OVERDRIVE5_TEMPERATURE_GET AdlOverDrive5TemperatureGet;
			static ADL_OVERDRIVE5_CURRENTACTIVITY_GET AdlOverDrive5CurrentActivityGet;

		private:
			bool InitAdlApi();	//初始化API
			bool isInit;	//是否已经初始化
		};

		//Nvidia显卡相关的类，继承GpuInterface类
		class NvidiaInterface: public GpuInterface
		{
		public:
			NvidiaInterface(void);	//构造函数
			virtual ~NvidiaInterface();
			virtual int GetGpuUsage();	//获取GPU利用率
			virtual int GetGpuTemp();	//获取GPU温度
			virtual bool InitInterface();
			
		private:
			bool InitNvApi();	//初始化N卡
			bool isInit;
			NvPhysicalGpuHandle phys;
		};

		class NvApiInterface: public GpuInterface{
		public:
			NvApiInterface();
			virtual ~NvApiInterface();
			virtual int GetGpuUsage();
			virtual int GetGpuTemp();
			virtual bool InitInterface();
		private:
			bool isInit;
			int data0;

			// function pointer
			void * call0;
			void * call1;
			void * call2;

			int buffer[1024];
		};

	}
}

#endif
