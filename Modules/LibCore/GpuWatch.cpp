
#include "GpuWatch.h"
#include "InfoRecorder.h"

namespace cg{
	namespace core{

#if 0
		//初始化静态变量
		int GpuWatch::gpuUtilization=-1;
		int GpuWatch::gpuTemp=-1;
		int GpuWatch::graNum=0;	//显卡数量
		//GpuInterface* GpuWatch::pInterface=NULL;
		std::string GpuWatch::graInfo="";
		int GpuWatch::type=-1;
#else
		GpuWatch * GpuWatch::gpuWatch = NULL;

#endif

		ADL_MAIN_CONTROL_CREATE AMDInterface::AdlMainControlCreate=NULL;
		ADL_MAIN_CONTROL_REFRESH AMDInterface::AdlMainControlRefresh=NULL;
		ADL_OVERDRIVE5_TEMPERATURE_GET AMDInterface::AdlOverDrive5TemperatureGet=NULL;
		ADL_OVERDRIVE5_CURRENTACTIVITY_GET AMDInterface::AdlOverDrive5CurrentActivityGet=NULL;

		/////////////////////////////////////////////
		//获取GPU温度、利用率信息的类
		GpuWatch::GpuWatch(void): isInit(false), gpuInterface(NULL)
		{
			InitGpuInterface();
		}

		GpuWatch::~GpuWatch(void)
		{
		}

		//初始化显卡
		bool GpuWatch::InitGpuInterface()
		{
			LPDIRECT3D9 pD3D=NULL;
			D3DADAPTER_IDENTIFIER9 di;
			pD3D=Direct3DCreate9(D3D_SDK_VERSION);	//创建D3D对象
			if(graNum == 0)
				graNum = pD3D->GetAdapterCount();	//获取显卡数量
			//不支持多显卡！！！！
#if 0
			if(graNum!=1)
			{
				/*pInterface = new GpuInterface;*/
				//infoRecorder->logError("[NvdiaInterface]: get adapter count failed, count:%d.\n", graNum);
				type=-1;
				return false;
			}
#endif
			//获得显卡信息
			HRESULT h=pD3D->GetAdapterIdentifier(D3DADAPTER_DEFAULT,0,&di);
			if(FAILED(h))
			{

				type=-1;
				return false;
			}
			//显卡描述信息
			graInfo = di.Description;

			pD3D->Release();	//释放D3D对象
			ChangeToLower(graInfo);
			
			//amd卡
			if( (graInfo.find("amd")!=-1) || (graInfo.find("ati")!=-1) ){
				/*pInterface = new AMDInterface;*/
				type=0;
				// create the  interface
				gpuInterface = new AMDInterface();
			}
			//N卡
			else {
				if( graInfo.find("nvidia")!=-1 ){
					type=1;
					// create the interface
					gpuInterface = new NvidiaInterface();
					//gpuInterface = new NvApiInterface();
				}
				else
				{
					/*pInterface = new GpuInterface;*/
					infoRecorder->logError("[GpuWatch]: type is unknown.\n");
					type=-1;
					return false;
				}
			}

			return true;
		}

		void GpuWatch::ChangeToLower(std::string &str)
		{
			auto iter=str.begin();
			while(iter!=str.end())
			{
				if((*iter>='A')&&(*iter<='Z'))
					*iter += 32;	//大写转换成小写
				iter++;
			}
		}

		//观测显卡利用率
		int GpuWatch::GetGpuUsage()
		{
			if(type==-1){
				gpuUtilization=-1;
			}

			if(gpuInterface){
				gpuUtilization = gpuInterface->GetGpuUsage();
			}
			return gpuUtilization;
		}

		//观测显卡温度
		int GpuWatch::GetGpuTemp()
		{
			/*if(pInterface)
			gpuTemp = pInterface->GetGpuTemp();
			else
			gpuTemp = -1;*/
			if(type == -1)
				gpuTemp = -1;


			if(gpuInterface){
				gpuTemp = gpuInterface->GetGpuTemp();
			}

			return gpuTemp;
		}

		//获取显卡信息
		void GpuWatch::GetGpuInformation(char *buf,int size)
		{
			if(type == -1)
				return;
			if(!graInfo.size())
				return;
			//显卡信息占的字节大于size
			if(graInfo.size()>size)
				return;
			strcpy_s(buf,size,graInfo.c_str());
		}

		AMDInterface::AMDInterface(): isInit(false)
		{
			InitAdlApi();
		}

		AMDInterface::~AMDInterface()
		{

		}

		//申请内存空间
		void* _stdcall AMDInterface::ADLMainMemoryAlloc(int size)
		{
			void *pbuf=NULL;
			pbuf = malloc(size);
			return pbuf;
		}

		//释放内存空间
		void _stdcall AMDInterface::ADLMainMemoryFree(void **buf)
		{
			if(*buf)
			{
				free(*buf);
				*buf=NULL;
			}
		}

		bool AMDInterface::InitInterface(){
			return InitAdlApi();
		}

		//初始化A卡
		bool AMDInterface::InitAdlApi()
		{
			//32位操作系统加载的库文件
			HMODULE hDLL=LoadLibraryA("atiadlxx.dll");
			if(!hDLL)
				hDLL=LoadLibraryA("atiadlxy.dll");	//64位操作系统
			if(!hDLL)
				return false;
			//获取函数入口地址
			AdlMainControlCreate = (ADL_MAIN_CONTROL_CREATE)GetProcAddress(hDLL,"ADL_Main_Control_Create");
			AdlOverDrive5TemperatureGet = (ADL_OVERDRIVE5_TEMPERATURE_GET)GetProcAddress(hDLL,"ADL_Overdrive5_Temperature_Get");
			AdlOverDrive5CurrentActivityGet = (ADL_OVERDRIVE5_CURRENTACTIVITY_GET)GetProcAddress(hDLL,"ADL_Overdrive5_CurrentActivity_Get");
			AdlMainControlRefresh = (ADL_MAIN_CONTROL_REFRESH)GetProcAddress(hDLL,"ADL_Main_Control_Refresh");

			//获取入口地址失败，初始化失败
			if(!AdlMainControlCreate || !AdlOverDrive5TemperatureGet || !AdlOverDrive5CurrentActivityGet || !AdlMainControlRefresh)
			{
				return false;
			}
			//调用ADLMainMemoryAlloc分配空间
			if(AdlMainControlCreate(ADLMainMemoryAlloc,1)!=ADL_OK)
			{
				return false;
			}

			if(AdlMainControlRefresh()!=ADL_OK)
			{
				return false;
			}
			isInit = true;
			return true;	
		}

		//获取GPU利用率
		int AMDInterface::GetGpuUsage()
		{
			int utilization = -1;
			ADLPMActivity activity;

			if(isInit==false)
				return -1;

			if(AdlOverDrive5CurrentActivityGet(0,&activity)!=ADL_OK)
			{
				utilization = -1;
			}
			utilization = activity.iActivityPercent;
			return utilization;
		}

		//获取GPU温度
		int AMDInterface::GetGpuTemp()
		{
			int temper;
			ADLTemperature tempStruct;
			if(isInit == false)
				return -1;
			if(AdlOverDrive5TemperatureGet(0,0,&tempStruct) != ADL_OK)
				return -1;
			temper=tempStruct.iTemperature/1000;
			return temper;
		}

		//NvidiaInterface类函数
		NvidiaInterface::NvidiaInterface():isInit(false),phys(NULL)
		{
			InitNvApi();	//初始化N卡
		}

		NvidiaInterface::~NvidiaInterface()
		{

		}

		bool NvidiaInterface::InitInterface(){
			return InitNvApi();
		}

		//初始化N卡
		bool NvidiaInterface::InitNvApi()
		{
			NvU32 nv;
			if(NvAPI_Initialize() != NVAPI_OK)
				return false;
			isInit = true;

			//获取物理GPU信息
			if(NvAPI_EnumPhysicalGPUs(&phys,&nv)!=NVAPI_OK)
			{
				phys=NULL;	//系统中无N卡
			}
			
			return true;
		}

		//获取GPU利用率
		int NvidiaInterface::GetGpuUsage()
		{
			NV_GPU_DYNAMIC_PSTATES_INFO_EX nvInfo;

			if ( (isInit == false) || (phys == NULL) ){
				return -1;
			}
			//版本信息
			nvInfo.version = NV_GPU_DYNAMIC_PSTATES_INFO_EX_VER;
			//获取GPU使用率失败
			NvAPI_Status status = NVAPI_OK;
			if((status = NvAPI_GPU_GetDynamicPstatesInfoEx(phys,&nvInfo))!=NVAPI_OK || nvInfo.utilization[0].bIsPresent ==0)
			{
				//NvAPI_ShortString str = {0};
				//NvAPI_GetErrorMessage(status, str);
				infoRecorder->logError("[NvidiaInterface]::GetGpuUsage(), NvAPI_GPU_GetDynamicPstatesInfoEx ret:%d, is present:%d.\n", status, nvInfo.utilization[0].bIsPresent);
				return -1;
			}
			//返回使用率
			infoRecorder->logTrace("[NvidiaInterface]: 0:%d, 1:%d, 2:%d, 3:%d, 4:%d, 5:%d, 6:%d, 7:%d.\n", nvInfo.utilization[0].percentage, nvInfo.utilization[1].percentage, nvInfo.utilization[2].percentage, nvInfo.utilization[3].percentage, nvInfo.utilization[4].percentage, nvInfo.utilization[5].percentage, nvInfo.utilization[6].percentage, nvInfo.utilization[7].percentage);
			return static_cast<int>(nvInfo.utilization[0].percentage);
		}

		//获取GPU温度
		int NvidiaInterface::GetGpuTemp()
		{
			NV_GPU_THERMAL_SETTINGS thermal;
			if( (isInit == false) || (phys==NULL) )
				return -1;
			thermal.version=NV_GPU_THERMAL_SETTINGS_VER;
			if(NvAPI_GPU_GetThermalSettings(phys,0,&thermal)!=NVAPI_OK)
			{
				return -1;
			}
			//返回温度
			return static_cast<int>(thermal.sensor[0].currentTemp);
		}



		// NvApiInterface
		NvApiInterface::NvApiInterface(): isInit(false), data0(0), call0(NULL), call1(NULL), call2(NULL){
			InitInterface();
		}
		NvApiInterface::~NvApiInterface(){}
		bool NvApiInterface::InitInterface(){

			HANDLE handle = LoadLibraryA("nvapi.dll");
			void * func = GetProcAddress((HMODULE)handle, "nvapi_QueryInterface");
			call0 = ((void *(*)(unsigned int))func)(22079528u);
			call1 = ((void *(*)(unsigned int))func)(3853292063u);
			call2 = ((void *(*)(unsigned int))func)(412753887u);

			int ret = ((int(*)())call0)();
			int data1 = 1;
			ret = ((int (*)(void *, void *))call1)(&data0, &data1);

			return true;
		}

		int NvApiInterface::GetGpuUsage(){
			buffer[0] = 65672;
			int ret = ((int (*)(int , int *))call2)(data0, buffer);
			if(ret != 0){
				infoRecorder->logError("[NvApiInterface]: get usage ret:%d.\n", ret);
			}
			int gpuLoad = buffer[3];
			return gpuLoad;
		}

		int NvApiInterface::GetGpuTemp(){
			return -1;
		}
	}


}