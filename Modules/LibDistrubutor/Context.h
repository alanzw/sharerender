#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include <event2/event.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/util.h>
#include <event2/listener.h>
#include <iostream>
#include <map>
#include <string>
#include "../LibCore/InfoRecorder.h"

using namespace std;

#ifndef BACKBUFFER_TEST
#define BACKBUFFER_TEST
#endif

#ifndef MAX_RENDER_COUNT
#define MAX_RENDER_COUNT 4
#endif

#ifndef MAX_CMD_LEN
#define MAX_CMD_LEN 50
#endif

#define INTERNAL_PORT 8558

#ifndef MAX_BUFFER_SIZE
#define MAX_BUFFER_SIZE 1024
#endif

#ifndef strncasecmp
#define strncasecmp(a, b, n) _strnicmp(a, b, n)
#endif

namespace cg{

	typedef evutil_socket_t IDENTIFIER;

	enum CMD_OPTION{
		UNKNOWN = -1,
		SETOFFLOAD = 1,
		ENCODEROPTION
	};


	//base context for DisContext and ProcessContext
	enum ContextType{
		UNKNOW_CONTEXT,
		DIS_CONTEXT,
		PROCESS_CONTEXT,
		RENDER_CONTEXT,
		CTRL_CONTEXT
	};

	class BaseContext{
	public:
		evutil_socket_t sock;
		ContextType contextType;
		struct bufferevent * bev;
		sockaddr_in * addr;

		char * cmd;
		char * url;
		char * data;

		char buffer[MAX_BUFFER_SIZE];
		char writeBuffer[MAX_BUFFER_SIZE];

		size_t length;// the length of data

		// for write buffer
		short writeLen;
		char * writeCurPtr;


		string getNetName(){

			return string(inet_ntoa(addr->sin_addr));
		}
		//virtual bool dealCmd() = 0;
		BaseContext(){
			cmd = (char*)malloc(sizeof(char) * MAX_CMD_LEN);
			data = NULL;
			length = 0;
			sock = 0;
			writeLen = 0;
			addr = NULL;
		}
		~BaseContext(){
			if(cmd){
				free(cmd);
				cmd = NULL;
			}
			data = NULL;
			length =0;
			if(data){
				free(data);
				data = NULL;
			}
			if(addr){
				free(addr);
				addr = NULL;
			}
		}
		void setCmd(char * _cmd){
			if(strlen(_cmd) >= MAX_CMD_LEN){
				printf("[BaseContext]: cmd is too long.\n");
			}
			cmd = strcpy(cmd, _cmd);
		}
		void setData(char * _data, size_t _len){
			memset(buffer, 0, MAX_BUFFER_SIZE);
			memcpy(buffer, _data, _len);
			length = _len;
		}
		inline char * getCmd(){ return cmd;}
		void readCmd(){
			memset(cmd, 0, MAX_CMD_LEN);
			char * t = buffer;
			while(*t && *t != '+')
				t++;
			strncpy(cmd, buffer, t - buffer);
			data = t + 1;
		}
		inline char * getData(){ return data; }

		inline void writeCmd(const char * cmd){
			memset(writeBuffer, 0, MAX_BUFFER_SIZE);
			writeLen = 0;
			sprintf(writeBuffer, "%s+", cmd);
			writeLen = strlen(writeBuffer);
			writeCurPtr = writeBuffer + writeLen;
		}
		inline void writeShort(short data){
			memcpy(writeCurPtr, &data, sizeof(short));
			writeCurPtr += sizeof(short);
			writeLen += sizeof(short);
		}
		inline void writeFloat(float data){
			memcpy(writeCurPtr, &data, sizeof(float));
			writeCurPtr += sizeof(float);
			writeLen += sizeof(float);
		}
		inline void writeIdentifier(IDENTIFIER id){
			memcpy(writeCurPtr, &id, sizeof(IDENTIFIER));
			writeCurPtr += sizeof(IDENTIFIER);
			writeLen += sizeof(IDENTIFIER);
		}

		inline void writeData(void * data, int len){
			memcpy(writeCurPtr, data, len);
			writeCurPtr += len;
			writeLen += len;
		}

		inline void writeToNet(int option){
			writeToNet(writeBuffer, writeLen, option);
			writeLen = 0;
			memset(writeBuffer, 0, sizeof(writeBuffer));
		}
		inline void writeToNet(char * data, int len, int option){
			if(send(sock, data, len, 0) == SOCKET_ERROR){
				cg::core::infoRecorder->logError("[BaseContext]: send data failed with:%d\n", WSAGetLastError());
			}
		}

		inline void writeToNet(){
			writeToNet(writeBuffer, writeLen);
			writeLen = 0;
			memset(writeBuffer, 0, sizeof(writeBuffer));
		}
		inline void writeToNet(char * data, int len){
			cg::core::infoRecorder->logTrace("[BaseContext]: send %d data, '%s'.\n", len, data);
#if 0
			if(send(sock, data, len, 0) == SOCKET_ERROR){
				printf("[BaseContext]: send failed with:%d.\n", WSAGetLastError());
			}
#else
			bufferevent_write(bev, (void *)data, len);
#endif

		}
		void readFromNet(char * dst, int &len);

	};

	enum TASK_STATUS{
		INIT = 10,
		ASSIGNED,
		STARTED,
		STOPPED
	};
	enum TASK_MODE{
		MODE_NONE = 20,
		MODE_NO_OFFLOAD,
		MODE_PARTIAL_OFFLOAD,
		MODE_FULL_OFFLOAD
	};

	// run on DisServer
	struct TaskInfo{
		IDENTIFIER id;
		TASK_STATUS status;
		TASK_MODE mode;    // the task mode, no offloading means logic server does everything, partial offload means render help to get the image, full offload means renders do all the rendering

		short portOffset;  // the offset of ports, each service has a port
		//static short offsetBase;

		bool remoteRender; // if render locally, set to false;

		float cpuRequire, gpuRequire;
		string taskName;
		char * path;

		BaseContext * client;

		BaseContext * logicCtx;
		bool logicReady;   // set true when recv logic ready cmd
		bool ctrlReady;    // whether the controller of the task is solved

		short renderCount;
		//short readyCount;
		BaseContext * renderCtx[MAX_RENDER_COUNT];
		bool renderReady[MAX_RENDER_COUNT]; // set true when recv render ready cmd
		bool isSent[MAX_RENDER_COUNT];  // set true when send the command
		BaseContext * toMigrate[MAX_RENDER_COUNT];  // check when receiving RENDER READY
		//map<BaseContext *, BaseContext *> migrateCtx;
		map<IDENTIFIER, BaseContext *> migrateCtx;  // ctx inside this will be removed when a IDENTIFIER hit the key value,client id to find ctx 

		void print(){
			cg::core::infoRecorder->logTrace("\t[id:%p]\t[renders:%d]\t", id, renderCount);
			std::cout << "\t[id: "<< id << "]\t[renders: " << renderCount << "]\t" << std::endl;
			for(int i =0; i < renderCount; i++){
				cg::core::infoRecorder->logTrace("\t[render %d:%p]", i, renderCtx[i]->sock);
				std::cout << "\t[render " << i << ": " << renderCtx[i]->sock << "]" << std::endl;
			}
		}

		TaskInfo(){
			id = 0;
			status = INIT;
			cpuRequire = 0.0f, gpuRequire = 0.0f;
			taskName = "";
			path = NULL;
			logicCtx = NULL;
			logicReady = false;
			ctrlReady = false;
			renderCount = 0;
			for (int i = 0; i < MAX_RENDER_COUNT; i++){
				renderCtx[i] = NULL;
				renderReady[i] = false;
				isSent[i] = false;
				toMigrate[i] = NULL;
			}

		}
		~TaskInfo(){}

		bool isMigrating(IDENTIFIER id){
			if (migrateCtx.find(id) != migrateCtx.end()){
				return true;
			}
			return false;
		}
		void removeCtx(BaseContext * ctx){
			int i = 0;
			for (i = 0; i < renderCount; i++){
				if (renderCtx[i] == ctx){
					// find the ctx
					break;
				}
			}
			for (int j = i; j < renderCount - 1; j++){
				renderCtx[j] = renderCtx[j + 1];
				renderReady[j] = renderReady[j + 1];
			}
			renderReady[renderCount - 1] = false;
			renderCount--;
		}

		void addRender(BaseContext * render){
			renderCtx[renderCount++] = render;
		}
		void setLogic(BaseContext * logic){
			logicCtx = logic;
			if (logic == NULL){
				printf("NULL logic is SET.\n");
			}
		}
	};

	struct RenderTaskInfo{
		IDENTIFIER identifier;

		BaseContext * renderCtx;
		short sameTask;   // how many sub task is the same

		RenderTaskInfo();
		~RenderTaskInfo();
	};
	enum DomainStatus{
		RENDER_MIGRATE,  // the render is under the processing of migrate
		RENDER_GREEN,
		RENDER_SERVE,   // the domain is the working node

		LOGIC_SERVE,
		LOGIC_MIGRATE,
		STATUS_NONE
	};

	class DomainInfo: public BaseContext{
	public:
		float cpuUsage;
		float gpuUsage;
		float memUsage;

		float fps, aveFps;   //
		double runTime;
		double lastUpdateTime;

		float aveCpu, aveGpu, aveMem;
		DomainStatus status;
		string domainType;

		//BaseContext * ctx;  // BaseContext for the domain
		map<IDENTIFIER, TaskInfo * > taskMap;   // tasks run on the domain

		void print(){
			cg::core::infoRecorder->logTrace("\t[cpu:%f\tgpu:%f\ttasks:%d]\n", cpuUsage, gpuUsage, taskMap.size());
			std::cout << "\t[type: " << domainType << " ip: " << getNetName() << " ID: " << sock <<"]\n\t[cpu: " << cpuUsage << "\tgpu: " << gpuUsage << "\ttasks: " << taskMap.size() << "]" << std::endl << "\ttasks: " << std::endl; 
			for(map<IDENTIFIER, TaskInfo *>::iterator it = taskMap.begin(); it != taskMap.end(); it++){
				it->second->print();
			}
		}

		void addTask(TaskInfo * task){
			taskMap[task->id] = task;
		}
		inline bool isTaskExist(IDENTIFIER id){
			if (taskMap.find(id) != taskMap.end()){
				return true;
			}
			return false;
		}
		bool isOverload(){
			// TODO, how to define overload, how to avoid fake overload ?????
			bool ret = false;


			return ret;
		}
		bool isGreen(){
			// TODO, make sure that the domain is green or not ???????
			bool ret = true;
			return ret;
		}

		TaskInfo * getUndivideTask(){
			for (map<IDENTIFIER, TaskInfo *>::iterator it = taskMap.begin(); it != taskMap.end(); it++){
				if (it->second->renderCount == 1){
					return it->second;
				}
			}
			return NULL;
		}
		TaskInfo * getLeastDivideTask(){
			TaskInfo * ret = taskMap.begin()->second;
			map<IDENTIFIER, TaskInfo *>::iterator it;
			for (it = taskMap.begin(); it != taskMap.end(); it++){
				if (ret->renderCount > it->second->renderCount){
					ret = it->second;
				}
			}
			return ret;
		}

		// constructor
		DomainInfo(){
			domainType = "";
			status = STATUS_NONE;
		}
		~DomainInfo(){}
		inline string toString(){
			return getNetName()+ " " + domainType;
		}
	};

	struct VideoItem{
		HWND windowHandle;
		void* device;
		HANDLE presentEvent;
		int winHeight, winWidth;

		VideoItem(){
			windowHandle = NULL;
			device = NULL;
			presentEvent = NULL;
			winHeight = 0;
			winWidth = 0;
		}
	};

	// video context manages the window handle and d3d device for video generator
	class VideoContext{
		map<IDENTIFIER, VideoItem*> itemMap;
		HANDLE notifier;    // the notifier for item adding

		VideoContext(){
			notifier = CreateEvent(NULL, FALSE, FALSE, NULL);

		}
		static VideoContext * ctx;
	public:
		static VideoContext * GetContext(){
			if(ctx == NULL){
				ctx = new VideoContext();
			}
			return ctx;
		}
		void WaitForNofify();
		inline void addMap(IDENTIFIER id, VideoItem* it){
			printf("[VideoContext]: trigger the video context event, id:%p, notify:%p.\n", id, notifier);
			SetEvent(notifier);
			itemMap.insert(map<IDENTIFIER, VideoItem *>::value_type(id, it));
		}
		inline VideoItem* findItem(IDENTIFIER id){
			map<IDENTIFIER, VideoItem*>::iterator it = itemMap.find(id);
			if(it != itemMap.end()){
				// find
				return it->second;
			}
			return NULL;
		}
		~VideoContext(){
			if(notifier){
				CloseHandle(notifier);
				notifier = NULL;
			}
		}
	};

	void RTSPListenerCB(struct evconnlistener * listerner, evutil_socket_t sock, struct sockaddr * sddr, int len, void * ctx);

	void RTSPAcceptErrorCB(struct evconnlistener * listener, void *ctx);

	struct evconnlistener * listenPort(int port, event_base * base, void * ctx);

}

#endif   // __CONTEXT_H__