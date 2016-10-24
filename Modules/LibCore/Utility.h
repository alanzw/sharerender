#ifndef __UTILITY__
#define __UTILITY__

#if 1
#include <WinSock2.h>
#endif

#include <stdio.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN    // fix the winsock error
#endif
#include <windows.h>

#include <stack>
#include "Opcode.h"
#include "InfoRecorder.h"

//#define __DEBUG__

#ifdef __DEBUG__
#include <vld.h>
#endif

using namespace std;

//////////////////////////////////////////////各种编译选项////////////////////////////////////////////////////////

//#define MEM_FILE_TEX
//#define SAVE_IMG
#define SERVER_INPUT
//#define USE_VARINT
//#define LOCAL_IMG
//#define USE_FLOAT_COMPRESS
#define BLOCKING_NETWORK

// enable use cache
//#define USE_CACHE

#define USE_BUFFER_EVEN_ODD

//#define USE_CLIENT_INPUT

//#define ENABLE_LOG

//#define ENABLE_NETWORK_COMPRESS
#define ENABLE_SERVER_RENDERING

// this definition will use SaveTextureToFileInMemory in stead
//#define SEND_FULL_TEXTURE
#define UPDATE_ALL

//#define Enable_Command_Validate

// this define will enable printing the first 256 bytes for each packet in each side
//#define ENABLE_NET_VALIDATION


#define BUFFER_AND_DELAY_UPDATE	// this will enable delay update of buffers

#define USE_CHAR_COMPRESS		// compare each char when update

//#define USE_SHORT_COMPRESS		// compare each short when update

//#define USE_INT_COMPRESS		// compare each int when update



////////////////编译选项结束//////////////////////////////

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "winmm.lib")
#pragma warning(disable : 4996)

#define eps (1e-4)

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 480
#define MAX_CHAR 1024
#define MAXPACKETSIZE 7500000
//#define MAXPACKETSIZE 20000000
#define MAX_FPS 20
#define BUFFER_BLOCK_SIZE 1
#define BUFFER_BLOCK_LEVEL 0

#define Cache_Length_Limit 100

#define Max_Func_Count 200
#define Max_Buf_Size 2600

#define MAX_SOURCE_COUNT 8

#define Cache_Use	0
#define Cache_Set	1


#define HASHSETSIZE 128
#define HASHSETMASK	127

#define CACHE_MODE_INIT 0
#define CACHE_MODE_COPY 1
#define CACHE_MODE_DIFF 2
#define CACHE_MODE_COM	3
#define CACHE_MODE_DIFF_AND_COM 4
#define CACHE_MODE_PRESS_NORMAL_TANGENT 5

//Object Type
#define DEVICE_OBJ				0
#define INDEX_BUFFER_OBJ		1
#define VERTEX_BUFFER_OBJ		2
#define PIXEL_SHADER_OBJ		3
#define	VERTEX_SHADER_OBJ		4
#define TEXTURE_OBJ				5
#define STATE_BLOCK_OBJ			6
#define VERTEX_DECL_OBJ			7

#define SAFE_RELEASE(obj) \
	if(obj) { delete obj; obj = NULL; }


struct BufferLockData
{
	UINT OffsetToLock;
	UINT SizeToLock;
	DWORD Flags;
	HANDLE Handle;
	VOID* pRAMBuffer;
	VOID* pVideoBuffer;
	bool Create;

	bool updated;   // indicate whether it is the first time to lock since last updating
	UINT updatedSize;
	UINT updatedOffset;
	UINT updatedSizeToLock;

	inline void updateClear(){
		OffsetToLock = 0;
		SizeToLock = 0;
		updatedSize = 0;

		updatedOffset = 0;
		updatedSizeToLock = 0;

		updated = true;
	}
	inline bool isCopy(UINT totalLen, short indexSizeForEach = 3){
		if(totalLen < updatedSize * indexSizeForEach)
			return false;
		else
			return true;
	}
	void updateLock(UINT offset, UINT size, DWORD flags);

};


#include "Log.h"
#include "hash.h"
#include "HashSet.h"
#include "Compressor.h"
#include "Config.h"
#include "Buffer.h"
#include "Network.h"
#include "CommandServer.h"
#include "CommandClient.h"

extern int writeSharedData(char * data, int size);
extern int readSharedData(char * dst, int size);
extern void startGameProcessWithDll(char * gameName);

// set the tcp sending buffer
extern bool setTcpBuffer(SOCKET s);
extern bool setNBIO(SOCKET s);

extern SOCKET DuplicateSocketFormProcess(DWORD processId, SOCKET old);

#endif
