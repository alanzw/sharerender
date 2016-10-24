#include "../../CloudGamingLiveMigrate/LibCore/CommonNet.h"
#include <process.h>
#include "../../CloudGamingLiveMigrate/LibCore/Config.h"
#include <WinSock2.h>
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderAPI.h"
#include "../../CloudGamingLiveMigrate/LibRender/LibRenderChannel.h"

#include "TestRenderProxy.h"
#include <string>

using namespace std;

#define maxl 10010

char buffer[maxl] = "hello, world";
char b2[maxl];
char b3[maxl];

bool client_render = true;
bool fromserver = false;
extern HANDLE videoInitMutex;
extern HANDLE presentEvent;
extern HANDLE presentMutex;

InfoRecorder * infoRecorder = NULL;

void usage(){
	printf("TestRenderProxy.exe [GameName]");
}

int main(int argc, char ** argv){
	WSADATA wsaData;
	WORD sockVersion = MAKEWORD(2,2);
	infoRecorder = new InfoRecorder("GameClient");

	if(WSAStartup(sockVersion, &wsaData) != 0){
		infoRecorder->logError("[LogicServer]: WSAStartup failed.\n");
		return 0;
	}

	string game_name;

	init_fptable();
	RenderChannel * rch = new RenderChannel();

	// connect to server
	SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if(s == INVALID_SOCKET){
		infoRecorder->logError("[GameClient]: create socket failed.\n");
		return -1;
	}

	

	sockaddr_in sin;
	sin.sin_family = AF_INET;
	sin.sin_port = htons(60000);
	sin.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");
	if(connect(s, (SOCKADDR *)&sin, sizeof(sin)) == -1){
		infoRecorder->logError("[GameClient]: connect failed.\n");
		return -1;
	}

	// set to async io
	u_long ul = 1; 
	ioctlsocket(s, FIONBIO, (u_long *)&ul);

	if(argc == 1){
		game_name = "trine.exe";

		rch->initRenderChannel(1, game_name, s);

	}else if( argc == 2){
		printf("start game: %s.\n", argv[1]);

		game_name = argv[1];


		rch->initRenderChannel(1, game_name, s);
	}else if(argc == 3){
		printf("start game:%s.\n", argv[1]);
		
		game_name  = string(argv[1]);
		int c_id = atoi((const char*)argv[3]);

		if(c_id > 1){
			client_render =false;
		}
		rch->initRenderChannel(c_id, game_name, s);
	}else{
		usage();
		return 0;
	}

	//
	infoRecorder->logError("[GameClient]: to start the render channel thread.\n");
	rch->startChannelThread();
	
	//
	infoRecorder->logError("[GameClient]: to wait the render thread exit!\n");
	WaitForSingleObject(rch->channelThreadHandle, INFINITE);
	return 0;
}
