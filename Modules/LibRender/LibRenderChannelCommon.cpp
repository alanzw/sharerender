#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include <string.h>
#include <time.h>
#include <map>

#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"
#include "../LibCore/CommandClient.h"
#include "../LibCore/CThread.h"
//#include "../LibDistrubutor/Context.h"
#include "../LibDistrubutor/Distributor.h"

#include "LibRenderAPI.h"
#include "LibRenderChannel.h"

using namespace cg;
using namespace cg::core;

HANDLE RenderChannel::presentMutex;

// this is for the common functions of render channel
RenderChannel::RenderChannel(){
	specialTag = 0;
	gD3d = NULL;
	hWnd = NULL;
	curDevice = NULL;
	cc = NULL;

	client_render = true;

	winWidth = 0;
	winHeight = 0;
	// D3D objects map
	for (int i = 0; i < Max_Obj_Cnt; i++){
		device_list[i] = NULL;
		vb_list[i] = NULL;
		ib_list[i] = NULL;
		vd_list[i] = NULL;
		vs_list[i] = NULL;
		ps_list[i] = NULL;
		tex_list[i] = NULL;
		sb_list[i] = NULL;
		ctex_list[i] = NULL;

		chain_list[i] = NULL;
		surface_list[i] = NULL;
	}

	// init the funciton table
	//initFptable();
	// map the functions to function array.
	//mapFunctions();

	// init the temp
	gap = 1.0f;
	videoInitMutex = NULL;
	presentEvent = NULL;
	DeviceHandle = NULL;
	hWnd = NULL;
	videoThread = NULL;
	channelThreadHandle = NULL;
	rtspObject = NULL;
	if (presentMutex == NULL){
		presentMutex = CreateMutex(NULL, FALSE, NULL);
	}
	presentEvent = NULL; //CreateEvent(NULL, FALSE, FALSE, NULL);

	videoItem = new VideoItem();
	//videoItem->presentEvent = presentEvent;

	windowCreated = false;
	generator = NULL;
	encoderOption = 1;
	isEncoding = false;
}

RenderChannel::~RenderChannel(){

	if (cc){
		delete cc;
		cc = NULL;
	}
}

bool RenderChannel::initRenderChannel(IDENTIFIER tid, string name, SOCKET s){
	// create the command client
	cc = new CommandClient();
	cc->set_connect_socket(s);
	gameName = name;
	taskId = tid;
	cg::core::infoRecorder->logTrace("[RenderChannel]: new CommandClient created: %p, socket set:%p, gameName:%s.\n", cc, s, gameName.c_str());

	return true;
}

void RenderChannel::dealControlCmd(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam){
	printf("WM_INPUT message.\n");
	//DefWindowProc(hwnd, uMsg, wParam, lParam);
}

char tempbuffer[7500000] = { '0' };
// for a seperated thread proc
DWORD RenderChannel::startChannelThread(){
	channelThreadHandle = chBEGINTHREADEX(NULL, 0, ChannelThreadProc, this, FALSE, &channelThreadId);
	return channelThreadId;
}

DWORD WINAPI RenderChannel::ChannelThreadProc(LPVOID param){
	cg::core::infoRecorder->logTrace("[RenderChannel]: enter channel thread proc.\n");
	RenderChannel * rch = (RenderChannel *)param;
	if (NULL == rch){
		cg::core::infoRecorder->logTrace("[RenderChannel]: NULL render channel given in thread proc.\n");
		return -1;
	}
	// send to logic the ADD_RENDER+task ID + Render ID

	char tm[100] = { 0 };
#if 1
	// need to select?
	fd_set fdSock;
	FD_ZERO(&fdSock);
	FD_SET(rch->cc->get_connect_socket(), &fdSock);
	int nRet = select(0, &fdSock, NULL, NULL, NULL);
	if(nRet){
		// to recv
		int r = 0;
		r = recv(rch->cc->get_connect_socket(), tm, 100, 0);
		tm[r] = 0;
		cg::core::infoRecorder->logTrace("[RenderChannel]: get %s from logic server.\n", tm);
		printf("[RenderChannel]: get %s from logic server.\n", tm);
	}
	else{
		printf("[RenderChannel]: recv feedback failed.\n");
		cg::core::infoRecorder->logError("[RenderChannel]: recv feedback failed.\n");
	}
#else
	send(rch->cc->get_connect_socket(), RENDER_STARTED, strlen(RENDER_STARTED), 0);
	//cc->send_raw_buffer((char *)RENDER_STARTED, strlen(RENDER_STARTED));
	char temp[50];
	int r = 0;
	fd_set fdSocket;
	FD_ZERO(&fdSocket);
	FD_SET(rch->cc->get_connect_socket(), &fdSocket);
	int nRet = select(0, &fdSocket, NULL, NULL, NULL);
	if (nRet > 0){
#if 0
		r = recv(rch->cc->get_connect_socket(), temp, 50, 0);
		temp[r] = 0;
		//cs.recv_raw_buffer(temp, 50);
		cg::core::infoRecorder->logTrace("[game_server]: get %s from render proxy.\n", temp);
#else
		r = recv(rch->cc->get_connect_socket(), tempbuffer, 7500000, 0);
		cg::core::infoRecorder->logTrace("[game_server]: get %d bytes\n", r);
#endif

	}
	//int r = recv(cc->get_connect_socket(), tm, 100, 0);
	cg::core::infoRecorder->logTrace("[RenderChannel]: recved %d chars from server, msg: %s. why cannot go to next line? error code:%d\n", r, temp, WSAGetLastError());
#endif

	// create the present event
	if (rch->presentEvent == NULL){
		rch->presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		if (!rch->presentEvent){
			cg::core::infoRecorder->logError("[RenderChannel]: create present event failed.\n");
		}
	}

	cg::core::infoRecorder->logTrace("[RenderChannel]: to start the encoding thread.\n");
	// create the video part
	// loop run
	MSG xmsg = {0};

#if 0
	while (xmsg.message != WM_QUIT){

		if(PeekMessage(&xmsg, NULL, 0U, 0U, PM_REMOVE)){
			TranslateMessage(&xmsg);
			DispatchMessage(&xmsg);
		}
		else{
			rch->cc->take_command(rch->op_code, rch->obj_id);

			if (rch->op_code >= 0 && rch->op_code < MaxSizeUntilNow_Opcode){
				if (rch->client_render){
					cg::core::infoRecorder->logTrace("[RenderChannel]: opcode:%d, cmd :%s.\n", rch->op_code, (funcs[rch->op_code].name));
					(*(funcs[rch->op_code].func))(rch);
				}
			}
			else{
				if (rch->op_code == MaxSizeUntilNow_Opcode){
					cg::core::infoRecorder->logError("[RenderChannel]: render proxy exit normally.\n");
				}
				else{
					cg::core::infoRecorder->logError("[RenderChannel]: render proxy exit, unexpected op_code:%d.\n", rch->op_code);
				}
				rch->cleanUp();
				break;
			}
		}
	}
#else

	 int callCount = 0;
	 int flag = 0, sleepCount = 0;

	while(true){
		if(PeekMessage(&xmsg, NULL, 0, 0, PM_REMOVE)){
#if 0
			TranslateMessage(&xmsg);
			DispatchMessage(&xmsg);
#else
			TranslateMessage(&xmsg);
			switch(xmsg.message){
				printf("mssage;%d.\n", xmsg.message);
			case WM_QUIT:
				if(rch){
					rch->cleanUp();
					return 0;
				}
				break;
			case WM_KEYUP:
				printf("keyup.\n");
				if(xmsg.wParam == VK_ESCAPE)
					PostQuitMessage(0);
				else if(xmsg.wParam == VK_ADD){
					// disable video generate
					rch->setEnableEncoding(false);
				}
				else if(xmsg.wParam == 0x5a){
					// key Z, for x264 encoder
					if(rch->generator){
						rch->generator->setEncoderType(X264_ENCODER);
					}
				}
				else if(xmsg.wParam == 0x58){
					// key X, for cuda encoder
					if(rch->generator){
						rch->generator->setEncoderType(CUDA_ENCODER);
					}
				}else if(xmsg.wParam == 0x43){
					// key C, for nvenc encoder
					if(rch->generator){
						rch->generator->setEncoderType(NVENC_ENCODER);
					}
				}
				break;
			default:
				DispatchMessage(&xmsg);
				break;
			}
#endif
		}
		// after dealing the message, to the rendering
		int func_remain = 0;
		//func_remain = rch->cc->fetch_stream_buffer();
		
		do{
			func_remain = rch->cc->take_command(rch->op_code, rch->obj_id);
			if (rch->op_code >= 0 && rch->op_code < MaxSizeUntilNow_Opcode){
				if (rch->client_render){
					//cg::core::infoRecorder->logTrace("[RenderChannel]: opcode:%d, cmd :%s.\n", rch->op_code, (funcs[rch->op_code].name));
					(*(funcs[rch->op_code].func))(rch);
				}
				if(rch->op_code == Present_Opcode){
					flag = 1;
				}
			}
			else{
				if (rch->op_code == MaxSizeUntilNow_Opcode){
					cg::core::infoRecorder->logError("[RenderChannel]: render proxy exit normally.\n");
				}
				else{
					cg::core::infoRecorder->logError("[RenderChannel]: render proxy exit, unexpected op_code:%d.\n", rch->op_code);
				}
				rch->cleanUp();
				delete rch;
				break;
			}
		}while(func_remain);
		// do the sleep to avoid waste

		callCount++;
#if 1
		if(callCount >= 100){
			callCount = 0;
			sleepCount++;
			Sleep(1);
		}
#endif
	}
#endif
	// on quit
	return 0;
}

// clean up when exit the render channel
void RenderChannel::cleanUp(){
	if(generator){
		generator->onQuit();
		delete generator;
		generator = NULL;
	}
	if(cc){
		delete cc;
		cc = NULL;
	}
	if(presentEvent){
		CloseHandle(presentEvent);
		presentEvent = NULL;
	}
	if(videoInitMutex){
		CloseHandle(videoInitMutex);
		videoInitMutex = NULL;
	}
	if(rtspObject){
		free(rtspObject);
		rtspObject = NULL;
	}
}

void RenderChannel::onPresent(unsigned int tags){
	// on present, check the generator, init the generator if not inited
#ifndef NO_VIDEO_GEN
	HRESULT hr = D3D_OK;
	if(!isEncoding)
		return;

	// generate video
	if(generator == NULL){
		generator= new cg::VideoGen(hWnd, curDevice, DX9, true,false, true);
		generator->setObject("/" + gameName);
	}
	if(generator && !generator->isInited()){
		cg::core::infoRecorder->logError("On Present, to init the generator, object:%s\n", gameName.c_str());
		//generator->setObject(gameName);
		IDirect3DSurface9 * rts = NULL;
		D3DSURFACE_DESC sdesc;
		hr = curDevice->GetRenderTarget(0, &rts);
		if(FAILED(hr)){
			cg::core::infoRecorder->logError("[RenderChannel]: on present, init the generator, get render target failed.\n");
			return;
		}
		if(FAILED(rts->GetDesc(&sdesc))){
			cg::core::infoRecorder->logError("[RenderChannel]: surface get Desc failed.\n");
			return;
		}
		imageWidth = sdesc.Width;
		imageHeight = sdesc.Height;

		if(rts){
			rts->Release();
			rts = NULL;
		}

		cg::core::infoRecorder->logError("[RenderChannel]: to init the generator, image width:%d, height:%d.\n", imageWidth, imageHeight);
		generator->setResolution(imageWidth, imageHeight);
		presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		generator->setPresentHandle(presentEvent);

		//encoderOption = ADAPTIVE_NVENC;
		//encoderOption = NVENC_ENCODER;
		//encoderOption = CUDA_ENCODER;
		//encoderOption = NVENC_ENCODER; 
		//encoderOption = X264_ENCODER; 
		
		switch(encoderOption){
		case CUDA_ENCODER:
			generator->setEncoderType(CUDA_ENCODER);
			break;
		case NVENC_ENCODER:
			generator->setEncoderType(NVENC_ENCODER);
			break;
		case ADAPTIVE_NVENC:
			generator->setEncoderType(ADAPTIVE_NVENC);
			break;
		case ADAPTIVE_CUDA:
			generator->setEncoderType(ADAPTIVE_CUDA);
			break;
		default:
		case X264_ENCODER:
			generator->setEncoderType(X264_ENCODER);
			break;
		}

		generator->onThreadStart();
		VideoGen::addMap((IDENTIFIER )taskId, generator);
	}
	else{
		//cg::core::infoRecorder->logError("on present: generator inited: %p.\n", generator);
	}
	if(generator){
		generator->setVideoTag(tags);
		if(this->specialTag){
			generator->setVideoSpecialTag(this->specialTag);
			specialTag = 0;
		}
		SetEvent(generator->getPresentEvent());
		generator->run();
	}
	// rate control
	cg::core::infoRecorder->onFrameEnd();
#endif
}
