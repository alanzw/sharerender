#ifndef __2DSERVER_H__
#define __2DSERVER_H__

// this is for the 2D server
// the 2d server is for the most complecated 3D games and 2d games, display the game via video streaming, we need to hook 3d or just capture window to get the game image

#include "videogenerator.h"
#include "CpuWatch.h"
#include "GpuWatch.h"

#define ASYNCHRONOUS_SOURCE   //   asynchronous
#define SOURCES 1

class VideoStream{
	VideoGenerator * gVideoGenerator;
	int vsourceInitialized;
	int resolutionRetrieved;
	int gameWidth, gameHeight;
	int encoderHeight, encoderWidth;
	int hookBoost;
	int noDefaultController;
	int enableServerRateControl;
	int serverNumTokenToFill;
	int serverTokenFillInterval;
	int serverMaxTokens;
	int videoFps;
	bool netStarted;

	struct pipeline * pipe[SOURCES];

	// the vent for each api
	HANDLE d9Event;
	HANDLE d10Event;
	HANDLE d101Event;
	HANDLE d11Event;
	HANDLE dxgiEvent;

	HANDLE windowEvent;
	HWND hWnd;

	ccgRect rect, * pRect;

	CpuWatch * cpuWatcher;
	GpuWatch * gpuWatcher;


	static VideoStream * videoStream;
	VideoStream();
	~VideoStream();

public:
	static VideoStream * GetStream(){ if(videoStream == NULL){ videoStream = new VideoStream(); } return videoStream;}



	// getter and setter
	inline int getVideoFps(){ return videoFps; }
	inline int getEncoderHeight(){ return encoderHeight; }
	inline int getEncoderWidth(){ return encoderWidth; }
	inline pipeline * getPipe(int index){ return pipe[index]; }
	inline int getResolutionRetrieved(){ return resolutionRetrieved; }
	inline HANDLE getWindowEvent(){ return windowEvent; }
	inline HWND getWnd(){ return hWnd; }
	inline ccgRect * getpRect(){ return pRect; }
	inline bool isNetStarted(){ return netStarted; }

	// functions
	int cropWindow();
};

#endif