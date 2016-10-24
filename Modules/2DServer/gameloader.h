#ifndef __GAMELOADER_H__
#define __GAMELOADER_H__
// this is for the game loader
// start a 2D game and the video server in a seperated process

#include "gamespider.h"

class GameLoader{
	GameSpider * gameSpider;


public:
	GameLoader();
	~GameLoader();

	int loadD3DGame(char * gameName);   // load a d3d game
	int load2DGame(char * gameName);    // load a 2d game and start the video server
	int loadGame(char * gameName);
};


#endif