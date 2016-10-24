#include "gameloader.h"

// this is for the game loader

GameLoader::GameLoader(){
	gameSpider = new GameSpider();

}
GameLoader::~GameLoader(){
	if(gameSpider){
		delete gameSpider;
		gameSpider = NULL;
	}
}

int GameLoader::load2DGame(char * gameName){

}


/// loading the 3D game may need the cmdline?
int GameLoader::loadD3DGame(char * gameName){

}