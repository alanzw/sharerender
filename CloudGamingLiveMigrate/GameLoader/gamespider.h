#ifndef __GAMEFINDER_H__
#define __GAMEFINDER_H__

// this is for a finder to scan the folder and register games
// read a file( format: [path] [game name] [support D3D?]

#include <iostream>

#include <map>
#include "../LibVideo/Config.h"

#define GAME_MAP "gamemap.txt"    // stores the game map information
#define D3D_SUFFIX "_support_d3d"   // 
#define HOOK_DLL "hook_dll"

using namespace std;

class GameSpider{
	char * basePath;    // the base folder to find games

	map<string, ConfVar> _confVars;
	map<string, ConfVar>::iterator vmi;

	char * confTrim(char * buf);    // trim the configuration
	int confParse(const char * filename, int lineno, char * buf);
	int confLoad(const char * filename);
	void confClear();
	char * confReadV(const char * key, char * store, int slen);
	int confReadBool(const char * key, int defVal);
	int confBoolVal(const char * ptr, int defVal);

	int confWriteV(const char * key, const char * value);
	void confErase(const char * key);

public:
	GameSpider(char * base);
	GameSpider();
	~GameSpider();

	char * getPath(char * gameName);
	char * getHookDllName();
	int isD3DSupported(char * gameName);

	bool changeCurrentDirectory(char * path);

};

#endif