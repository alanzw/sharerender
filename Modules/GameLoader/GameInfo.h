#ifndef __GAMEINFO_H__
#define __GAMEINFO_H__
// store the game information

#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

#define GAME_MAP "gamemap.txt" // store the game map information
#define HOOK_DLL "hook_dll"   // not all games support migration


using namespace std;

class GameInfoItem{
public:
	string gameName;
	string gamePath;
	string hookDll;   // name of the dll to use
	bool supportD3D;

	GameInfoItem(string _name, string _path, string _dll, bool _support): gameName(_name), gamePath(_path), hookDll(_dll), supportD3D(_support){}
	void show(){
		cout << "[Item]:\t" << gameName << " " << gamePath << " " << hookDll << " " << supportD3D <<endl;
	}
};

// game info
class GameInfo{
	string mapName;


	map<string, GameInfoItem *> infoMap;   // game name to find the information
	static GameInfo * gameInfo;
	GameInfo(string _mapName){
		mapName = _mapName;
	}

	
	
public:
	static GameInfo * GetGameInfo(string mapName = ""){
		if(mapName == ""){
			if(gameInfo == NULL){
				cout << "[GameInfo]: load game info file:" << GAME_MAP << endl;
				gameInfo = new GameInfo(GAME_MAP);
			}
		}
		else{
			if(gameInfo == NULL){
				cout << "[GameInfo]: load game info file:" << mapName<<endl;
				gameInfo = new GameInfo(mapName);
			}
		}
		return gameInfo;
	}
	bool loadInfo(){
		ifstream ifs(mapName);
		if(!ifs){
			cerr << "[GameInfo]: ERROR, unable to open game map file." << endl;
			return false;
		}
		string line;
		string gameName, path, dllname, d3dOption;
		bool supportD3D = false;
		int infoCount = 0;
		while(getline(ifs, line)){
			// do per-line processing
			istringstream stream(line);  // bind stream to the line we read
			if(line.c_str()[0] == '#'){
				// comments
				continue;
			}
			else{
				// valid content
				stream >> path >> gameName >> dllname >> d3dOption;
				cout << infoCount++ << " [" << path << "] ["<< gameName << "] ["<< dllname << "] [" << d3dOption << "]" << endl;  

				if(!strcmp(d3dOption.c_str(), "true") || !strcmp(d3dOption.c_str(), "TRUE")){
					supportD3D = true;
				}
				else if(!strcmp(d3dOption.c_str(), "false") || !strcmp(d3dOption.c_str(), "FALSE")){
					supportD3D = false;
				}
				else{
					cerr << "[GameInfo]: error, unknown d3d option." << endl;
					continue;
				}
				// format the game name
				string formatedName = formatGameName(gameName);
				// add to map
				GameInfoItem * item = new GameInfoItem(formatedName, path, dllname, supportD3D);
				infoMap[formatedName] = item;
			}
		}
		return true;
	}

	// format the game name to lower case and full name with suffix(.exe)
	string formatGameName(string name){
		//string lowerCase = name.tolower();
		cout<<"[GameInfo]: game name " << name << " to lower case, ";
		std::transform(name.begin(), name.end(), name.begin(), ::tolower);
		cout<< name << endl;
		int index = name.find_last_of('.');
		cout<< "[GameInfo]: find dot at " << index<<endl;
		if(index != -1 && name.substr(index + 1) == "exe"){
			// got a post suffix

		}
		else{
			
			name.append(".exe");
			cout<< "[GameInfo]: add suffix to game name. now: "<< name <<endl;
		}
		return name;
	}

	GameInfoItem * findGameInfo(string name){
		//deal with game name, cases and the .exe dealing, all convert to lower case
		string gameName = formatGameName(name);

		map<string, GameInfoItem *>::iterator it = infoMap.find(gameName);
		if(it != infoMap.end()){
			// find
			cout << "[GameInfo]: find the game info for '" << name << "'."<<endl;
			GameInfoItem * item = it->second;
			item->show();
			return it->second;
		}
		else{
			cerr << "[GameInfo]: cannot find the information for game '"<< name<<"'"<<endl;
			return NULL;
		}
	}

	string findGamePath(string name){
		GameInfoItem * item = findGameInfo(name);
		if(item){
			return item->gamePath;
		}
		return "";
	}
	string findGameDll(string name){
		GameInfoItem * item = findGameInfo(name);
		if(item){
			return item->hookDll;
		}
		else
			return "";
	}
	bool isD3DSupported(string name){
		GameInfoItem * item = findGameInfo(name);
		if(item)
			return item->supportD3D;
		else
			return false;
	}

	// print all the information
	void showAllInfo(){
		GameInfoItem * item = NULL;
		int count = 0;
		map<string, GameInfoItem *>::iterator it;
		cout << "[GameInfo]: start show info.\n" << endl;
		for(it = infoMap.begin(); it != infoMap.end(); it++){

			item = it->second;
			cout << "[" << count++<< "]:\tpath:["<<item->gamePath<<"]\n\tname:[" << item->gameName<<"]\n\tdll name:["<< item->hookDll<<"]\n\tsupport d3d:["<< (item->supportD3D == true ? "true" : "false" )<<"]"<< endl; 
		}
		cout << "[GameInfo]: end show info.\n" <<endl;
	}
};


#endif