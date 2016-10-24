#include "gamespider.h"
#include "../LibCore/Log.h"
#include "../LibVideo/Commonwin32.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

// this is for the game spider

// constructor and the destructor
GameSpider::GameSpider(char * base){
	basePath = _strdup(base);
	// find the configure file in the base folder
	if(confLoad(GAME_MAP) == -1){
		infoRecorder->logError("[GameSpider]: load the map file failed.\n");
		
	}


}

// default, use the current directory 
GameSpider::GameSpider(){
	char path[MAX_PATH] = {0};
	GetCurrentDirectory(MAX_PATH, path);
	basePath = _strdup(path);

	// load the configure file 
	if(confLoad(GAME_MAP) == -1){
		infoRecorder->logError("[GameSpider]: load the map file failed.\n");
	}
}

GameSpider::~GameSpider(){
	if(basePath){
		free(basePath);
		basePath = NULL;
	}
}

char * GameSpider::getPath(char * gameName){
	/// game name is key to lookup the map
	char path[128] = { 0 };
	confReadV(gameName, path, 128);
	infoRecorder->logError("[GameSpider]: the path for game '%s' is '%s'.\n", gameName, path);
	return _strdup(path);
}

int GameSpider::isD3DSupported(char *gameName){
	char key[128] = {0};
	sprintf(key, "%s%s", gameName, D3D_SUFFIX);
	return confReadBool(key, 0);
}

char * GameSpider::getHookDllName(){
	char name[64] = {0};
	confReadV(HOOK_DLL, name, 64);
	infoRecorder->logError("[GameSpider]: get the hook dll name '%s'.\n", name);
	return _strdup(name);
}


// file operator

int GameSpider::confBoolVal(const char * ptr, int defVal){
	if (strcasecmp(ptr, "true") == 0
		|| strcasecmp(ptr, "1") == 0
		|| strcasecmp(ptr, "y") == 0
		|| strcasecmp(ptr, "yes") == 0
		|| strcasecmp(ptr, "enabled") == 0
		|| strcasecmp(ptr, "enable") == 0)
		return 1;
	if (strcasecmp(ptr, "false") == 0
		|| strcasecmp(ptr, "0") == 0
		|| strcasecmp(ptr, "n") == 0
		|| strcasecmp(ptr, "no") == 0
		|| strcasecmp(ptr, "disabled") == 0
		|| strcasecmp(ptr, "disable") == 0)
		return 0;
	return defVal;
}

int GameSpider::confReadBool(const char * key, int defVal){
	char buf[64];
	char *ptr = confReadV(key, buf, sizeof(buf));
	if (ptr == NULL)
		return defVal;
	return confBoolVal(ptr, defVal);
}

char * GameSpider::confTrim(char * buf){
	char *ptr;
	// remove head spaces
	while (*buf && isspace(*buf))
		buf++;
	// remove section
	if (buf[0] == '[') {
		buf[0] = '\0';
		return buf;
	}
	// remove comments
	if ((ptr = strchr(buf, '#')) != NULL)
		*ptr = '\0';
	if ((ptr = strchr(buf, ';')) != NULL)
		*ptr = '\0';
	if ((ptr = strchr(buf, '/')) != NULL) {
		if (*(ptr + 1) == '/')
			*ptr = '\0';
	}
	// move ptr to the end, again
	for (ptr = buf; *ptr; ptr++)
		;
	--ptr;
	// remove comments
	while (ptr >= buf) {
		if (*ptr == '#')
			*ptr = '\0';
		ptr--;
	}
	// move ptr to the end, again
	for (ptr = buf; *ptr; ptr++)
		;
	--ptr;
	// remove tail spaces
	while (ptr >= buf && isspace(*ptr))
		*ptr-- = '\0';
	//
	return buf;
}

int GameSpider::confParse(const char * filename, int lineno, char * buf){
	char *option, *token; //, *saveptr;
	char *leftbracket, *rightbracket;
	ConfVar gcv;
	//
	option = buf;
	if ((token = strchr(buf, '=')) == NULL) {
		return 0;
	}
	if (*(token + 1) == '\0') {
		return 0;
	}
	*token++ = '\0';
	//
	option = confTrim(option);
	if (*option == '\0')
		return 0;
	//
	token = confTrim(token);
	if (token[0] == '\0')
		return 0;
	// check if its a include
	if (strcmp(option, "include") == 0) {
#ifdef WIN32
		char incfile[_MAX_PATH];
		char tmpdn[_MAX_DIR];
		char drive[_MAX_DRIVE], tmpfn[_MAX_FNAME];
		char *ptr = incfile;
		if (token[0] == '/' || token[0] == '\\' || token[1] == ':') {
			strncpy(incfile, token, sizeof(incfile));
		}
		else {
			_splitpath(filename, drive, tmpdn, tmpfn, NULL);
			_makepath(incfile, drive, tmpdn, token, NULL);
		}
		// replace '/' with '\\'
		while (*ptr) {
			if (*ptr == '/')
				*ptr = '\\';
			ptr++;
		}
#else
		char incfile[PATH_MAX];
		char tmpdn[PATH_MAX];
		//
		strncpy(tmpdn, filename, sizeof(tmpdn));
		if (token[0] == '/') {
			strncpy(incfile, token, sizeof(incfile));
		}
		else {
			snprintf(incfile, sizeof(incfile), "%s/%s", dirname(tmpdn), token);
		}
#endif
		infoRecorder->logTrace("# include: %s\n", incfile);
		return confLoad(incfile);
	}
	// check if its a map
	if ((leftbracket = strchr(option, '[')) != NULL) {
		rightbracket = strchr(leftbracket + 1, ']');
		if (rightbracket == NULL) {
			infoRecorder->logTrace("# %s:%d: malformed option (%s without right bracket).\n",
				filename, lineno, option);
			return -1;
		}
		// no key specified
		if (leftbracket + 1 == rightbracket) {
			infoRecorder->logTrace("# %s:%d: malformed option (%s without a key).\n",
				filename, lineno, option);
			return -1;
		}
		// garbage after rightbracket?
		if (*(rightbracket + 1) != '\0') {
			infoRecorder->logTrace("# %s:%d: malformed option (%s?).\n",
				filename, lineno, option);
			return -1;
		}
		*leftbracket = '\0';
		leftbracket++;
		*rightbracket = '\0';
	}
	// its a map
	if (leftbracket != NULL) {
		infoRecorder->logError("[ccgConfig]: %s[%s] = %s\n", option, leftbracket, token);
		Log::logscreen("[ccgConfig]: %s[%s] = %s\n", option, leftbracket, token);
		_confVars[option][leftbracket] = token;
	}
	else {
		infoRecorder->logError("[ccgConfig]: %s = %s\n", option, token);
		Log::logscreen("[ccgConfig]: %s = %s\n", option, token);
		_confVars[option] = token;
	}
	return 0;

}

int GameSpider::confLoad(const char * filename){
	FILE *fp;
	char buf[8192];
	int lineno = 0;
	//
	if (filename == NULL)
		return -1;
	if ((fp = fopen(filename, "rt")) == NULL) {
		infoRecorder->logError("[ccg config]: open coinfig file: %s failed\n", filename);
		return -1;
	}
	while (fgets(buf, sizeof(buf), fp) != NULL) {
		lineno++;
		if (confParse(filename, lineno, buf) < 0) {
			fclose(fp);
			return -1;
		}
	}
	fclose(fp);
	return lineno;
}

void GameSpider::confClear(){
	_confVars.clear();
	vmi = _confVars.begin();
	return;

}

char * GameSpider::confReadV(const char * key, char * store, int slen){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(key)) == _confVars.end())
		return NULL;
	if (mi->second.value().c_str() == NULL)
		return NULL;
	if (store == NULL)
		return _strdup(mi->second.value().c_str());
	strncpy(store, mi->second.value().c_str(), slen);
	return store;
}

int GameSpider::confWriteV(const char * key, const char * value){
	_confVars[key] = value;
	return 0;
}

void GameSpider::confErase(const char * key){
	_confVars.erase(key);
	return;
}

// change the execute path
bool GameSpider::changeCurrentDirectory(char * path){
	char curDir[MAX_PATH] = {0};
	if(SetCurrentDirectory(path)){
		if(GetCurrentDirectory(MAX_PATH, curDir)){
			if(strncmp(path, curDir, strlen(path))){
				infoRecorder->logError("[GameSpider]: path not right.\n");
				return false;
			}
			else{
				infoRecorder->logError("[GameSpider]:set the directory succeeded.\n");
				return true;
			}
		}
		else{
			infoRecorder->logError("[GameSpider]: set and get the current directory failed.\n");
			return false;
		}
	}
	return false;
}