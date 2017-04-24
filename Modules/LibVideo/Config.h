#ifndef __CCG_CONFIG__
#define __CCG_CONFIG__

//#include "utility.h"
#include <string>
#include <map>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

using namespace std;

#if 0
class Config {
public:

	Config(char fname[]);
	~Config();

	DWORD read_property(LPCTSTR lpAppName, LPCTSTR lpKeyName, LPTSTR lpReturnedString);
	DWORD read_property(LPCTSTR lpAppName, LPCTSTR lpKeyName, int& retval);
	DWORD read_property(LPCTSTR lpAppName, LPCTSTR lpKeyName, float& retval);
	BOOL write_property(LPCTSTR lpAppName, LPCTSTR lpKeyName, LPTSTR lpString);

private:
	char fname_[100];
};

class ServerConfig : public Config {
public:
	ServerConfig(char fname[]);

	void load_config();
	void show_config();

	int command_port_;
	int max_fps_;

	int mesh_low_;
	int mesh_up_;
	float mesh_ratio_;

	int ban_ib_;
};

class ClientConfig : public Config {
public:
	ClientConfig(char fname[]);

	void load_config(int client_num);
	void show_config();

	char srv_ip_[100];
	int srv_port_;
};

#endif

#if 0
class ConfVar {
private:
	std::string data;
	std::map<std::string, std::string> mapdata;
	std::map<std::string, std::string>::iterator mi;
	void clear();
public:
	ConfVar();
	std::string value();
	ConfVar& operator=(const char *value);
	ConfVar& operator=(const std::string value);
	ConfVar& operator=(const ConfVar var);
	std::string& operator[](const char *key);
	std::string& operator[](const std::string key);
	bool haskey(const char *key);
	// iteratively access to the map
	int msize();
	void mreset();
	std::string mkey();
	std::string mvalue();
	std::string mnextkey();
};

class ccgConfig{
	map<string, ConfVar> _confVars;
	map<string, ConfVar>::iterator vmi;

public:
	ccgConfig(char * filename);
	~ccgConfig();

	char *confTrim(char *buf);
	int confParse(const char * filename, int lineno, char *buf);
	int confLoad(const char * filename);
	int UrlParse(const char * url);
	void confClear();
	char * confReadV(const char * key, char *store, int slen);
	int confReadInt(const char * key);
	int confMultipleInt(char * buf, int *val, int n);
	int confReadInts(const char * key, int *val, int n);
	int confBoolVal(const char *ptr, int defVal);
	int confReadBool(const char * key, int defVal);
	int confWriteV(const char * key, const char * value);
	void confErase(const char * key);

	int confIsMap(const char * key);
	int confHashKey(const char * mapName, const char *key);
	int confMapSize(const char * mapName);
	char * confMapReadV(const char * nameName, const char *key, char *store, int slen);
	int confMapReadInt(const char *mapName, const char *key);
	int confMapReadInts(const char * mapName, const char *key, int *val, int n);
	int confReadBool(const char * mapName, const char * key, int defVal);
	int confMapWriteV(const char * mapName, const char * key, const char * value);
	void confMapErase(const char * mapName, const char * key);

	void confMapReset(const char * mapName);
	char * confMapKey(const char * mapName, char *keyStore, int kLen);
	char * confMapValue(const char * mapName, char * valStore, int vLen);
	char * confMapNextKey(const char * mapName, char * keyStore, int kLen);
	void confReset();
	const char * confKey();
	const char * confNextKey();
};
#endif

#endif
