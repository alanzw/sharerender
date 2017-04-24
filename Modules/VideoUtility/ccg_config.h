#ifndef __CCG_CONFIG__
#define __CCG_CONFIG__

#include <string>
#include <map>

namespace cg{
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
		void print();
	};

	class ccgConfig{
	protected:
		std::string configFileName;
		std::map<std::string, ConfVar> _confVars;
		std::map<std::string, ConfVar>::iterator vmi;

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

		void print();
	};
}

#endif
