#include "Config.h"
#include "Commonwin32.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"

#define PATH_MAX 128

/////////////////////////////////////////////////////////////////////

ccgConfig::ccgConfig(char * filename){

}
ccgConfig::~ccgConfig(){

}

char * ccgConfig::confTrim(char *buf){
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
int ccgConfig::confParse(const char * filename, int lineno, char *buf){
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
		infoRecorder->logTrace("[ccgConfig]: %s[%s] = %s\n", option, leftbracket, token);
		Log::logscreen("[ccgConfig]: %s[%s] = %s\n", option, leftbracket, token);
		_confVars[option][leftbracket] = token;
	}
	else {
		infoRecorder->logTrace("[ccgConfig]: %s = %s\n", option, token);
		Log::logscreen("[ccgConfig]: %s = %s\n", option, token);
		_confVars[option] = token;
	}
	return 0;
}
int ccgConfig::confLoad(const char * filename){
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
int ccgConfig::UrlParse(const char * url){
	char *ptr, servername[1024], serverport[64];
	//
	if (url == NULL)
		return -1;
	if (strncasecmp("rtsp://", url, 7) != 0)
		return -1;
	strncpy(servername, url + 7, sizeof(servername));
	for (ptr = servername; *ptr; ptr++) {
		if (*ptr == '/') {
			*ptr = '\0';
			break;
		}
		if (*ptr == ':') {
			unsigned i;
			*ptr = '\0';
			for (++ptr, i = 0;
				isdigit(*ptr) && i < sizeof(serverport)-1;
				i++) {
				//
				serverport[i] = *ptr++;
			}
			serverport[i] = '\0';
			confWriteV("server-port", serverport);
			break;
		}
	}
	confWriteV("server-url", url);
	confWriteV("server-name", servername);
	return 0;
}
void ccgConfig::confClear(){
	_confVars.clear();
	vmi = _confVars.begin();
	return;
}
char * ccgConfig::confReadV(const char * key, char *store, int slen){
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
int ccgConfig::confReadInt(const char * key){
	char buf[64];
	char *ptr = confReadV(key, buf, sizeof(buf));
	if (ptr == NULL)
		return INT_MIN;
	return strtol(ptr, NULL, 0);
}
int ccgConfig::confMultipleInt(char * buf, int *val, int n){
	int reads = 0;
	char *endptr, *ptr = buf;
	while (reads < n) {
		val[reads] = strtol(ptr, &endptr, 0);
		if (ptr == endptr)
			break;
		ptr = endptr;
		reads++;
	}
	return reads;
}
int ccgConfig::confReadInts(const char * key, int *val, int n){
	char buf[1024];
	char *ptr = confReadV(key, buf, sizeof(buf));
	if (ptr == NULL)
		return 0;
	return confMultipleInt(buf, val, n);
}
int ccgConfig::confBoolVal(const char *ptr, int defVal){
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
int ccgConfig::confReadBool(const char * key, int defVal){
	char buf[64];
	char *ptr = confReadV(key, buf, sizeof(buf));
	if (ptr == NULL)
		return defVal;
	return confBoolVal(ptr, defVal);
}
int ccgConfig::confWriteV(const char * key, const char * value){
	_confVars[key] = value;
	return 0;
}
void ccgConfig::confErase(const char * key){
	_confVars.erase(key);
	return;
}


int ccgConfig::confIsMap(const char * key){
	return confMapSize(key) > 0 ? 1 : 0;
}
int ccgConfig::confHashKey(const char * mapName, const char *key){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return 0;
	return mi->second.haskey(key);
}
int ccgConfig::confMapSize(const char * mapName){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return 0;
	return mi->second.msize();
}
char * ccgConfig::confMapReadV(const char * mapName, const char *key, char *store, int slen){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return NULL;
	if ((mi->second)[key] == "")
		return NULL;
	if (store == NULL)
		return _strdup((mi->second)[key].c_str());
	strncpy(store, (mi->second)[key].c_str(), slen);
	return store;
}
int ccgConfig::confMapReadInt(const char *mapName, const char *key){
	char buf[64];
	char *ptr = confMapReadV(mapName, key, buf, sizeof(buf));
	if (ptr == NULL)
		return INT_MIN;
	return strtol(ptr, NULL, 0);
}
int ccgConfig::confMapReadInts(const char * mapName, const char *key, int *val, int n){
	char buf[1024];
	char *ptr = confMapReadV(mapName, key, buf, sizeof(buf));
	if (ptr == NULL)
		return 0;
	return confMultipleInt(buf, val, n);
}
int ccgConfig::confReadBool(const char * mapName, const char * key, int defVal){
	char buf[64];
	char *ptr = confMapReadV(mapName, key, buf, sizeof(buf));
	if (ptr == NULL)
		return defVal;
	return confBoolVal(ptr, defVal);
}
int ccgConfig::confMapWriteV(const char * mapName, const char * key, const char * value){
	_confVars[mapName][key] = value;
	return 0;
}
void ccgConfig::confMapErase(const char * mapName, const char * key){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return;
	_confVars.erase(mi);
	return;
}

void ccgConfig::confMapReset(const char * mapName){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return;
	mi->second.mreset();
	return;
}
char * ccgConfig::confMapKey(const char * mapName, char *keyStore, int kLen){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return NULL;
	if (mi->second.mkey() == "")
		return NULL;
	if (keyStore == NULL)
		return _strdup(mi->second.mkey().c_str());
	strncpy(keyStore, mi->second.mkey().c_str(), kLen);
	return keyStore;
}
char * ccgConfig::confMapValue(const char * mapName, char * valStore, int vLen){
	map<string, ConfVar>::iterator mi;
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return NULL;
	if (mi->second.mkey() == "")
		return NULL;
	if (valStore == NULL)
		return _strdup(mi->second.mvalue().c_str());
	strncpy(valStore, mi->second.mvalue().c_str(), vLen);
	return valStore;
}
char * ccgConfig::confMapNextKey(const char * mapName, char * keyStore, int kLen){
	map<string, ConfVar>::iterator mi;
	string k = "";
	//
	if ((mi = _confVars.find(mapName)) == _confVars.end())
		return NULL;
	k = mi->second.mnextkey();
	if (k == "")
		return NULL;
	if (keyStore == NULL)
		return _strdup(k.c_str());
	strncpy(keyStore, k.c_str(), kLen);
	return keyStore;
}
void ccgConfig::confReset(){
	vmi = _confVars.begin();
}
const char * ccgConfig::confKey(){
	if (vmi == _confVars.end())
		return NULL;
	return vmi->first.c_str();
}
const char * ccgConfig::confNextKey(){
	if (vmi == _confVars.end())
		return NULL;
	// move forward
	vmi++;
	//
	if (vmi == _confVars.end())
		return NULL;
	return vmi->first.c_str();
}




///////// for Class ConfVar ////////////////////
void ConfVar::clear(){
	this->data = "";
	this->mapdata.clear();
	this->mi = this->mapdata.begin();
}

ConfVar::ConfVar() {
	this->clear();
}

string
ConfVar::value() {
	return this->data;
}

ConfVar&
ConfVar::operator=(const char *value) {
	this->data = value;
	this->mapdata.clear();
	this->mi = this->mapdata.begin();
	return *this;
}

ConfVar&
ConfVar::operator=(const string value) {
	this->data = value;
	this->mapdata.clear();
	this->mi = this->mapdata.begin();
	return *this;
}

ConfVar&
ConfVar::operator=(const ConfVar var) {
	this->data = var.data;
	this->mapdata = var.mapdata;
	this->mi = this->mapdata.begin();
	return *this;
}

string&
ConfVar::operator[](const char *key) {
	return mapdata[key];
}

string&
ConfVar::operator[](const string key) {
	return mapdata[key];
}

bool
ConfVar::haskey(const char *key) {
	return (mapdata.find(key) != mapdata.end());
}

int
ConfVar::msize() {
	return this->mapdata.size();
}

void
ConfVar::mreset() {
	this->mi = this->mapdata.begin();
	return;
}

string
ConfVar::mkey() {
	if (this->mi == this->mapdata.end())
		return "";
	return mi->first;
}

string
ConfVar::mvalue() {
	if (this->mi == this->mapdata.end())
		return "";
	return mi->second;
}

string
ConfVar::mnextkey() {
	if (this->mi == this->mapdata.end())
		return "";
	// move forward
	this->mi++;
	//
	if (this->mi == this->mapdata.end())
		return "";
	return mi->first;
}
