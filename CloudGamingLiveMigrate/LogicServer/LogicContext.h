#ifndef __LOGICCONTEXT_H__
#define __LOGICCONTEXT_H__

// logic context to store all the object and the flags for each clients
// use map , and use the hash

#include <map>
#include "../LibDistrubutor/Context.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

using namespace std;

///////////////////////////// Process Context /////////////////////////////
// process context runs inside game process, only connect to loader and execute the cmd from loader
// define the cmd between loader and game process

// feedback
extern const char * PROCESS_STARTED;
extern const char * RENDER_EXIT;

///////////////////////////// LogicContext /////////////////////////////
template <class Type> 
class LogicContext{
	static map<void *, void *> objMap;   // all the object map, Wrapper to find IDirect
	map<void *, bool> creationMap; /// the object create flag map for each context
	Type identifier;   // the identifier of the context
	bool frameLock;   // identify that the logic context is from the begin scene or not
public:
	// initially the context for a client is locked, and can be unlocked in present, then, everything is OK,
	inline bool lock(){
		// lock the context
		// called by initializing
		frameLock = true;
	}
	// called by present
	inline bool unlock(){
		// unlock the context
		frameLock = false;
		return true;
	}

	inline bool isLocked(){ return frameLock; }
	inline bool ableToSend(bool sceneBegined){ }
	// set the frame lock to true only when the context call begin scene
	inline void setLock(bool value){ frameLock =  value; }
	inline void setIdentifier(Type s){ this->identifier = s; }
	inline Type getIdentifier(){ return this->identifier; }
	inline void addMap(void * key, bool value){ creationMap[key] = value; }

	LogicContext<Type>(Type ident): identifier(ident){
		frameLock = false;
	}
	~LogicContext(){ creationMap.clear(); }

	bool isCreated(void *key){
		map<void *, bool>::iterator mi;
		mi = this->creationMap.find(key);
		if(mi->second == true){
			return true;
		}
		else{
			return false;
		}
	}
};

#endif
