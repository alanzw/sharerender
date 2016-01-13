#ifndef __INITIALIZER_H__
#define __INITIALIZER_H__

// the initializer for float context

// this definition will put all the objects that created before the first call to BeginScene to the initializer, if not define, only the Render Target and Depth Stencil Surface will.
//#define INITIAL_ALL_RESOURCE

#include <list>
#include "GameServer.h"
class Initializer : public IdentifierBase{
	typedef std::list<IdentifierBase *>::iterator iterator;
	std::list<IdentifierBase *> objList;
	//unsigned int initialFlags;   // the flag for each context

	static bool initialized; 
	static bool initilizeBegan;
	static Initializer *initializer;

	Initializer();
	// push the object to initializer, record until begin scene called
	void pushObj(IdentifierBase * obj);
	//bool checkObjs();
public:

	static Initializer * GetInitializer();
	virtual ~Initializer();

	static bool EndInitialize();
	static void BeginInitalize();
	static void PushObj(IdentifierBase *obj);
	static void Reset();

	inline iterator begin(){ return objList.begin(); }
	inline iterator end(){ return objList.end(); }

	// inherate from parenet
	virtual int sendCreation(void * ctx);
	virtual int sendUpdate(void * ctx);
	virtual int checkCreation(void * ctx);
	virtual int checkUpdate(void * ctx);
};

#endif