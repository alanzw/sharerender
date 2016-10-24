#ifndef __TESTSMALLHASH_H__
#define __TESTSMALLHASH_H__

// for testing the small hash

#include "../../CloudGamingLiveMigrate/LogicServer/SmallHash.h"

// the test class type
class TestClass{
	char * context;
	int index;

public:
	TestClass(int _index, char * _context){
		context = NULL;
		index = _index;
		context = _strdup(_context);
	}
	~TestClass(){
		if(context){
			free(context);
			context = NULL;
		}
		index = 0;
	}
	inline char * getContext(){ return context; }
	inline int getIndex(){ return index; }
	inline bool display(){
		printf("[TestClass]: (index = %d, context = %s ", index, context);
		return true;
	}
};


void TestSmallHash();

#endif