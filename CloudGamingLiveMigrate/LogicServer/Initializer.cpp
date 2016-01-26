//#include "Initializer.h"
#include "CommandServerSet.h"

// the initializer

Initializer * Initializer::initializer = NULL;
bool Initializer::initialized = false;
bool Initializer::initilizeBegan = false;

Initializer::Initializer(){
	
}

Initializer::~Initializer(){

}

// push object to list
void Initializer::pushObj(IdentifierBase * obj){
	objList.push_back(obj);
}


// only creation need to be checked
int Initializer::sendCreation(void * ctx){
	infoRecorder->logError("[Initializer]: send creation. %d objects should be created in initialization.\n", objList.size());
	for(iterator it = objList.begin(); it != objList.end(); it++){
		(*it)->checkCreation(ctx);
		(*it)->checkUpdate(ctx);
	}

	return 0;
}
int Initializer::sendUpdate(void * ctx){
	infoRecorder->logTrace("[Initializer]: send update, TODO.\n");
	return 0;
}
int Initializer::checkCreation(void * ctx){
	infoRecorder->logTrace("[Initializer]: check creation.\n");
	ContextAndCache * cc = (ContextAndCache *)ctx;
	int ret = 0;
	if(!cc->isChanged(creationFlag)){
		ret = sendCreation(ctx);	
		cc->setCreation(creationFlag);
		ret = 1;
	}
	return ret;
}
int Initializer::checkUpdate(void * ctx){
	infoRecorder->logTrace("[Initializer]: check update, TODO.\n");
	int ret = 0;
	return ret;
}


// static functions
bool Initializer::EndInitialize(){
	infoRecorder->logTrace("[Initializer]: end initializer, to check all of the objects in initialize list, total:%d.\n", initializer->objList.size());
	//initializer->checkObjs();

	initialized = true;
	return true;
}

void Initializer::BeginInitalize(){
	initilizeBegan = true;
	if(!initializer){
		initializer = new Initializer();
	}
}
void Initializer::PushObj(IdentifierBase *obj){
	if(initilizeBegan){
		if(!initialized)
			initializer->pushObj(obj);
		else{
			// already initialized, cannot push object
		}
	}
}
// if initialized not complete, cannot the object
Initializer * Initializer::GetInitializer(){
	if(!initialized){
		infoRecorder->logTrace("[Initializer]: not initialized yet, will return NULL.\n");
		if(!initializer){
			initializer = new Initializer();
		}
		else{
			//

		}
		return NULL;
	}
	else{
		if(!initializer){
			// error
			infoRecorder->logError("[Initializer]: initialize done, but get NULL pointer, ERROR.\n");
		}
		return initializer;
	}

}

void Initializer::Reset(){
	initialized = false;
	initilizeBegan = false;
	if(initializer){
		initializer->objList.clear();
	}
}