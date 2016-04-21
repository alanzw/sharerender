//#include "Initializer.h"
#include "CommandServerSet.h"

// the initializer

Initializer * Initializer::initializer = NULL;
bool Initializer::initialized = false;
bool Initializer::initilizeBegan = false;

Initializer::Initializer(){
	cmdBuf_ = NULL;
	cmdBuffer = new Buffer(30000);
	cmdBuffer->clear();
	cmdBuffer->get_cur_ptr(2);
	sv_ptr = NULL;
	func_count = 0;
}

Initializer::~Initializer(){
	if(cmdBuffer){
		delete cmdBuffer;
		cmdBuffer = NULL;
	}
	if(cmdBuf_){
		delete cmdBuf_;
		cmdBuf_ = NULL;
	}
	sv_ptr = NULL;
	func_count = 0;
}

// push object to list
void Initializer::pushObj(IdentifierBase * obj){
	objList.push_back(obj);
}


// only creation need to be checked
int Initializer::sendCreation(void * ctx){
	infoRecorder->logError("[Initializer]: send creation. %d objects should be created in initialization.\n", objList.size());
	ContextAndCache * ctx_ = (ContextAndCache *)ctx;
	ctx_->flush();

	for(iterator it = objList.begin(); it != objList.end(); it++){
		(*it)->checkCreation(ctx);
		(*it)->checkUpdate(ctx);
	}
	// send the command buffer
	
	ctx_->flush();
	if(cmdBuf_)
		ctx_->send_packet(cmdBuf_);

	return 1;
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
bool Initializer::endInitialize(){
	cmdBuffer->set_count_part(func_count);
	if(func_count){
		cmdBuf_ = cmdBuffer->dumpFixedBuffer();
		cmdBuffer->clear();
		cmdBuffer->get_cur_ptr(2);
		func_count = 0;
	}
	return true;
}

// static functions
bool Initializer::EndInitialize(){
	infoRecorder->logTrace("[Initializer]: end initializer, to check all of the objects in initialize list, total:%d.\n", initializer->objList.size());
	//initializer->checkObjs();

	initialized = true;
#if 0
	initializer->cmdBuffer->set_count_part(initializer->func_count);
	initializer->func_count = 0;
	initializer->cmdBuf_ = initializer->cmdBuffer->dumpFixedBuffer();
	initializer->cmdBuffer->clear();
	initializer->cmdBuffer->get_cur_ptr(2);
#endif
	bool ret = true;
	ret = initializer->endInitialize();

	return ret;
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

void Initializer::BeginCommand(int op_code, int obj_id){
	if(initilizeBegan && !initialized){
		initializer->beginCommand(op_code,obj_id);
	} 
}
void Initializer::EndCommand(){
	if(initilizeBegan && !initialized){
		initializer->endCommand();
	}
}
void Initializer::WriteVec(int op_code, float *vec, int size){
	if(initilizeBegan && !initialized){
		initializer->writeVec(op_code, vec, size);
	}
}
void Initializer::WriteInt(int data){
	if(initilizeBegan && !initialized){
		initializer->writeInt(data);
	}
}

void Initializer::WriteUInt(UINT data){
	if(initilizeBegan && !initialized){
		initializer->writeUInt(data);
	}
}
void Initializer::WriteChar(char data){
	if(initilizeBegan && !initialized){
		initializer->writeChar(data);
	}
}
void Initializer::WriteUChar(UCHAR data){
	if(initilizeBegan && !initialized){
		initializer->writeUChar(data);
	}
}
void Initializer::WriteShort(short data){
	if(initilizeBegan && !initialized){
		initializer->writeShort(data);
	}
}
void Initializer::WriteUShort(USHORT data){
	if(initilizeBegan && !initialized){
		initializer->writeUShort(data);
	}
}
void Initializer::WriteFloat(float data){
	if(initilizeBegan && !initialized){
		initializer->writeFloat(data);
	}

}
void Initializer::WriteByteArr(char *src, int length){
	if(initilizeBegan && !initialized){
		initializer->writeByteArr(src, length);
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


// member function
void Initializer::beginCommand(int op_code, int obj_id){
	sv_ptr = cmdBuffer->get_cur_ptr();
	if(obj_id == 0){
		cmdBuffer->write_uchar((op_code << 1) | 0);

	}
	else{
		cmdBuffer->write_uchar((op_code << 1) | 1);
		cmdBuffer->write_ushort(obj_id);
	}
	func_count++;
}
void Initializer::endCommand(){
	// end of a piece of command
}

void Initializer::writeVec(int op_code, float *vec, int size){
	cmdBuffer->write_byte_arr((char *)vec, size);
}
void Initializer::writeInt(int data){
	cmdBuffer->write_int(data);
}
void Initializer::writeUInt(UINT data){
	cmdBuffer->write_uint(data);
}
void Initializer::writeChar(char data){
	cmdBuffer->write_char(data);
}
void Initializer::writeUChar(UCHAR data){
	cmdBuffer->write_uchar(data);
}
void Initializer::writeShort(short data){
	cmdBuffer->write_short(data);
}
void Initializer::writeUShort(USHORT data){
	cmdBuffer->write_ushort(data);
}
void Initializer::writeFloat(float data){
	cmdBuffer->write_float(data);
}
void Initializer::writeByteArr(char *src, int length){
	cmdBuffer->write_byte_arr(src, length);
}