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
	Buffer * cmdBuf_;  // extra command before any beginScene
	Buffer * cmdBuffer;   // tmp buffer to recorde command

	char * sv_ptr;
	short func_count;


	static bool initialized; 
	static bool initilizeBegan;
	static Initializer *initializer;

	Initializer();
	// push the object to initializer, record until begin scene called
	void pushObj(IdentifierBase * obj);
	//bool checkObjs();
	void beginCommand(int op_code, int obj_id);
	void endCommand();
	
	void writeVec(int op_code, float *vec, int size);
	void writeInt(int data);
	void writeUInt(UINT dawta);
	void writeChar(char data);
	void writeUChar(UCHAR data);
	void writeShort(short data);
	void writeUShort(USHORT data);
	void writeFloat(float data);
	void writeByteArr(char *src, int length);

	bool endInitialize();

public:

	static Initializer * GetInitializer();
	virtual ~Initializer();

	static bool EndInitialize();
	static void BeginInitalize();
	static void PushObj(IdentifierBase *obj);
	static void Reset();

	static void BeginCommand(int op_code, int obj_id);
	static void EndCommand();
	static void WriteVec(int op_code, float *vec, int size);
	static void WriteInt(int data);
	static void WriteUInt(UINT dawta);
	static void WriteChar(char data);
	static void WriteUChar(UCHAR data);
	static void WriteShort(short data);
	static void WriteUShort(USHORT data);
	static void WriteFloat(float data);
	static void WriteByteArr(char *src, int length);
	

	static bool InitializerEnabled(){ return !initialized; }

	inline iterator begin(){ return objList.begin(); }
	inline iterator end(){ return objList.end(); }

	// inherate from parenet
	virtual int sendCreation(void * ctx);
	virtual int sendUpdate(void * ctx);
	virtual int checkCreation(void * ctx);
	virtual int checkUpdate(void * ctx);
};

#endif