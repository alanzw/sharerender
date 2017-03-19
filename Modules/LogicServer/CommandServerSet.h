#ifndef __COMMAND_SERVER_SET_H__
#define __COMMAND_SERVER_SET_H__

#include "../LibCore/HashSet.h"
#include "LogicContext.h"
#include "../LibCore/CommandServer.h"
#include "../LibCore/SmallHash.h"
#include "../LibCore/LinkedList.h"
#include "TaskQueue.h"
//#include "GameServer.h"
#include "Initializer.h"
#include "../LibCore/Network.h"
#include "../LibCore/Buffer.h"
#include "../LibCore/InfoRecorder.h"
#include "../libCore/TimeTool.h"

using namespace std;
using namespace cg;
using namespace cg::core;

// define the hot plug, this definition will enable logic server to hot plug render proxy anytime

// HOT_PLUG means add render proxy whenever needed
//#define ENABLE_HOT_PLUG
#define SINGLE_CONTEXT

// context lock will lock the context when needed.
//#define _CONTEXT_LOCK_

#define BUFFER_UNLOCK_UPDATE   // this definition will enable system memory to buffer the udpate of VB and IB

#ifndef MULTI_CLIENTS
#define MULTI_CLIENTS
#endif

// this definition will enable log for ref count
#if 1
#ifndef LOG_REF_COUNT
#define LOG_REF_COUNT
#endif
#endif



//#define USE_CACHE
//#define DISABLE_SHADOW_OBJECT

/// logs
#if 0
#define ENBALE_DEVICE_LOG
#define ENABLE_DEVICE_LOG
#define ENABLE_STATE_BLOCK_LOG
#define ENABLE_PIXEL_SHADER_LOG
#define ENABLE_INDEX_LOG
#define ENABLE_SURFACE_LOG
#define ENABLE_SWAP_CHAIN_LOG
#define ENABLE_TEXTURE_LOG
#define ENABLE_VERTEX_BUFFER_LOG
#define ENABLE_VERTEX_DECLARATION_LOG
#define ENABLE_VERTEX_SHADER_LOG
#endif

//#define ENABLE_TEXTURE_LOG
//#define ENBALE_DEVICE_LOG
//#define ENABLE_VERTEX_BUFFER_LOG
//#define ENABLE_INDEX_LOG

enum CtxStatus{
	CTX_INIT,   // the initial status
	CTX_PREPARE, // in the proceeding of preparation
	CTX_READY   // data OK.
};

enum ServerStatus{
	SERVER_INIT,
	SERVER_READY
};

// thread safe queue

class ContextAndCache : public cg::core::Network, public cg::core::Buffer{
public:
	int				index;   // the index of the context
	Cache *			cache_mgr_;
	CtxStatus		status;

	// helper task queue to check the creation when the ctx is INIT, and check the update when READY
	TaskQueue *		taskQueue;
	HANDLE			lockMutex;   // the mutex to lock the context

	int				op_code;
	int				obj_id;
	int				sv_obj_id;
	char *			sv_ptr;
	char *			cm_len_ptr;

	char *			rid_pos;
	short			func_count_;
	int				size_limit_;

	// the preparation time measurment
	DWORD			start, end;
	inline void		preperationStart(){ start = GetTickCount(); }
	inline void		preperationEnd(){ 
		end = GetTickCount(); 
		infoRecorder->logError("ctx init(ms): %d.\n", end - start );
		taskQueue->framePrint();
	}

	int				getCacheIndex(const char * buffer, size_t size, int & hit_id, int & rep_id);

	inline void		setIndex(int i){ index = i;}
	inline int		getIndex(){ return index; }
	inline void		lock(){
#ifdef _CONTEXT_LOCK_
		WaitForSingleObject(lockMutex, INFINITE); 
		infoRecorder->logTrace("[ContextAndCache]: lock the context.\n");
#endif
	}
	inline void		unlock(){ 
#ifdef _CONTEXT_LOCK_
		ReleaseMutex(lockMutex); 
		infoRecorder->logTrace("[ContextAndCache]: unlock the context.\n");
#endif
	}
	ContextAndCache();
	ContextAndCache(int _index);
	~ContextAndCache();

	// for bit map
	inline bool		isCreated(unsigned int mask){ return (bool)((1 << index) & mask); }
	inline void		resetCreation(unsigned int & mask){ mask &= ((-1)^(1<<index)); }
	inline void		setCreation(unsigned int & mask){ mask |= (1 << index); }
	inline bool		isChanged(unsigned int mask){ return (1<<index)&mask;	}
	inline bool		isChecked(unsigned int mask){ return (1<<index)&mask;   }
	inline void		setChecked(unsigned int &mask){ mask |= (1 << index);   }

	// set bit to 0
	inline void		resetChecked(unsigned int &mask){ mask &= ((-1)^(1<<index)); }
	inline void		resetChanged(unsigned int &mask){ mask &= ((-1)^(1<<index)); }
	inline void		setChanged(unsigned int &mask){ mask |= (1 << index);}
	inline void		setChangedToAll(unsigned int &mask){ mask = 0x8fffffff; }

	inline bool		isSend(unsigned int mask){ return (1 << index) & mask; }
	inline void		setSend(unsigned int &mask){ mask |= (1 << index);}

	void			write_vec(int op_code, float *vec, int size, CommandRecorder *cr_);

	int				flush(CommandRecorder *cr);
	int				flush();

	inline void		shutDown(CommandRecorder *cr){
		beginCommand(MaxSizeUntilNow_Opcode, 0);
		endCommand(cr, 1);
	}
	inline int		getCommandLength(){ return get_cur_ptr() - sv_ptr; }
	inline char *	getCurPtr(int length){ return get_cur_ptr(length);	}

	Buffer *		dumpBuffer();

	// the begin command and the end command
	void			beginCommand(int _op_code, int id);
	void			endCommand(int force_flush = 0);
	void			endCommand(CommandRecorder * cr, int force_flush = 0);
	inline void		cancelCommand(){
		go_back(sv_ptr);
		func_count_--;
		unlock();
	}
	void			eraseFlag();    // this function achieved in ServerInit.cpp
	void			checkFlags();   // check all related objects
	void			checkObj(IdentifierBase * obj);  // check the obj, created? 
	void			updateObj(IdentifierBase * obj);   // check the object's update
	void			pushUpdate(IdentifierBase * obj);   // push the object to the queue
	void			pushSync(IdentifierBase * obj); //push the sync object to the queue

	// override
	bool operator < (ContextAndCache & t1)const{
		if(index < t1.index)
			return true;
		else
			return false;
	}

	bool operator > (ContextAndCache &t1)const{
		if(index > t1.index)
			return true;
		else
			return false;
	}

	void write_packed_byte_arr(char * src, int length);
};

// the empty context is for standalone use or test the context build time
class EmptyContext : ContextAndCache{

public:

};

class IndexManager{
	static IndexManager * _indexMgr;
	unsigned int	_maxIndex;
	unsigned int	_bitmap;
	IndexManager(int maxIndex);
public:
	static IndexManager *GetIndexManager(int maxIndex);
	int				getAvailableIndex();
	bool			resetIndex(int index);
};

class IndexedContextPool{
	int ctxCount;
	ContextAndCache * array[MAX_RENDER_COUNT * 2];
	int curIndex;
	ContextAndCache * curCtx;
	CRITICAL_SECTION section;

public:
	IndexedContextPool(){
		ctxCount = 0;
		curIndex = -1;
		curCtx = NULL;
		InitializeCriticalSection(&section);

		for(int i = 0; i< MAX_RENDER_COUNT * 2; i++){
			array[i] = NULL;
		}
	}
	~IndexedContextPool(){}
	int getCtxCount(){
		lock();
		unlock();
		return ctxCount;
	}
	void lock(){ EnterCriticalSection(&section); }
	void unlock(){ LeaveCriticalSection(&section); }
	// re-order the list, by the context's id
	bool sort(){
		ContextAndCache * t = NULL;
		for(int i = 1; i < ctxCount; i++){
			if(array[0]->getIndex() > array[i]->getIndex()){
				t = array[0];
				array[0] = array[i];
				array[i] = t;
			}
		}
		return true;
	}

	bool add(ContextAndCache * ctx){
		bool ret = false;
		lock();
		if((ctxCount + 1) < (MAX_RENDER_COUNT * 2)){
			array[ctxCount++ ] = ctx;
			ret = true;
		}
		else{
			// too many context.
			ret = false;
		}
		unlock();
		return ret;
	}
	ContextAndCache * remove(ContextAndCache * ctx){
		lock();
		int i = 0;
		ContextAndCache * toRemove = NULL;
		// find the ctx
		for(i = 0; i < ctxCount; i++){
			if(array[i] == ctx){
				toRemove = array[i];
				break;
			}
		}
		if(toRemove){
			infoRecorder->logTrace("[IndexContextPool]: find the ctx to remove, idx:%d.\n", i);
		}
		// remove from array
		array[i] = NULL;
		for(int j = i; j < ctxCount - 1; j ++){
			array[j] = array[j + 1];
			array[j + 1] = 0;
		}
		ctxCount--;
		sort();
		unlock();
		return toRemove;
	}

	// for test
	ContextAndCache * remove(int index){
		lock();
		index = (index % ctxCount);
		ContextAndCache * toRemove = array[index];
		for(int j = 0; j < ctxCount - 1; j ++){
			array[j] = array[j + 1];
		}
		ctxCount--;
		sort();
		unlock();
		return toRemove;
	}

	ContextAndCache * getCurCtx(){
		return curCtx;
	}

	ContextAndCache * getNextCtx(){
		lock();
		if(ctxCount <= 0){
			unlock();
			return NULL;
		}
		// get the current work context
		if(curIndex == -1){
			curIndex = 0;
		}
		else{
			curIndex++;
			int t = curIndex;
			curIndex = t % ctxCount;
		}
		curCtx = array[curIndex];
		unlock();
		return curCtx;
	}

	void printError(){
		infoRecorder->logError("[IndexContextPool]: ctx count:%d, cur index:%d, cur ctx:%p.\n", ctxCount, curIndex, array[curIndex]);
	}

	ContextAndCache * getCtx(int index){
		//print();
		return array[index];
	}
	// print the pool
	void print(){
		infoRecorder->logTrace("[IndexedContextPool]: the pool has '%d' contexts.\n", ctxCount);
	}
};

class ContextManager{
	int ctxCount;   //the count of ctxs in pool;
	IndexedContextPool ctx_pool;     // hold the context with ready status
	IndexedContextPool ctx_init;    // hold the context that do the init job
	IndexedContextPool ctx_buff;	// hold the new added context which is not insert into the init pool
	ContextAndCache * _ctx_cache;   // current ContextAndCache, status is READY
	SmallHash<SOCKET, bool> socketMap;   /// to find whether the socket exist
	//ContextAndCache * ctx_array[MAX_RENDER_COUNT * 2];   // all ctx is in the array

	// helper 
	CommandRecorder * cr_;
public:

	// for travel
	ContextManager(){
		infoRecorder->logTrace("[CotextManager]: constructor.\n");
		cr_ = new CommandRecorder();
		_ctx_cache = NULL;
		ctxCount = 0;
	}
	~ContextManager(){
		if(cr_){
			delete cr_;
			cr_= NULL;
		}
	}
	bool declineCtx(SOCKET s);
	// switch the current work context, select another context with CTX_READY in ctx_pool
	bool switchCtx();
	// add a new exist context to init pool
	int addCtx(ContextAndCache  * _ctx);

	//// work for current ctx and cache
	inline bool isCached(int _op_code){
		return _ctx_cache ? isCached(_op_code, _ctx_cache) : false;
		//return isCached(_op_code, _ctx_cache);
	}

	inline void getCacheIndex(const char * buffer, size_t size, int &hit_id, int &rep_id){
		return getCacheIndex(buffer, size, hit_id, rep_id, _ctx_cache);
	}

	inline int sendPacket(Buffer * buf){
		return _ctx_cache ? sendPacket(buf, _ctx_cache) : 0;
	}
	// called when init
	inline void setCacheFilter(){
		if(_ctx_cache)
			return setCacheFilter(_ctx_cache);
	}

	inline bool isSocketMapped(SOCKET s){
		return socketMap.getValue(s);
	}
	inline bool mapSocket(SOCKET s){
		return socketMap.addMap(s, true);
	}
	inline bool unmapSocket(SOCKET s){
		return socketMap.unMap(s);
	}

	inline SOCKET getCtxSocket(){ return _ctx_cache ? _ctx_cache->get_connect_socket() : NULL; }
	
	inline bool isCreated(unsigned int flag){
		return _ctx_cache ? isCreated(flag, _ctx_cache) : true;
	}

	inline void setCreation(unsigned int &flag){
		if(_ctx_cache)
			setCreation(flag, _ctx_cache);
	}

	void checkObj(IdentifierBase * obj);  // check the obj in each context
	void pushSync(IdentifierBase * obj); //push the sync object to the queue
	void checkCreation(IdentifierBase * obj);// only check the creation for the object.
	// for buffers that need to update or send data for once
	inline bool		isChanged(unsigned int flag){
		return _ctx_cache ? isChanged(flag, _ctx_cache) : false;
	}
	inline bool		isSend(unsigned int flag){
		return _ctx_cache ? isSend(flag, _ctx_cache) : true;
	}
	inline void		setSend(unsigned int &flag){
		if(_ctx_cache)
			setSend(flag, _ctx_cache);
	}
	inline void		resetChanged(unsigned int &flag){
		if(_ctx_cache)
			resetChanged(flag, _ctx_cache);
	}
	inline void		setChanged(unsigned int &flag){
		if(_ctx_cache)
			setChanged(flag, _ctx_cache);
	}
	inline void		setChangedToAll(unsigned int &flag){
		if(_ctx_cache)
			setChangedToAll(flag, _ctx_cache);
	}

	// buffer operation
	///// buffer operation
	inline int		readInt(){
		return _ctx_cache ? readInt(_ctx_cache) : -1;
	}
	inline UINT		readUInt(){
		return _ctx_cache ? readUInt(_ctx_cache) : 0;
	}
	inline char		readChar(){
		return _ctx_cache ? readChar(_ctx_cache) : 0;
	}
	inline UCHAR	readUChar(){
		return _ctx_cache ? readUChar(_ctx_cache) : 0;
	}

	inline short	readShort(){
		return _ctx_cache ? readShort(_ctx_cache) : -1;
	}
	inline USHORT	readUShort(){
		return _ctx_cache ? readUShort(_ctx_cache) : 0;
	}
	inline float	readFloat(){
		return _ctx_cache ? readFloat(_ctx_cache) : (float)-1.0;
	}
	inline void		readByteArr(char * dst, int length){
		if(_ctx_cache)
			readByteArr(dst, length, _ctx_cache);
	}

	// write to all clients, write the common buffer inside command server
	inline void		writeInt(int data){
		if(_ctx_cache)
		writeInt(data, _ctx_cache);
	}

	inline void		writeUInt(UINT data){
		if(_ctx_cache)
		writeUInt(data, _ctx_cache);
	}
	inline void		writeChar(char data){
		if(_ctx_cache)
		writeChar(data, _ctx_cache);
	}
	inline void		writeUChar(UCHAR data){
		if(_ctx_cache)
		writeUChar(data, _ctx_cache);
	}
	inline void		writeShort(short data){
		if(_ctx_cache)
		writeShort(data, _ctx_cache);
	}

	inline void		writeUShort(USHORT data){
		if(_ctx_cache)
		writeUShort(data, _ctx_cache);
	}
	inline void		writeFloat(float data){
		if(_ctx_cache)
		writeFloat(data, _ctx_cache);
	}
	inline void		writeByteArr(char * src, int length){
		if(_ctx_cache)
		writeByteArr(src, length, _ctx_cache);
	}
	inline void		writePackedByteArr(char *src, int len){
		if(_ctx_cache)
			writePackedByteArr(src, len, _ctx_cache);
	}
	inline void		writeVec(int op_code, float * vec, int size = 16){
		if(_ctx_cache)
		writeVec(op_code, vec, _ctx_cache, size);
	}
	// TODO
	inline void		shutDown(){
		if(_ctx_cache)
		shutDown(_ctx_cache);
		cr_->print_info();
	}
	void			shutDownAll(){
		// shutdown all

		cr_->print_info();
	}

	inline int			getCommandLength(){
		
		return _ctx_cache ? getCommandLength(_ctx_cache) : 0;
	}
	inline char *		getCurPtr(int length){
		return _ctx_cache ? getCurPtr(length, _ctx_cache) : NULL;
	}
	inline int			flush(){
		return _ctx_cache  ? flush(_ctx_cache) : 0;
	}

	inline Buffer *		dumpBuffer(){
		return _ctx_cache ? dumpBuffer(_ctx_cache) : NULL;
	}

	// the begin command and the end command
	inline void			beginCommand(int op_code, int id){
		//infoRecorder->logTrace("[ContextManager]: begin command.\n");
		if(_ctx_cache)
			beginCommand(op_code, id, _ctx_cache);
	}
	inline void			endCommand(int force_flush = 0){
		if(_ctx_cache)
			endCommand(_ctx_cache, force_flush);
	}
	inline void			cancelCommand(){
		if(_ctx_cache)
			cancelCommand(_ctx_cache);
	}
	inline void printRecord(){
		cr_->print_info();
	}

	
private:
	///// work for certain context
	inline bool			isCached(int _op_code, ContextAndCache * ctx){
		infoRecorder->logTrace("[ContextManager]: isCached.\n");
		return ctx->cache_filter[_op_code];
	}
	inline void			getCacheIndex(const char * buffer, size_t size, int &hit_id, int &rep_id, ContextAndCache * ctx){
		infoRecorder->logTrace("[ContextManager]: getCacheIndex.\n");
		ctx->getCacheIndex(buffer, size, hit_id, rep_id);
		//return ctx->cache_mgr_->get_cache_index(buffer, size, hit_id, rep_id);
	}
	inline int			sendPacket(Buffer * buf, ContextAndCache * ctx){
		infoRecorder->logTrace("[ContextManager]: send packet.\n");
		return ctx->send_packet(buf);
	}
	// called when init
	inline void			setCacheFilter(ContextAndCache * ctx){ return ctx->set_cache_filter();}
	inline bool			isCreated(unsigned int flag, ContextAndCache * ctx){ return ctx->isCreated(flag); }
	inline void			setCreation(unsigned int & flag, ContextAndCache * ctx){
		ctx->setCreation(flag);
	}
	inline bool			isSend(unsigned int flag, ContextAndCache * ctx){
		return ctx->isSend(flag);
	}
	inline bool			isChanged(unsigned int flag, ContextAndCache * ctx){
		return ctx->isChanged(flag);
	}
	inline void			setSend(unsigned int & flag, ContextAndCache * ctx){
		ctx->setSend(flag);
	}
	inline void			setChanged(unsigned int &flag, ContextAndCache * ctx){
		ctx->setChanged(flag);
	}
	inline void			resetChanged(unsigned int &flag, ContextAndCache * ctx){
		ctx->resetChanged(flag);
	}
	inline void			setChangedToAll(unsigned int & flag, ContextAndCache * ctx){
		ctx->setChangedToAll(flag);
	}
	///// buffer operation
	inline int			readInt(ContextAndCache *ctx){
		return ctx->read_int();
	}
	inline UINT			readUInt(ContextAndCache *ctx){
		return ctx->read_uint();
	}
	inline char			readChar(ContextAndCache *ctx){
		return ctx->read_char();
	}
	inline UCHAR		readUChar(ContextAndCache *ctx){
		return ctx->read_uchar();
	}

	inline short		readShort(ContextAndCache *ctx){
		return ctx->read_short();
	}
	inline USHORT		readUShort(ContextAndCache *ctx){
		return ctx->read_ushort();
	}
	inline float		readFloat(ContextAndCache *ctx){
		return ctx->read_float();
	}
	inline void			readByteArr(char * dst, int length, ContextAndCache *ctx){
		ctx->read_byte_arr(dst, length);
	}

	// write to all clients, write the common buffer inside command server
	inline void			writeInt(int data, ContextAndCache *ctx){
		ctx->write_int(data);
	}
	inline void			writeUInt(UINT data, ContextAndCache *ctx){
		ctx->write_uint(data);
	}
	inline void			writeChar(char data, ContextAndCache *ctx){
		ctx->write_char(data);
	}
	inline void			writeUChar(UCHAR data, ContextAndCache *ctx){
		ctx->write_uchar(data);
	}
	inline void			writeShort(short data, ContextAndCache *ctx){
		ctx->write_short(data);
	}
	inline void			writeUShort(USHORT data, ContextAndCache *ctx){
		ctx->write_ushort(data);
	}
	inline void			writeFloat(float data, ContextAndCache *ctx){
		ctx->write_float(data);
	}
	inline void			writeByteArr(char * src, int length, ContextAndCache *ctx){ ctx->write_byte_arr(src, length); }
	inline void			writePackedByteArr(char *src, int len, ContextAndCache *ctx){
		ctx->write_packed_byte_arr(src, len); }

	inline void			writeVec(int op_code, float * vec, ContextAndCache * ctx, int size = 16){ ctx->write_vec(op_code, vec, size,  cr_); }
	inline void			shutDown(ContextAndCache *ctx){ ctx->shutDown(cr_); }
	inline int			getCommandLength(ContextAndCache *ctx){ return ctx->getCommandLength(); }
	inline char *		getCurPtr(int length, ContextAndCache *ctx){ return ctx->getCurPtr(length); }
	inline int			flush(ContextAndCache *ctx){ return ctx->flush(cr_); }
	inline Buffer *		dumpBuffer(ContextAndCache *ctx){ return ctx->dumpBuffer();}

	// the begin command and the end command
	inline void			beginCommand(int op_code, int id, ContextAndCache *ctx){ ctx->beginCommand(op_code, id); }
	inline void			endCommand(ContextAndCache * ctx, int force_flush = 0){	ctx->endCommand(cr_, force_flush); }
	inline void			cancelCommand(ContextAndCache *ctx){ return ctx->cancelCommand(); }
};

// this is for the command server set to support multiple clients
class CommandServerSet{
	ContextManager * ctx_mgr_;   // the context manager
	ServerConfig * config_;

	int op_code;

	int clients;
	fd_set fdSocket;

	HANDLE mutex;
	CommandServerSet(int _limit_size = Max_Buf_Size);
	// prepare another buffer, send the creation information to the non-active context, dynamically create and destroy

	HANDLE ctxEvent;

public:
	bool addNewRender; // the flag to identify whether there is a new added render proxy, set true in unlockNew when there is any new render added
	bool checkFlag;  // in SwitchCtx, when addNewRender is true, the next frame will check checkFlag, if true, prepare the data for new added render proxy

	ServerStatus status;

	// the buffers for new added servers
	int newAddedClients;
	SOCKET newAddedSocket[MAX_CLIENTS];
	SmallHash<SOCKET, CommandServer *> newAddedSet;
	SmallHash<SOCKET, LogicContext<SOCKET> *> newAddedContextSet;

	// lock all 
	bool			lock();
	void			unlock();
	inline void		waitCtxEvent(){
		infoRecorder->logTrace("[CommnadServerSet]: wait event:%p.\n", ctxEvent);
		WaitForSingleObject(ctxEvent, INFINITE);
	}
	inline void		setCtxEvent(){
		// start queue proc to deal with update or create task for objects
		SetEvent(ctxEvent);
	}
	
	~CommandServerSet();

	static CommandServerSet * csSet;
	static CommandServerSet * GetServerSet(){
		if(!csSet){
			csSet = new CommandServerSet();
		}
		return csSet;
	}

	// create a new command server using a socket
	int				addServer(SOCKET s);
	bool			isSocketMapped(SOCKET s);
	inline SOCKET	getCurCtxSocket(){ return ctx_mgr_->getCtxSocket(); }
	int				declineServer(SOCKET s);

	inline bool		isCreated(unsigned int flag){ return ctx_mgr_->isCreated(flag); }
	inline void		setCreation(unsigned int &flag){ ctx_mgr_->setCreation(flag); }

	// for buffers that need to update or send data for once
	inline bool		isChanged(unsigned int flag){ return ctx_mgr_->isChanged(flag); }
	inline bool		isSend(unsigned int flag){ return ctx_mgr_->isSend(flag); }
	inline void		setSend(unsigned int &flag){ ctx_mgr_->setSend(flag);}
	inline void		setChanged(unsigned int &flag){	ctx_mgr_->setChanged(flag); }
	inline void		resetChanged(unsigned int &flag){ ctx_mgr_->resetChanged(flag);}
	inline void		setChangedToAll(unsigned int &flag){ ctx_mgr_->setChangedToAll(flag); }
	inline int		getFps(){ if(config_){return config_->max_fps_; }else{ return 25;}}

	void			checkObj(IdentifierBase * obj); // check the current object
	void			pushSync(IdentifierBase * obj); //push the sync object to the queue

	void			checkCreation(IdentifierBase * obj); // only check the creation of an object

	// interface from buffer

	inline int		readInt(){ return ctx_mgr_->readInt();}
	inline UINT		readUInt(){ return ctx_mgr_->readUInt(); }
	inline char		readChar(){ return ctx_mgr_->readChar(); }
	inline UCHAR	readUChar(){ return ctx_mgr_->readUChar(); }

	inline short	readShort(){ return ctx_mgr_->readShort();}
	inline USHORT	readUShort(){ return ctx_mgr_->readUShort(); }
	inline float	readFloat(){ return ctx_mgr_->readFloat();}
	inline void		readByteArr(char * dst, int length){ return ctx_mgr_->readByteArr(dst, length);}

	// for single client
	inline void		writeInt(int data){return ctx_mgr_->writeInt(data);}
	inline void		writeUInt(UINT data){ return ctx_mgr_->writeUInt(data);}
	inline void		writeChar(char data){ return ctx_mgr_->writeChar(data);}
	inline void		writeUChar(UCHAR data){ return ctx_mgr_->writeUChar(data);}
	inline void		writeShort(short data){ return ctx_mgr_->writeShort(data);}

	inline void		writeUShort(USHORT data){ return ctx_mgr_->writeUShort(data);}
	inline void		writeFloat(float data){ return ctx_mgr_->writeFloat(data);}
	inline void		writeByteArr(char * src, int length){ return ctx_mgr_->writeByteArr(src, length);}
	inline void		writePackedByteArr(char *src, int length){ return ctx_mgr_->writePackedByteArr(src, length); }
	inline void		writeVec(int op_code, float * vec, int size = 16){ return ctx_mgr_->writeVec(op_code, vec, size);	}
	
	void			shutDown();
	int				getCommandLength();
	char *			getCurPtr(int length);
	int				flush();

	Buffer *		dumpBuffer();

		// the begin command and the end command
	void			beginCommand(int op_code, int id);
	void			endCommand(int force_flush = 0);
	void			commit();   // commit at present
	void			cancelCommand();
	void			printRecord();


	HANDLE notifier;
	inline HANDLE	getNotifier(){return notifier;}
};

extern CommandServerSet * csSet;
extern cg::core::PTimer * pTimer;

#endif  // __COMMAND_SEVER_SET__