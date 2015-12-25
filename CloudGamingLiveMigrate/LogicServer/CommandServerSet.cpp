//#include "../LibVideo/Commonwin32.h"
#include "../LibCore/Utility.h"
#include "CommandServerSet.h"
#include "../LibCore/CmdHelper.h"

// this define will disable update worker thread
//#define SINGLE_CONTEXT
//#define ENABLE_SET_LOCK

//#define ENABLE_SET_LOG		// log for command server set
#define ENABLE_CTX_LOG      // log for context
//#define ENABLE_MGR_LOG      // log for context manager


////////////////////// context and cache ///////////////////////////
ContextAndCache::ContextAndCache(int _index){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: constructor with limit size:%d, index:%d.\n", Max_Buf_Size, _index);
#endif

	// reserve 2 bytes for func_count
	get_cur_ptr(2);
	obj_id = -1;
	func_count_ = 0;

	status = CtxStatus::CTX_INIT;
	op_code = -1;
	sv_obj_id = -1;
	sv_ptr = NULL;
	cm_len_ptr = NULL;
	rid_pos = NULL;
	index = _index;

	size_limit_ = Max_Buf_Size;
	cache_mgr_ = new Cache();
	taskQueue = new TaskQueue();
	set_cache_filter();
	taskQueue->setStatus(QUEUE_INVALID);

	// create the mutex
	lockMutex = CreateMutex(NULL, FALSE, NULL);
}
ContextAndCache::ContextAndCache(){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: constructor with limit size:%d.\n", Max_Buf_Size);
#endif

	// reserve 2 bytes for func_count
	clear();
	get_cur_ptr(2);
	obj_id = -1;
	func_count_ = 0;

	status = CtxStatus::CTX_INIT;
	op_code = -1;
	sv_obj_id = -1;
	sv_ptr = NULL;
	cm_len_ptr = NULL;
	rid_pos = NULL;
	index = 0;

	size_limit_ = Max_Buf_Size;
	cache_mgr_ = new Cache();
	taskQueue = new TaskQueue();
	set_cache_filter();
	taskQueue->setStatus(QUEUE_INVALID);

	// create the mutex
	lockMutex = CreateMutex(NULL, FALSE, NULL);
}

ContextAndCache::~ContextAndCache(){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logError("[ContextAndCache]: destructor.\n");
#endif
	if(cache_mgr_){
		delete cache_mgr_;
		cache_mgr_ = NULL;
	}
	if(lockMutex){
		CloseHandle(lockMutex);
	}
	if(taskQueue){
		delete taskQueue;
		taskQueue = NULL;
	}
	close_socket(connect_socket);
	close_socket(listen_socket);
	//clean_up();
}


int ContextAndCache::getCacheIndex(const char * buffer, size_t size, int & hit_id, int & rep_id){
	cache_mgr_->get_cache_index(buffer, size, hit_id, rep_id);
	return 0;
}

void ContextAndCache::checkObj(IdentifierBase * obj){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: ctx '%d' check the object creation.\n",index);
#endif
	obj->checkCreation((void *)this);
}

void ContextAndCache::updateObj(IdentifierBase * obj){
#ifdef ENABLE_CTX_LOG

	infoRecorder->logTrace("[ContextAndCache]: ctx '%d' update object.\n",index);
#endif
	obj->checkUpdate((void *)this);

}

void ContextAndCache::pushSync(IdentifierBase * obj){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: ctx '%d' push sync.\n",index);
#endif
	taskQueue->add(obj);
}

void ContextAndCache::pushUpdate(IdentifierBase * obj){
#ifdef  ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: ctx '%d' push update.\n", index);
#endif
	taskQueue->add(obj);
}

void ContextAndCache::write_vec(int op_code, float * vec, int size, CommandRecorder* cr_){
#ifdef USE_CACHE
	int hit_id = -1, rep_id = -1;
	getCacheIndex((char *)vec, size, hit_id, rep_id);
	if(hit_id != -1){
		write_ushort((hit_id << 1) | Cache_Use);
		cr_->cache_hit(op_code);
	}
	else{
		write_ushort((rep_id << 1) | Cache_Set);
		write_byte_arr((char *)vec, size);
		cr_->cache_miss(op_code);
	}
#else
	write_byte_arr((char *)vec, size);
#endif
}

int ContextAndCache::flush(CommandRecorder * cr){
	set_count_part(func_count_);
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: flush, function cout:%d.\n", func_count_);
#endif
	int len = this->send_packet(this);
	cr->add_lengh(len);
	func_count_ = 0;
	clear();
	get_cur_ptr(2);
	return len;
}
int ContextAndCache::flush(){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: flush, function count:%d.\n", func_count_);
#endif
	set_count_part(func_count_);
	int len = this->send_packet(this);
	func_count_ = 0;
	clear();
	get_cur_ptr(2);
	return len;
}
Buffer * ContextAndCache::dumpBuffer(){
	Buffer * buf = new Buffer(*this);
	buf->get_cur_ptr(2);
	return buf;
}

void ContextAndCache::beginCommand(int _op_code, int id){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: begin command:%d, obj_id:%d.\n", _op_code, id);
#endif
	// lock the context
	lock();

	op_code = _op_code;
	obj_id = id;
	// record current ptr for canceling the command
	sv_ptr = get_cur_ptr();

#ifdef Enable_Command_Validate
	cm_len_ptr = get_cur_ptr(4);
#endif
	if(obj_id == 0){
		write_uchar((op_code << 1) | 0);
	}
	else{
		write_uchar((op_code << 1) | 1);
		write_ushort(obj_id);
	}
#ifdef USE_CACHE
	// append or replace cache info
	//int hit_id = -1, rep_id = -1;
	if(cache_filter[_op_code]){
		rid_pos = get_cur_ptr(2);
		memset(rid_pos, 0, 2);
	}
#endif
	func_count_++;
}

void ContextAndCache::endCommand(int force_flush/* = 0*/){	
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: end command.\n");
#endif
#ifdef USE_CACHE
	int hit_id = -1, rep_id = -1;
	if(cache_filter[op_code]){
		getCacheIndex(sv_ptr, getCommandLength(), hit_id, rep_id);
		if(hit_id != -1){
#ifdef Enable_Command_Validate
			if(obj_id)
				go_back(sv_ptr + 7);
			else
				go_back(sv_ptr + 5);
#else
			if(obj_id)
				go_back(sv_ptr + 3);
			else
				go_back(sv_ptr + 1);
#endif  // enable command validate
			write_ushort((hit_id << 1) | Cache_Use);
		}
		else
		{
			*((unsigned short *)rid_pos) = (rep_id << 1) | Cache_Set;
		}
	}
#endif

#ifdef Enable_Command_Validate
	int cm_len = get_cur_ptr() - sv_ptr;
	memcpy(cm_len_ptr, &cm_len, sizeof(int));

#endif // enable command validate

	if(force_flush || get_size() >= size_limit_){
		int len = flush();
		if(len <= 0){
			infoRecorder->logTrace("[ContextAndCache]: end_command, len <= 0.\n");
		}
	}
	// unlock the context
	unlock();
}

void ContextAndCache::endCommand(CommandRecorder *cr, int force_flush/*= 0*/){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: end command with recorder.\n");
#endif
#ifdef USE_CACHE
	int hit_id = -1, rep_id = -1;
	if(cache_filter[op_code]){
		getCacheIndex(sv_ptr, getCommandLength(), hit_id, rep_id);
		if(hit_id != -1){
#ifdef Enable_Command_Validate
			if(obj_id)
				go_back(sv_ptr + 7);
			else
				go_back(sv_ptr + 5);
#else
			if(obj_id)
				go_back(sv_ptr + 3);
			else
				go_back(sv_ptr + 1);
#endif  // enable command validate
			write_ushort((hit_id << 1) | Cache_Use);
			cr->cache_hit(op_code);
		}
		else
		{
			*((unsigned short *)rid_pos) = (rep_id << 1) | Cache_Set;
			cr->cache_miss(op_code);
		}
	}
#endif

#ifdef Enable_Command_Validate
	int cm_len = get_cur_ptr() - sv_ptr;
	memcpy(cm_len_ptr, & cm_len, sizeof(int));

#endif // enable command validate
	if(!force_flush){
		cr->add_record(op_code, getCommandLength());
	}
	if(force_flush || get_size() >= size_limit_){
		int len = flush();
		if(len <= 0){
			infoRecorder->logTrace("[ContextAndCache]: end_command, len <= 0.\n");
		}
	}
	unlock();
}


/////////////////// context manager /////////////////////////

void ContextManager::checkObj(IdentifierBase * obj){
#ifdef ENABLE_MGR_LOG
	infoRecorder->logTrace("[ContextManager]: check the object.\n");
#endif
	// for current context, check the creation now
	// QUEUE_CREATE status
	if(_ctx_cache){
		_ctx_cache->checkObj(obj);   // check the creation of the object in current context
		_ctx_cache->updateObj(obj);
	}
	// update it elsewhere
	// all context, queue work
	// update check for all

#ifndef SINGLE_CONTEXT
	// set the status to QUEUE_CREATE
	if(obj->stable){
		//DebugBreak();
		ContextAndCache * otherCtx = NULL; //ctx_init.getNextCtx();
		while(otherCtx = ctx_init.getNextCtx()){
			otherCtx->pushUpdate(obj);	
		}
	}
#else
	if(_ctx_cache){
		_ctx_cache->updateObj(obj);
	}
#endif
}

// only check the creation of the object
void ContextManager::checkCreation(IdentifierBase * obj){
	if(_ctx_cache){
		_ctx_cache->checkObj(obj);
	}
}


void ContextManager::pushSync(IdentifierBase * obj){
#ifdef ENABLE_MGR_LOG
	infoRecorder->logTrace("[ContextManager]: push sync object.\n");
#endif
	if(_ctx_cache)
		_ctx_cache->pushSync(obj);

}

// TODO
// get another available context in the ctx_pool
bool ContextManager::switchCtx(){
#ifdef ENABLE_MGR_LOG
	infoRecorder->logTrace("[ContextManager]: switch context.\n");
#endif
	bool ret = false;
	ContextAndCache * cur_ctx = NULL;

	for(int i = 0; i < ctx_init.getCtxCount(); i++){
		cur_ctx = ctx_init.getCtx(i);
		if(cur_ctx->status == CtxStatus::CTX_INIT){
			// the next frame to prepare data
			cur_ctx->status = CTX_PREPARE;
			cur_ctx->taskQueue->setStatus(QUEUE_CREATE);
		}
		else if(cur_ctx->status == CtxStatus::CTX_READY){
			infoRecorder->logTrace("[ContextManager]: context is ready for switching.\n");
			// add to the ctx_pool
			// remove from init pool
			ctx_init.remove(cur_ctx);
			ctx_pool.add(cur_ctx);

			// the render is ready, change the render step
			cg::core::CmdController * cmdCtrrl = cg::core::CmdController::GetCmdCtroller();

			cmdCtrrl->addRenderConnection();

			cur_ctx->preperationEnd();
			//addCtxToPool(cur_ctx);   // available to switch
		}
		else if(cur_ctx->status == CtxStatus::CTX_PREPARE && cur_ctx->taskQueue->isDone()){
			// the next frame to render
			cur_ctx->status = CTX_READY;
			cur_ctx->taskQueue->setStatus(QUEUE_UPDATE);  // now, only update
			// add ctx to pool
		}
	}

	// get a context with status CTX_READY
	ContextAndCache * n = NULL;
	n = ctx_pool.getNextCtx();
	//n = getNextInPool(_ctx_cache);
	if(n){
		infoRecorder->logTrace("[ContextManager]: switch get another available context to use.\n");
		ret = true;
		_ctx_cache = n;
	}
	else{
		// always means that there's only one context
		infoRecorder->logTrace("[ContextManager]: switch get No available context to use.\n");
		// TODO, should never be here
		//DebugBreak();
		ret = false;
		// no change to _ctx_cache
	}
	return ret;
}

// add context to init pool, not ready for using
int ContextManager::addCtx(ContextAndCache * _ctx){
	int ret = 0;
#ifdef ENABLE_MGR_LOG

	infoRecorder->logTrace("[ContextManager]: add ctx:%p.\n", _ctx);
#endif
#ifndef ENABLE_HOT_PLUG
	if(_ctx_cache){
		// when exist at least one context, the new one should add to init
		ctx_init.add(_ctx);
	}
	else{
		// this is the first context
		_ctx->status = CTX_READY;
		_ctx->taskQueue->setStatus(QUEUE_UPDATE); 
		ctx_pool.add(_ctx);
		// also set to  the current work context
		_ctx_cache = _ctx;
	}
#else
	// assume that every context that added is not ready to render
	infoRecorder->logTrace("[ContextManager]: after add ctx to init array.\n");
	ctx_init.add(_ctx);
	
#endif

	ctxCount ++;
	_ctx->preperationStart();
#ifndef SINGLE_CONTEXT
	// start the queue thread
	infoRecorder->logTrace("[ContextManager]: start the task queue proc.\n");
	// add the context to task queue
	_ctx->taskQueue->setContext(_ctx);
	_ctx->taskQueue->startThread();
#endif

	return ret;
}


//////////////////////////// CommandServerSet////////////////////////////
CommandServerSet * CommandServerSet::csSet;

// I need to add some code to lock and unlock, all operations between BeginScene/EndSecene should check the BeginScene call ( TODO)

// I need to make sure the image sequence among multi-renders, It's the job of distributor
bool CommandServerSet::lock(){
#ifdef ENABLE_SET_LOCK
	WaitForSingleObject(mutex, INFINITE);
#endif
	return true;
}
void CommandServerSet::unlock(){
#ifdef ENABLE_SET_LOCK
	ReleaseMutex(mutex);
#endif
}


// this file is for the command server set
// constructor and destructor
CommandServerSet::CommandServerSet(int _limit_size /* Max_Buf_Size */)
	: clients(0){

		infoRecorder->logTrace("[CommandServerSet]: constructor.\n");
		if(clients > MAX_CLIENTS){
			infoRecorder->logError("[CommandServerSet]: init the server set with '%d' clients, but max is '%d'.\n", clients, MAX_CLIENTS);
			return;
		}

		//this->clients = 0;
		FD_ZERO(&fdSocket);

		ctx_mgr_ = new ContextManager();
#if 0
		if(ctx_mgr_){
			ctx_mgr_->setCacheFilter();
		}
#endif
		ctxEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
		ResetEvent(ctxEvent);

		infoRecorder->logTrace("[CommandServerSet]: construct finished.\n");
}

CommandServerSet::~CommandServerSet(){
	infoRecorder->logTrace("[CommandServerSet]: destructor.\n");
	if(this->ctx_mgr_){
		delete ctx_mgr_;
		ctx_mgr_ = NULL;
	}

	clients = 0;
};


// function to add a new render server( actually a new render client), called by distributor
int CommandServerSet::addServer(SOCKET s){
	infoRecorder->logTrace("[CommandServerSet]: add server.\n");
	// create cache context with socket s

	// find the socket, if not exist
	if(!ctx_mgr_->isSocketMapped(s)){
		ContextAndCache * ctx = new ContextAndCache();
		//set cache filter for new context and cache
		ctx->set_cache_filter();
		infoRecorder->logError("[CommandServerSet]: to set the connection socket.\n");
		ctx->set_connect_socket(s);
		ctx_mgr_->addCtx(ctx);

		// send something
		char msg[100]= {0};
		sprintf(msg, "Hello, Everybody!");
		send(s, msg, strlen(msg), 0);
		setCtxEvent();
	}
	else{
		infoRecorder->logTrace("[CommandServerSet]: the socket is already mapped.\n");
	}

	return 0;
}

// delete a server from normal set
int CommandServerSet::declineServer(SOCKET s){
	infoRecorder->logTrace("[CommandServerSet]: decline server.\n");
	ContextAndCache * ctx = NULL;
	if(!clients)
		return -1;
	if(ctx_mgr_->isSocketMapped(s)){
		// to decline
		ctx = ctx_mgr_->declineCtx(s);
		
		if(!ctx){
			// error
			infoRecorder->logError("[CommandServerSet]: remove ctx failed, but the socket is mapped.\n");
			return -1;
		}
		// release the context
		infoRecorder->logError("[CommandServerSet]: to release the context with sock:%p, ctx:%p.\n", s, ctx);
		delete ctx;
		ctx = NULL;

		//unmap the socket
		if(!ctx_mgr_->unmapSocket(s)){
			infoRecorder->logError("[CommandServerSet]: unmap the socket is failed.\n");
			return -1;
		}
	}
	else{
		// invalid socket
		infoRecorder->logTrace("[CommandServerSet]: the socket did not exist in map.\n");
		return -1;
	}
	return 0;
}

// interface from Buffer


//


char * CommandServerSet::getCurPtr(int length){
#ifdef ENABLE_SET_LOG

	infoRecorder->logTrace("[CommandServerSet]: get cur ptr: len:%d.\n", length);
#endif
	return ctx_mgr_->getCurPtr(length);
}

// the begin command and  the end command,

void CommandServerSet::beginCommand(int _op_code, int id){
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommnadServerSet]: Begin Command, op_code:%d, obj id:%d.\n", _op_code, id);
#endif
	op_code = _op_code;

	if(op_code == BeginScene_Opcode){

	}
	// recorder current ptr for canceling the command
	ctx_mgr_->beginCommand(op_code, id);
}

// when need to flush, set the force_flush flag to 1, switch the context when present called.
void CommandServerSet::endCommand(int force_flush){
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommandServerSet]: End Command, force flush:%d.\n", force_flush);
#endif
	if(op_code == MaxSizeUntilNow_Opcode){
		// shut down all render proxy
		infoRecorder->logTrace("[CommandServerSet]: to shut down all render proxy.\n");
		ctx_mgr_->shutDownAll();
	}
	ctx_mgr_->endCommand(force_flush);
	// switch the context
	if(op_code == Present_Opcode){
		infoRecorder->logTrace("[CommandServerSet]: switch context.\n");
		if(ctx_mgr_->switchCtx()){
			infoRecorder->logTrace("[CommnandServerSet]: switch the context succeeded.\n");
		}
		else{
			infoRecorder->logTrace("[CommandServerSet]: switch the context failed. no change to current context.\n");
		}
	}
	else if(op_code == SwapChainPresent_Opcode){
		infoRecorder->logTrace("[CommandServerSet]: switch context, swap chain.\n");
		if(ctx_mgr_->switchCtx()){
			infoRecorder->logTrace("[CommnandServerSet]: switch the context succeeded.\n");
		}
		else{
			infoRecorder->logTrace("[CommandServerSet]: switch the context failed. no change to current context.\n");
		};
	}
}

int CommandServerSet::flush(){
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommandServerSet]: flush.\n");
#endif
	int len = ctx_mgr_->flush();
	return len;
}

void CommandServerSet::cancelCommand(){
#ifdef ENABLE_SET_LOG

	infoRecorder->logTrace("[CommandServerSet]: cancel command.\n");
#endif
	ctx_mgr_->cancelCommand();
}

int CommandServerSet::getCommandLength(){
	return ctx_mgr_->getCommandLength();
}

void CommandServerSet::shutDown(){
	infoRecorder->logTrace("[CommandServerSet]: shutDown\n");
	ctx_mgr_->shutDown();
}

Buffer * CommandServerSet::dumpBuffer(){
	infoRecorder->logTrace("[CommandServerSet]: dump buffer.\n");
	return ctx_mgr_->dumpBuffer();
}

void CommandServerSet::printRecord(){
	ctx_mgr_->printRecord();
}

void CommandServerSet::checkObj(IdentifierBase * obj){
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommandServerSet]: check the object.\n");
#endif
	// for current context, check the creation now
	ctx_mgr_->checkObj(obj);
}


void CommandServerSet::checkCreation(IdentifierBase * obj){
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommandServerSet]: check creation.\n");
#endif
	ctx_mgr_->checkCreation(obj);
}

void CommandServerSet::pushSync(IdentifierBase * obj){
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommandServerSet]: push sync object.\n");
#endif
	ctx_mgr_->pushSync(obj);
}