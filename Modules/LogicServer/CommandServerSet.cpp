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

IndexManager * IndexManager::_indexMgr;

IndexManager::IndexManager(int maxIndex): _maxIndex(maxIndex), _bitmap(0){

}

IndexManager * IndexManager::GetIndexManager(int maxIndex){
	if(!_indexMgr){
		_indexMgr = new IndexManager(maxIndex);
	}
	return _indexMgr;
}

// return an available index for a context
int IndexManager::getAvailableIndex(){
	unsigned int mask = 1;
	int ret = 0;
	bool find = false;
	for(ret = 0; ret < sizeof(int)*8; ret++){
		if(!(_bitmap & (mask << ret))){
			find = true;
			break;
		}
	}
	if(find)
		return ret;
	else
		return -1;
}

bool IndexManager::resetIndex(int index){
	if(index >= sizeof(int) * 8)
		return false;
	unsigned int mask = 1 << index;
	unsigned int remask = ~mask;

	_bitmap &= remask;
	return true;
}

////////////////////// context and cache ///////////////////////////
ContextAndCache::ContextAndCache(int _index){
#ifdef ENABLE_CTX_LOG
	infoRecorder->logTrace("[ContextAndCache]: constructor with limit size:%d, index:%d.\n", Max_Buf_Size, _index);
#endif

	// reserve 2 bytes for func_count
	get_cur_ptr(2);
	obj_id = -1;
	func_count_ = 0;

	status = CTX_INIT;
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

	status = CTX_INIT;
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
#if 0
		if(taskQueue->isStart()){
			taskQueue->stop();
		}
#endif
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

void ContextAndCache::write_packed_byte_arr(char * src, int length){
	// flush the existing commands
	//   +++++++++++++++++++++++++++
	//   |cnt |idx |size| data ... | 
	//   +++++++++++++++++++++++++++
	int prefixSize = 10; // 3 * sizeof(USHORT) + sizeof(int)
	int maxDataLen = 1460;

	if(func_count_ && flush() <= 0){
		infoRecorder->logError("[ContextAndCache]: before write packed byte arr, len <= 0, error:%d.\n", WSAGetLastError());
	}
	// packet the array data
	clear();
	USHORT packetCount = 0, index = 0, sizeToSend = 0, defaultSizeToSend = maxDataLen;
	int remainLen = length, totalSend = 0;
	// calculate the packet count
	packetCount = (length + maxDataLen - 1) / (maxDataLen);

	sizeToSend = maxDataLen;
	int ret = 0;

	for(index = 0; index < packetCount; index++){
		write_ushort(packetCount);
		write_ushort(index);
		sizeToSend = remainLen > defaultSizeToSend ? defaultSizeToSend : remainLen;
		write_ushort(sizeToSend);
		write_byte_arr(src, sizeToSend);
		src += sizeToSend;

		remainLen -= sizeToSend;
		totalSend += sizeToSend;
		ret = send_packet(this);

		if(ret == -1){
			infoRecorder->logError("[ContextAndCache]: socket error:%d.\n", WSAGetLastError());
		}

		clear();
		//infoRecorder->logError("[ContextAndCache]: send packed byte arr, idx:%d, total count: %d, data len:%d, succ send:%d.\n", index, packetCount, sizeToSend, ret);
	}

#ifdef ENABLE_SET_LOG
	infoRecorder->logError("[ContextAndCache]: write_packed_byte_arr, total packet:%d, total send:%d.\n",packetCount, totalSend);
#endif
	// unlock the context
	get_cur_ptr(2);
	unlock();
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

	
	if(func_count_ == 0){
		return 0;
	}

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
	infoRecorder->logError("[ContextManager]: check the object, type:%s, id:%d, frame check flag:0x%x.\n", typeid(*obj).name(), obj->getId(), obj->frameCheckFlag);
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
	//if(obj->stable){
		ContextAndCache * otherCtx = NULL; //ctx_init.getNextCtx();
		// ERROR
		for(int j = 0; j < ctx_init.getCtxCount(); j++){
			otherCtx = ctx_init.getCtx(j);
			if(!otherCtx->isChecked(obj->frameCheckFlag)){
				otherCtx->pushUpdate(obj);
				otherCtx->setChecked(obj->frameCheckFlag);
			}
		}
	//}
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
		ContextAndCache * otherCtx = NULL; //ctx_init.getNextCtx();
		// ERROR
		for(int j = 0; j < ctx_init.getCtxCount(); j++){
			otherCtx = ctx_init.getCtx(j);
			
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
	// move all buffered ctx to init pool, in the next, these ctx will begin to start reconstruction
	ContextAndCache * cur_ctx = NULL;
	while(ctx_buff.getCtxCount()){
		cur_ctx = ctx_buff.getCtx(0);
		ctx_buff.remove(cur_ctx);
		ctx_init.add(cur_ctx);
	}

	for(int i = 0; i < ctx_init.getCtxCount(); i++){
		cur_ctx = ctx_init.getCtx(i);
		if(cur_ctx->status == CTX_INIT){
			// the next frame to prepare data
			cur_ctx->status = CTX_PREPARE;
			cur_ctx->taskQueue->setStatus(QUEUE_CREATE);
		}
		else if(cur_ctx->status == CTX_READY && cur_ctx->taskQueue->isDone()){
			infoRecorder->logError("[ContextManager]: context is ready for switching.\n");
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
		else if(cur_ctx->status == CTX_PREPARE && cur_ctx->taskQueue->isDone()){
			// the next frame to render
			cur_ctx->status = CTX_READY;
			cur_ctx->taskQueue->setStatus(QUEUE_UPDATE);  // now, only update
			// add ctx to pool
			
		}else{
			infoRecorder->logError("[ContextManager]: ctx id:%d, status:%s, queen len:%d.\n", cur_ctx->getIndex(), cur_ctx->status == CTX_INIT ? "CTX_INIT" : (cur_ctx->status == CTX_PREPARE ? "CTX_PREPARE" : (cur_ctx->status == CTX_READY ? "CTX_READY" :"NONE")), cur_ctx->taskQueue->getCount());
		}
	}

	// get a context with status CTX_READY
	if(!ctx_pool.getCtxCount()){
		infoRecorder->logTrace("[ContextManager]: no context available.\n");
		_ctx_cache = NULL;
		return false;
	}
	ContextAndCache * n = NULL;
	n = ctx_pool.getNextCtx();
	if(n){
		//infoRecorder->logError("[ContextManager]: switch get another available context to use.\n");
		ret = true;
		_ctx_cache = n;
	}
	else{
		// always means that there's only one context
		//infoRecorder->logTrace("[ContextManager]: switch get No available context to use. Should never be here\n");
		// TODO, should never be here
		ctx_pool.printError();
		ret = false;
	}
	return ret;
}

bool ContextManager::declineCtx(SOCKET s){
	ContextAndCache * ret = NULL;
	ContextAndCache * ctx = NULL;
	infoRecorder->logError("[ContextManager]: remove context in init.\n");
	for(int i = 0; i < ctx_init.getCtxCount(); i++){
		ctx = ctx_init.getCtx(i);
		if(ctx && ctx->connect_socket == s){
			ctx_init.remove(ctx);
			ret  = ctx;
			goto REMOVAL;
		}
	}
	infoRecorder->logError("[ContextManager]: remove context in pool.\n");
	for(int i = 0; i < ctx_pool.getCtxCount(); i++){
		ctx = ctx_pool.getCtx(i);
		if(ctx && ctx->connect_socket == s){
			ctx_pool.remove(ctx);
			ret = ctx;
			goto REMOVAL;
		}
	}
	return false;
REMOVAL:
	if(ret == _ctx_cache){
		_ctx_cache = NULL;
	}
	// clear the flags in each object with Wrappered directx
	ret->eraseFlag();
	IndexManager * indexMgr = IndexManager::GetIndexManager(sizeof(int)*8);
	indexMgr->resetIndex(ret->getIndex());
	// wait the thread to exit
	if(ret->taskQueue->isStart()){
		ret->taskQueue->stop();
		WaitForSingleObject(ret->taskQueue->getThread(), 1000);
	}

	//release the ctx
	delete ret;
	ret = NULL;
	// here always means return NULL
	infoRecorder->logError("[ContextManager]: current ctx:%p\n", _ctx_cache);
	return true;
}

// add context to init pool, not ready for using
int ContextManager::addCtx(ContextAndCache * _ctx){
	int ret = 0;
	infoRecorder->logError("[ContextManager]: add ctx:%p to ctx buffer.\n", _ctx);
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
	infoRecorder->logTrace("[ContextManager]: after add ctx to init array, current ctx: %p.\n", this->_ctx_cache);
	//ctx_init.add(_ctx);
	ctx_buff.add(_ctx);			// insert to buffer
#endif
	ctxCount ++;
	_ctx->checkFlags();
	_ctx->preperationStart();
#ifndef SINGLE_CONTEXT
	// start the queue thread
	infoRecorder->logError("[ContextManager]: start the task queue proc.\n");
	// add the context to task queue
	_ctx->taskQueue->setContext(_ctx);
	_ctx->taskQueue->start();
	//_ctx->taskQueue->startThread();
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
	infoRecorder->logTrace("[CommandServerSet]: add server, current clients:%d.\n", clients);
	// create cache context with socket s

	// find the socket, if not exist
	if(!ctx_mgr_->isSocketMapped(s)){
		ctx_mgr_->mapSocket(s);
		IndexManager * indexMgr = IndexManager::GetIndexManager(sizeof(int) * 8);
		ContextAndCache * ctx = new ContextAndCache(indexMgr->getAvailableIndex());
		//set cache filter for new context and cache
		clients++;
		ctx->set_cache_filter();
		infoRecorder->logError("[CommandServerSet]: add ctx with index:%d.\n", ctx->getIndex());
		ctx->set_connect_socket(s);
		//DebugBreak();
		ctx_mgr_->addCtx(ctx);

		infoRecorder->logError("[CommandServerSet]: to send msg back to render.\n");
		// send something
		char msg[100]= {0};
		sprintf(msg, "Hello, Everybody!");
		send(s, msg, strlen(msg), 0);
		setCtxEvent();
	}
	else{
		infoRecorder->logError("[CommandServerSet]: the socket is already mapped.\n");
	}
	infoRecorder->logError("[CommandServerSet]: add server succ.\n");
	return 0;
}

// delete a server from normal set
int CommandServerSet::declineServer(SOCKET s){
	infoRecorder->logError("[CommandServerSet]: decline server, sock:%p, clients:%d.\n", s, clients);

	ContextAndCache * ctx = NULL;
	if(!clients)
		return -1;
	if(ctx_mgr_->isSocketMapped(s)){
		// to decline
		if(ctx_mgr_->declineCtx(s)){
			clients--;
		}
		else{
			infoRecorder->logError("[CommandServerSet]: remove ctx failed, but the socket is mapped.\n");
			return -1;
		}
		
		//unmap the socket
		if(!ctx_mgr_->unmapSocket(s)){
			infoRecorder->logError("[CommandServerSet]: unmap the socket is failed.\n");
			return -1;
		}
	}
	else{
		// invalid socket
		infoRecorder->logError("[CommandServerSet]: the socket did not exist in map.\n");
		return -1;
	}
	return 0;
}

// interface from Buffer
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


#if 0
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
#endif
}
/*
commit the server set at the end of each frame, because the sending step will cause the cs set send the in N frames
*/
void CommandServerSet::commit(){
	infoRecorder->logTrace("[CommandServerSet]: commit.\n");
	if(ctx_mgr_->switchCtx()){
		infoRecorder->logTrace("[CommandServerSet]: switch the context succeeded.\n");
	}
	else{
		infoRecorder->logTrace("[CommandServerSet]: switch the cntext failed. no change to current context.\n");
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
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommandServerSet]: shutDown\n");
#endif
	ctx_mgr_->shutDown();
}

Buffer * CommandServerSet::dumpBuffer(){
#ifdef ENABLE_SET_LOG
	infoRecorder->logTrace("[CommandServerSet]: dump buffer.\n");
#endif
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