#include "WrapDirect3dstateblock9.h"
#include "WrapDirect3ddevice9.h"
#include "../LibCore/Opcode.h"
#include "CommandServerSet.h"
#ifndef MULTI_CLIENTS
#define MULTI_CLIENTS
#endif
#ifdef MULTI_CLIENTS

#ifdef MULTI_CLIENTS


//#define ENABLE_STATE_BLOCK_LOG

// build the state block with a list, the list contains the dependency of this state block
StateBlock::StateBlock(int id, Buffer * buf, list<IdentifierBase *>& dlist): id_(id), cmdBuf_(buf){
	// copy the list
	for(list<IdentifierBase *>::iterator it = dlist.begin(); it != dlist.end(); it++){
		dependencyList.push_back(*it);
	}
	
}

bool StateBlock::sendCreation(ContextAndCache * ctx){
	// check the dependency list
#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[StateBlock]: send creation.\n");
#endif
	IdentifierBase * obj = NULL;
	if(!dependencyList.empty()){
		ctx->flush();
	}
	while(!dependencyList.empty()){
		obj = dependencyList.front();
		obj->checkCreation(ctx);
		dependencyList.pop_front();
	}
#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[StateBlock]: to send the buffer.\n");
#endif
	ctx->send_packet(cmdBuf_);
	return true;
}


///////////////////////// StateBlockRecorder /////////////////////

StateBlockRecorder * StateBlockRecorder::recorder = NULL;
Buffer * StateBlockRecorder::stateBuffer = NULL;
std::list<IdentifierBase *> StateBlockRecorder::dependencyList;

StateBlock * StateBlockRecorder::StateBlockEnd(int serialNumber){
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[StateBlockRecorder]: state block function:%d.\n", func_count);
#endif
	StateBlock * ret = NULL;
	stateBuffer->set_count_part(func_count);
	func_count = 0;

	Buffer * buf = stateBuffer->dumpFixedBuffer();
	buf->print();
	// clean the buffer and status
	stateBuffer->clear();
	stateBuffer->get_cur_ptr(2);

	if(isIndependent()){
		#ifdef ENABLE_STATE_BLOCK_LOG
		infoRecorder->logTrace("[StateBlockRecorder]: build state block without any dependency.\n");
#endif
		ret = new StateBlock(serialNumber, buf);
	}else{
		#ifdef ENABLE_STATE_BLOCK_LOG
		infoRecorder->logTrace("[StateBlockRecorder]: build state block with dependency, dependency size:%d.\n", dependencyList.size());
#endif
		ret =  new StateBlock(serialNumber, buf, dependencyList);
	}
	hasDependency = false;
	dependencyList.clear();
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[StateBlockRecorder]: dependency list size:%d after dump the buffer.\n", dependencyList.size());
	#endif
	return ret;
}

StateBlockRecorder::StateBlockRecorder(){
	stateBlockBegined = false;
	// create the static buffer
	if(!stateBuffer){
		stateBuffer = new Buffer(30000);
		stateBuffer->clear();
		stateBuffer->get_cur_ptr(2);   // the function count poart
	}
	hasDependency = false;
	sv_ptr = NULL;
	func_count = 0;
}
StateBlockRecorder::~StateBlockRecorder(){
	if(stateBuffer){
		delete stateBuffer;
		stateBuffer = NULL;
	}
	if(!isIndependent()){
	
	}
	stateBlockBegined = false;
	hasDependency = false;
}

void StateBlockRecorder::BeginCommand(int op_code, int obj_id){
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[StateBlockRecorder]: Begin command, op_code:%d, obj_id:%d.\n", op_code, obj_id);
#endif
	sv_ptr = stateBuffer->get_cur_ptr();
	if(obj_id == 0){
		stateBuffer->write_uchar((op_code << 1) | 0);
	}else{
		stateBuffer->write_uchar((op_code << 1) | 1);
		stateBuffer->write_ushort(obj_id);
	}
	func_count++;
}

void StateBlockRecorder::EndCommand(){
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[StateBlockRecorder]: End command.\n");
#endif
}
void StateBlockRecorder::WriterVec(int op_code, float * vec, int size){
	stateBuffer->write_byte_arr((char *)vec, size);
}



#endif  // MULTI_CLIENTS

int WrapperDirect3DStateBlock9::sendCreation(void *ctx){
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[WrapperDirect3DStateBlock9]: send creation, creation command:%s.\n", creationCommand == CreateStateBlock_Opcode ? "CreateStateBlock_Opcode" :(creationCommand == EndStateBlock_Opcode ? "EndStateBlock_Opcode" : "Unknown"));
#endif
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(creationCommand == CreateStateBlock_Opcode){
		c->beginCommand(CreateStateBlock_Opcode, getDeviceId());
		c->write_uint(type);
		c->write_int(this->id);
		c->endCommand();
	}else if(creationCommand == EndStateBlock_Opcode){
		// use the recorder to rebuild the state block
		// TODO, how to solve the dependency ?
		if(stateBlock){
			stateBlock->sendCreation(c);
		}else{
			return -1;
		}
	}

	return 0;
}

int WrapperDirect3DStateBlock9::checkCreation(void *ctx){
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[WrapperDirect3DStateBlock9]: call check creation.\n");
#endif
	int ret = 0;
	ContextAndCache * c = (ContextAndCache *)ctx;
	if(!c->isCreated(creationFlag)){
		#ifdef ENABLE_STATE_BLOCK_LOG
		infoRecorder->logTrace("[WrapperDirct3DStateBlock9]: to send creation.\n");
#endif
		ret = sendCreation(ctx);
		if(!ret)
			c->setCreation(creationFlag);
	}
	return ret;

}
int WrapperDirect3DStateBlock9::checkUpdate(void *ctx){
	int ret = 0;
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[WrapperDirect3DStateBlock9]: check update.TODO\n");
#endif

	return ret;
}
int WrapperDirect3DStateBlock9::sendUpdate(void *ctx){
	int ret= 0;
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("[WrapeprDirect3DStateBlock9]: send update, TODO.\n");
#endif

	return ret;
}

#endif

WrapperDirect3DStateBlock9::WrapperDirect3DStateBlock9(IDirect3DStateBlock9* _sb, int _id): m_sb(_sb), id(_id){
	creationFlag = 0;
	updateFlag = 0x8fffffff;
	stable = true;
}

int WrapperDirect3DStateBlock9::GetID() {
	return this->id;
}

void WrapperDirect3DStateBlock9::SetID(int id) {
	this->id = id;
}

IDirect3DStateBlock9* WrapperDirect3DStateBlock9::GetSB9() {
	return this->m_sb;
}


/*** IUnknown methods ***/
STDMETHODIMP WrapperDirect3DStateBlock9::QueryInterface(THIS_ REFIID riid, void** ppvObj) {
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("WrapperDirect3DStateBlock9::QueryInterface() called\n");
#endif
	HRESULT hr = m_sb->QueryInterface(riid, ppvObj);
	*ppvObj = this;
	return hr;
}

STDMETHODIMP_(ULONG) WrapperDirect3DStateBlock9::AddRef(THIS) {
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("WrapperDirect3DStateBlock9::AddRef() called\n");
#endif
	return m_sb->AddRef();
}

STDMETHODIMP_(ULONG) WrapperDirect3DStateBlock9::Release(THIS) {
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("WrapperDirect3DStateBlock9::Release() called\n");
#endif
	ULONG hr = m_sb->Release();
#ifdef LOG_REF_COUNT
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logError("WrapperDirect3DStateBlock9::Release(),id:%d, ref:%d.\n", id, hr);
#endif
#endif
	return hr;
}

/*** IDirect3DStateBlock9 methods ***/
STDMETHODIMP WrapperDirect3DStateBlock9::GetDevice(THIS_ IDirect3DDevice9** ppDevice) {
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("WrapperDirect3DStateBlock9::GetDevice() called\n");
#endif

	IDirect3DDevice9* base = NULL;
	HRESULT hr = this->m_sb->GetDevice(&base);
	WrapperDirect3DDevice9 * ret = WrapperDirect3DDevice9::GetWrapperDevice9(base);
	#ifdef ENABLE_STATE_BLOCK_LOG
	if(ret == NULL){
		infoRecorder->logTrace("WrapperDirect3DStateBlock9::GetDevice return NULL!\n");
	}
#endif
	*ppDevice = ret;
	return hr;
	//return m_sb->GetDevice(ppDevice);
}

STDMETHODIMP WrapperDirect3DStateBlock9::Capture(THIS) {
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("WrapperDirect3DStateBlock9::Capture() called, id:%d, creation flag:0x%p\n", id, creationFlag);
#endif
	
#ifndef MULTI_CLIENTS
	cs.begin_command(StateBlockCapture_Opcode, id);
	cs.end_command();
#else

	// before use the state block, check creation
	csSet->checkCreation(this);
	csSet->beginCommand(StateBlockCapture_Opcode, id);
	csSet->endCommand();
#endif
	return m_sb->Capture();
}

STDMETHODIMP WrapperDirect3DStateBlock9::Apply(THIS) {
	#ifdef ENABLE_STATE_BLOCK_LOG
	infoRecorder->logTrace("WrapperDirect3DStateBlock9::Apply() called, id:%d, creation flat:0x%x\n", id, creationFlag);
#endif
	
#ifndef MULTI_CLIENTS
	cs.begin_command(StateBlockApply_Opcode, id);
	cs.end_command();
#else
	// before using the state block, check creation
	csSet->checkCreation(this);
	csSet->beginCommand(StateBlockApply_Opcode, id);
	csSet->endCommand();
#endif
	return m_sb->Apply();
}