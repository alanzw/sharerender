#ifndef __WRAP_DIRECT3DSTATEBLOCK9__
#define __WRAP_DIRECT3DSTATEBLOCK9__

#include "CommandServerSet.h"

#ifdef MULTI_CLIENTS

// linear memory to record all the functions
class StateBlock{
	int id_;   // the id for state block
	// buf format: [cmd count][
	Buffer* cmdBuf_;   // the buffer, which record the all cmds for the state block
	list<IdentifierBase *> dependencyList;   // all lobject in this list should be checked when using the state block	
public:
	StateBlock():id_(-1), cmdBuf_(NULL){}
	// build the state block without dependency list
	StateBlock(int id, Buffer * buf): id_(id), cmdBuf_(buf){}
	// build the state block with a dependency list
	StateBlock(int id, Buffer * buf, list<IdentifierBase *>& dlist);
	~StateBlock(){ if(cmdBuf_) delete cmdBuf_; }
	Buffer * getBuffer(){ return cmdBuf_; }
	int getSize(){ return cmdBuf_->get_size(); }
	bool sendCreation(ContextAndCache * ctx);
};

// record the state block, all the status
class StateBlockRecorder{
	static Buffer* stateBuffer;
	static std::list<IdentifierBase *> dependencyList;

	static StateBlockRecorder* recorder;
	bool stateBlockBegined;
	bool hasDependency;    // whether the state block has a dependency list

	char * sv_ptr;
	short func_count;

	StateBlockRecorder();
public:
	
	virtual ~StateBlockRecorder();
	static StateBlockRecorder * GetRecorder(){
		if(!recorder){
			recorder = new StateBlockRecorder();
		}
		return recorder;
	}
	// functions
	inline bool				isStateBlockBegined(){ return stateBlockBegined; }
	inline bool				isIndependent(){ return !hasDependency; }
	inline void				StateBlockBegin(){ stateBlockBegined = true; }
	StateBlock *			StateBlockEnd(int serialNumber);

	// function to record the command
	void BeginCommand(int op_code, int obj_id);
	void EndCommand();
	void WriterVec(int op_code, float * vec, int size);
	inline void WriteInt(int data){ stateBuffer->write_int(data); }
	inline void WriteUInt(UINT data){ stateBuffer->write_uint(data); }
	inline void WriteChar(char data){ stateBuffer->write_char(data); }
	inline void WriterUChar(UCHAR data){ stateBuffer->write_uchar(data); }
	inline void WriteShort(short data){ stateBuffer->write_short(data); }
	inline void WriteUShort(USHORT data){ stateBuffer->write_ushort(data); }
	inline void WriteFloat(float data){ stateBuffer->write_float(data); }
	inline void WriteByteArr(char * src, int length){ stateBuffer->write_byte_arr(src, length); }
	char * GetCurPtr(int length){ return stateBuffer->get_cur_ptr(length);}

	inline void pushDependency(IdentifierBase * obj){ if(!hasDependency) hasDependency = true; dependencyList.push_back(obj); }
};

#endif // MULTI_CLIENTS

class WrapperDirect3DStateBlock9 : public IDirect3DStateBlock9
#ifdef MULTI_CLIENTS
	, public IdentifierBase
#endif
{
private:
	IDirect3DStateBlock9* m_sb;
public:

#ifdef MULTI_CLIENTS
	UINT type;   // creation type when using CreateStateBlock
	int creationCommand;   // create state block or begin state block
	StateBlock * stateBlock;  // when created with begin/end state block, the stateBlock record all commands for the state block
#endif

	static HashSet m_list;
	static int ins_count;
	WrapperDirect3DStateBlock9(IDirect3DStateBlock9* _sb, int _id);
	static WrapperDirect3DStateBlock9* GetWrapperStateBlock9(IDirect3DStateBlock9 *ptr);
	IDirect3DStateBlock9* GetSB9();
#ifdef MULTI_CLIENTS
	
	virtual int checkCreation(void *ctx);
	virtual int sendCreation(void *ctx);
	virtual int checkUpdate(void *ctx);
	virtual int sendUpdate(void * ctx);
#endif
public:
	/*** IUnknown methods ***/
	COM_METHOD(HRESULT, QueryInterface)(THIS_ REFIID riid, void** ppvObj);
	COM_METHOD(ULONG,AddRef)(THIS);
	COM_METHOD(ULONG,Release)(THIS);

	/*** IDirect3DStateBlock9 methods ***/
	COM_METHOD(HRESULT, GetDevice)(THIS_ IDirect3DDevice9** ppDevice);
	COM_METHOD(HRESULT, Capture)(THIS);
	COM_METHOD(HRESULT, Apply)(THIS);
};

#endif
