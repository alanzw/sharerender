#include <WinSock2.h>
#include "Utility.h"

using namespace cg;
using namespace cg::core;

CommandClient::CommandClient(SOCKET s) :func_count(0), op_code_(0), obj_id_(0){
	config_ = new ClientConfig("graphic.client.conf");

	sv_ptr = NULL;
	cache_mgr_ = new ClientCache();
	last_command_ = new Command(-1, get_buffer() + 6, 0);
	set_cache_filter();

}

CommandClient::CommandClient(): func_count(0), op_code_(0), obj_id_(0) {
	config_ = new ClientConfig("graphic.client.conf");
	

	sv_ptr = NULL;

	cache_mgr_ = new ClientCache();
	last_command_ = new Command(-1, get_buffer() + 6, 0);
	set_cache_filter();
}

void CommandClient::load_port(int client_num) {
	if(client_num < 1 || client_num > 8) {
		printf("client_num should be among [1-8]\n");
		return;
	}
	config_->load_config(client_num);
}

void CommandClient::init() {
	if(!init_socket()) return;

#if 1
	printf("[CommandClient:ip:%s port:%d\n", config_->srv_ip_, config_->srv_port_);

#endif

	connect_socket = socket(AF_INET, SOCK_STREAM, 0);
	const char ch_opt = 1;
	int err = setsockopt(connect_socket, IPPROTO_TCP, TCP_NODELAY, &ch_opt, sizeof(char));

	SOCKADDR_IN addr_in;
	memset(&addr_in, 0, sizeof(SOCKADDR_IN));
	addr_in.sin_addr.S_un.S_addr = inet_addr(config_->srv_ip_);
	addr_in.sin_family = AF_INET;
	addr_in.sin_port = htons(config_->srv_port_);

	int connect_ret_val = connect(connect_socket, (SOCKADDR*)&addr_in, sizeof(SOCKADDR));
}

CommandClient::~CommandClient() {
	close_socket(connect_socket);
	clean_up();
}

bool CommandClient::validate_last_command() {
	if(last_command_->op_code_ == -1) return true;

	int consume_length = get_cur_ptr() - last_command_->offest_;

	if(consume_length != last_command_->length_) {
		infoRecorder->logError("=========================================================================\n");
		infoRecorder->logError("The following command has errors, please check your code!\n");
		infoRecorder->logError("\top_code=%d, expected length=%d, consumed length=%d\n", last_command_->op_code_, last_command_->length_, consume_length);
	}
	assert(consume_length == last_command_->length_);

	return (consume_length == last_command_->length_);
}

void CommandClient::record_last_command() {
	last_command_->offest_ = get_cur_ptr();

	char* data = get_cur_ptr(4);
	last_command_->length_ = *((int*)data);
}

void CommandClient::recv_packed_byte_arr(char * dst, int length){
	unsigned char D = '0';
	unsigned char endFlag = '0';
	unsigned short idx = 0;
	short data_len = 0;
	int total_len = 0;
	
	do{
		int len = recv_packet(this);
		if(len <= 0){
			// recv failed
			return;
		}
		D = read_uchar();
		endFlag = read_uchar();
		idx = read_ushort();
		data_len = read_short();
		infoRecorder->logError("[CommandClient]: recv_packed_byte_arr, D:%c, end flag:%c, idx:%d, data len:%d.\n", D, endFlag, idx, data_len);
		read_byte_arr(dst, data_len);
		total_len += data_len;
		dst += data_len;
	}while(endFlag != '1');
	if(total_len != length){
		// error
		infoRecorder->logError("[CommandClient]: recv_packed_byte_arr, total len:%d != data len:%d, cur function count:%d.\n", total_len, length, func_count);
	}
	infoRecorder->logError("[CommandClient]: after recv_packed_byte_arr, cur function count:%d.\n", func_count);
}

void CommandClient::take_command(int& op_code, int& obj_id) {
	if (sv_ptr){
		infoRecorder->logTrace("[CommandClinet]: set sv_ptr to %p.\n", sv_ptr);
		cur_ptr = sv_ptr;
	}

#ifdef Enable_Command_Validate
	if(!validate_last_command()) {
		op_code = obj_id = -1;
		return;
	}
#endif

	infoRecorder->logTrace("[CommandClient]: take_command, func count:%d.\n", func_count);
	
	if(func_count == 0) {
		int len = recv_packet(this);

		if(len <= 0) {
			op_code = obj_id = -1;
			return;
		}

		//cout << "len part: " << get_length_part() << endl;

		func_count = get_count_part();
		//infoRecorder->logTrace("[CommandClient]: read func count:%d.\n", func_count);
		//cout << "count: " << func_count << endl;
		move_over(2);
	}

	--func_count;

#ifdef Enable_Command_Validate
	record_last_command();
#endif

	op_code = read_uchar();
	if(op_code & 1) {
		obj_id = read_ushort();
	}
	else {
		obj_id = 0;
	}
	op_code >>= 1;

#ifdef USE_CACHE

	if(cache_filter[op_code]) {
		int cache_id = read_ushort();
		
		if((cache_id & 1) == Cache_Use) {
			infoRecorder->logTrace("cache hited\n");
			sv_ptr = cur_ptr;
			cur_ptr = cache_mgr_->get_cache(cache_id >> 1);
		}
		else {
			//set cache
			infoRecorder->logTrace("cache miss\n");
			sv_ptr = NULL;
			cache_mgr_->set_cache((cache_id >> 1), get_cur_ptr(), 200);
		}
	}
	else {
		infoRecorder->logTrace("[CommandClient]: set sv_ptr NULL, cache filter failed.\n");
		sv_ptr = NULL;
	}

#endif

	last_command_->op_code_ = op_code;

	//infoRecorder->logTrace("[CommandClient]: opcode:%d, obj_id:%d.\n", op_code, obj_id);

}

void CommandClient::read_vec(float* vec, int size) {

#ifdef USE_CACHE

	int cache_id = read_ushort();
	if((cache_id & 1) == Cache_Use) {
		char* p = cache_mgr_->get_cache(cache_id >> 1);
		memcpy((char*)vec, p, size);  //size = 16 or 8
	}
	else {
		read_byte_arr((char*)vec, size);
		cache_mgr_->set_cache((cache_id >> 1), (char*)vec, size);
	}

#else

	read_byte_arr((char *)vec, size);

#endif

}

