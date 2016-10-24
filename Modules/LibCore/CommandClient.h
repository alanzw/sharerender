#ifndef __COMMAND_CLIENT__
#define __COMMAND_CLIENT__
#if 0
#include <WinSock2.h>
#include "Utility.h"
#endif
#include <iostream>
using namespace std;

#include "ClientCache.hpp"

namespace cg{
	namespace core{

		struct Command {
			int op_code_;
			char* offest_;
			int length_;

			Command(int op_code, char* offest, int length): op_code_(op_code), offest_(offest), length_(length) {

			}
		};

		class CommandClient : public Network, public Buffer {
		public:
			CommandClient();
			CommandClient(SOCKET s);
			~CommandClient();

			void load_port(int client_num);
			void init();

			void read_vec(float* vec, int size=16);

			void record_last_command();
			bool validate_last_command();


			int fetch_stream_buffer();
			int take_command(int& op_code, int& obj_id);
			void recv_packed_byte_arr(char * dst, int length);

		private:
			char* sv_ptr;
			short func_count;

			int op_code_;
			int obj_id_;

			ClientCache* cache_mgr_;
			ClientConfig* config_;
			Command* last_command_;
		};

	}
}

#endif

