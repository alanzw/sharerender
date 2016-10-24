#ifndef __COMMAND_SERVER__
#define __COMMAND_SERVER__
#include "../LibCore/CommandRecorder.h"
#include "../LibCore/Utility.h"
#include "../LibCore/Cache.hpp"
#include "../libCore/Network.h"
#include "../libCore/Buffer.h"
#include <WinSock2.h>

#ifndef MULTI_CLIENTS
#define MULTI_CLIENTS
#define MAX_CLIENTS 8
#endif
namespace cg{
	namespace core{

		enum COMMAND_SERVER_STATUS{
			ALL_READY, // all ready means that all data needed for rendering is ready at client
			NON_READY, // this is for the new added command server, the data needed by client is not ready
			NO_STATUS   /// error status
		};

		class CommandServer : public Network, public Buffer {
		public:
			CommandServer();
			CommandServer(int size_limit);
			~CommandServer();

			Buffer* dump_buffer();

			void write_vec(int op_code, float* vec, int size=16);
			void init();
			void start_up();
			void shut_down();

			void accept_client();
			void begin_command(int op_code, int id);
			void end_command(int force_flush = 0);
			void cancel_command();
			int get_command_length();
			void print_record();

			int flush();

			ServerConfig* config_;

#ifdef MULTI_CLIENTS

			// new added by alan 2014/8/27 night
			COMMAND_SERVER_STATUS status;
			bool frameStarted; // indicate that a new frame is ready to go, if the server status is ALL_READY, the send the render commands

			inline void setFrameStarted(){ frameStarted = true; }
			//int ready; // 0 for the server that all data for rendering is ready, -1 for new added server, no data is reay, for this kind of server, we must do the render from the beginning of a frame

			void init(SOCKET s);
#endif

		private:
			int op_code;
			int obj_id;
			int sv_obj_id;
			char* sv_ptr;  // the sv_ptr point to the begin of a command
			char* cm_len_ptr;

			char* rid_pos;

			short func_count_;
			int size_limit_;

			CommandRecorder* cr_;

			Cache* cache_mgr_;
		};
	}

}

#endif