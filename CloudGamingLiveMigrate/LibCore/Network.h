#ifndef __NETWORK__
#define __NETWORK__

//#define MULTI_CLIENTS 8

#include <WinSock2.h>
#include "Opcode.h"
#define MAX_CLIENT_COUNT 8

namespace cg{
	namespace core{
		class Buffer;

		class Network {
		public:
			Network();
			Network(SOCKET s);
			~Network();

			bool init_socket();
			void close_socket(SOCKET socket);
			void clean_up();
			void Accept();

			void set_cache_filter();

			int select_recv(SOCKET s, char * buf, int size);

			int send_packet(Buffer* buffer);
			int recv_packet(Buffer* buffer);
			int recv_n_byte(char* buffer, int nbytes);

			SOCKET connect_socket;
			SOCKET listen_socket;

#ifdef MULTI_CLIENTS
			int set_size; // the valid socket set
			SOCKET socket_set[MAX_CLIENT_COUNT];

			SOCKET & get_connect_socket(int index);
			void set_connect_sockets(SOCKET * set, int size);

			void send_raw_buffer(SOCKET s, string buf);
			void send_raw_buffer(SOCKET s, char * buf, int len);
			void recv_raw_buffer(SOCKET s, string& buf, int& len);
			int recv_raw_buffer(SOCKET s, char * buf, int len);

			int send_packet(SOCKET s, Buffer* buffer);
			int send_packet(SOCKET s[], int size, Buffer* buffer);

#endif

			bool cache_filter[MaxSizeUntilNow_Opcode + 10];

			SOCKET& get_connect_socket();
			SOCKET& get_listen_socket();
			void set_connect_socket(const SOCKET connect_socket);

			void send_raw_buffer(string buf);
			void send_raw_buffer(char * buf, int len);
			void recv_raw_buffer(string& buf, int& len);
			int recv_raw_buffer(char * buf, int len);
			//void send_raw_buffer(char* buf, int len);
			//void recv_raw_buffer(char* dst, int len);
		};

	}
}
#endif


