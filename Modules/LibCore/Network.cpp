#include <WinSock2.h>
#include "Utility.h"
#include "../LibCore/InfoRecorder.h"


char tmp_buffer[100000];



// print the tcp error string
void ErrorToString(int err){
	switch(err){
	case WSANOTINITIALISED:
		TRACE_LOG("[Net]: WSANOTINITIALISED.\n");
		//TRACE_LOG("");
		break;
	case WSAENETDOWN:
		TRACE_LOG("[Net]: WSAENETDOWN.\n");
		break;
	case WSABASEERR:
		TRACE_LOG("[Net]: WSABASSERR.\n");
	case WSAEINTR:
		TRACE_LOG("[Net]: WSAEINTR.\n");
		break;
	case WSAEINPROGRESS:
		TRACE_LOG("[Net]: WSAEINPROGRESS.\n");
		break;
	case WSAEFAULT:
		TRACE_LOG("[Net]: WSAEFAULT.\n");
		break;
	case WSAENETRESET:
		TRACE_LOG("[Net]:WSAENETRESET\n");
		break;
	case WSAENOBUFS:
		TRACE_LOG("[Net]:WSAENOBUFS\n");
		break;
	case WSAENOTCONN:
		TRACE_LOG("[Net]:WSAENOTCONN\n");
		break;
	case WSAENOTSOCK:
		TRACE_LOG("[Net]:WSAENOTSOCK\n");
		break;
	case WSAEOPNOTSUPP:
		TRACE_LOG("[Net]:WSAEOPNOTSUPP\n");
		break;
	case WSAESHUTDOWN:
		TRACE_LOG("[Net]:WSAESHUTDOWN\n");
		break;
	case WSAEWOULDBLOCK:
		TRACE_LOG("[Net]:WSAEWOULDBLOCK\n");
		break;
	case WSAEMSGSIZE:
		TRACE_LOG("[Net]:WSAEMSGSIZE\n");
		break;
	case WSAEINVAL:
		TRACE_LOG("[Net]:WSAEINVAL\n");
		break;
	case WSAECONNABORTED:
		TRACE_LOG("[Net]:WSAECONNABORTED\n");
		break;
	case WSAECONNRESET:
		TRACE_LOG("[Net]:WSAECONNRESET\n");
		break;
	default:
		TRACE_LOG("[Net]:invalid error code.\n");
		break;
	}
}

// print [count] bytes in [data]
void printBytes(char * data, int count){
#ifdef ENABLE_NET_VALIDATION
	cg::core::infoRecorder->logTrace("[Global]: print %d bytes.\n", count);

	// each line 16 bytes
	int lines = count / 16;
	for(int i = 0; i< lines; i++){
		for(int j = 0; j < 16; j++){
			cg::core::infoRecorder->logTrace("%d\t", data[i * 16 + j]);
		}
		cg::core::infoRecorder->logTrace("\n");
	}
	for(int j = 0; j < count % 16; j++){
		cg::core::infoRecorder->logTrace("%d\t", data[lines * 16 + j]);
	}
	cg::core::infoRecorder->logTrace("\n");
#endif
}
string sockopterrorstr(int err){
	string ret;
	switch(err){
	case WSANOTINITIALISED:
		ret = string("WSANOTINITIALISED");
		break;
	case WSAENETDOWN:
		ret = string("WSAENETDOWN");
		break;
	case WSAEFAULT:
		ret = string("WSAEFAULT");
		break;
	case WSAEINPROGRESS:
		ret = string("WSAEINPROGRESS");
		break;
	case WSAEINVAL:
		ret = string("WSAEINVAL");
		break;
	case WSAENETRESET:
		ret = string("WSAENETRESET");
		break;
	case WSAENOPROTOOPT:
		ret = string("WSAENOPROTOOPT");
		break;
	case WSAENOTCONN:
		ret = string("WSAENOTCONN");
		break;
	case WSAENOTSOCK:
		ret = string("WSAENOTSOCK");
		break;
	}
	return ret;
}

bool setTcpBuffer(SOCKET s){
	bool ret = false;
	int sndWnd = 8388608; // 8MB
	int socketret= 0;
	//SOCKET_ERROR
	if ((socketret = setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char *)&sndWnd, sizeof(sndWnd))) == 0){
		cg::core::infoRecorder->logError("[SOCKET]: set the TCP sending buffer success\n");
		ret = true;
	}
	else{
		socketret = GetLastError();
		cg::core::infoRecorder->logError("[SOCKET]: set the TCP sending buffer on socket:%p failed with:%d, str:%s, str error:%s.\n", s, socketret, sockopterrorstr(socketret).c_str(), strerror(socketret));

		ret = false;
	}

	int nRecvBuf = 8388608;//ÉèÖÃÎª32K 
	int leng = 4;
	socketret = 0;
	if ((socketret = setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*)&nRecvBuf, sizeof(int)))== 0){
		cg::core::infoRecorder->logError("[SOCKET]: set TCP recv buffer succeeded.\n");
		ret = true;
	}
	else{
		ret = false;
		socketret = GetLastError();
		cg::core::infoRecorder->logError("[SOCKET]: set TCP recv buffer on socket:%p failed with:%d, str:%s.\n", s, socketret, sockopterrorstr(socketret).c_str());
	}
	return ret;
}

bool setNBIO(SOCKET s){
	bool ret = true;
	u_long ul = 1;
	cg::core::infoRecorder->logError("[Network]: set nonblocking io.\n");
	if(ioctlsocket(s, FIONBIO, (u_long *)&ul) == SOCKET_ERROR){
		// error
		int err = GetLastError();
		cg::core::infoRecorder->logError("[Network]: set nonblocking on socket:%p failed with:%d, str:%s.\n", s, err, sockopterrorstr(err));
		ret = false;
	}
	return ret;
}

namespace cg{
	namespace core{
		Network::Network() {

		}

		Network::Network(SOCKET s){

		}

		Network::~Network() {

		}

		void Network::set_cache_filter() {
			cg::core::infoRecorder->logTrace("[Network]: set cache filter.\n");
			memset(cache_filter, 0, sizeof(bool) * (MaxSizeUntilNow_Opcode + 10));
			cache_filter[SetStreamSource_Opcode] = 1;
			cache_filter[DrawIndexedPrimitive_Opcode] = 1;

#if 0
			cache_filter[DrawPrimitive_Opcode] = 1;
			cache_filter[DrawPrimitiveUP_Opcode] = 1;
			cache_filter[DrawIndexedPrimitiveUP_Opcode] = 1;
#endif

			cache_filter[SetRenderState_Opcode] = 1;
			cache_filter[SetRenderTarget_Opcode] = 1;
			cache_filter[SetTexture_Opcode] = 1;

			cache_filter[SetPixelShader_Opcode] = 1;
			cache_filter[SetVertexShader_Opcode] = 1;
			cache_filter[SetSamplerState_Opcode] = 1;
			cache_filter[SetViewport_Opcode] = 1;
			cg::core::infoRecorder->logTrace("[Network]: set cache filter done.\n");
		}

		int Network::send_packet(Buffer* buffer) {
			cg::core::infoRecorder->logTrace("[Network]: send packet.\n");
			if(connect_socket == -1) return -1;

#ifndef ENABLE_NETWORK_COMPRESS
			buffer->set_length_part();
			int len  = 0;
			int err = -1;
			do{
				len = send(connect_socket, buffer->get_buffer(), buffer->get_size(), 0);
				if(len == SOCKET_ERROR){
					fd_set writeSet;
					int nRec = 0;
					err = WSAGetLastError();
					if(err == WSAEWOULDBLOCK){

						FD_ZERO(&writeSet);
						FD_SET(connect_socket, &writeSet);

						nRec = select(0, NULL, &writeSet, NULL, NULL);
						if(nRec > 0){
							continue;  // ready to send
						}
					}else if(err == WSAENOTSOCK){
						return buffer->get_size();
					}
				}else{
					break;
				}
			}while(true);
			//printf("Network::send_packet(), %d bytes sent.\n", len);
			return len;
#else
			int out_len = 4;
			//cg::core::infoRecorder->logTrace("Network::send_packet(), buffer size=%d\n", buffer->get_size());
			Compressor::lzo_encode(buffer->get_buffer() + 4, buffer->get_size() - 4, buffer->com_buffer_ + 4, out_len);

#ifdef BLOCKING_NETWORK
			out_len += 4;
			memcpy(buffer->com_buffer_, &out_len, 4);

			LARGE_INTEGER start, end; 
			QueryPerformanceCounter(&start);

			int len = send(connect_socket, buffer->com_buffer_, out_len, 0);
			QueryPerformanceCounter(&end);
			cg::core::infoRecorder->logTrace("[Network]::send_packet(), %d bytes send, org size:%d, time:%d\n", len, buffer->get_size(), end.QuadPart-start.QuadPart);
			return len;
#else

			// use select mode to send
			fd_set writeSet;
			FD_ZERO(&writeSet);
			FD_SET(connect_socket, &writeSet);
			int nRec = 0;
			int k = 0;
			do{
				nRec = select(0, NULL, &writeSet, NULL, NULL);
				if(nRec <= 0){
					cg::core::infoRecorder->logError("[Network]: socket %p not available to write.\n");
					continue;
				}
				send(connect_socket, (char *)&out_len, 4, 0);
				k = send(connect_socket, buffer->com_buffer_ + 4, out_len, 0);
				cg::core::infoRecorder->logError("[Network]: send_packet, %d bytes sent, expected %d.\n", k, out_len);
				if(k == SOCKET_ERROR && WSAGetLastError() == WSAEWOULDBLOCK){
					nRec =select(0, NULL, &writeSet, NULL, NULL); 
					if(nRec <=0){}
					else
						break;
				}
				break;
			}while(true);


			if(k == SOCKET_ERROR){
				// send failed, check the error code
				int err = WSAGetLastError();
				ErrorToString(err);
			}
			return out_len + 4;
#endif

#endif
		}

		int Network::select_recv(SOCKET s, char * buf, int size){
			fd_set fdSocket;
			FD_ZERO(&fdSocket);
			FD_SET(s, &fdSocket);
			int nRec = select(0, &fdSocket, NULL, NULL, NULL);
			if (nRec <= 0){
				cg::core::infoRecorder->logError("[Network]: recv select failed.\n");
				return nRec;
			}
			// now to recv the size lenght
		}

		int Network::recv_packet(Buffer* buffer) {
			if(connect_socket == -1) return -1;
			int k = 0;
#ifndef ENABLE_NETWORK_COMPRESS
			k = recv_n_byte(buffer->get_buffer(), 4);
			if(k <= 0) return k;

			int total_len = buffer->get_length_part();
			k = recv_n_byte(buffer->get_buffer() + 4, total_len - 4);

			if(k <= 0) return k;
			buffer->go_back(buffer->get_buffer() + 4);

			return total_len;
#else

#ifndef BLOCKING_NETWORK
			fd_set fdSocket;
			FD_ZERO(&fdSocket);
			FD_SET(connect_socket, &fdSocket);

			cg::core::infoRecorder->logTrace("[Network]: wait for data.\n");
			int nRec = select(0, &fdSocket, NULL, NULL, NULL);
			if (nRec <= 0){
				cg::core::infoRecorder->logError("[Network]: recv select failed.\n");
				return nRec;
			}
			// now to recv the size lenght

			int k, data_len;
			k = recv(connect_socket, (char *)&data_len, 4, 0);
			cg::core::infoRecorder->logTrace("[Network]: next packet size is: %d.\n", data_len);
			if(k == SOCKET_ERROR){
				ErrorToString(WSAGetLastError());
			}
			if (k <= 0)return k;

#if 1
			int x = recv_n_byte(buffer->com_buffer_, data_len);
#else
			int x = select_recv(connect_socket, buffer->com_buffer_, data_len);
#endif


			cg::core::infoRecorder->logTrace("[Network]: recved %d bytes, and expected %d bytes.\n", x, data_len);
			if (x <= 0 || x != data_len){
				cg::core::infoRecorder->logTrace("[Network]: recv error code:%d.\n", WSAGetLastError());
			}

			int out_len = 0;
			Compressor::lzo_decode(buffer->com_buffer_, data_len, buffer->get_buffer() + 4, out_len);

			printBytes(buffer->get_buffer() + 4, 256);

			buffer->go_back(buffer->get_buffer() + 4);
			cg::core::infoRecorder->logTrace("[Network]: recv len:%d, org total len:%d.\n",data_len, out_len);
			return out_len;
#else
			k = recv_n_byte(buffer->com_buffer_, 4);
			if(k <=0)return k;
#if 0
			k = recv(connect_socket, buffer->com_buffer_, 100000, 0);
			if(k <= 0) return k;
#endif

			int total_len = *((int*)(buffer->com_buffer_));
			//cg::core::infoRecorder->logTrace("[Network]: plan to recv %d bytes, recved %d bytes.\n", total_len, k);
			int x = recv_n_byte(buffer->com_buffer_ + 4, total_len - 4);
			//cg::core::infoRecorder->logTrace("[Network]: plan to recv %d bytes, recved %d bytes.\n", total_len, x);
			int out_len = 0;
			Compressor::lzo_decode(buffer->com_buffer_ + 4, total_len - 4, buffer->get_buffer() + 4, out_len);

			buffer->go_back(buffer->get_buffer() + 4);
			cg::core::infoRecorder->logTrace("[Network]: recv len:%d, org total len:%d.\n",total_len, out_len + 4);
			return out_len + 4;

#endif   // BLOCKING_NETWORK
#endif   // ENABLE_NETWORK_COMPRESS
		}

		int Network::recv_n_byte(char* buffer, int nbytes) {
			int recvlen = 0;

#ifndef BLOCKING_NETWORK
#if 0
			fd_set fdSocket;
			FD_ZERO(&fdSocket);
			FD_SET(connect_socket, &fdSocket);
			int nRec = select(0, &fdSocket, NULL, NULL, NULL);
			if (nRec > 0){
				//int t = recv(connect_socket, buffer )
				while (recvlen != nbytes) {
					int t = recv(connect_socket, buffer + recvlen, nbytes - recvlen, 0);

					if (t <= 0){

						if (WSAGetLastError() == WSAEWOULDBLOCK){

						}
						else{

							return t;
						}
					}

					recvlen += t;
				}
			}
#else
			fd_set fdSocket;
			FD_ZERO(&fdSocket);
			FD_SET(connect_socket, &fdSocket);
			int nRec = 0; // select(0, &fdSocket, NULL, NULL, NULL);

			while (recvlen != nbytes) {
				int t = recv(connect_socket, buffer + recvlen, nbytes - recvlen, 0);

				if (t <= 0){
					// if data not arriving, wait the data
					if (WSAGetLastError() == WSAEWOULDBLOCK){
						int nRec = select(0, &fdSocket, NULL, NULL, NULL);
						if (nRec > 0){
							t = 0;
							continue;
						}
						else{
							t = 0;
						}
					}
					else{
						cg::core::infoRecorder->logTrace("[Network]: net error code:%d.\n", WSAGetLastError());
						return t;
					}
				}

				recvlen += t;
			}
#endif
#else
			while (recvlen != nbytes) {
				int t = recv(connect_socket, buffer + recvlen, nbytes - recvlen, 0);
				//cg::core::infoRecorder->logTrace("[Network]: recv_n_byte, recved %d byte.\n", t);
				if (t <= 0) {
					cg::core::infoRecorder->logError("[Netowrk]: recv error, error code:%d.\n", WSAGetLastError());
					return t;
				}

				recvlen += t;
			}
#endif

			return recvlen;
		}

		void Network::send_raw_buffer(string buf) {
			if(connect_socket == -1) return;

			int len = send(connect_socket, buf.c_str(), buf.length() + 1, 0);
		}
		void Network::send_raw_buffer(char * buf, int len){
			if(connect_socket == -1)return;

			int len1 = send(connect_socket, buf, len, 0);
			//cg::core::infoRecorder->logError("send size:%d\n", len1);
		}

		void Network::recv_raw_buffer(string& buf, int& len) {
			if(connect_socket == -1) return;

			char tmp[100 + 10];
			len = recv(connect_socket, tmp, 100, 0);
			buf = tmp;
		}
		int Network::recv_raw_buffer(char * buf, int len){
			if( connect_socket == -1) return 0;

			//char tmp[10000];
			return recv(connect_socket, buf, len, 0);
		}

		bool Network::init_socket() {
			WORD version;
			WSAData wsaData;
			//version = MAKEWORD(1, 1);
			version = MAKEWORD(2, 2);

			int err = WSAStartup(version, &wsaData);
			if(err) {
				printf("Network::init_socket(), socket start failed\n");
				return false;
			}
			else {
				printf("Network::init_socket(), socket start success\n");
				return true;
			}
		}

		void Network::Accept() {
			SOCKADDR_IN addr_in;
			memset(&addr_in, 0, sizeof(SOCKADDR_IN));
			int len = sizeof(SOCKADDR);
			printf("Network::Accept() called\n");
			connect_socket = accept(listen_socket, (SOCKADDR*)&addr_in, &len);
			printf("Network::Accept() end\n");
		}

		void Network::clean_up() {
			WSACleanup();
		}

		void Network::close_socket(SOCKET socket) {
			closesocket(socket);
		}

		SOCKET& Network::get_connect_socket() {
			return this->connect_socket;
		}

		SOCKET& Network::get_listen_socket() {
			return this->listen_socket;
		}

		void Network::set_connect_socket(const SOCKET _connect_socket) {
			//setNBIO(_connect_socket);
#if 1
			if (setTcpBuffer(_connect_socket)){

			}
			else{
				cg::core::infoRecorder->logError("[Network]: sending buffer set failed.\n");
			}
#endif
			cg::core::infoRecorder->logError("[Network]: set the connection socket:%p.\n", _connect_socket);
			this->connect_socket = _connect_socket;
		}

	}
}

