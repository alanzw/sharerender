#ifndef __DISNETWORK_H__
#define __DISNETWORK_H__

#include "Buffer.h"
#define DIS_MAX_MSG_LEN 1024
namespace cg{
	namespace core{

		class Buffer;
		class DisNetwork{

			SOCKET sock;
			Buffer * buf;

			int _send_packet(Buffer * buffer);
			int _recv_packet(Buffer * buffer);
			int _recv_n_byte(char * buffer, int nbytes);

		public:
			DisNetwork();
			~DisNetwork();

			inline void setSocket(SOCKET s){ sock = s; }
			inline SOCKET getSocket(){ return sock; }

			int sendPacket();
			int recvPacket();

			void writeInt(int data);
			void writeShort(short data);
			void writeUInt(unsigned int data); 
			void writeUShort(unsigned short data);
			void writeChar(char data);
			void writeUChar(unsigned char data);
			void writeFloat(float data);
			void writeDouble(double data);

			int readInt();
			short readShort();
			unsigned int readUInt();
			unsigned short readUShort();
			char readChar();
			unsigned char readUChar();
			float readFloat();
			double readDouble();

			void writeArray(unsigned char * src, int size);
			int readArray(unsigned char * dst, int len);
		};
	}
}
#endif