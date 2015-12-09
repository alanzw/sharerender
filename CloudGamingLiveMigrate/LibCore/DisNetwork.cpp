#include <WinSock2.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include "DisNetwork.h"
//#include "compressor.h"
using namespace cg;
using namespace cg::core;

DisNetwork::DisNetwork(){
	buf = new Buffer(DIS_MAX_MSG_LEN);
}
DisNetwork::~DisNetwork(){
	if (buf){
		delete buf;
		buf = NULL;
	}
	
}

int DisNetwork::_send_packet(Buffer *buffer){
	if (sock == -1)return -1;
	buffer->set_length_part();
	int len = send(sock, buffer->get_buffer(), buffer->get_size(), 0);
	return len;
}

int DisNetwork::_recv_packet(Buffer * buffer){
	if (sock == -1)return -1;
	int k = 0;
	k = _recv_n_byte(buffer->get_buffer(), 4);
	if (k <= 0)return k;

	int total_len = buffer->get_length_part();
	k = _recv_n_byte(buffer->get_buffer() + 4, total_len - 4);

	if (k <= 0)return k;
	buffer->go_back(buffer->get_buffer() + 4);
	return total_len;
}

int DisNetwork::_recv_n_byte(char *buffer, int nbytes){
	int recvLen = 0;
	while(recvLen != nbytes){
		int t = recv(sock, buffer + recvLen, nbytes - recvLen, 0);
		if (t <= 0)return t;
		recvLen += t;
	}
	return recvLen;
}

int DisNetwork::sendPacket(){
#if 0
	if (sock == -1){
		return -1;
	}
	buf->set_length_part();
	int len = send(sock, buf->get_buffer(), buf->get_size(), 0);
#else
	if (buf){
		return _send_packet(buf);
	}
	else
		return -1;
#endif
}

int DisNetwork::recvPacket(){
	if (buf){
		return _recv_packet(buf);
	}
	else
		return -1;
}

void DisNetwork::writeInt(int data){
	return buf->write_int(data);
}

void DisNetwork::writeShort(short data){
	return buf->write_short(data);
}
void DisNetwork::writeUInt(unsigned int data){
	return buf->write_uint(data);
}
void DisNetwork::writeUShort(unsigned short data){
	return buf->write_ushort(data);
}
void DisNetwork::writeChar(char data){
	return buf->write_char(data);
}
void DisNetwork::writeUChar(unsigned char data){
	return buf->write_uchar(data);
}
void DisNetwork::writeFloat(float data){
	return buf->write_float(data);
}
void DisNetwork::writeDouble(double data){
	return buf->write_float(data);
}

int DisNetwork::readInt(){
	return buf->read_int();
}
short DisNetwork::readShort(){
	return buf->read_short();
}
unsigned int DisNetwork::readUInt(){
	return buf->read_uint();
}
unsigned short DisNetwork::readUShort(){
	return buf->read_ushort();
}
char DisNetwork::readChar(){
	return buf->read_char();
}
unsigned char DisNetwork::readUChar(){
	return buf->read_uchar();
}
float DisNetwork::readFloat(){
	return buf->read_float();
}
double DisNetwork::readDouble(){
	return buf->read_float();
}

void DisNetwork::writeArray(unsigned char * src, int size){
	return buf->write_byte_arr((char *)src, size);
}
int DisNetwork::readArray(unsigned char *dst, int maxLen){
	buf->read_byte_arr((char *)dst, maxLen);
	return 0;
}
