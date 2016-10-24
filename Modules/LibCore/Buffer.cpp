#include <WinSock2.h>
#include "Utility.h"
#include "../libCore/InfoRecorder.h"

// this define will enable the log in the buffer layer.
//#define ENABLE_LOG_IN_BUFFER_LAYER
namespace cg{
	namespace core{
		Buffer::Buffer(int capacity): capacity_(capacity) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: construct with size:%d.\n", capacity);
#endif
			buffer_ = new char[capacity_ + 105];
			com_buffer_ = new char[capacity_ + 105];

			cur_ptr = buffer_ + 4;
			size_ = 0;
		}

		Buffer::Buffer(): capacity_(MAXPACKETSIZE) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: construct with default size:%d.\n", MAXPACKETSIZE);
#endif
			buffer_ = new char[capacity_ + 105];
			com_buffer_ = new char[capacity_ + 105];

			for (int i = 0; i < MAXPACKETSIZE; i += 4 * 1024){
				com_buffer_[i] = 0;
				buffer_[i] = 0;
			}

			cur_ptr = buffer_ + 4;
			size_ = 0;
		}

		Buffer::Buffer(const Buffer& other) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]::Buffer(const Buffer&) called\n");
#endif
			capacity_ = other.capacity_;
			buffer_ = new char[other.capacity_ + 105];
			com_buffer_ = new char[other.capacity_ + 105];
			memcpy(buffer_, other.buffer_, other.capacity_);
			cur_ptr = buffer_ + 4;

			size_ = other.size_;
		}

		Buffer::~Buffer() {
			if(buffer_) {
				delete[] buffer_;
				buffer_ = NULL;
			}

			if(com_buffer_) {
				delete[] com_buffer_;
				com_buffer_ = NULL;
			}
		}

		void Buffer::clear() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]::clear to %p.\n", buffer_ + 4);
#endif
			cur_ptr = buffer_ + 4;
		}

		void Buffer::move_over(int length) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: move over %d.\n", length);
#endif
			cur_ptr += length;
		}

	char* Buffer::get_cur_ptr(int length) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: get cur ptr with len:%d.\n", length);
#endif

			char* ptr = cur_ptr;
			cur_ptr += length;
			return ptr;
		}

	char* Buffer::get_cur_ptr() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: get cur ptr.\n");
#endif
			return cur_ptr;
		}

	void Buffer::set_length_part() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: set lenght.\n");
#endif
			size_ = cur_ptr - buffer_;
			memcpy(buffer_, &size_, sizeof(int));
		}

	int Buffer::get_length_part() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: get len.\n");
#endif
			int ret = *( (int*)(buffer_) );
			return ret;
		}

	void Buffer::set_count_part(USHORT func_count) {
			memcpy(buffer_ + 4, &func_count, sizeof(short));
		}

	USHORT Buffer::get_count_part() {
			USHORT ret = *( (USHORT*)(buffer_ + 4) );
			return ret;
		}

	char* Buffer::get_buffer() {
			return buffer_;
		}

	int Buffer::get_size() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: get_size:%d.\n", cur_ptr - buffer_);
#endif
			return size_ = (cur_ptr - buffer_);
		}

	void Buffer::go_back(char* sv_ptr) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: go back to %p.\n", sv_ptr);
#endif
			cur_ptr = sv_ptr;
		}

	void Buffer::write_int(int data) {

#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write int:%d.\n", data);
#endif
#ifdef USE_VARINT
			Compressor::encode_int(data, cur_ptr);
#else
			*( (int*)(cur_ptr) ) = data;
			cur_ptr += sizeof(int);
#endif
		}

	void Buffer::write_uint(UINT data) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write uint:%u.\n", data);
#endif
#ifdef USE_VARINT
			Compressor::encode_uint(data, cur_ptr);
#else
			*( (UINT*)(cur_ptr) ) = data;
			cur_ptr += sizeof(UINT);
#endif
		}

	void Buffer::write_char(char data) {

#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write char :%d.\n", data);
#endif
			*( (char*)(cur_ptr) ) = data;
			cur_ptr += sizeof(char);
		}

	void Buffer::write_uchar(unsigned char data) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write uchar:%u.\n", data);
#endif
			*( (unsigned char*)(cur_ptr) ) = data;
			cur_ptr += sizeof(unsigned char);
		}

	void Buffer::write_float(float data) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write float:%f.\n", data);
#endif
			*( (float*)(cur_ptr) ) = data;
			cur_ptr += sizeof(float);
		}

	void Buffer::write_short(short data) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write short:%d.\n", data);
#endif
			*( (short*)(cur_ptr) ) = data;
			cur_ptr += sizeof(short);
		}

	void Buffer::write_ushort(USHORT data) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write ushort:%u.\n", data);
#endif
			*( (USHORT*)(cur_ptr) ) = data;
			cur_ptr += sizeof(USHORT);
		}

	void Buffer::write_byte_arr(char* data, int length) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: write %d bytes.\n", length);
#endif
			memcpy(cur_ptr, data, length);
			cur_ptr += length;
		}

	int Buffer::read_int() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read int.\n");
#endif
#ifdef USE_VARINT
			int data;
			Compressor::decode_int(cur_ptr, data);
			return data;
#else
			int data = *( (int*)(cur_ptr) );
			cur_ptr += sizeof(int);
			return data;
#endif
		}

	UINT Buffer::read_uint() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read uint.\n");
#endif
#ifdef USE_VARINT
			UINT data;
			Compressor::decode_uint(cur_ptr, data);
			return data;
#else

#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read location offset: %d, cur ptr: %p, base: %p.\n", get_size(), cur_ptr, buffer_);
#endif
			UINT data = *( (UINT*)(cur_ptr) );

			cur_ptr += sizeof(UINT);
			return data;
#endif
		}

	char Buffer::read_char() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read char.\n");
#endif
			char data = *( (char*)(cur_ptr) );
			cur_ptr += sizeof(char);
			return data;
		}

	UCHAR Buffer::read_uchar() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read uchar.\n");
#endif
			UCHAR data = *( (UCHAR*)(cur_ptr) );
			cur_ptr += sizeof(UCHAR);
			return data;
		}

	float Buffer::read_float() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read float.\n");
#endif
			float data = *( (float*)(cur_ptr) );
			cur_ptr += sizeof(float);
			return data;
		}

	short Buffer::read_short() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read short.\n");
#endif
			short data = *( (short*)(cur_ptr) );
			cur_ptr += sizeof(short);
			return data;
		}

	USHORT Buffer::read_ushort() {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read ushort.\n");
#endif
			USHORT data = *( (USHORT*)(cur_ptr) );
			cur_ptr += sizeof(USHORT);
			return data;
		}

	void Buffer::read_byte_arr(char* dst, int length) {
#ifdef ENABLE_LOG_IN_BUFFER_LAYER
			infoRecorder->logTrace("[Buffer]: read %d bytes.\n",length);
#endif
			memcpy(dst, cur_ptr, length);
			cur_ptr += length;
		}

		// dump the buffer, but the buffer will never change, so, the buf len is much more smaller
	Buffer * Buffer::dumpFixedBuffer(){
			int size = this->get_size();
			Buffer * ret =  new Buffer(size);
			ret->size_ = size;
			memcpy(ret->buffer_, buffer_, size);
			ret->get_cur_ptr(size);
			infoRecorder->logTrace("[Buffer]: dump buffer, size:%d.\n", size);
			return ret;
		}


	void Buffer::print(){
		short func_count = *(short *)(buffer_ + 4);

		// get the first opcode
		unsigned char opcode = (*(unsigned char *)(buffer_ + 6));
		int obj_id = -1;
		if(opcode & 1){
			// get the first obj id;
			obj_id = *(int *)(buffer_ + 7);
		}
		infoRecorder->logTrace("[Buffer]: print, function count:%d, first op_code:%d, obj_id = %d.\n", func_count, opcode >> 1, obj_id);
	}
	}
}