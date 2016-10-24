#ifndef __TAGDRIVER_H__
#define __TAGDRIVER_H__

#ifndef TAG_SIZE
#define TAG_SIZE  255
#endif

// this is for the tag driver, generate the syn tag for multi-sourced graphic server and client
#include <stdio.h>

namespace cg{
	namespace core{

		// node definition
		template<typename T>
		struct Node{
			T * data;
			Node * next;
			Node * pre;
			bool operator< (Node<T> & t1)const{
				if (*data < *t1.data)
					return true;
				else
					return false;
			}
			bool operator> (Node<T> & t2)const{
				if (*data > *(t2.data))
					return true;
				else
					return false;
			}
			bool operator == (Node<T> & t2) const{
				if(*data == *(t2.data)){
					return true;
				}
				else
					return false;
			}
			bool operator <= (Node<T> &t2) const{
				if(*data <= *(t2.data)){
					return true;
				}
				else
					return false;
			}

			Node(T * _data){ data = _data; next = NULL; pre = NULL; }
			Node(){ data = (T*)(0), next = NULL; pre = NULL; }
		};

		typedef void(*func)(void * p);

		// link list definition
		template <typename T>
		struct DoubleLinkList{
			Node<T> * header;

			int size;

			DoubleLinkList(){
				header = new Node<T>();   // allocate a empty node for header
				header->next = header;
				header->pre = header;    // to header itself
				size = 0;
			}

			void trivalAndDO(func f){
				Node<T> * tem = header->next;
				while (tem != header){
					f(tem);
					tem = tem->next;
				}
			}

			Node<T> * getAndRemoveHead(){
				puts("get and remove head.");
				if (header->next == header){
					// null list
					size = 0;
					return NULL;
				}
				Node<T> * ret = header->next;

				header->next = ret->next;
				ret->next->pre = header;

				if (header->next == header)
					size = 0;
				else{
					size--;
				}


			map<int, int> t;
			

				return ret;
			}

			Node<T> * getHead(){
				if(header->next == header){
					size = 0;
					return NULL;
				}
				Node<T> * ret = header->next;
				return ret;
			}

			void insert(Node<T> * t){
				Node<T> * temp = header->next;

				while (temp!= header){
					if (*t < *temp){
						// add t before temp
						t->next = temp;
						t->pre = temp->pre;
						temp->pre->next = t;
						temp->pre = t;
						size++;
						return;  // succeeded
					}
					temp = temp->next;
				}
				// t is the largest, add to tail
				if (temp == header){
					puts("add to tail");
					// null list
					temp->pre->next = t;
					t->next = temp;
					t->pre = temp->pre;
					temp->pre = t;
					size++;
				}
				printf("list size:%d.\n", size);
			}

			// get and remove an element that is larger than lowBound, and delete all elements before 
			Node<T> * getAndTrim(Node<T> * lowBound){

				Node<T> * ret = header->next;

				if(lowBound == NULL){
					puts("return head.");
					//ret = header->next;
					if(ret != header)
						return getAndRemoveHead();
					else
						return NULL;
				}

				while (ret != header){
					if(*ret <= *lowBound){
						printf("trim node.\n");
						header->next = ret->next;
						ret->next->pre = header;
						//header->pre = ret->pre;
						delete ret;
						size--;
						// move to next node
						ret = header->next;
					}
					else{
						puts("found ret.");
						break;
					}
				}
				if (ret == header){
					// list NULL
					printf("get trim null list.\n");
					size = 0;
					return NULL;
				}
				else{
					// ret is the found node
					// return but not delete
					header->next = ret->next;
					ret->next->pre = header;
					size--;
					ret->next = NULL;
					ret->pre = NULL;

					printf("size after trim:%d.\n", size);
					return ret;
				}
				printf("size after trim:%d.\n", size);
			}
		};

		/////////////////////////////////////////////////////

		template< typename Type, typename DataType>
		struct IndexedFrame{
			Type frameIndex;
			Type tag;
			bool displayed;
			DataType  frame;


			bool operator < (IndexedFrame & t1) const{
				if (tag < t1.tag){
					return true;
				}
				else if (tag == t1.tag && frameIndex < t1.frameIndex){
					return true;
				}
				else
					return false;
			}
			bool operator > (IndexedFrame &t1) const{
				if (tag > t1.tag)
					return true;
				else if (tag == t1.tag && frameIndex > t1.frameIndex)
					return true;
				else
					return false;
			}

			bool operator == (IndexedFrame & t1) const{
				if(tag == t1.tag && frameIndex == t1.frameIndex){
					return true;
				}
				else
					return false;
			}

			bool operator <= (IndexedFrame &t1)const{
				if(tag <= t1.tag)
					return true;
				else if(tag == t1.tag && frameIndex <= t1.frameIndex)
					return true;
				else
					return false;
			}

			IndexedFrame(Type _frameIndex, Type _tag, DataType _data){
				frameIndex = _frameIndex;
				tag = _tag;
				frame = _data;
				displayed = false;
			}
			IndexedFrame(){
				frameIndex = Type( -1);
				tag = Type(-1);
				frame = DataType(0);
				displayed = false;
			}
			~IndexedFrame(){
				if (frame){
					//delete data;
				}
			}
		};

		// use a linked list to store the frame, O(1) to search

		template<typename DataType>
		class IndexedFrameMatrix{

			DoubleLinkList<IndexedFrame<unsigned char, DataType>> pool;
			//IndexedFrame<unsigned char, char *> * header;
			//Node<IndexedFrame<unsigned char, char *> *> * rowPointer;

			//short rows, cols;  // the rows and cols for the matrix
		public:
			//IndexedFrameMatrix();

			int addFrame(IndexedFrame<unsigned char, DataType> * toAdd){
				Node<IndexedFrame<unsigned char, DataType>> * addNode = new Node<IndexedFrame<unsigned char, DataType>>(toAdd);
				pool.insert(addNode);
				return pool.size;
			}
			IndexedFrame<unsigned char, DataType> * getFrame(IndexedFrame<unsigned char, DataType> * lowBound)  /// get the latest frame
			{
				Node<IndexedFrame<unsigned char, DataType>> * retNode = NULL;
				Node<IndexedFrame<unsigned char, DataType>> * lowBoundNode = NULL;

				if(lowBound == NULL){
					// get the header as the lowBound;
					printf("[IndexFrameMatrix]: getFrame(), NULL lowbound.\n");
				}
				else{
					printf("[IndexFrameMatrix]: low bound: index = %d, tag = %d.\n", lowBound->frameIndex, lowBound->tag);
					lowBoundNode = new Node<IndexedFrame<unsigned char, DataType>>(lowBound);
				}

				retNode = pool.getAndTrim(lowBoundNode);

				IndexedFrame<unsigned char, DataType> * ret = NULL;
				if (retNode){
					ret = retNode->data;
					delete retNode;
				}else{
					printf("[IndexFrameMatrix]: getFrame(), getAndTrime ret NULL, low bound: index = %d, tag = %d.\n", lowBoundNode->data->frameIndex, lowBoundNode->data->tag);
				}
				return ret;
			}
			void showAll(func f){
				pool.trivalAndDO(f);
			}
		};
	}
}

#endif