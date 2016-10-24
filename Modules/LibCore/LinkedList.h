#ifndef __LINKEDLIST_H__
#define __LINKEDLIST_H__

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
				if (*data > *(t1.data))
					return true;
				else
					return false;
			}

			Node(T * _data){ data = _data; next = NULL; pre = NULL; }
			Node(){ data = (T*)(0), next = NULL; pre = NULL; }

			~Node(){
				if(data){
					delete data;
					data = NULL;
				}
			}
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
				header->data = (T*)0;
				size = 0;
			}

			~DoubleLinkList(){
				Node<T> * tem = header->next;
				while(tem != header){
					delete tem;
					tem = tem->next;
				}
			}

			void trivalAndDO(func f){
				Node<T> * tem = header->next;
				while (tem != header){
					f(tem);
					tem = tem->next;
				}
			}

			Node<T> * getAndRemoveHead(){
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

				}
			}

			void remove(Node<T> * cur){
				Node<T> * temp = header->next;
				while(temp != header){
					if(cur == temp){

						temp->pre->next = temp->next;
						temp->next->pre = temp->pre;
					}
				}
			}
			Node<T> * getNext(Node<T> * cur = NULL){
				if(cur == NULL){
					// return the header
					if(header->next != header)
						return header->next;
					else
						return NULL;
				}
				else{
					// return cur->next

					if(header->next == header){
						// NULL list
						return NULL;
					}else{
						// list not NULL
						if(cur->next != header)
							return cur->next;
						else
							return header->next;
					}
				}
			}

			// get and remove an element that is larger than lowBound, and delete all elements before 
			Node<T> * getAndTrim(Node<T> * lowBound){
				Node<T> * ret = header->next;

				while (ret != header && *ret < *lowBound){
					header->next = ret->next;
					ret->next->pre = header;
					//header->pre = ret->pre;
					delete ret;
					size--;
					// move to next node
					ret = header->next;
				}
				if (ret == header){
					// list NULL
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
					return ret;
				}


			}
		};
	}
}

#endif