#ifndef __HASH_SET__
#define __HASH_SET__

//#include "utility.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#ifndef HASHSETMASK
#define HASHSETMASK 127
#endif

#ifndef HASHSETSIZE
#define HASHSETSIZE 128
#endif
namespace cg{
	namespace core{

		struct HashLinkedList {
			int id;
			PVOID pKey;
			PVOID pData;
			HashLinkedList* pNext;
		};

		class HashSet {
		public:
			
			
			HashSet();
			PVOID GetDataPtr(UINT id);
			PVOID GetDataPtr(PVOID pKey);
			//inline PVOID& operator[](int id){ return GetDataPtr(id);}
			//inline PVOID& operator[](PVOID pKey){ return GetDataPtr(pKey); }

			bool AddMember(UINT id, PVOID pData);
			bool AddMember(PVOID pKey, PVOID pData);
			bool DeleteMember(PVOID pKey);
			bool DeleteMember(UINT id);
			__forceinline UINT GetHash(PVOID pKey) {
				DWORD Key = (DWORD)pKey;
				return (( Key >> 3 ^ Key >> 7 ^ Key >> 11 ^ Key >> 17 ) & HASHSETMASK);
			}
			__forceinline UINT GetHash(UINT id) {
				return (( id >> 3 ^ id >> 7 ^ id >> 11 ^ id >> 17 ) & HASHSETMASK);
			}

			class iterator{
				HashSet *set_;
				HashLinkedList * cur_;
				int index;  // the array index
			public:
				iterator(HashSet * set, HashLinkedList * p, int idx = 0)throw(): set_(set), cur_(p), index(idx){}
				iterator():set_(NULL), cur_(NULL), index(-1){}
				// copy constructor
				iterator(const iterator & iter){
					set_ = iter.set_;
					cur_ = iter.cur_;
					index = iter.index;
				}
				bool operator==(const iterator &iter) const throw()
					{return (cur_ == iter.cur_) && (index == iter.index); }
				bool operator!=(const iterator &iter) const throw()
				{
					return !(*this == iter);
				}
				iterator & operator++(){
					// get the next node
					cur_ = cur_->pNext;
					// if cur_ is NULL, find the next NOT NULL node
					while(!cur_){
						index++;
						if(index < HASHSETSIZE){
							cur_ = set_->m_pHead[index];
						}
						else{
							// to the end
							cur_ = NULL; 
							break;
						}
					}
					return *this;
				}
				iterator operator++(int){
					iterator temp(*this);
					operator++();
					return temp;
				}

				HashLinkedList & operator*() throw(){
					return *cur_;
				}
				HashLinkedList operator*() const{
					return *cur_;
				}
				HashLinkedList * operator->(){ return cur_; }
				const HashLinkedList * operator->() const {return cur_; }

			};

			iterator begin() throw(){
				// get the first not NULL node
				int idx = 0;
				HashLinkedList * head = m_pHead[idx];
				while(!head){
					idx++;
					if(idx < HASHSETSIZE){
						head = m_pHead[idx];
					}else{
						head = NULL;
						break;
					}
				}
				return iterator(this, head, idx);
			}

			iterator end() throw(){
				return iterator(this, NULL, HASHSETSIZE);
			}
#if 0
			iterator begin() const throw(){
				// get the first not NULL node
				int idx = 0;
				HashLinkedList * head = m_pHead[idx];
				while(!head){
					idx++;
					if(idx < HASHSETSIZE){
						head = m_pHead[idx];
					}else{
						head = NULL;
						break;
					}
				}
				return iterator(this, head, idx);	
			}
			iterator end() const throw(){
				return iterator(this, NULL, HASHSETSIZE);
			}
  #endif
			
			
		private:
			HashLinkedList* m_pHead[HASHSETSIZE];
		};
	}
}

#endif