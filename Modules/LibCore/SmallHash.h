#ifndef __SMALLHASH_H__
#define __SMALLHASH_H__

// small hash
#include <map>

//#include <Utility>
#define DEFAULT_MAP_SIZE 8
using namespace std;
namespace cg{
	namespace core{

		template <class KeyType, class MapType> struct Item{
			KeyType key;
			MapType value;

			template <class KeyType, class MapType> Item<KeyType, MapType> & operator[](int index);
		};


		template <class KeyType, class MapType> class SmallHash{
			KeyType * keyArray;
			MapType * mapArray;

			short count;    // valid values
			short size; //// size of the items

		public:
			SmallHash(){
				//items = NULL;
				size = DEFAULT_MAP_SIZE;

				keyArray = NULL;
				mapArray = NULL;

				keyArray = (KeyType *)malloc(sizeof(KeyType) * size);
				mapArray = (MapType *)malloc(sizeof(MapType) * size);

				count = 0;
			}
			SmallHash(int _size){
				//items = NULL;
				keyArray = NULL;
				mapArray = NULL;

				size = _size;
				//items = (Item<KeyType, MapType> **)malloc(sizeof(Item<KeyType, MapType>) * size);
				keyArray = (KeyType *)malloc(sizeof(KeyType) * size);
				mapArray = (MapType *)malloc(sizeof(MapType) * size);
				count = 0;
			}
			~SmallHash(){
				if (keyArray){
					free(keyArray);
					keyArray = NULL;
				}
				if (mapArray){
					free(mapArray);
					mapArray = NULL;
				}
				count = 0;
				size = 0;
			}

			int clear(){ return 0;}
			// return the whole size of the hash
			int getSize(){ return size; }

			// return the mapped pairs in the hash
			int getCount(){
				return count;
			}

			KeyType getKey(int index){
				if (index < count){
					return keyArray[index];
				}
				else{
					return -1;
				}
			}
			MapType getMapValue(int index){
				if (index < count){
					return mapArray[index];
				}
				else{
					return (MapType)0;
				}
			}


			MapType getValue(KeyType key){
				for (int i = 0; i < count; i++){
					if (keyArray[i] == key){
						return mapArray[i];
					}

				}
				return (MapType)0;
			}
			bool addMap(KeyType key, MapType value){
				if (count < size){
					keyArray[count] = key;
					mapArray[count] = value;
					count++;
					return true;
				}
				else{
					return false;
				}
				return false;
			}
			// unmap the key, and return the whether succeeded or failed
			bool unMap(KeyType key){
				int i = 0;
				int find = 0;
				for (i = 0; i < count; i++){
					if (keyArray[i] == key){
						find = 1;
						break;
					}
				}
				if (find){
					for (int j = i; j < count - 1; j++){
						keyArray[j] = keyArray[j + 1];
						mapArray[j] = mapArray[j + 1];
					}
					count--;
					return true;
				}
				return false;
			}
#if 1
			MapType & operator[](KeyType key){

				int i = 0;
				for (i = 0; i < count; i++){
					if (keyArray[i] == key){
						return mapArray[i];
					}
				}
				//i++;
				if (i < size){
					printf("add map %d\n", i);
					mapArray[i] = (KeyType)0;
					keyArray[i] = key;
					count++;
					return mapArray[i];
				}
				else{
					printf("realloc %d.\n", i);
					// enlarge the container
					MapType * n = (MapType *)malloc(size * 2 * sizeof(MapType));
					KeyType * k = (KeyType *)malloc(size * 2 * sizeof(KeyType));
					memset(n, 0, size * 2 * sizeof(MapType));
					memset(k, 0, size * 2 * sizeof(KeyType));
					for (int m = 0; m < this->count; m++){
						k[m] = keyArray[m];
						n[m] = mapArray[m];
					}
					//
					free(mapArray);
					free(keyArray);
					mapArray = n;
					keyArray = k;

					mapArray[i] = (KeyType)0;
					keyArray[i] = key;
					count++;
					size *= 2;
				}
				return mapArray[i];
			}
#else
#endif
		};
	}
}
#endif