#include "TestSmallHash.h"

// the function to test the small hash
#define TEST_MAX 9

void TestSmallHash(){
	int index = 0;
	char context[100] = {0};
	SmallHash<int , TestClass *> testMap;
	TestClass * tClass = NULL;
	for(int i = 0; i < TEST_MAX; i++){
		sprintf(context, "%d-class", i);
		tClass = new TestClass(i, context);
		if(!testMap.addMap(i, tClass)){
			printf("add testClass(index:%d, context:%s) failed.\n", tClass->getIndex(), tClass->getContext());
		}
	}

	// test the get value
	for(int j = 0; j < TEST_MAX; j++){
		tClass = testMap.getValue(j);
		if(tClass){
			tClass->display();
			printf("\n");
		}
		else{
			printf("getValue(%d) failed.\n", j);
		}

		tClass = testMap[j];
		if(tClass){
			tClass->display();
			printf("\n");
		}
		else{
			printf("[%d] failed.\n", j);
		}
	}

	for(int j = 0; j < testMap.count; j++)
	{
		tClass = testMap[j];
		if(tClass){
			tClass->display();
		}
		else{
			printf("[%d] is NULL.\n", j);
		}
		printf("\n");
	}
}