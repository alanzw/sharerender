#if 1
#include "TestLogicServer.h"

// the main function for TestLogicServer
int main(int argc, char ** argv){
	printf("enter test logic server.\n");
	TestSmallHash();
	printf("exit test logic server.\n");
	getchar();
}
#else
#include <stdio.h>
int main(){
	printf("hello world.\n");

	getchar();

	return 1;
}
#endif