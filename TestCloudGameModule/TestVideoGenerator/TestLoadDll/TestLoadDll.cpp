#include <stdio.h>
#include <Windows.h>

#ifndef MAT_CONT_FLAG_SHIFT
#define MAT_CONT_FLAG_SHIFT 14
#endif

#ifndef MAT_CONT_FLAG
#define MAT_CONT_FLAG ( 1 << MAT_CONT_FLAG_SHIFT)
#endif

#ifndef SUBMAT_FLAG_SHIFT
#define SUBMAT_FLAG_SHIFT 15
#endif

#ifndef SUBMAT_FLAG
#define SUBMAT_FLAG ( 1 << SUBMAT_FLAG_SHIFT)
#endif

#ifndef CN_MAX
#define CN_MAX 512
#endif
#ifndef CN_SHIFT
#define CN_SHIFT 3
#endif

#ifndef DEPTH_MAX
#define DEPTH_MAX (1 << CN_SHIFT)
#endif

#ifndef MAT_CN_MASK
#define MAT_CN_MASK ((CN_MAX - 1) <<CN_SHIFT)
#endif

#ifndef MAT_CN
#define MAT_CN(flags)			((((flags) & MAT_CN_MASK)>>CN_SHIFT) + 1)
#endif

#ifndef MAT_TYPE_MASK
#define MAT_TYPE_MASK			(DEPTH_MAX * CN_MAX - 1)
#endif

#ifndef MAT_TYPE
#define MAT_TYPE(flags)			((flags) & MAT_TYPE_MASK)
#endif

#ifndef MAT_DEPTH_MASK
#define MAT_DEPTH_MASK			(DEPTH_MAX -1)
#endif

#ifndef MAT_DEPTH
#define MAT_DEPTH(flags)		((flags) & MAT_DEPTH_MASK)
#endif

#ifndef ELEM_SIZE1
#define ELEM_SIZE1(type) \
	((((sizeof(size_t)<<28)|0x8442211) >> MAT_DEPTH(type)* 4) & 15)
#endif

#ifndef ELEM_SIZE
#define ELEM_SIZE(type) \
	(MAT_CN(type) << ((((sizeof(size_t)/4+1) * 16384|0x3a50) >> MAT_DEPTH(type)* 2)&3))
#endif

int main(int argc, char ** argv){
#if 0
	HMODULE hr = NULL;
	char * dllName = NULL;
	if(argc == 1){
		dllName = _strdup("GameVideoGenerator.dll");
		hr = LoadLibrary("GameVideoGenerato.dll");
	}else if(argc > 1){
		dllName = _strdup(argv[1]);
		hr = LoadLibrary(argv[1]);
	}

	if(hr == NULL){
		printf("load dll: '%s' failed.\n", dllName);
		DWORD err= GetLastError();
		printf("error code:%d.\n", err);
	}
	else{
		FARPROC addr = GetProcAddress(hr, "hook_D3D9DevicePresent");
		if(addr == NULL){
			printf("get addr failed. err code:%d.\n", GetLastError());
		}
		printf("get addr for hook_createDXGIFactory:0x%p.\n", addr);
	}

#else
	for(int i = 0; i < 100; i++){
		printf("index :%d, ELEM_SIZE:%d.", i, ELEM_SIZE(i));
		printf(" channels:%d.\n", MAT_CN(i));
	}
	getchar();
#endif
	return 0;
}