#if 0
//--------------------
// PROGRAM: PEDUMP
// FILE:    PEDUMP.C
// AUTHOR:  Matt Pietrek - 1993
//--------------------
#include <windows.h>
#include <stdio.h>
//#include "objdump.h"
//#include "exedump.h"
//#include "extrnvar.h"

// Global variables set here, and used in EXEDUMP.C and OBJDUMP.C
BOOL fShowRelocations = FALSE;
BOOL fShowRawSectionData = FALSE;
BOOL fShowSymbolTable = FALSE;
BOOL fShowLineNumbers = FALSE;

char HelpText[] = 
	"PEDUMP - Win32/COFF .EXE/.OBJ file dumper - 1993 Matt Pietrek\n\n"
	"Syntax: PEDUMP [switches] filename\n\n"
	"  /A    include everything in dump\n"
	"  /H    include hex dump of sections\n"
	"  /L    include line number information\n"
	"  /R    show base relocations\n"
	"  /S    show symbol table\n";

// Open up a file, memory map it, and call the appropriate dumping routine
void DumpFile(LPSTR filename)
{
	HANDLE hFile;
	HANDLE hFileMapping;
	LPVOID lpFileBase;
	PIMAGE_DOS_HEADER dosHeader;

	hFile = CreateFile(filename, GENERIC_READ, FILE_SHARE_READ, NULL,
		OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

	if ( hFile == INVALID_HANDLE_VALUE )
	{   printf("Couldn't open file with CreateFile()\n");
	return; }

	hFileMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if ( hFileMapping == 0 )
	{   CloseHandle(hFile);
	printf("Couldn't open file mapping with CreateFileMapping()\n");
	return; }

	lpFileBase = MapViewOfFile(hFileMapping, FILE_MAP_READ, 0, 0, 0);
	if ( lpFileBase == 0 )
	{
		CloseHandle(hFileMapping);
		CloseHandle(hFile);
		printf("Couldn't map view of file with MapViewOfFile()\n");
		return;
	}

	printf("Dump of file %s\n\n", filename);

	dosHeader = (PIMAGE_DOS_HEADER)lpFileBase;
	if ( dosHeader->e_magic == IMAGE_DOS_SIGNATURE )
	{ DumpExeFile( dosHeader ); }
	else if ( (dosHeader->e_magic == 0x014C)    // Does it look like a i386
		&& (dosHeader->e_sp == 0) )        // COFF OBJ file???
	{
		// The two tests above aren't what they look like.  They're
		// really checking for IMAGE_FILE_HEADER.Machine == i386 (0x14C)
		// and IMAGE_FILE_HEADER.SizeOfOptionalHeader == 0;
		DumpObjFile( (PIMAGE_FILE_HEADER)lpFileBase );
	}
	else
		printf("unrecognized file format\n");
	UnmapViewOfFile(lpFileBase);
	CloseHandle(hFileMapping);
	CloseHandle(hFile);
}

// process all the command line arguments and return a pointer to
// the filename argument.
PSTR ProcessCommandLine(int argc, char *argv[])
{
	int i;

	for ( i=1; i < argc; i++ )
	{
		strupr(argv[i]);

		// Is it a switch character?
		if ( (argv[i][0] == '-') || (argv[i][0] == '/') )
		{
			if ( argv[i][1] == 'A' )
			{   fShowRelocations = TRUE;
			fShowRawSectionData = TRUE;
			fShowSymbolTable = TRUE;
			fShowLineNumbers = TRUE; }
			else if ( argv[i][1] == 'H' )
				fShowRawSectionData = TRUE;
			else if ( argv[i][1] == 'L' )
				fShowLineNumbers = TRUE;
			else if ( argv[i][1] == 'R' )
				fShowRelocations = TRUE;
			else if ( argv[i][1] == 'S' )
				fShowSymbolTable = TRUE;
		}
		else    // Not a switch character.  Must be the filename
		{   return argv[i]; }
	}
}

int main(int argc, char *argv[])
{
	PSTR filename;

	if ( argc == 1 )
	{   printf(    HelpText );
	return 1; }

	filename = ProcessCommandLine(argc, argv);
	if ( filename )
		DumpFile( filename );
	return 0;
}
#endif


//#include "../../CloudGamingLiveMigrate/LibDistrubutor/Context.h"
#include "../../CloudGamingLiveMigrate/VideoGen/generator.h"


#ifndef _DEBUG
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.lib")
#pragma comment(lib, "libgroupsock.lib")
#pragma comment(lib, "libBasicUsageEnvironment.lib")
#pragma comment(lib, "libUsageEnvironment.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")
#else

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "libliveMedia.d.lib")
#pragma comment(lib, "libgroupsock.d.lib")
#pragma comment(lib, "libBasicUsageEnvironment.d.lib")
#pragma comment(lib, "libUsageEnvironment.d.lib")
#pragma comment(lib, "SDL2.lib")
#pragma comment(lib, "swscale.lib")
#pragma comment(lib, "swresample.lib")
#pragma comment(lib, "postproc.lib")
#pragma comment(lib, "avdevice.lib")
#pragma comment(lib, "avfilter.lib")
#pragma comment(lib, "avformat.lib")
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avutil.lib")

#endif
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib,"ws2_32.lib")
#pragma comment(lib,"d3d9.lib")
#pragma comment(lib,"d3dx9.lib")

int main(){
	
	infoRecorder = new InfoRecorder("test_gpu_encoding");
	IDirect3D9 * gD3D = Direct3DCreate9(D3D_SDK_VERSION);
	D3DPRESENT_PARAMETERS d3dpp;

	IDirect3DDevice9 * gDevice = NULL;
	ZeroMemory(&d3dpp, sizeof(d3dpp));
	d3dpp.Windowed = true;
	d3dpp.BackBufferFormat = D3DFMT_X8R8G8B8;
	d3dpp.BackBufferWidth  = 640;
	d3dpp.BackBufferHeight = 480;
	d3dpp.BackBufferCount  = 1;
	d3dpp.SwapEffect = D3DSWAPEFFECT_COPY;
	d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
	d3dpp.Flags = D3DPRESENTFLAG_VIDEO;//D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
	DWORD dwBehaviorFlags = D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED | D3DCREATE_HARDWARE_VERTEXPROCESSING;

	 gD3D->CreateDevice(0,
		D3DDEVTYPE_HAL,
		NULL,
		dwBehaviorFlags,
		&d3dpp,
		&gDevice);
			
	cg::VideoGen * gen = new cg::VideoGen((HWND)NULL, gDevice, DX9);
	gen->setResolution(800, 600);
	gen->setOutputFileName("test_gpu_encoding.video.264");

	HANDLE presentEvent = CreateEvent(NULL, FALSE, FALSE, NULL);

	gen->setPresentHandle(presentEvent);
#if 0
	gen->setSourceType(SURFACE);
	gen->initVideoGen();
	gen->activeEncoder(NVENC_ENCODER);
#endif

	gen->onThreadStart();
	gen->run();
}