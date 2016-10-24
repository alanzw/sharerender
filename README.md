# ShareRender manual
=========
ShareRender is a cloud gaming system that enables fine-grained resource sharing at the frame-level.

##Set up

###Prerequisites

####DirectX SDK
Microsoft Visual Studio (c++) 2010 (If you want to quick start, you can install Microsoft Visual C++ 2010 Redistributable Package, http://www.microsoft.com/en-us/download/details.aspx?id=5555)

DirectX SDK 9.0 c (Recommended: DXSDK_Jun10 version, the version released in June 2010, http://www.microsoft.com/en-us/download/details.aspx?id=6812)

####CUDA SDK
To use NVCUVENC encoding library, CUDA 6.0 is required. And for lastest GPU such maxwell, we can use NVENC instead.

###Configuration
Add DirectX install path to System Environment as "DXSDK_DIR" and add CUDA 6.0 install path to "CUDA_PATH".

###Build

Start to build the soluation in VS2010, if gets warnings about missing libraries, add them to linker input.

##Quick Start

You can start with our pre-compiled exe files for Windows X86(with visual studio c++ 2010, the reason is that some other dependency library was compiled with VS2010, if using other version, 
you need to re-compile all libraries).

1. you need to write the configure file which named gamemap.txt in default. The configure file includes exe file for game, the path, support D3D or not, the interception dll to use(use LogicServer.dll in default)

2. install the compiled files, double click DebugCopyToBin.bat when use debug version and double click ReleaseCopyToBin.bat when use release version.

3. copy all files in bin to game directory for simple(These files are dependecies, copy to game dirctory is the easiest way to make it work).

4. Start DisManager.exe, which act as scheduler and accept the regiesteration of gameloader and render proxy.

5. Start gameloader.exe with IP of DisManager in VM. Gameloader waill load game and inject LogicServer.dll to game process.

6. Start RenderProxy.exe with IP of DisManager in any physic server.

7. Start client with request of RTSP and rqeust of game. (e.g. )

8. Some other functions is integrated to system for convenience to query performance data. Please contact author for detailed information.

##Game test cases


SprillRitchie: http://sprill-ritchie-abenteuerliche-zeitreise.software.informer.com/1.0/

Shadowrun returns: http://harebrained-schemes.com/shadowrun/

CastleStorm: http://www.castlestormgame.com/

Trine: http://trine-thegame.com/site/

Unity Angry Bots: http://unity3d.com/showcase/live-demos#angrybots

# ShareRender architecture
========
