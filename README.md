# FGCG manual
=========
Fine grained scheduling framework for cloud gaming system.

##Set up

###Prerequisites

####DirectX SDK
Microsoft Visual Studio (c++) 2010 (If you want to quick start, you can install Microsoft Visual C++ 2010 Redistributable Package, http://www.microsoft.com/en-us/download/details.aspx?id=5555)

DirectX SDK 9.0 c (Recommended: DXSDK_Jun10 version, the version released in June 2010, http://www.microsoft.com/en-us/download/details.aspx?id=6812)

####CUDA SDK
To use NVCUVENC encoding library, CUDA 6.0 is required. And for lastest GPU such maxwell, we can use NVENC instead.

###Build

###Configuration
Add DirectX install path to System Environment as "DXSDK_DIR" and add CUDA 6.0 install path to "CUDA_PATH".

##Quick Start

You can start with our pre-compiled exe files for Windows X86(with visual studio c++ 2010, the reason is that some other dependency library was compiled with VS2010, if using other version, 
you need to re-compile all libraries).

1. you need to write the configure file which named gamemap.txt in default. The configure file includes exe file for game, the path, support D3D or not, the interception dll to use(
use LogicServer.dll in default)

2. install the compiled files, double click DebugCopyToBin.bat when use debug version and double click ReleaseCopyToBin.bat when use release version.

3. copy all files in bin to game directory for simple.

4.  

##Game test cases


SprillRitchie: http://sprill-ritchie-abenteuerliche-zeitreise.software.informer.com/1.0/

Shadowrun returns: http://harebrained-schemes.com/shadowrun/

CastleStorm: http://www.castlestormgame.com/

Trine: http://trine-thegame.com/site/

Unity Angry Bots: http://unity3d.com/showcase/live-demos#angrybots

# FGCG architecture
========
