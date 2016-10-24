#ifndef __NVENCENCODER_H__
#define __NVENCENCODER_H__
// for the NVENC encoder for MAXWELL GPUs
#include "NVEncodeAPI.h"
#include "Defines.h"
#include <cuda.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "helper_timer.h"
#include "../LibCore/InfoRecorder.h"
#include "Pipeline.h"
#include "CThread.h"
#include "CudaFilter.h"
#include "VideoCommon.h"

// from libCore
#include "../LibCore/Glob.hpp"
#include "../LibCore/GpuMat.hpp"
#include "../LibCore/DXDataTypes.h"

#include "threads\NvThreadingClasses.h"


#define MAX_ENCODERS 16
#define MAX_RECONFIGURATION 10

#ifndef NV_WINDOWS
#define NV_WINDOWS
#endif

#ifndef max
#define max(a, b) (a > b ? a: b)
#endif

#define MAX_INPUT_QUEUE 32
#define MAX_OUTPUT_QUEUE 32

#define SET_VER(configStruct, type) {configStruct.version = type##_VER; }

static const GUID NV_ENC_H264_PROFILE_INVALID_GUID = {
	0x0000000, 0x0000, 0x0000, { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }
};

static const GUID NV_ENC_PRESET_GUID_NULL =
{
	0x0000000, 0x0000, 0x0000, { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }
};

//#define (NV_WINDOWS)
#define NVENCAPI __stdcall
//#endif
typedef struct{
	int param;
	char name[256];
} param_desc;

const param_desc framefieldmode_names[] =
{
	{ 0,                                    "Invalid Frame/Field Mode" },
	{ NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME, "Frame Mode"               },
	{ NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD, "Frame Mode"               },
	{ NV_ENC_PARAMS_FRAME_FIELD_MODE_MBAFF, "MB adaptive frame/field"  }
};

const param_desc ratecontrol_names[] =
{
	{ NV_ENC_PARAMS_RC_CONSTQP,                 "Constant QP Mode"                        },
	{ NV_ENC_PARAMS_RC_VBR,                     "VBR (Variable Bitrate)"                  },
	{ NV_ENC_PARAMS_RC_CBR,                     "CBR (Constant Bitrate)"                  },
	{ 3,                                        "Invalid Rate Control Mode"               },
	{ NV_ENC_PARAMS_RC_VBR_MINQP,               "VBR_MINQP (Variable Bitrate with MinQP)" },
	{ 5,                                        "Invalid Rate Control Mode"               },
	{ 6,                                        "Invalid Rate Control Mode"               },
	{ 7,                                        "Invalid Rate Control Mode"               },
	{ NV_ENC_PARAMS_RC_2_PASS_QUALITY,          "Two-Pass Prefered Quality Bitrate"       },
	{ 9,                                        "Invalid Rate Control Mode"               },
	{ 10,                                       "Invalid Rate Control Mode"               },
	{ 11,                                       "Invalid Rate Control Mode"               },
	{ 12,                                       "Invalid Rate Control Mode"               },
	{ 13,                                       "Invalid Rate Control Mode"               },
	{ 14,                                       "Invalid Rate Control Mode"               },
	{ 15,                                       "Invalid Rate Control Mode"               },
	{ NV_ENC_PARAMS_RC_2_PASS_FRAMESIZE_CAP,    "Two-Pass Prefered Frame Size Bitrate"    },
	{ 17,                                       "Invalid Rate Control Mode"               },
	{ 18,                                       "Invalid Rate Control Mode"               },
	{ 19,                                       "Invalid Rate Control Mode"               },
	{ 20,                                       "Invalid Rate Control Mode"               },
	{ 21,                                       "Invalid Rate Control Mode"               },
	{ 22,                                       "Invalid Rate Control Mode"               },
	{ 23,                                       "Invalid Rate Control Mode"               },
	{ 24,                                       "Invalid Rate Control Mode"               },
	{ 25,                                       "Invalid Rate Control Mode"               },
	{ 26,                                       "Invalid Rate Control Mode"               },
	{ 27,                                       "Invalid Rate Control Mode"               },
	{ 28,                                       "Invalid Rate Control Mode"               },
	{ 29,                                       "Invalid Rate Control Mode"               },
	{ 30,                                       "Invalid Rate Control Mode"               },
	{ 31,                                       "Invalid Rate Control Mode"               },
	{ NV_ENC_PARAMS_RC_2_PASS_VBR,              "Two-Pass (Variable Bitrate)"             }
};

const param_desc encode_picstruct_names[] =
{
	{ 0,                                    "0 = Invalid Picture Struct"                },
	{ NV_ENC_PIC_STRUCT_FRAME,              "1 = Progressive Frame"                     },
	{ NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM,   "2 = Top Field interlaced frame"            },
	{ NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP,   "3 = Bottom Field first inerlaced frame"    },
};

/**
*  * Input picture type
*   */
const param_desc encode_picture_types[] =
{
	{ NV_ENC_PIC_TYPE_P,             "0 = Forward predicted"                              },
	{ NV_ENC_PIC_TYPE_B,             "1 = Bi-directionally predicted picture"             },
	{ NV_ENC_PIC_TYPE_I,             "2 = Intra predicted picture"                        },
	{ NV_ENC_PIC_TYPE_IDR,           "3 = IDR picture"                                    },
	{ NV_ENC_PIC_TYPE_BI,            "4 = Bi-directionally predicted with only Intra MBs" },
	{ NV_ENC_PIC_TYPE_SKIPPED,       "5 = Picture is skipped"                             },
	{ NV_ENC_PIC_TYPE_INTRA_REFRESH, "6 = First picture in intra refresh cycle"           },
	{ NV_ENC_PIC_TYPE_UNKNOWN,       "0xFF = Picture type unknown"                        }
};

/**
*  * Input slice type
*   */
const param_desc encode_slice_type[] =
{
	{ NV_ENC_SLICE_TYPE_DEFAULT, "0 = Slice type is same as picture type" },
	{ 1   ,                      "1 = Invalid slice type mode"            },
	{ NV_ENC_SLICE_TYPE_I,       "2 = Intra predicted slice"              },
	{ NV_ENC_SLICE_TYPE_UNKNOWN, "0xFF = Slice type unknown"              }
};

/**
*  * Motion vector precisions
*   */
const param_desc encode_precision_mv[] =
{
	{ 0,                               "0 = Invalid encode MV precision" },
	{ NV_ENC_MV_PRECISION_FULL_PEL,    "1 = Full-Pel    Motion Vector precision" },
	{ NV_ENC_MV_PRECISION_HALF_PEL,    "2 = Half-Pel    Motion Vector precision" },
	{ NV_ENC_MV_PRECISION_QUARTER_PEL, "3 = Quarter-Pel Motion Vector precision" },
};

typedef struct
{
	GUID id;
	char name[256];
	unsigned int  value;
} guid_desc;

enum
{
	NV_ENC_PRESET_DEFAULT                   =0,
	NV_ENC_PRESET_LOW_LATENCY_DEFAULT       =1,
	NV_ENC_PRESET_HP                        =2,
	NV_ENC_PRESET_HQ                        =3,
	NV_ENC_PRESET_BD                        =4,
	NV_ENC_PRESET_LOW_LATENCY_HQ            =5,
	NV_ENC_PRESET_LOW_LATENCY_HP            =6,
	NV_ENC_PRESET_LOSSLESS_DEFAULT          =8,
	NV_ENC_PRESET_LOSSLESS_HP               =9
};

const guid_desc codec_names[] =
{
	{ NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 0},
	{ NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 1},
	{ NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 2},
	{ NV_ENC_CODEC_H264_GUID, "Invalid Codec Setting" , 3},
	{ NV_ENC_CODEC_H264_GUID, "H.264 Codec"           , 4}
};

const guid_desc codecprofile_names[] =
{
	{ NV_ENC_H264_PROFILE_BASELINE_GUID, "H.264 Baseline", 66 },
	{ NV_ENC_H264_PROFILE_MAIN_GUID,     "H.264 Main Profile", 77 },
	{ NV_ENC_H264_PROFILE_HIGH_GUID,     "H.264 High Profile", 100 },
	{ NV_ENC_H264_PROFILE_STEREO_GUID,   "H.264 Stereo Profile", 128 },
	{ NV_ENC_H264_PROFILE_HIGH_444_GUID,   "H.264 444 Profile", 244 }
};

const guid_desc preset_names[] =
{
	{ NV_ENC_PRESET_DEFAULT_GUID,                               "Default Preset",  0},
	{ NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID,                   "Low Latancy Default Preset", 1 },
	{ NV_ENC_PRESET_HP_GUID,                                    "High Performance (HP) Preset", 2},
	{ NV_ENC_PRESET_HQ_GUID,                                    "High Quality (HQ) Preset", 3 },
	{ NV_ENC_PRESET_BD_GUID,                                    "Blue Ray Preset", 4 },
	{ NV_ENC_PRESET_LOW_LATENCY_HQ_GUID,                        "Low Latancy High Quality (HQ) Preset", 5 },
	{ NV_ENC_PRESET_LOW_LATENCY_HP_GUID,                        "Low Latancy High Performance (HP) Preset", 6 },
	{ NV_ENC_PRESET_GUID_NULL,                                  "Reserved Preset", 7},
	{ NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID,                      "Lossless Default Preset", 8 },
	{ NV_ENC_PRESET_LOSSLESS_HP_GUID,                           "Lossless (HP) Preset", 9 }

};

inline bool compareGUIDs(GUID guid1, GUID guid2)
{
	if (guid1.Data1    == guid2.Data1 &&
		guid1.Data2    == guid2.Data2 &&
		guid1.Data3    == guid2.Data3 &&
		guid1.Data4[0] == guid2.Data4[0] &&
		guid1.Data4[1] == guid2.Data4[1] &&
		guid1.Data4[2] == guid2.Data4[2] &&
		guid1.Data4[3] == guid2.Data4[3] &&
		guid1.Data4[4] == guid2.Data4[4] &&
		guid1.Data4[5] == guid2.Data4[5] &&
		guid1.Data4[6] == guid2.Data4[6] &&
		guid1.Data4[7] == guid2.Data4[7])
	{
		return true;
	}

	return false;
}

inline void printGUID(int i, GUID *id)
{
	printf("GUID[%d]: %08X-%04X-%04X-%08X", i, id->Data1, id->Data2, id->Data3, *((unsigned int *)id->Data4));
}

inline void printPresetGUIDName(GUID guid)
{
	int loopCnt = sizeof(preset_names)/ sizeof(guid_desc);

	for (int cnt = 0; cnt < loopCnt; cnt++)
	{
		if (compareGUIDs(preset_names[cnt].id, guid))
		{
			printf(" \"%s\"\n", preset_names[cnt].name);
		}
	}
}

inline void printProfileGUIDName(GUID guid)
{
	int loopCnt = sizeof(codecprofile_names)/ sizeof(guid_desc);

	for (int cnt = 0; cnt < loopCnt; cnt++)
	{
		if (compareGUIDs(codecprofile_names[cnt].id, guid))
		{
			printf(" \"%s\"\n", codecprofile_names[cnt].name);
		}
	}
}

typedef enum _NvEncodeCompressionStd
{
	NV_ENC_Unknown=-1,
	NV_ENC_H264=4      // 14496-10
} NvEncodeCompressionStd;

typedef enum _NvEncodeInterfaceType
{
	NV_ENC_DX9=0,
	NV_ENC_DX11=1,
	NV_ENC_CUDA=2, // On Linux, CUDA is the only NVENC interface available
	NV_ENC_DX10=3,
} NvEncodeInterfaceType;

const param_desc nvenc_interface_names[] =
{
	{ NV_ENC_DX9,   "DirectX9"  },
	{ NV_ENC_DX11,  "DirectX11" },
	{ NV_ENC_CUDA,  "CUDA"      },
	{ NV_ENC_DX10,  "DirectX10" }
};

typedef enum _CommandLineArgFlags
{
	//Flag                          Bit
	CMD_0_INTERFACE_TYPE,
	CMD_0_PRESET,
	CMD_0_CHROMA_FORMAT_IDC,
	CMD_0_FIELD_MODE,
	CMD_0_BOTTOM_FIELD_FIRST,
	CMD_0_BITRATE,
	CMD_0_MAX_BITRATE,
	CMD_0_RC_MODE,
	CMD_0_NUM_B_FRAMES,
	CMD_0_FRAME_RATE_NUM,
	CMD_0_FRAME_RATE_DEN,
	CMD_0_GOP_LENGTH,
	CMD_0_ENABLE_INITIAL_RCQP,
	CMD_0_INITIAL_QPI,
	CMD_0_INITIAL_QPP,
	CMD_0_INITIAL_QPB,
	CMD_0_PROFILE,
	CMD_0_LEVEL,
	CMD_0_SLICE_MODE,
	CMD_0_SLICE_MODE_DATA,
	CMD_0_VBV_BUFFERSIZE,
	CMD_0_VBV_INITIAL_DELAY,
	CMD_0_ENABLE_PTD,
	CMD_0_SEPARATE_COLOUR_PLANE_FLAG,
	CMD_0_ENABLE_AQ,
	CMD_0_INTRA_REFRESH_COUNT,
	CMD_0_INTRA_REFRESH_PERIOD,
	CMD_0_IDR_PERIOD,
	CMD_0_INPUT_FILE,
	CMD_0_SYNC_MODE,
	CMD_0_WIDTH,
	CMD_0_HEIGHT,

	CMD_1_MAXWIDTH       =0,
	CMD_1_MAXHEIGHT      =1,
	CMD_1_CABAC_ENABLE   =2,
	CMD_1_MV_PRECISION   =3
}CommandLineArgFlags;

typedef enum _H264SliceMode
{
	H264_SLICE_MODE_MBS       = 0x0,
	H264_SLICE_MODE_BYTES     = 0x1,
	H264_SLICE_MODE_MBROW     = 0x2,
	H264_SLICE_MODE_NUMSLICES = 0x3,
}H264SliceMode;

struct EncodeConfig
{
	NvEncodeCompressionStd      codec;
	unsigned int                profile;
	unsigned int                level;
	unsigned int                width;
	unsigned int                height;
	unsigned int                maxWidth;
	unsigned int                maxHeight;
	unsigned int                frameRateNum;
	unsigned int                frameRateDen;
	unsigned int                avgBitRate;
	unsigned int                peakBitRate;
	unsigned int                gopLength;
	unsigned int                enableInitialRCQP;
	NV_ENC_QP                   initialRCQP;
	unsigned int                numBFrames;
	unsigned int                fieldEncoding;
	unsigned int                bottomFieldFrist;
	unsigned int                rateControl; // 0= QP, 1= CBR. 2= VBR
	int                         numSlices;
	unsigned int                vbvBufferSize;
	unsigned int                vbvInitialDelay;
	NV_ENC_MV_PRECISION         mvPrecision;
	unsigned int                enablePTD;
	int                         preset;
	int                         syncMode;
	NvEncodeInterfaceType       interfaceType;
	unsigned int                useMappedResources;

	unsigned int				useDeviceMemory;

	char                        InputClip[256];
	FILE                        *fOutput;
	unsigned int                endFrame;
	unsigned int                chromaFormatIDC;
	unsigned int                separateColourPlaneFlag;
	unsigned int                enableAQ;
	unsigned int                intraRefreshCount;
	unsigned int                intraRefreshPeriod;
	unsigned int                idr_period;
	unsigned int                vle_cabac_enable;
	unsigned int                sliceMode;
	unsigned int                sliceModeData;
};

struct EncodeInputSurfaceInfo
{
	unsigned int      dwWidth;
	unsigned int      dwHeight;
	unsigned int      dwLumaOffset;
	unsigned int      dwChromaOffset;
	void              *hInputSurface;
	unsigned int      lockedPitch;
	NV_ENC_BUFFER_FORMAT bufferFmt;
	void              *pExtAlloc;
	unsigned char     *pExtAllocHost;
	unsigned int      dwCuPitch;
	NV_ENC_INPUT_RESOURCE_TYPE type;
	void              *hRegisteredHandle;

	GpuMat *		dstMat;
};

struct EncodeOutputBuffer
{
	unsigned int     dwSize;
	unsigned int     dwBitstreamDataSize;
	void             *hBitstreamBuffer;
	HANDLE           hOutputEvent;
	bool             bWaitOnEvent;
	void             *pBitstreamBufferPtr;
	bool             bEOSFlag;
	bool             bReconfiguredflag;
};

struct EncoderThreadData
{
	EncodeOutputBuffer      *pOutputBfr;
	EncodeInputSurfaceInfo  *pInputBfr;
};

#define DYN_DOWNSCALE 1
#define DYN_UPSCALE   2

struct EncodeFrameConfig
{
	unsigned char *yuv[3];
	unsigned int stride[3];
	unsigned int width;
	unsigned int height;
	NV_ENC_PIC_STRUCT picStruct;
	bool         bReconfigured;
};

struct FrameThreadData
{
	HANDLE        hInputYUVFile;
	unsigned int  dwFileWidth;
	unsigned int  dwFileHeight;
	unsigned int  dwSurfWidth;
	unsigned int  dwSurfHeight;
	unsigned int  dwFrmIndex;
	void          *pYUVInputFrame;
};

struct EncoderGPUInfo
{
	char gpu_name[256];
	unsigned int device;
};
struct configs_s
{
	const char *str;
	int type;
	int offset;
};

class CNvEncoderThread;

// The main Encoder Class interface
class CNvEncoder : public CThread
{
public:
	CNvEncoder();
	virtual ~CNvEncoder();
protected:
	void                                                *m_hEncoder;
#if defined (NV_WINDOWS)
	IDirect3D9                                          *m_pD3D;
	IDirect3DDevice9                                    *m_pD3D9Device;
	ID3D10Device                                        *m_pD3D10Device;
	ID3D11Device                                        *m_pD3D11Device;
#endif
	CUcontext                                            m_cuContext;
	unsigned int                                         m_dwEncodeGUIDCount;
	GUID                                                 m_stEncodeGUID;
	unsigned int                                         m_dwCodecProfileGUIDCount;
	GUID                                                 m_stCodecProfileGUID;
	GUID                                                 m_stPresetGUID;
	unsigned int                                         m_encodersAvailable;
	unsigned int                                         m_dwInputFmtCount;
	NV_ENC_BUFFER_FORMAT                                *m_pAvailableSurfaceFmts;
	NV_ENC_BUFFER_FORMAT                                 m_dwInputFormat;
	NV_ENC_INITIALIZE_PARAMS                             m_stInitEncParams;
	NV_ENC_RECONFIGURE_PARAMS                            m_stReInitEncParams;
	NV_ENC_CONFIG                                        m_stEncodeConfig;
	NV_ENC_PRESET_CONFIG                                 m_stPresetConfig;
	NV_ENC_PIC_PARAMS                                    m_stEncodePicParams;
	bool                                                 m_bEncoderInitialized;
	EncodeConfig                                         m_stEncoderInput[MAX_RECONFIGURATION];

	EncodeInputSurfaceInfo                               m_stInputSurface[MAX_INPUT_QUEUE];
	EncodeOutputBuffer                                   m_stBitstreamBuffer[MAX_OUTPUT_QUEUE];
	CNvQueue<EncodeInputSurfaceInfo *, MAX_INPUT_QUEUE>   m_stInputSurfQueue;
	CNvQueue<EncodeOutputBuffer *, MAX_OUTPUT_QUEUE>      m_stOutputSurfQueue;
	unsigned int                                         m_dwMaxSurfCount;
	unsigned int                                         m_dwCurrentSurfIdx;
	unsigned int                                         m_dwFrameWidth;
	unsigned int                                         m_dwFrameHeight;
	unsigned int                                         m_uMaxHeight;
	unsigned int                                         m_uMaxWidth;

	unsigned int                                         m_uRefCount;
	configs_s                                            m_configs[50];

	FILE                                                *m_fOutput;
	FILE                                                *m_fInput;

	CNvEncoderThread									*m_pEncoderThread;

	unsigned char                                       *m_pYUV[3];
	bool                                                 m_bAsyncModeEncoding; // only avialable on Windows Platforms
	unsigned char                                        m_pUserData[128];
	NV_ENC_SEQUENCE_PARAM_PAYLOAD                        m_spspps;
	EncodeOutputBuffer                                   m_stEOSOutputBfr;
	CNvQueue<EncoderThreadData, MAX_OUTPUT_QUEUE>        m_pEncodeFrameQueue;
	unsigned int                                         m_dwReConfigIdx;
	unsigned int                                         m_dwNumReConfig;
	virtual bool                                         ParseConfigFile(const char *file);
	virtual void                                         ParseCommandlineInput(int argc, const char *argv[], unsigned int *cmdArgFlags);
	virtual bool                                         ParseConfigString(const char *str);
	virtual bool                                         ParseReConfigFile(char *reConfigFile);
	virtual bool                                         ParseReConfigString(const char *str);
	virtual HRESULT                                      LoadCurrentFrame(unsigned char *yuvInput[3] , HANDLE hInputYUVFile,unsigned int dwFrmIndex);
	virtual void                                         DisplayEncodingParams(EncoderAppParams pEncodeAppParams, int numConfigured);
	virtual HRESULT                                      OpenEncodeSession(int argc, const char *argv[],unsigned int deviceID = 0);
	virtual HRESULT										 OpenEncodeSession(unsigned int deviceID = 0);
	virtual HRESULT                                      LoadCurrentFrame(unsigned char *yuvInput[3] , HANDLE hInputYUVFile, unsigned int dwFrmIndex,
		unsigned int dwFileWidth, unsigned int dwFileHeight, unsigned int dwSurfWidth, unsigned int dwSurfHeight,
		bool bFieldPic, bool bTopField, int FrameQueueSize, int chromaFormatIdc = 1) = 0;
	
	virtual void                                         PreloadFrames(unsigned int frameNumber, unsigned int numFramesToEncode, unsigned int gpuid, HANDLE hInput) = 0;
	virtual int                                          CalculateFramesFromInput(HANDLE hInputFile, const char *filename, int width, int height) = 0;
	virtual HRESULT                                      InitializeEncoder() = 0;
	virtual void                                         PreInitializeEncoder(unsigned int *cmdArgFlags) = 0;
	virtual HRESULT                                      ReconfigureEncoder(EncodeConfig EncoderReConfig) = 0;
	virtual HRESULT                                      EncodeFrame(EncodeFrameConfig *pEncodeFrame, bool bFlush=false) = 0;
	virtual HRESULT                                      DestroyEncoder() = 0;

public:
	virtual int                                          EncoderMain(EncoderGPUInfo encoderInfo, EncoderAppParams appParams, int argc,const char *argv[]) =0 ;
	virtual HRESULT                                      CopyBitstreamData(EncoderThreadData stThreadData);
	virtual HRESULT                                      CopyFrameData(FrameThreadData stFrameData);
	virtual HRESULT                                      QueryEncodeCaps(NV_ENC_CAPS caps_type, int *p_nCapsVal);

protected:
#if defined (NV_WINDOWS) // Windows uses Direct3D or CUDA to access NVENC
	HRESULT                                              InitD3D9(unsigned int deviceID = 0);
	HRESULT                                              InitD3D10(unsigned int deviceID = 0);
	HRESULT                                              InitD3D11(unsigned int deviceID = 0);
#endif
	HRESULT                                              InitCuda(unsigned int deviceID = 0);
	HRESULT                                              AllocateIOBuffers(unsigned int dwInputWidth, unsigned int dwInputHeight, unsigned int maxFrmCnt);
	HRESULT                                              ReleaseIOBuffers();

	unsigned char                                       *LockInputBuffer(void *hInputSurface, unsigned int *pLockedPitch);
	HRESULT                                              UnlockInputBuffer(void *hInputSurface);
	unsigned int                                         GetCodecType(GUID encodeGUID);
	unsigned int                                         GetCodecProfile(GUID encodeGUID);
	GUID                                                 GetCodecProfileGuid(unsigned int profile);
	HRESULT                                              GetPresetConfig(int iPresetIdx);

	HRESULT                                              FlushEncoder();
	HRESULT                                              ReleaseEncoderResources();
	HRESULT                                              WaitForCompletion();

	// from cthread
	virtual void run() = 0;
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam) = 0;
	virtual void onQuit() = 0;
	//virtual BOOL stop() = 0;
	virtual void onThreadStart() = 0;

	NV_ENC_REGISTER_RESOURCE                             m_RegisteredResource;

public:
	NV_ENCODE_API_FUNCTION_LIST                         *m_pEncodeAPI;
	HINSTANCE                                            m_hinstLib;
	bool                                                 m_bEncodeAPIFound;

	StopWatchInterface                                  *m_timer;
	

	GpuMat * dstMat;
	int interopWithD3D;     // if true, the source is from Device memroy( the D3D rendered surface), or, the source is from system memory.
};



#define QUERY_PRINT_CAPS(pEnc, CAPS, VAL) pEnc->QueryEncodeCaps(CAPS, &VAL); printf("Query %s = %d\n", #CAPS, VAL); infoRecorder->logTrace("[CNvEncoder]: Query %s = %d\n", #CAPS, VAL);

void queryAllEncoderCaps(CNvEncoder *pEncoder);

typedef NVENCSTATUS(__stdcall *MYPROC)(NV_ENCODE_API_FUNCTION_LIST *);



// globals
extern InfoRecorder *infoRecorder;

// for H264 encoder

class CNvEncoderH264 : public CNvEncoder{
public:
	CNvEncoderH264();
	CNvEncoderH264(int _w, int _h, char * name);
	~CNvEncoderH264();


	
protected:
	char * outputName;
	// for the source
	pipeline * imagePipe;
	pipeline * cudaPipe;

	int srcWidth, srcHeight;

	CudaFilter * cuFilter;

	EncoderGPUInfo * pEncoderInfo, encoderInfo[16];
	EncodeFrameConfig stEncodeFrame, * pStEncodeFrame;
	EncoderAppParams * pAppParams, appParams;

	int curYuvframecnt;
	double total_encode_time, sum;
	HANDLE sourceNotifier;  // the present event notifier
	DXDevice dxDevice;
	DX_VERSION dxVer;
	// original config
	int frameCount;
	int frameNumber;
	int encoderID;

	HANDLE condMutex;

	bool                                                 m_bMVC;
	unsigned int                                         m_dwViewId;
	unsigned int                                         m_dwPOC;
	unsigned int                                         m_dwFrameNumSyntax;
	unsigned int                                         m_dwIDRPeriod;
	unsigned int                                         m_dwNumRefFrames[2];
	unsigned int                                         m_dwFrameNumInGOP;

	char                                                 m_outputFilename[256];

	void                                                 InitDefault();
	virtual HRESULT                                      InitializeEncoder();
	virtual HRESULT                                      LoadCurrentFrame(unsigned char *yuvInput[3] , HANDLE hInputYUVFile, unsigned int dwFrmIndex,
		unsigned int dwFileWidth, unsigned int dwFileHeight, unsigned int dwSurfWidth, unsigned int dwSurfHeight,
		bool bFieldPic, bool bTopField, int FrameQueueSize, int chromaFormatIdc = 1);
	virtual HRESULT										 LoadCurrentFrame(CUdeviceptr dstptr);
	virtual HRESULT										LoadCurrentFrame(GpuMat & mat);

	virtual void                                         PreloadFrames(unsigned int frameNumber, unsigned int numFramesToEncode, unsigned int gpuid, HANDLE hInput);
	virtual int                                          CalculateFramesFromInput(HANDLE hInputFile, const char *filename, int width, int height);
	virtual HRESULT                                      EncodeFrame(EncodeFrameConfig *pEncodeFrame, bool bFlush=false);
	virtual HRESULT                                      DestroyEncoder();
	virtual void                                         PreInitializeEncoder(unsigned int *cmdArgFlags);
	virtual void 
		PreInitializeEncoder();
	virtual HRESULT                                      ReconfigureEncoder(EncodeConfig EncoderReConfig);
public:
	virtual int                                          EncoderMain(EncoderGPUInfo encoderInfo, EncoderAppParams appParams, int argc,const char *argv[]);


	// from cthread
	virtual void run();
	virtual void onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam);
	virtual void onQuit();
	//virtual BOOL stop() = 0;
	virtual void onThreadStart();


public:
	inline void setSourceNotifier(HANDLE h){ this->sourceNotifier = h; }
	inline HANDLE getSourceNotifier(){ return sourceNotifier; }
	inline void setDXDevice(DX_VERSION ver, DXDevice *d){
		dxDevice.d10Device = d->d10Device;
		dxVer = ver;
	}
	inline void setCudaPipe(pipeline * p){ cudaPipe = p; }
	inline void setImagePipe(pipeline * p){ imagePipe = p; }

	int InitCudaFilter(DX_VERSION ver, void * d);
};


class CNvEncoderThread: public CNvThread{
public:
	CNvEncoderThread(CNvEncoder *pOwner, U32 dwMaxQueuedSamples)
		:   CNvThread("Encoder Output Thread")
		,   m_pOwner(pOwner)
		,   m_dwMaxQueuedSamples(dwMaxQueuedSamples)
	{
		// Empty constructor
	}

	// Overridden virtual functions
	virtual bool ThreadFunc();
	// virtual bool ThreadFini();

	bool QueueSample(EncoderThreadData &sThreadData);
	int GetCurrentQCount()
	{
		return m_pEncoderQueue.GetCount();
	}
	bool IsIdle()
	{
		return m_pEncoderQueue.GetCount() == 0;
	}
	bool IsQueueFull()
	{
		return m_pEncoderQueue.GetCount() >= m_dwMaxQueuedSamples;
	}

protected:
	CNvEncoder *const m_pOwner;
	CNvQueue<EncoderThreadData, MAX_OUTPUT_QUEUE> m_pEncoderQueue;
	U32 m_dwMaxQueuedSamples;
};


// tool function remained to original
#ifndef STRNCASECMP
#define STRNCASECMP _strnicmp

#if 0
// CUDA Utility Helper Functions
inline int 
	stringRemoveDelimiter(
	char delimiter, 
	const char *string)
{
	int string_start = 0;

	while (string[string_start] == delimiter)
	{
		string_start++;
	}

	if (string_start >= (int)strlen(string)-1)
	{
		return 0;
	}

	return string_start;
}
#endif
#endif

#if 0
// This function wraps the CUDA Driver API into a template function
template <class T>
inline bool 
	getCmdLineArgumentValue(
	const int argc, const char **argv, 
	const char *string_ref, T *value)
{
	bool bFound = false;

	if (argc >= 1)
	{
		for (int i=1; i < argc; i++)
		{
			int string_start = stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];
			int length = (int)strlen(string_ref);

			if (!STRNCASECMP(string_argv, string_ref, length))
			{
				if (length+1 <= (int)strlen(string_argv))
				{
					int auto_inc = (string_argv[length] == '=') ? 1 : 0;
					*value = (T)atoi(&string_argv[length + auto_inc]);
				}

				bFound = true;
				i=argc;
			}
		}
	}

	return bFound;
}

#endif

#if 0
inline bool getCmdLineArgumentString(
	const int argc, const char **argv,
	const char *string_ref, char **string_retval)
{
	bool bFound = false;

	if (argc >= 1)
	{
		for (int i=1; i < argc; i++)
		{
			int string_start = stringRemoveDelimiter('-', argv[i]);
			char *string_argv = (char *)&argv[i][string_start];
			int length = (int)strlen(string_ref);

			if (!STRNCASECMP(string_argv, string_ref, length))
			{
				*string_retval = &string_argv[length+1];
				bFound = true;
				continue;
			}
		}
	}

	if (!bFound)
	{
		*string_retval = NULL;
	}

	return bFound;
}


inline int 
	getFileExtension(
	char *filename, 
	char **extension)
{
	int string_length = (int)strlen(filename);

	while (filename[string_length--] != '.')
	{
		if (string_length == 0)
			break;
	}

	if (string_length > 0) string_length += 2;

	if (string_length == 0)
		*extension = NULL;
	else
		*extension = &filename[string_length];

	return string_length;
}
#endif
#endif