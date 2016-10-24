#if defined(NV_WINDOWS)

#include <d3d9.h>
#include <d3d10_1.h>
#include <d3d11.h>

#pragma warning(disable : 4996)
#endif


#include "../common/NvHWEncoder.h"
#include "../common/CNvEncoder.h"

class CNvEncoderImpl{
public:
	//CNvEncoderImpl());
	virtual ~CNvEncoderImpl();
};