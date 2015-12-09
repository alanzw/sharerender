#include "check_d3d9.h"
#include <Windows.h>
#include <stdio.h>
#include "inforecoder.h"


extern InfoRecorder * infoRecorder;

void _checkD3DError(HRESULT hr, const char * file, const int line){
	char err_msg[200] = {0};
	switch(hr){
	case D3DOK_NOAUTOGEN:
		sprintf(err_msg, "D3DOK_NOAUTOGEN");
		break;
	case D3DERR_CONFLICTINGRENDERSTATE:
		sprintf(err_msg, "D3DERR_CONFLICTINGRENDERSTATE");
		break;
	case D3DERR_CONFLICTINGTEXTUREFILTER:
		sprintf(err_msg, "D3DERR_CONFLICTINGTEXTUREFILTER");
		break;
	case D3DERR_CONFLICTINGTEXTUREPALETTE:
		sprintf(err_msg, "D3DERR_CONFLICTINGTEXTUREPALETTE");
		break;
	case D3DERR_DEVICEHUNG:
		sprintf(err_msg, "D3DERR_DEVICEHUNG");
		break;
	case D3DERR_DEVICELOST:
		sprintf(err_msg, "D3DERR_DEVICELOST");
		break;
	case D3DERR_DEVICENOTRESET:
		sprintf(err_msg, "D3DERR_DEVICENOTRESET");
		break;
	case D3DERR_DEVICEREMOVED:
		sprintf(err_msg, "D3DERR_DEVICEREMOVED");
		break;
	case D3DERR_DRIVERINTERNALERROR:
		sprintf(err_msg, "D3DERR_DRIVERINTERNALERROR");
		break;
	case D3DERR_INVALIDCALL:
		sprintf(err_msg, "D3DERR_INVALIDCALL");
		break;
	case D3DERR_DRIVERINVALIDCALL:
		sprintf(err_msg, "D3DERR_DRIVERINVALIDCALL");
		break;
	case D3DERR_INVALIDDEVICE:
		sprintf(err_msg, "D3DERR_INVALIDDEVICE");
		break;
	case D3DERR_MOREDATA:
		sprintf(err_msg, "D3DERR_MOREDATA");
		break;
	case D3DERR_NOTAVAILABLE:
		sprintf(err_msg, "D3DERR_NOTAVAILABLE");
		break;
	case D3DERR_NOTFOUND:
		sprintf(err_msg, "D3DERR_NOTFOUND");
		break;
	case D3D_OK:
		sprintf(err_msg, "D3D_OK");
		break;
	case D3DERR_OUTOFVIDEOMEMORY:
		sprintf(err_msg, "D3DERR_OUTOFDEVICEMEMORY");
		break;
	case D3DERR_TOOMANYOPERATIONS:
		sprintf(err_msg, "D3DERR_TOOMANYOPERATIONS");
		break;
	case D3DERR_UNSUPPORTEDALPHAARG:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDALPHAARG");
		break;
	case D3DERR_UNSUPPORTEDALPHAOPERATION:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDALPHAOPERATION");
		break;
	case D3DERR_UNSUPPORTEDCOLORARG:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDCOLORARG");
		break;
	case D3DERR_UNSUPPORTEDCOLOROPERATION:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDCOLOROPERATION");
		break;
	case D3DERR_UNSUPPORTEDFACTORVALUE:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDFACTORVALUE");
		break;
	case D3DERR_WASSTILLDRAWING:
		sprintf(err_msg, "D3DERR_WASSTILLDRAWING");
		break;
	case D3DERR_WRONGTEXTUREFORMAT:
		sprintf(err_msg, "D3DERR_WRONGTEXTUREFORMAT");
		break;
	case E_FAIL:
		sprintf(err_msg, "E_FAIL");
		break;
	case E_INVALIDARG:
		sprintf(err_msg, "E_INVALIDARG");
		break;
	case E_NOINTERFACE:
		sprintf(err_msg, "E_NOINTERFACE");
		break;
	case E_NOTIMPL:
		sprintf(err_msg, "E_NOTIMPL");
		break;
	case E_OUTOFMEMORY:
		sprintf(err_msg, "OUTPUTMEMORY");
		break;
#if 0
	case S_OK:
		sprintf(err_msg, "S_OK");
		break;
#endif
	case S_NOT_RESIDENT:
		sprintf(err_msg, "S_NOT_RESIDENT");
		break;
	case S_RESIDENT_IN_SHARED_MEMORY:
		sprintf(err_msg, "S_RESIDENT_IN_SHARED_MEMROY");
		break;
	case S_PRESENT_MODE_CHANGED:
		sprintf(err_msg, "S_PRESENT_MODE_CHANGED");
		break;
	case S_PRESENT_OCCLUDED:
		sprintf(err_msg, "S_PRESENT_OCLLUDED");
		break;
	case D3DERR_UNSUPPORTEDOVERLAY:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDOVERLAY");
		break;
	case D3DERR_UNSUPPORTEDOVERLAYFORMAT:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDOVERLAYFORAMT");
		break;
	case D3DERR_CANNOTPROTECTCONTENT:
		sprintf(err_msg, "D3DERR_CANNOTPROTECTCONTENT");
		break;
	case D3DERR_UNSUPPORTEDCRYPTO:
		sprintf(err_msg, "D3DERR_UNSUPPORTEDCRYPTO");
		break;
	case D3DERR_PRESENT_STATISTICS_DISJOINT:
		sprintf(err_msg, "D3DERR_PRESENT_STATISTICS_DISJOINT");
		break;
	default:
		sprintf(err_msg, "UNKNOWN RET VALUE");
	}

	infoRecorder->logError("[D3D]: return %s from file: %s, line: %d.\n", err_msg, file, line);
}