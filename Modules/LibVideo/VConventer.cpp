#include "AVCodecCommon.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"
#include "VConventer.h"

#if 1
inline bool 
operator<(const struct VConventerConfig &a,const struct VConventerConfig &b){
	if (a.width < b.width) return true;
	if (a.width > b.width) return false;
	if (a.height < b.height) return true;
	if (a.height > b.height) return false;
	if (a.fmt < b.fmt) return true;
	if (a.fmt > b.fmt) return false;
	return false;
}
#endif

map<struct VConventerConfig, struct SwsContext *> Conventer::conventers;

struct SwsContext * Conventer::lookupFrameConventerInternal(struct VConventerConfig * ccfg){
	map<struct VConventerConfig, struct SwsContext *>::iterator mi;
	if ((mi = conventers.find(*ccfg)) != conventers.end()){
		return mi->second;
	}
	return NULL;
}
struct SwsContext * Conventer::lookupFrameConventer(int w, int h, PixelFormat fmt){
	struct VConventerConfig ccfg;

	ccfg.width = w;
	ccfg.height = h;
	ccfg.fmt = fmt;

	return lookupFrameConventerInternal(&ccfg);

}

struct SwsContext * Conventer::lookupFrameConventer(int w, int h, PixelFormat fmt, PixelFormat dstFmt){
	struct VConventerConfig ccfg;

	ccfg.width = w;
	ccfg.height = h;
	ccfg.fmt = fmt;
	ccfg.dstFmt = dstFmt;
	return lookupFrameConventerInternal(&ccfg);
}

struct SwsContext * Conventer::createFrameConventer(int srcw, int srch, PixelFormat srcfmt,
	int dstw, int dsth, PixelFormat dstfmt){
	map<struct VConventerConfig, struct SwsContext * >::iterator mi;
	struct VConventerConfig ccfg;
	struct SwsContext * ctx;

	ccfg.width = srcw;
	ccfg.height = srch;
	ccfg.fmt = srcfmt;
	//DebugBreak();
	if ((ctx = lookupFrameConventerInternal(&ccfg)) != NULL){
		return ctx;
	}
	if ((ctx = AVCodecCommon::SwScaleInit(srcfmt, srcw, srch, dstfmt, dstw, dsth)) == NULL){
		infoRecorder->logError("[conventer]: SwScaleInit failed.\n");
		return NULL;

	}

	conventers[ccfg] = ctx;
	infoRecorder->logTrace("Frame Conventer created: from (%d, %d)[%d] -> (%d, %d)[%d]\n",
		(int)srcw, (int)srch, (int)srcfmt, (int)dstw, (int)dsth, (int)dstfmt);
	return ctx;
}
struct SwsContext * Conventer::createFrameConventer(VConventerConfig * src, VConventerConfig * dst){
	map< struct VConventerConfig, struct SwsContext * >::iterator mi;
	struct SwsContext * ctx;

	if ((ctx = lookupFrameConventerInternal(src)) != NULL){
		return NULL;
	}
	if ((ctx = AVCodecCommon::SwScaleInit(src->fmt, src->width, src->height, dst->width, dst->height)) == NULL)
		return NULL;

	conventers[*src] = ctx;
	
	return ctx;
}