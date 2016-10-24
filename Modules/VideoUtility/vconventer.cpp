#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "avcodeccommon.h"
#include "vconventer.h"
#include "../LibCore/InfoRecorder.h"

namespace cg{
#if 1
	inline bool 
		operator<(const struct VConventerConfig &a,const struct VConventerConfig &b){
			if (a.width < b.width) return true;
			if (a.width > b.width) return false;
			if (a.height < b.height) return true;
			if (a.height > b.height) return false;
			if (a.fmt < b.fmt) return true;
			if (a.fmt > b.fmt) return false;
			if (a.dstFmt < b.dstFmt) return true;
			if (a.dstFmt > b.dstFmt) return false;
			return false;
	}
#endif

	std::map<struct VConventerConfig, struct SwsContext *> Conventer::conventers;

	struct SwsContext * Conventer::lookupFrameConventerInternal(struct VConventerConfig * ccfg){
		cg::core::infoRecorder->logTrace("[conventer]: look up frame conventer internal.\n");
		std::map<struct VConventerConfig, struct SwsContext *>::iterator mi;
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
			std::map<struct VConventerConfig, struct SwsContext * >::iterator mi;
			struct VConventerConfig ccfg;
			struct SwsContext * ctx;

			ccfg.width = srcw;
			ccfg.height = srch;
			ccfg.fmt = srcfmt;
			ccfg.dstFmt = dstfmt;

			cg::core::infoRecorder->logTrace("[conventer]: creat frame conventor src (%d x %d) to dst (%d x %d), dst format:%s.\n", srcw, srch, dstw, dsth, dstfmt == PIX_FMT_NV12 ? "NV12": "YUV420P");

			//DebugBreak();
			if ((ctx = lookupFrameConventerInternal(&ccfg)) != NULL){
				cg::core::infoRecorder->logError("[conventer]: the conventer config exist.\n");
				return ctx;
			}
			if ((ctx = AVCodecCommon::SwScaleInit(srcfmt, srcw, srch, dstfmt, dstw, dsth)) == NULL){
				cg::core::infoRecorder->logError("[conventer]: SwScaleInit failed.\n");
				return NULL;

			}

			conventers[ccfg] = ctx;
			cg::core::infoRecorder->logTrace("Frame Conventer created: from (%d, %d)[%d] -> (%d, %d)[%d]\n",
				(int)srcw, (int)srch, (int)srcfmt, (int)dstw, (int)dsth, (int)dstfmt);
			cg::core::infoRecorder->logTrace("[conventer]: swscontext:%p.\n", ctx);
			return ctx;
	}
	struct SwsContext * Conventer::createFrameConventer(VConventerConfig * src, VConventerConfig * dst){
		std::map< struct VConventerConfig, struct SwsContext * >::iterator mi;
		struct SwsContext * ctx;

		if ((ctx = lookupFrameConventerInternal(src)) != NULL){
			return NULL;
		}
		if ((ctx = AVCodecCommon::SwScaleInit(src->fmt, src->width, src->height, dst->width, dst->height)) == NULL)
			return NULL;

		conventers[*src] = ctx;

		return ctx;
	}

}