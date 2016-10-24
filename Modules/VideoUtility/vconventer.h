#ifndef __VCONVENTER_H__
#define __VCONVENTER_H__


#define CUDA_OP
#ifdef CUDA_OP
//the cuda headers
#endif
namespace cg{
	struct VConventerConfig{
		int width;
		int height;
		PixelFormat fmt;
		PixelFormat dstFmt;  // the destination format
	};

	class Conventer{
		static std::map<struct VConventerConfig, struct SwsContext *> conventers;
		static struct SwsContext * lookupFrameConventerInternal(struct VConventerConfig * ccfg);

	public :
		static struct SwsContext * lookupFrameConventer(int w, int h, PixelFormat fmt);

		static struct SwsContext * lookupFrameConventer(int w, int h, PixelFormat fmt, PixelFormat dstFmt);

		static struct SwsContext * createFrameConventer(int srcw, int srch, PixelFormat srcfmt,
			int dstw, int dsth, PixelFormat dstfmt);
		static struct SwsContext * createFrameConventer(VConventerConfig * src, VConventerConfig * dst);
	};

}


#endif