#ifndef __VIDEOWRITER_H__
#define __VIDEOWRITER_H__

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "../VideoGen/rtspcontext.h"


namespace cg{
	class VideoWriter{
	protected:
		static HANDLE			syncMutex;
		static bool				syncReset;
		static struct timeval	syncTv;
		long long				basePts, newPts, pts, ptsSync;    // PTS sync
		unsigned int			tags;

		bool					isChanged;
		DWORD					encodeStart, encodeEnd;


		bool					enableWriteToNet;
		bool					enableWriteToFile;
		FILE *					m_fOutput;
		cg::rtsp::RTSPContext *	ctx;

		int						writeToFile(FILE * file, AVPacket * pkt);
		int						sendPacket(int channelId, cg::rtsp::RTSPContext * rtsp, AVPacket * pkt, int64_t encoderPts);
		void					print();

		////////////////// public functions /////////////////
	public:
		// help changing encoder
		inline void setChanged(bool val){ isChanged = val; encodeStart = GetTickCount(); }

		VideoWriter(): m_fOutput(NULL), enableWriteToFile(false), enableWriteToNet(false),ctx(NULL){
			print();
		}
		VideoWriter(FILE * out): m_fOutput(out), ctx(NULL), enableWriteToFile(true), enableWriteToNet(false){
			print();
		}

		VideoWriter(cg::rtsp::RTSPContext * _ctx): ctx(_ctx), m_fOutput(NULL), enableWriteToNet(true), enableWriteToFile(false){
			print();
		}
		VideoWriter(FILE *out, cg::rtsp::RTSPContext *_ctx): ctx(_ctx), m_fOutput(out), enableWriteToFile(true), enableWriteToNet(true){
			print();
		}


		virtual ~VideoWriter(){ if(m_fOutput){fflush(m_fOutput); fclose(m_fOutput); m_fOutput = NULL;}}

		int						ptsSynchronize(int sampleRate);
		virtual int				sendPacket(int channelId, AVPacket * pkt, int64_t encoderPts);
		int64_t					updataPts(int64_t basePts, float fps = (30.0f) );
		int64_t					getUpdatedPts(){ return pts; }
		void					setTags(unsigned int tag){ tags =  tag; }
	};

}


#endif // __VIDEOWRITER_H__
