#ifndef __VIDEOWRITER_H__
#define __VIDEOWRITER_H__

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "../VideoGen/rtspcontext.h"
#include "../LibCore/TimeTool.h"

namespace cg{
	class VideoWriter{
	protected:
		static HANDLE			syncMutex;
		static bool				syncReset;
		static struct timeval	syncTv;
		long long				basePts, newPts, pts, ptsSync;    // PTS sync
		unsigned int			tags;
		unsigned char			specialTag;   // used for response delay
		bool					specialTagValid;
		unsigned char			valueTag;

		bool					isChanged;
		DWORD					encodeStart, encodeEnd;


		bool					enableWriteToNet;
		bool					enableWriteToFile;
		FILE *					m_fOutput;
		cg::rtsp::RTSPContext *	ctx;
		cg::core::PTimer *		pTimer;
		UINT					writeTime;

		int						writeToFile(FILE * file, AVPacket * pkt);
		int						sendPacket(int channelId, cg::rtsp::RTSPContext * rtsp, AVPacket * pkt, int64_t encoderPts);
		void					print();
		
		////////////////// public functions /////////////////
	public:

		inline float getWriteTime(){ return (float)1000.0 * writeTime / pTimer->getFreq(); }

		// help changing encoder
		inline void setChanged(bool val){ isChanged = val; encodeStart = GetTickCount(); }

		VideoWriter(): m_fOutput(NULL), enableWriteToFile(false), enableWriteToNet(false),ctx(NULL), specialTag(0),specialTagValid(false), valueTag(0){
			print();
		}
		VideoWriter(FILE * out): m_fOutput(out), ctx(NULL), enableWriteToFile(true), enableWriteToNet(false), specialTag(0),specialTagValid(false), valueTag(0){
			print();
		}

		VideoWriter(cg::rtsp::RTSPContext * _ctx): ctx(_ctx), m_fOutput(NULL), enableWriteToNet(true), enableWriteToFile(false), pTimer(NULL), writeTime(0), specialTag(0),specialTagValid(false), valueTag(0){
			print();
			pTimer = new cg::core::PTimer();
		}
		VideoWriter(FILE *out, cg::rtsp::RTSPContext *_ctx): ctx(_ctx), m_fOutput(out), enableWriteToFile(true), enableWriteToNet(true), pTimer(NULL), writeTime(0), specialTag(0),specialTagValid(false), valueTag(0){
			print();
			pTimer = new cg::core::PTimer();
		}


		virtual ~VideoWriter(){ 
			if(m_fOutput){
				fflush(m_fOutput); 
				fclose(m_fOutput); 
				m_fOutput = NULL;
			}
			if(pTimer){
				delete pTimer; 
				pTimer = NULL;
			}
		}
		inline unsigned char	getSpecialTag(){ return specialTag; }
		void					setSpecialTag(unsigned char val);
		void					setValueTag(unsigned char tag);
		int						ptsSynchronize(int sampleRate);
		virtual int				sendPacket(int channelId, AVPacket * pkt, int64_t encoderPts);
		int64_t					updataPts(int64_t basePts, float fps = (30.0f) );
		int64_t					getUpdatedPts(){ return pts; }
		void					setTags(unsigned int tag){ tags =  tag; }
	};

}


#endif // __VIDEOWRITER_H__
