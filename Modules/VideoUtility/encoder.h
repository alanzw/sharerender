#ifndef __ENCODER_H__
#define __ENCODER_H__
#include <cuda.h>
#include "helper_cuda_drvapi.h"

#include "videocommon.h"
#include "..\LibCore\CThread.h"
#include "..\VideoUtility\VideoWriter.h"
#include "..\LibCore\TimeTool.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include "pipeline.h"

// this file is for the generous encoder
//int ffio_open_dyn_packet_buf(AVIOContext **, int);

// this will output the proper CUDA error string in the event that a CUDA host call returns an error
//#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
typedef DWORD (WINAPI *EncoderThreadProc)(_In_  LPVOID lpParameter);

#define	IMAGE_SOURCE_CHANNEL_MAX	4	// temp define here
#define ENCODER_POOL_SIZE			4	// encoder pool size 

namespace cg{

	class Encoder: public cg::core::CThread{
	protected:
		bool				assigned;

		pipeline *			sourcePipe;
		int					id;				// the id of the encoder

		int					encoderWidth, encoderHeight;
		ENCODER_TYPE		useEncoderType;	// use the x264 encoder or the cuda encoder
		//SOURCE_TYPE			useSourceType;	// use source type, IMAGE or SURFACE
		HANDLE				cond, condMutex;// notifier and mutex
		bool				inited;
		VideoWriter*		writer;
		
	protected:
		cg::core::PTimer *	pTimer;
		cg::core::PTimer *	refIntraMigrationTimer;   // set to NULL every time it is used
		UINT				encodeTime;
		UINT				packTime;

		// functions
		virtual  pooldata * loadFrame();
		virtual void		releaseData(struct pooldata * data);

	public:
		inline float		getEncodeTime(){ return (float)1000.0 * encodeTime / pTimer->getFreq(); }
		virtual void		setRefIntraMigrationTimer(cg::core::PTimer * refTimer){ refIntraMigrationTimer = refTimer; }

		Encoder(int _id, int _height, int _width,/* SOURCE_TYPE srcType, */ENCODER_TYPE _type, pipeline * _imgPipe, VideoWriter *_writer);
		virtual ~Encoder();

		// thread functions

		virtual BOOL		run() = 0;
		virtual void		onThreadMsg(UINT msg, WPARAM wParam, LPARAM lParam) = 0;
		virtual BOOL		onThreadStart() = 0;
		virtual void		onQuit();
		

		inline ENCODER_TYPE getEncoderType(){		return useEncoderType; }
		void				registerEvent();
		void				unregisterEvent();
		inline bool			isAssigned(){			return assigned; }
		inline bool			isInited(){				return inited; }


		inline void			encoderChanged(){
			if(writer){
				writer->setChanged(true);
			}
		}
	};
}
#endif
