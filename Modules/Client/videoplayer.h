#ifndef __VIDEOPLAYER_H__
#define __VIDEOPLAYER_H__

// a video player, the video is the rtsp frame, first, store the data into pipeline, then a seperate thread
// will player the video,
// note: the video pipeline is sequenced according to timestamp
// frame will be dropped when the timestamp is too old

// the video pipeline

struct FrameData{
	struct timeval pts;
	unsigned char * frameData;
	struct FrameData * next;
};

class FramePipeline{

};


#endif