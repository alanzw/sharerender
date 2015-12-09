#ifndef __CHANNELMANAGER_H__
#define __CHANNELMANAGER_H__


class ChannelManager{
	static int channels;

	int maxWidth, maxHeight, maxStride;
	//Channel ** channelArray;   // array for many channels
	map<int, Channel *> channelArray;
	map<RTSPContext *, Channel *> contextMap;

	static ChannelManager * manager;
	ChannelManager(){}
	~ChannelManager(){}

public:
	//ChannelManager();

	//~ChannelManager();
	//static ChannelManager * CreateChannelManager();
	static ChannelManager * GetChannelManager();
	bool ReleaseChannelManager();

	bool init(int maxWidth, int maxHeight, int maxStride);
	static bool Release();   // release all
	bool release(int id);   // release a channel
	int setupEx(const char * pipeFormat, struct VsourceConfig ** config, int nConfig = 0);

	Channel * getChannel(int channelId);

	int getMaxWidth(int channelId);
	int getMaxHeight(int channelId);
	int getMaxStride(int channelId);

	static int videoSourceChannels();

	// determine the channel charactors
	ENCODER_TYPE getNewChannelType(); // the encoder type for the 
	// create new channel
	Channel * getNewChannel();
	// thread functions
	void mapRtspContext(Channel * channel, RTSPContext * context){
		contextMap[context] = channel;
	}
};

#endif