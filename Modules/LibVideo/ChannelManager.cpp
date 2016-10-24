#include "Commonwin32.h"
#include "Pipeline.h"

#include "Config.h"
#include "RtspConf.h"
#include "RtspContext.h"
#include "../LibCore/Log.h"
#include "../LibCore/InfoRecorder.h"
#include "Encoder.h"
#include "FilterRGB2YUV.h"
#include "VSource.h"
#include "ChannelManager.h"

int ChannelManager::channels = 0;
ChannelManager * ChannelManager::manager;// = NULL;

ChannelManager * ChannelManager::GetChannelManager(){
	if (!manager){
		return new ChannelManager();
	}
	else
		return manager;
}

bool ChannelManager::ReleaseChannelManager(){
	// release all the channel
	for (int i = 0; i < channels; i++)
		delete channelArray[i];
	// release the manager
	//~ChannelManager();
	return true;
}

bool ChannelManager::init(int maxWidth, int maxHeight, int maxStride){
	this->maxWidth = maxWidth;
	this->maxHeight = maxHeight;
	this->maxStride = maxStride;
	return true;
}
// release all
bool ChannelManager::Release(){
	if (manager){
		manager->ReleaseChannelManager();
		manager = NULL;
	}
	return true;
}

bool ChannelManager::release(int channelId){
	if (channelId > channels){
		infoRecorder->logError("release the channel %d is failed, max channel id is:%d", channelId, channels);
		return false;
	}
	else
		delete channelArray[channelId];
	return true;
}

//int ChannelManager::setupEx(const char * pipeFormat, struct VsourceConfig **config, int nConfig = 0){}

Channel * ChannelManager::getChannel(int channelId){
	if (channelId > channels){
		infoRecorder->logError("get the channel source %d failed, the max channel id is %d\n", channelId, channels);
		return NULL;
	}
	else
		return channelArray[channelId];
}

int ChannelManager::getMaxWidth(int channelId){
	if (channelId > channels){
		infoRecorder->logError("get the channel source %d failed, the max channel id is %d\n", channelId, channels);
		return -1;
	}
	else
		return channelArray[channelId]->getMaxWidth();
}

int ChannelManager::getMaxHeight(int channelId){
	if (channelId > channels){
		infoRecorder->logError("get the channel source %d failed, the max channel id is %d\n", channelId, channels);
		return -1;
	}
	else
		return channelArray[channelId]->getMaxHeight();
}

int ChannelManager::getMaxStride(int channelId){
	if (channelId > channels){
		infoRecorder->logError("get the channel source %d failed, the max channel id is %d\n", channelId, channels);
		return -1;
	}
	else
		return channelArray[channelId]->getMaxStride();
}

int ChannelManager::videoSourceChannels(){
	return channels;
}


Channel * ChannelManager::getNewChannel(){
	int channelId = this->channels++;
	ENCODER_TYPE type = this->getNewChannelType();
	Channel * chn = new Channel(type);  // create a new channel
	channelArray[channelId] = chn;  // add to map
	return chn;
}

ENCODER_TYPE ChannelManager::getNewChannelType(){

	return ENCODER_TYPE::X264_ENCODER;
}