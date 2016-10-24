#ifndef __FAKEGAME_H__
#define __FAKEGAME_H__

// this is the fake game, we need a game program which can occupy cpu and gpu to the given value with given parameters


// include the resouce monitor since we need to monitor hardware usage
#include "../LibCore/CpuWatch.h"
#include "../LibCore/GpuWatch.h"
#include "CpuMonster.h"
#include "GpuMonster.h"

// need a recorder to record the cpu/gpu usage, the desired one, and the drawing parameter which controls the usage

// the config class for parameters, read from a config file
// it's the parameters given from different cpu and gpu device

class GameConfig{
	char * gpuName;



};

class GameEntity{
	int cpuLoad;
	int gpuLoad;

	float cpuUsage;
	float gpuUsage;
	float cpuDesire;
	float gpuDesire;

	GameConfig config;
};

#endif