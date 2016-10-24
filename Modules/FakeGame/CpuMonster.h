#ifndef __CPUMONSTER_H__
#define __CPUMONSTER_H__

#include "../LibCore/CpuWatch.h"
#include "ResourceMonster.h"
// this file is for a procedure that eats up the CPU to given usage
class CPUMonster: public ResourceMonster{
	cg::core::CpuWatch * cpuWatch;   // moniter the cpu
public:

};

#endif