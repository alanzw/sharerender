#ifndef __RESOURCEMONSTER_H__
#define __RESOURCEMONSTER_H__

// the base for resource monster, which eats up the resource to give usage

class ResourceMonster{
	float usage;

public:

	ResourceMonster(){
		usage = 0.0f;
	}
	inline float getUsage(){ return usage; }
	inline void setUsage(float u){ usage = u; }

};

#endif