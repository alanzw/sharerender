#include "CNvEncoder.h"
namespace cg{
	namespace nvenc{
		CNvEncoder::~CNvEncoder(){
			if(m_pNvHWEncoder){
				delete m_pNvHWEncoder;
				m_pNvHWEncoder = NULL;
			}

		}
	}
}