#ifndef __CMDCONTROLLER_H__
#define __CMDCONTROLLER_H__
// cmd controller for dis manager, recv the cmd from console and do the work

#include "../LibDistrubutor/Distributor.h"


	class CmdDealer{

		static CmdDealer * contorller;

		cg::DisServer * server;

	public:
		inline void setServer(cg::DisServer * _server){ server = _server; }
		static CmdDealer* GetDealer(){
			if(!contorller){
				contorller = new CmdDealer();

			}
			return contorller;
		}
		void print();
		bool addRender(int id);
		bool changeEncoder(int id);
	};



#endif