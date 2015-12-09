This project is for the games that do not use GPU.

The project contains 
	1. a 2D game loader to start the game, 
	2. a game spider to correct the path problem

the game loader and the game spider will be added to Loader and  RenderProxy.
the whole system contains three types of domains, the task scheduler, the Logic Server and the Render Proxy. However, the Task Scheduler will never responsible for running a game, so that, Logic Server and the Render Proxy will be able to run any kind of games.