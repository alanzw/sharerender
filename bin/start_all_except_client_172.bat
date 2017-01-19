start DisManager.exe config/server.distributor.conf
ping -n 3 127.0.0.1>null
start GameLoader.exe config/gameloader.conf
#ping -n 3 127.0.0.1>null
start RenderProxy.exe -m 0 -c config/server.render.conf -r 1
::start RenderProxyLoader.exe

::ping -n 3 127.0.0.1>null
::start client config/client.ref.conf 192.168.100.100 Trine
